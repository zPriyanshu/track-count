import cv2
import json
import csv
import os
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


INPUT_VIDEO   = "data/a.mp4"       # Source video file
MODEL_PATH    = "model/yolo11m.pt" # YOLO model
TRACKER       = "bytetrack.yaml"   # Tracker = bytetrack (Every frame of video)
CONF_THRESH   = 0.25               # Detection confidence threshold
IOU_THRESH    = 0.50               # Intersection over Union threshold 

# Counting line Y-coordinate (horizontal line vehicles must cross to be counted)

COUNTING_LINE_Y = None   # ← None = auto-center; or set e.g. 287 manually

# BOTH directions (True) or only top→bottom (False)
COUNT_BOTH_DIRECTIONS = True

DEBUG = False

# Only count these YOLO class names (None = count all detected objects)
VEHICLE_CLASSES = {"car", "truck","motorcycle", "bicycle"}      #  "bus", 

# Output paths
OUTPUT_VIDEO_PATH  = "output/output_detected.mp4"    # Annotated video
OUTPUT_JSON_PATH   = "output/vehicle_count.json"     # Count summary (JSON)
OUTPUT_CSV_PATH    = "output/vehicle_count.csv"      # Count summary (CSV)


def save_results(count: int, class_counts: dict, input_video: str):
    """
    Saves the vehicle count summary to both JSON and CSV files.

    Args:
        count        : Total number of unique vehicles that crossed the line.
        class_counts : Dict of {class_name: count} for each vehicle type.
        input_video  : Name of the source video file (stored in the report).
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── JSON ──────────────────────────────────
    json_data = {
        "timestamp"    : timestamp,
        "source_video" : input_video,
        "total_vehicles_counted": count,
        "by_class"     : class_counts,
    }
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(json_data, f, indent=4)
    print(f"[INFO] Count summary saved → {OUTPUT_JSON_PATH}")

    # ── CSV ───────────────────────────────────
    file_exists = os.path.isfile(OUTPUT_CSV_PATH)
    with open(OUTPUT_CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Source Video", "Total Count", "By Class"])
        writer.writerow([timestamp, input_video, count, str(class_counts)])
    print(f"[INFO] Count summary appended → {OUTPUT_CSV_PATH}")


# ─────────────────────────────────────────────




def run_detection():
    """
    Main pipeline:
      1. Load YOLO model
      2. Stream tracking results frame-by-frame
      3. Draw bounding boxes, IDs, and counting line on each frame
      4. Count vehicles crossing the line (one count per unique tracker ID)
      5. Write annotated frames to output video
      6. Save count summary to JSON + CSV
    """

    # ── Step 1: Validate Input ─────────────────
    if not os.path.exists(INPUT_VIDEO):
        print(f"[ERROR] Input video not found: {INPUT_VIDEO}")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model weights not found: {MODEL_PATH}")
        return

    # ── Step 2: Load YOLO Model ────────────────
    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("[INFO] Model loaded successfully.")

    # ── Step 3: Get Video Metadata for Writer ──
    cap_probe = cv2.VideoCapture(INPUT_VIDEO)
    frame_width  = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_probe.release()

    print(f"[INFO] Video: {frame_width}x{frame_height} @ {fps:.1f} FPS | {total_frames} frames")

    # ── Auto-set counting line if not manually specified ─
    global COUNTING_LINE_Y
    if COUNTING_LINE_Y is None:
        COUNTING_LINE_Y = frame_height // 2
        print(f"[INFO] COUNTING_LINE_Y auto-set to vertical center: {COUNTING_LINE_Y}")
    else:
        print(f"[INFO] COUNTING_LINE_Y set to: {COUNTING_LINE_Y}")

    # ── Step 4: Setup Video Writer ─────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
    print(f"[INFO] Output video will be saved to: {OUTPUT_VIDEO_PATH}")

    # ── Step 5: Tracking State Variables ───────
    vehicle_count  = 0                # Total vehicles that crossed the line
    tracker_memory = {}               # {tracker_id: previous_cy} — tracks last Y position
    counted_ids    = set()            # Set of tracker IDs already counted
    class_counts   = {}               # {class_name: count} per vehicle type
    frame_idx      = 0

    # ── Step 6: Stream Model Tracking ──────────
    print("[INFO] Starting vehicle tracking... Press ESC to quit early.")

    results_stream = model.track(
        source  = INPUT_VIDEO,
        persist = True,             # Persist tracker state across frames
        conf    = CONF_THRESH,
        iou     = IOU_THRESH,
        tracker = TRACKER,
        stream  = True,             # IMPORTANT: stream=True avoids RAM accumulation
    )

    for result in results_stream:
        frame_idx += 1
        # Get the original frame (numpy array, BGR)
        frame = result.orig_img.copy()
        
        # ── Per-Frame: Process Detections ───────
        if result.boxes.id is not None:
            boxes   = result.boxes.xyxy.cpu().numpy()           # [x1,y1,x2,y2]
            ids     = result.boxes.id.cpu().numpy().astype(int) # Tracker IDs
            classes = result.boxes.cls.cpu().numpy().astype(int)# Class indices

            for box, obj_id, cls_idx in zip(boxes, ids, classes):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2   # Bounding box center X
                cy = (y1 + y2) // 2   # Bounding box center Y

                class_name = model.names[cls_idx]   # e.g. "car", "truck"

                # Skip non-vehicle classes if filter is active
                if VEHICLE_CLASSES and class_name not in VEHICLE_CLASSES:
                    continue

                # ── Crossing Line Logic ──────────────────────────────────
                # A vehicle is counted when its center (cy) crosses COUNTING_LINE_Y.
                # COUNT_BOTH_DIRECTIONS=True  → count top→bottom AND bottom→top
                # COUNT_BOTH_DIRECTIONS=False → count only top→bottom crossings
                if obj_id in tracker_memory:
                    prev_cy = tracker_memory[obj_id]

                    crossed = False
                    direction = ""

                    # Top → Bottom crossing (vehicle moves downward in frame)
                    if prev_cy < COUNTING_LINE_Y and cy >= COUNTING_LINE_Y:
                        crossed = True
                        direction = "↓ DOWN"

                    # Bottom → Top crossing (vehicle moves upward in frame)
                    elif COUNT_BOTH_DIRECTIONS and prev_cy > COUNTING_LINE_Y and cy <= COUNTING_LINE_Y:
                        crossed = True
                        direction = "↑ UP"
                        
                    if crossed and obj_id not in counted_ids:
                        vehicle_count += 1
                        counted_ids.add(obj_id)
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                        print(f"[COUNT] Frame {frame_idx:4d} | ID {obj_id:3d} | "
                              f"{class_name} | {direction} | Total: {vehicle_count}")

                # Debug: print cy of every tracked vehicle so you can verify line position
                if DEBUG:
                    print(f"  [DBG] Frame {frame_idx:4d} | ID {obj_id:3d} | "
                          f"{class_name} | cy={cy} | line_y={COUNTING_LINE_Y}")

                # Store last known Y position for next frame comparison
                tracker_memory[obj_id] = cy

                # ── Draw Bounding Box & Label ────
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID:{obj_id} {class_name}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw center point of bounding box
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # ── Draw Counting Line ───────────────────
        # Yellow horizontal line that vehicles must cross to be counted
        cv2.line(frame,
                 (0, COUNTING_LINE_Y),
                 (frame_width, COUNTING_LINE_Y),
                 (0, 255, 255), 2)

        # ── Draw HUD Overlay ─────────────────────
        # Semi-transparent background for readability
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 65), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, f"Total Vehicles: {vehicle_count}",
                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # Frame progress indicator
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}",
                    (frame_width - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # ── Write Frame to Output Video ──────────
        writer.write(frame)

        # ── Live Preview Window ──────────────────
        # Wrapped in try/except so the script still works in headless environments
        try:
            cv2.imshow("Vehicle Detection & Counting", frame)
            if cv2.waitKey(1) & 0xFF == 27:   # ESC to exit early
                print("[INFO] Early exit requested by user.")
                break
        except cv2.error:
            pass  # No display available (e.g. running via terminal without GUI)

    # ── Step 7: Cleanup ─────────────────────────
    writer.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] Processed {frame_idx} frames.")
    print(f"[DONE] Total vehicles counted: {vehicle_count}")
    print(f"[DONE] By class: {class_counts}")
    print(f"[DONE] Annotated video saved → {OUTPUT_VIDEO_PATH}")

    # ── Step 8: Save Results to File ────────────
    save_results(vehicle_count, class_counts, INPUT_VIDEO)







if __name__ == "__main__":
    run_detection()






# COUNTING_LINE_Y = 287 was hardcoded but the video is only 
# 360px tall — so cy values ranged
#  120–340, vehicles crossed fine but the OLD single-direction
#  logic had an inversion issue	    

# Solution= frame_height // 2 = 180

# I was counting vehicles from top to bottom, but the vehicles were coming from bottom to top, so thats why its was showing 0 vehicles counted.