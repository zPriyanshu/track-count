import cv2
import json
import csv
import os
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


INPUT_VIDEO   = "data/xyz.mp4"     
MODEL_PATH    = "model/yolo11x.pt" 
TRACKER       = "my_tracker.yaml"   # Tracker = bytetrack (Every frame of video)
CONF_THRESH   = 0.35              # Detection confidence threshold         
IOU_THRESH    = 0.50               # Intersection over Union threshold     

DEBUG = False

VEHICLE_CLASSES = {"car", "truck", "motorcycle", "bicycle","bus"}

# Output paths
OUTPUT_VIDEO_PATH  = "output/output_detected.mp4"    
OUTPUT_JSON_PATH   = "output/vehicle_count.json"     
OUTPUT_CSV_PATH    = "output/vehicle_count.csv"


def save_results(count: int, class_counts: dict, input_video: str):
   
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
    if not os.path.exists(INPUT_VIDEO):
        print(f"[ERROR] Input video not found: {INPUT_VIDEO}")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model weights not found: {MODEL_PATH}")
        return

    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("[INFO] Model loaded successfully.")

    cap_probe = cv2.VideoCapture(INPUT_VIDEO)
    frame_width  = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_probe.release()

    print(f"[INFO] Video: {frame_width}x{frame_height} @ {fps:.1f} FPS | {total_frames} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
    print(f"[INFO] Output video will be saved to: {OUTPUT_VIDEO_PATH}")

    vehicle_count  = 0                
    counted_ids    = set()            
    class_counts   = {}              
    frame_idx      = 0


    print("[INFO] Starting vehicle tracking... Press ESC to quit early.")

    results_stream = model.track(
        source  = INPUT_VIDEO,
        persist = True,             # Persist tracker state across frames
        conf    = CONF_THRESH,
        iou     = IOU_THRESH,
        tracker = TRACKER,
        stream  = True,            
    )

    for result in results_stream:
        frame_idx += 1

        frame = result.orig_img.copy()
        
        if result.boxes.id is not None:
            boxes   = result.boxes.xyxy.cpu().numpy()
            ids     = result.boxes.id.cpu().numpy().astype(int) 
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for box, obj_id, cls_idx in zip(boxes, ids, classes):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2   
                cy = (y1 + y2) // 2   

                class_name = model.names[cls_idx]   

                if VEHICLE_CLASSES and class_name not in VEHICLE_CLASSES:
                    continue

                if obj_id not in counted_ids:
                    vehicle_count += 1
                    counted_ids.add(obj_id)
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    print(f"[COUNT] Frame {frame_idx:4d} | ID {obj_id:3d} | "
                          f"{class_name} | NEW | Total: {vehicle_count}")

                if DEBUG:
                    print(f"  [DBG] Frame {frame_idx:4d} | ID {obj_id:3d} | "
                          f"{class_name} | cx={cx} cy={cy}")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID:{obj_id} {class_name}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 65), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, f"Total Vehicles: {vehicle_count}",
                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)


        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}",
                    (frame_width - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        writer.write(frame)

        try:
            cv2.imshow("Vehicle Detection & Counting", frame)
            if cv2.waitKey(1) & 0xFF == 27:   # ESC to exit early
                print("[INFO] Early exit requested by user.")
                break
        except cv2.error:
            pass 


    writer.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] Processed {frame_idx} frames.")
    print(f"[DONE] Total vehicles counted: {vehicle_count}")
    print(f"[DONE] By class: {class_counts}")
    print(f"[DONE] Annotated video saved → {OUTPUT_VIDEO_PATH}")


    save_results(vehicle_count, class_counts, INPUT_VIDEO)







if __name__ == "__main__":
    run_detection()



