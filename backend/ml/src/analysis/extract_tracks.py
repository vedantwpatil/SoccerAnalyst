import csv
from ultralytics import YOLO
import supervision as sv
from tqdm import tqdm

VIDEO_PATH = "../../data/08fd33_4.mp4"
MODEL_PATH = "../../data/best.pt"
OUTPUT_CSV = "tracks_raw.csv"  # Output file
CONFIDENCE = 0.25  # Keep low for ByteTrack to work
IMG_SIZE = 1280  # Critical for small objects (ball)


def main():
    model = YOLO(MODEL_PATH)

    with open(OUTPUT_CSV, mode="w", newline="") as f:
        writer = csv.writer(f)
        # Write Header
        writer.writerow(
            ["frame", "track_id", "class_id", "x1", "y1", "x2", "y2", "conf"]
        )

        # stream=True creates a generator (memory efficient)
        # persist=True keeps the tracker state alive between frames
        results_generator = model.track(
            source=VIDEO_PATH,
            stream=True,
            persist=True,
            tracker="bytetrack.yaml",  # Built-in Ultralytics tracker config
            conf=CONFIDENCE,
            imgsz=IMG_SIZE,
            verbose=False,  # Silence the per-frame logging
        )

        print(f"Processing {VIDEO_PATH}...")
        for frame_idx, result in enumerate(tqdm(results_generator)):
            detections = sv.Detections.from_ultralytics(result)

            for i in range(len(detections)):
                # Parse fields
                box = detections.xyxy[i]  # [x1, y1, x2, y2]
                cls_id = int(detections.class_id[i])
                conf = float(detections.confidence[i])

                # Get Track ID (might be None if tracker failed to associate)
                # We default to -1 for untracked objects (like the ball sometimes)
                track_id = (
                    int(detections.tracker_id[i])
                    if detections.tracker_id is not None
                    else -1
                )

                # Format: frame, track_id, class_id, x1, y1, x2, y2, conf
                writer.writerow(
                    [
                        frame_idx,
                        track_id,
                        cls_id,
                        round(box[0], 2),
                        round(box[1], 2),
                        round(box[2], 2),
                        round(box[3], 2),
                        round(conf, 4),
                    ]
                )

    print(f"Done! Raw tracks saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
