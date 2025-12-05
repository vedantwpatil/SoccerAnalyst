import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

# --- CONFIGURATION ---
VIDEO_PATH = "../../data/Inter v Barcelona 2nd League  UCL  Tactical Camera Full Match - THOLIUNATHI23 (720p, h264).mp4"
INPUT_CSV = "tracks_raw.csv"
OUTPUT_CSV = "tracks_pitch.csv"
MODEL_KEYPOINT_PATH = "../../data/keypoint_best.pt"

# Physical World Coordinates (Meters)
# We define a standard pitch dictionary to map detected classes to real locations (FIFA Standard 105x68m)
# Key: Class ID from the model, Value: (x, y) in meters
# Origin (0,0) is Top-Left Corner Flag
PITCH_KEYPOINTS_MAP = {
    0: (0, 0),  # Top-Left Corner
    1: (0, 13.84),  # Left Penalty Box Top (Goal Line)
    2: (16.5, 13.84),  # Left Penalty Box Top Corner
    3: (16.5, 54.16),  # Left Penalty Box Bottom Corner
    4: (0, 54.16),  # Left Penalty Box Bottom (Goal Line)
    5: (0, 24.84),  # Left Goal Area Top (Goal Line)
    6: (5.5, 24.84),  # Left Goal Area Top Corner
    7: (5.5, 43.16),  # Left Goal Area Bottom Corner
    8: (0, 43.16),  # Left Goal Area Bottom (Goal Line)
    9: (0, 68),  # Bottom-Left Corner
    10: (11.0, 34),  # Left Penalty Spot (approx)
    # Center Line (x=52.5)
    11: (52.5, 0),  # Center Line Top
    12: (52.5, 68),  # Center Line Bottom
    13: (52.5, 24.85),  # Center Circle Top (y=34-9.15)
    14: (52.5, 43.15),  # Center Circle Bottom (y=34+9.15)
    15: (52.5, 34),  # Center Spot (Kickoff)
    # Right Side (x=105)
    16: (105, 0),  # Top-Right Corner
    17: (105, 13.84),  # Right Penalty Box Top (Goal Line)
    18: (88.5, 13.84),  # Right Penalty Box Top Corner (105-16.5)
    19: (88.5, 54.16),  # Right Penalty Box Bottom Corner
    20: (105, 54.16),  # Right Penalty Box Bottom (Goal Line)
    21: (105, 24.84),  # Right Goal Area Top
    22: (99.5, 24.84),  # Right Goal Area Top Corner (105-5.5)
    23: (99.5, 43.16),  # Right Goal Area Bottom Corner
    24: (105, 43.16),  # Right Goal Area Bottom
    25: (105, 68),  # Bottom-Right Corner
    26: (94.0, 34),  # Right Penalty Spot (approx)
    # Center Circle Sides
    27: (43.35, 34),  # Center Circle Left (52.5-9.15)
    28: (61.65, 34),  # Center Circle Right (52.5+9.15)
}

# The 4 Virtual Corners we want to track/smooth (Even if off-screen)
VIRTUAL_CORNERS_WORLD = np.array(
    [
        [0, 0],  # TL
        [0, 68],  # BL
        [105, 68],  # BR
        [105, 0],  # TR
    ],
    dtype=np.float32,
)


class AdaptiveHomographySmoother:
    """
    Stabilizes the camera view using a Kalman Filter on the 4 VIRTUAL pitch corners.
    """

    def __init__(self, init_src_pts, process_noise=1e-4, measurement_noise=1e-2):
        # State: [x1, y1, x2, y2, x3, y3, x4, y4] (8 variables)
        self.kf = cv2.KalmanFilter(8, 8, 0)
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        self.kf.measurementMatrix = np.eye(8, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(8, dtype=np.float32) * measurement_noise

        # Initialize State with the first frame's detection
        flat_pts = init_src_pts.flatten()
        self.kf.statePost = flat_pts.reshape(8, 1).astype(np.float32)
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)

    def update_and_get_h(self, observed_virtual_corners):
        """
        Updates filter with 'observed' virtual corners and returns smooth H.
        """
        # 1. Predict next state
        self.kf.predict()

        # 2. Correct (if we have a valid observation this frame)
        if observed_virtual_corners is not None:
            measurement = (
                observed_virtual_corners.flatten().reshape(8, 1).astype(np.float32)
            )
            self.kf.correct(measurement)

        # 3. Retrieve Smoothed Corners from State
        smoothed_flat = self.kf.statePost.flatten()
        smoothed_src_pts = smoothed_flat.reshape(4, 2)

        # 4. Compute Homography: Smoothed Pixels -> Fixed World Meters
        H_smooth, _ = cv2.findHomography(smoothed_src_pts, VIRTUAL_CORNERS_WORLD)
        return H_smooth


def get_raw_homography(frame, model):
    """
    Runs inference on the frame, finds visible keypoints, and computes H_raw.
    Returns None if < 4 points found.
    """
    # 1. Run Inference (Force high res for keypoints)
    results = model.predict(frame, imgsz=1280, conf=0.25, verbose=False)[0]

    if results.boxes is None:
        return None

    src_pts = []
    dst_pts = []

    # Iterate through detections and match Class ID to PITCH_KEYPOINTS_MAP
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in PITCH_KEYPOINTS_MAP:
            continue

        # Center of the box is the keypoint
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        src_pts.append([cx, cy])
        dst_pts.append(PITCH_KEYPOINTS_MAP[cls_id])

    if len(src_pts) < 4:
        return None  # Not enough points for Homography

    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)

    # 3. Compute Raw Homography
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    return H


def project_virtual_corners(H_raw, width, height):
    """
    Uses the raw homography to guess where the 4 pitch corners are in the IMAGE space.
    (They might be negative or huge if off-screen).
    Inverse Logic: World -> Pixel
    """
    if H_raw is None:
        return None

    # We need H_inv: World -> Pixel
    try:
        H_inv = np.linalg.inv(H_raw)
    except np.linalg.LinAlgError:
        return None

    # Project World Corners -> Pixel Corners
    # Reshape for perspectiveTransform: (N, 1, 2)
    world_corners = VIRTUAL_CORNERS_WORLD.reshape(-1, 1, 2)
    virtual_pixels = cv2.perspectiveTransform(world_corners, H_inv)

    return virtual_pixels.reshape(4, 2)


def main():
    print("Loading tracks and model...")
    df = pd.read_csv(INPUT_CSV)

    # Load Keypoint Model (YOLOv8 Pose or Detection)
    # NOTE: Ensure this path exists!
    try:
        kp_model = YOLO(MODEL_KEYPOINT_PATH)
    except Exception as e:
        print(
            f"WARNING: Model not found at {MODEL_KEYPOINT_PATH}. Logic will fail without real keypoints."
        )
        print(e)
        kp_model = None

    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 1. Initialization (Find H in first frame to start Filter)
    print("Initializing stabilization...")
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read video.")
        return

    H_init = get_raw_homography(first_frame, kp_model)

    if H_init is None:
        print("FATAL: Could not find 4 keypoints in first frame. Cannot initialize.")
        # Fallback: Identity or manual hardcode
        return

    init_virtual_corners = project_virtual_corners(H_init, width, height)
    smoother = AdaptiveHomographySmoother(init_virtual_corners)

    # Prepare storage
    smoothed_coords = []

    # 2. FRAME-BY-FRAME Processing
    print("Processing frames...")
    unique_frames = sorted(df["frame"].unique())

    # Reset video to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    current_frame_idx = 0

    for target_frame_idx in tqdm(unique_frames):
        # Fast forward video to target frame (if gaps exist)
        while current_frame_idx < target_frame_idx:
            cap.read()
            current_frame_idx += 1

        ret, frame = cap.read()
        if not ret:
            break
        current_frame_idx += 1

        # A. Detect Raw H -> Virtual Corners
        H_raw = get_raw_homography(frame, kp_model)
        raw_virtual_corners = project_virtual_corners(H_raw, width, height)

        # B. Update Kalman Filter (Handles missing detections gracefully)
        # If raw_virtual_corners is None, filter predicts based on history
        H_smooth = smoother.update_and_get_h(raw_virtual_corners)

        # C. Transform Tracks
        frame_tracks = df[df["frame"] == target_frame_idx].copy()
        if frame_tracks.empty:
            continue

        # Use feet position (Center-Bottom of BBox)
        pixel_pts = np.column_stack(
            [(frame_tracks["x1"] + frame_tracks["x2"]) / 2, frame_tracks["y2"]]
        ).astype(np.float32)

        pixel_pts_r = pixel_pts.reshape(-1, 1, 2)
        world_pts = cv2.perspectiveTransform(pixel_pts_r, H_smooth)
        world_pts = world_pts.reshape(-1, 2)

        frame_tracks["pitch_x"] = world_pts[:, 0]
        frame_tracks["pitch_y"] = world_pts[:, 1]
        smoothed_coords.append(frame_tracks)

    # 3. Merge & Save
    if smoothed_coords:
        final_df = pd.concat(smoothed_coords)
        final_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved to {OUTPUT_CSV}")
    else:
        print("No tracks processed.")

    cap.release()


if __name__ == "__main__":
    main()
