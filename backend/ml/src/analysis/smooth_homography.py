import cv2
import numpy as np
import pandas as pd
from collections import deque
from tqdm import tqdm

VIDEO_PATH = "../../data/08fd33_4.mp4"
INPUT_CSV = "tracks_raw.csv"
OUTPUT_CSV = "tracks_pitch.csv"

# You need 4 corresponding points: (Pixel_X, Pixel_Y) <-> (Pitch_X_Meters, Pitch_Y_Meters)
# Example: Left-Top-Corner, Left-Bottom-Corner, Center-Top, Center-Bottom
# THESE ARE PLACEHOLDERS. You must measure these from your video's first frame.
SRC_POINTS = np.array(
    [
        [450, 300],  # Point 1: Pixel (x,y)
        [200, 900],  # Point 2: Pixel (x,y)
        [1400, 300],  # Point 3: Pixel (x,y)
        [1700, 900],  # Point 4: Pixel (x,y)
    ],
    dtype=np.float32,
)

# Standard Pitch Dimensions (105m x 68m)
DST_POINTS = np.array(
    [
        [0, 0],  # Point 1: Pitch (0,0) is top-left corner
        [0, 68],  # Point 2: Pitch (0,68) is bottom-left corner
        [52.5, 0],  # Point 3: Pitch (52.5,0) is center-line top
        [52.5, 68],  # Point 4: Pitch (52.5,68) is center-line bottom
    ],
    dtype=np.float32,
)


class SmoothViewTransformer:
    """
    Temporal Smoothing Wrapper for Homography.
    Uses a deque to average the H-matrix over the last 'buffer_size' frames.
    """

    def __init__(self, src_pts, dst_pts, buffer_size=10):
        self.buffer = deque(maxlen=buffer_size)
        self.src = src_pts
        self.dst = dst_pts

    def update(self):
        # In a real dynamic camera system, you'd redetect keypoints here.
        # For a STATIC camera (like the Metrica/SoccerTrack wide view),
        # we calculate H once and just keep it stable.

        # If your camera MOVES, you must detect keypoints per frame here
        # and replace self.src with new_keypoints.

        H_raw, _ = cv2.findHomography(self.src, self.dst, cv2.RANSAC)
        if H_raw is not None:
            self.buffer.append(H_raw)

    def transform_points(self, points):
        if len(self.buffer) == 0:
            return points

        # Average the matrices
        H_smooth = np.mean(self.buffer, axis=0)

        # Reshape for cv2.perspectiveTransform: (N, 1, 2)
        points_reshaped = points.reshape(-1, 1, 2).astype(np.float32)

        # Apply Transform
        transformed = cv2.perspectiveTransform(points_reshaped, H_smooth)
        return transformed.reshape(-1, 2)


def main():
    # 1. Load Raw Tracks
    print("Loading raw tracks...")
    df = pd.read_csv(INPUT_CSV)

    # 2. Prepare Transformer
    transformer = SmoothViewTransformer(SRC_POINTS, DST_POINTS, buffer_size=30)

    # Pre-fill the buffer (Since we assume static camera/points for this demo)
    # If dynamic, you'd update this inside the loop based on feature matching
    for _ in range(30):
        transformer.update()

    # 3. Calculate "Foot" Position
    # We track the bounding box center-bottom (the feet), not the center.
    # x_foot = (x1 + x2) / 2
    # y_foot = y2 (bottom of box)
    df["x_pixel"] = (df["x1"] + df["x2"]) / 2
    df["y_pixel"] = df["y2"]

    # 4. Apply Transformation Batch-wise (Faster)
    # Note: For truly dynamic cameras, you must do this frame-by-frame.
    # Since we are using a static-assumption for the demo points above:
    print("Projecting to pitch coordinates...")

    pixel_points = df[["x_pixel", "y_pixel"]].to_numpy()
    pitch_points = transformer.transform_points(pixel_points)

    df["pitch_x"] = pitch_points[:, 0]
    df["pitch_y"] = pitch_points[:, 1]

    # 5. Save
    output_cols = ["frame", "track_id", "class_id", "pitch_x", "pitch_y"]
    df[output_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"Done! Pitch coordinates saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
