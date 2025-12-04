import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

INPUT_CSV = "tracks_pitch.csv"
OUTPUT_STATS_CSV = "player_stats.csv"
FPS = 25  # Frame rate of your video (Check this!)
SMOOTH_WINDOW = 25  # Window size for Sav-Gol (must be odd). 25 = 1 second at 25fps.
POLY_ORDER = 2  # Quadratic fit (preserves acceleration curves)


def calculate_kinematics(group):
    """
    Function applied to each Player ID group.
    Handles smoothing and derivative calculation (Velocity).
    """
    group = group.sort_values("frame")

    # Reindex to handle missing frames (Occlusions)
    # Creates rows for missing frames with NaN, then fills them
    full_idx = range(group["frame"].min(), group["frame"].max() + 1)
    group = group.set_index("frame").reindex(full_idx)

    # Interpolate Missing Data (Linear fill for gaps)
    group["pitch_x"] = group["pitch_x"].interpolate(method="linear")
    group["pitch_y"] = group["pitch_y"].interpolate(method="linear")

    # Apply Savitzky-Golay Smoothing
    # Need minimum data length > window_size
    if len(group) > SMOOTH_WINDOW:
        group["x_smooth"] = savgol_filter(
            group["pitch_x"], window_length=SMOOTH_WINDOW, polyorder=POLY_ORDER
        )
        group["y_smooth"] = savgol_filter(
            group["pitch_y"], window_length=SMOOTH_WINDOW, polyorder=POLY_ORDER
        )
    else:
        # Fallback for very short tracks
        group["x_smooth"] = group["pitch_x"]
        group["y_smooth"] = group["pitch_y"]

    # 5. Calculate Velocity (First Derivative)
    # dx / dt (Time between frames = 1/FPS)
    dt = 1 / FPS

    group["vx"] = group["x_smooth"].diff() / dt  # Velocity X (m/s)
    group["vy"] = group["y_smooth"].diff() / dt  # Velocity Y (m/s)

    # 6. Speed (Magnitude of velocity vector)
    group["speed_mps"] = np.sqrt(group["vx"] ** 2 + group["vy"] ** 2)
    group["speed_kmh"] = group["speed_mps"] * 3.6  # Km/Hour

    return group.reset_index()  # Bring 'frame' back as a column


def main():
    print("Loading pitch tracks...")
    df = pd.read_csv(INPUT_CSV)

    # 1. Process Players (Class 0)
    players = df[df["class_id"] == 0].copy()
    stats_players = (
        players.groupby("track_id").apply(calculate_kinematics).reset_index(drop=True)
    )

    # 2. Process Ball (Class 1)
    ball = df[df["class_id"] == 1].copy()

    # The ball track ID might change frequently or be -1.
    # We treat all ball detections as a single 'track' for smoothing purposes
    ball["track_id"] = -999  # Force a single ID for the ball

    # We use a smaller smoothing window for the ball because it moves jerkily (kicks/bounces)
    # Ball physics are different from humans.
    stats_ball = calculate_kinematics(ball)

    # 3. Combine
    combined_stats = pd.concat([stats_players, stats_ball], ignore_index=True)

    combined_stats.to_csv(OUTPUT_STATS_CSV, index=False)


if __name__ == "__main__":
    main()
