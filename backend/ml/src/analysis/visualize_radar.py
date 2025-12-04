import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
from sklearn.cluster import KMeans
from tqdm import tqdm

INPUT_CSV = "player_stats.csv"  # The file from analyze.py
VIDEO_PATH = "../../data/08fd33_4.mp4"  # Needed to detect jersey colors
OUTPUT_VIDEO = "tactical_radar_teams.mp4"
INPUT_RAW_CSV = "tracks_raw.csv"  # Needed to get bounding boxes for color extraction

PITCH_LENGTH = 105
PITCH_WIDTH = 68
FPS = 25
BALL_COLOR = (1.0, 1.0, 0.0, 1.0)  # Opaque Yellow


def assign_teams_by_color(stats_df, raw_df):
    """
    1. Looks at the middle frame of each track.
    2. Crops the player's jersey.
    3. Clusters all players into 2 teams based on color.
    """
    print("Assigning teams based on jersey colors...")

    # Open Video
    cap = cv2.VideoCapture(VIDEO_PATH)

    # Get unique tracks
    unique_tracks = stats_df["track_id"].unique()
    player_colors = []
    valid_track_ids = []

    for tid in tqdm(unique_tracks):
        # Find a sample frame for this player (middle of their track)
        track_instances = raw_df[raw_df["track_id"] == tid]
        if track_instances.empty:
            continue

        mid_idx = len(track_instances) // 2
        sample = track_instances.iloc[mid_idx]

        frame_id = int(sample["frame"])
        box = [
            int(sample["x1"]),
            int(sample["y1"]),
            int(sample["x2"]),
            int(sample["y2"]),
        ]

        # Read Frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            continue

        # Crop Player (Center of torso)
        # We take the center 1/3rd of the box to avoid grass/head
        h, w = box[3] - box[1], box[2] - box[0]
        cy, cx = box[1] + h // 2, box[0] + w // 2
        crop = frame[cy - 10 : cy + 10, cx - 10 : cx + 10]  # Small 20x20 patch

        if crop.size == 0:
            continue

        # Calculate Average Color (Lab space is better for perception, but RGB works)
        avg_color = crop.mean(axis=0).mean(axis=0)
        player_colors.append(avg_color)
        valid_track_ids.append(tid)

    cap.release()

    # K-Means Clustering (2 Teams)
    print("Clustering colors...")
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    labels = kmeans.fit_predict(player_colors)

    # Create a mapping dictionary
    team_map = dict(zip(valid_track_ids, labels))

    # Map back to DataFrame
    stats_df["team_id"] = stats_df["track_id"].map(team_map)

    # Handle NaNs (tracks that failed color detection) -> Assign to Team -1
    stats_df["team_id"] = stats_df["team_id"].fillna(-1).astype(int)

    return stats_df


def get_color(team_id, speed, max_speed=30):
    """
    Returns an RGBA color based on Team and Speed.
    Team 0 = Blues, Team 1 = Reds
    Faster = More Opaque/Intense
    """
    # Normalize speed (0 to 1.0)
    # We clamp it so even standing players are slightly visible (alpha 0.3)
    intensity = 0.3 + 0.7 * (min(speed, max_speed) / max_speed)

    if team_id == 0:
        # Blue Team (R=0, G=0, B=1)
        return (0.1, 0.1, 1.0, intensity)
    elif team_id == 1:
        # Red Team (R=1, G=0, B=0)
        return (1.0, 0.1, 0.1, intensity)
    else:
        # Unknown (Grey)
        return (0.5, 0.5, 0.5, 0.3)


def draw_pitch(ax):
    plt.plot([0, 0, 105, 105, 0], [0, 68, 68, 0, 0], color="black")
    plt.plot([52.5, 52.5], [0, 68], color="black")
    circle = Circle((52.5, 34), 9.15, color="black", fill=False)
    ax.add_patch(circle)
    ax.add_patch(Rectangle((0, 13.84), 16.5, 40.32, fill=False, color="black"))
    ax.add_patch(Rectangle((105 - 16.5, 13.84), 16.5, 40.32, fill=False, color="black"))
    ax.add_patch(Rectangle((-2, 30.34), 2, 7.32, fill=True, color="black", alpha=0.3))
    ax.add_patch(Rectangle((105, 30.34), 2, 7.32, fill=True, color="black", alpha=0.3))


def main():
    print("Loading stats...")
    df = pd.read_csv(INPUT_CSV)
    raw_df = pd.read_csv(INPUT_RAW_CSV)

    df = assign_teams_by_color(df, raw_df)

    # 3. Setup Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(-5, PITCH_LENGTH + 5)
    ax.set_ylim(-5, PITCH_WIDTH + 5)
    ax.set_aspect("equal")
    ax.axis("off")
    draw_pitch(ax)

    # Initialize Scatter: Note 'animated=True' is optimization for blitting
    scat = ax.scatter([], [], s=120, edgecolors="black", linewidth=1.5)
    time_text = ax.text(0.5, 1.05, "", transform=ax.transAxes, ha="center", fontsize=12)

    # Get unique frames and sort them
    frames = sorted(df["frame"].unique())

    def update(frame_num):
        current_data = df[df["frame"] == frame_num]

        # Optimize: Pre-allocate lists or use numpy arrays if strict on perf
        offsets = []
        face_colors = []
        sizes = []

        if current_data.empty:
            return scat, time_text

        for _, row in current_data.iterrows():
            offsets.append([row["x_smooth"], row["y_smooth"]])

            # Ball Logic
            if row["track_id"] == -999:
                face_colors.append(BALL_COLOR)
                sizes.append(60)
            # Player Logic
            else:
                c = get_color(row["team_id"], row["speed_kmh"])
                face_colors.append(c)
                sizes.append(120)

        # Update Artist Properties
        scat.set_offsets(offsets)
        scat.set_facecolor(face_colors)  # Preserves black edges
        scat.set_sizes(sizes)

        time_text.set_text(f"Frame: {frame_num}")
        return scat, time_text

    print(f"Rendering {len(frames)} frames...")
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000 / FPS,
        blit=True,
    )

    anim.save(OUTPUT_VIDEO, fps=FPS, extra_args=["-vcodec", "libx264"])
    print(f"Saved to {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
