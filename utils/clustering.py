import cv2
import numpy as np
from sklearn.cluster import KMeans

def identify_jerseys_and_roles(frame, player_detections, field_dimensions):
    # Extract color information from player detections
    player_colors = []
    player_positions = []
    for detection in player_detections:
        x, y, w, h = detection
        player_roi = frame[y:y+h, x:x+w]
        average_color = np.mean(player_roi, axis=(0, 1))
        player_colors.append(average_color)
        player_positions.append((x + w/2, y + h/2))  # Center point of player

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(player_colors)

    # Identify clusters
    cluster_sizes = np.bincount(kmeans.labels_)
    outfield_team1, outfield_team2 = np.argsort(cluster_sizes)[-2:]
    goalkeeper_clusters = [i for i in range(5) if i not in [outfield_team1, outfield_team2]]

    # Determine team sides (assuming teams are on opposite sides of the field)
    team1_x = np.mean([pos[0] for pos, label in zip(player_positions, kmeans.labels_) if label == outfield_team1])
    team2_x = np.mean([pos[0] for pos, label in zip(player_positions, kmeans.labels_) if label == outfield_team2])
    team1_side = 'left' if team1_x < team2_x else 'right'

    # Assign team and role to each player
    player_assignments = []
    for i, (label, position) in enumerate(zip(kmeans.labels_, player_positions)):
        x, y = position
        if label == outfield_team1:
            role = "Team 1 Outfield"
        elif label == outfield_team2:
            role = "Team 2 Outfield"
        elif label in goalkeeper_clusters:
            # Determine goalkeeper team based on position
            if (team1_side == 'left' and x < field_dimensions[0] / 2) or \
               (team1_side == 'right' and x > field_dimensions[0] / 2):
                role = "Team 1 Goalkeeper"
            else:
                role = "Team 2 Goalkeeper"
        else:
            role = "Referee"
        player_assignments.append(role)

    return player_assignments
