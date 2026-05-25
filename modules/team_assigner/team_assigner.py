import logging
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        logger.info("TeamAssigner inicializado")

    def get_clustering_model(self, image):
        # Reshape image to 2d array
        image_2d = image.reshape(-1, 3)

        # Perform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans
    
    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_image = image[0:int(image.shape[0]/2),:]

        # Get clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1],
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 if non_player_cluster == 0 else 0

        return kmeans.cluster_centers_[player_cluster]

    def assign_team_color(self, frame, player_detections):
        logger.info(f"Atribuindo cores de time para {len(player_detections)} jogadores no frame inicial")
        player_colors = []

        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]
        logger.info(f"Cores dos times definidas — Time 1: {self.team_colors[1].astype(int).tolist()} | Time 2: {self.team_colors[2].astype(int).tolist()}")

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1

        self.player_team_dict[player_id] = team_id
        logger.debug(f"Jogador {player_id} atribuído ao time {team_id}")

        return team_id