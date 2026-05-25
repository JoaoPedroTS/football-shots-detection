import cv2
import logging
import numpy as np

logger = logging.getLogger(__name__)

class ViewTransformer():
    def __init__(self):
        logger.info("Inicializando ViewTransformer")
        court_width = 68
        court_length = 23.32

        self.pixels_vertices = np.array([
            [110, 1035],
            [265, 275],
            [910, 260],
            [1640, 915]
        ])

        self.target_vertices = np.array([
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ])

        self.pixels_vertices = self.pixels_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixels_vertices, self.target_vertices)
        logger.info(f"Transformação de perspectiva configurada: campo {court_length}m x {court_width}m")
    
    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixels_vertices, p, False) >= 0
        if not is_inside:
            return None
        
        reshape_point = point.reshape(-1, 1, 2).astype(np.float32)
        transform_point = cv2.perspectiveTransform(reshape_point, self.perspective_transformer)
        return transform_point.reshape(-1, 2)
    
    def add_transformed_position_to_tracks(self, tracks):
        logger.info("Transformando posições para coordenadas reais do campo")
        total = inside = outside = 0

        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info["position_adjusted"]
                    position = np.array(position)
                    position_transformed = self.transform_point(position)
                    
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                        tracks[object][frame_num][track_id]["position_transformed"] = position_transformed
        logger.info(
            f"Transformação concluída: {total} tracks — "
            f"{inside} dentro do campo ({100 * inside // total if total else 0}%), "
            f"{outside} fora do campo"
        )