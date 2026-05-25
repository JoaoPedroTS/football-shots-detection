import os
import cv2
import pickle
import logging
import numpy as np
import sys
sys.path.append("../")
from utils import measure_distance, measure_xy_distance

logger = logging.getLogger(__name__)

class CameraMovementEstimator():
    def __init__(self, frame):
        logger.info("Inicializando CameraMovementEstimator")
        self.minimun_distance = 5

        self.lk_params = dict(
            winSize = (15, 15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1

        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance = 3,
            blockSize = 7,
            mask = mask_features
        )
        logger.info("CameraMovementEstimator inicializado")
    
    def add_adjust_position_to_tracks(self, tracks, camera_movement_per_frame):
        logger.info("Ajustando posições dos trackers com base no movimento de câmera")
        total_adjusted = 0

        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info["position"]
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0] - camera_movement[0], position[1] - camera_movement[1])
                    tracks[object][frame_num][track_id]["position_adjusted"] = position_adjusted
                    total_adjusted += 1
        logger.info(f"Posições ajustadas: {total_adjusted} tracks processados")

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # Read the stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            logger.info(f"Carregando movimento de câmera para {len(frames)} frames")
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        logger.info(f"Calculando movimento de cÂmera para {len(frames)} frames")
        camera_movement = [[0, 0]]*len(frames)

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)
        logger.debug(f"Features detectadas no frame 0: {len(frames) if old_features is not None else 0}")

        frames_with_movement = 0

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_distance(new_features_point, old_features_point)

                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)

            if max_distance > self.minimun_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                frames_with_movement += 1
                logger.debug(f"Frame {frame_num}: movimento (x={camera_movement_x:.2f}, y={camera_movement_y:.2f}, dist={max_distance:.2f}px)")

            old_gray = frame_gray.copy()
        logger.info(f"Movimento de câmera calculado: {frames_with_movement}/{len(frames)} frames com movimento significativo")

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(camera_movement, f)
            logger.info(f"Stub salvo em: {stub_path}")
        
        return camera_movement

    def draw_camera_movement(self, frame, camera_movement_per_frame):
        logger.info(f"Desenhando movimento de cÂmera em {len(frame)} frames")
        output_frames = []

        for frame_num, frame in enumerate(frame):
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (0, 0),
                (500, 100),
                (250, 250, 250),
                -1
            )
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(
                frame,
                f"Camera Movement X: {x_movement:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                3
            )
            
            frame = cv2.putText(
                frame,
                f"Camera Movement Y: {y_movement:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                3
            )
            output_frames.append(frame)

        logger.info("Anotações de movimento de câmera concluídas")
        return output_frames