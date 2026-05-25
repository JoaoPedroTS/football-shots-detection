from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import cv2
import logging
import pickle
import os
import sys

sys.path.append("../")
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

logger = logging.getLogger(__name__)

class Tracker:
    def __init__(self, model_path: str):
        logger.info(f"Carregando modelo: {model_path}")
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        logger.info("Modelo e tracker iniciandos.")

    def add_position_to_tracks(self, tracks):
        logger.info("Adicionando posições aos tracks")
        total = 0        
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]
                    if object == "ball":
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]["position"] = position
        logger.info(f"Posições adicionadas: {total} tracks processados")

    def interpolate_ball_positions(self, ball_positions):
        logger.info("Interpolando posições da bola")
        missing_before = sum(1 for x in ball_positions if not x.get(1, {}).get("bbox"))

        ball_positions = [x.get(1, {}).get("bbox", [])for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        logger.info(f"Interpolação concluída: {missing_before} frames sem bola preenchidos")
        return ball_positions


    def detect_frames(self, frames: list) -> list:
        batch_size = 20
        detections = []
        total_batches = (len(frames) + batch_size - 1) // batch_size
        logger.info(f"Iniciando detecção: {len(frames)} frames em {total_batches} batches")

        for i in range(0, len(frames), batch_size):
            batch_num = i // batch_size + 1
            logger.info(f"Processando batch {batch_num}/{total_batches} (frames {i}–{min(i + batch_size, len(frames)) - 1})")

            detections_batch = self.model.predict(
                frames[i:i+batch_size], conf=0.1
            )
            detections += detections_batch
        logger.info(f"Detecção concluída: {len(detections)} frames processados")
        return detections

    def get_object_tracks(self, frames: list, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            logger.info(f"Carregando tracks do stub: {stub_path}")
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            logger.info(f"Stub carregado: {len(tracks['players'])} frames")
            return tracks

        logger.info("Iniciando rastreamento de objetos")
        detections = self.detect_frames(frames)
        tracks = {"players": [], "goalkeepers": [], "referees": [], "ball": []}
        total_players = total_goalkeepers = total_referees = total_ball = 0

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)
            logger.debug(f"Frame {frame_num}: {detection_supervision}")
            
            # Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["goalkeepers"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                    total_players += 1
                
                if cls_id == cls_names_inv["goalkeeper"]:
                    tracks["goalkeepers"][frame_num][track_id] = {"bbox": bbox}
                    total_goalkeepers += 1
                
                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                    total_referees += 1
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}
                    total_ball += 1

        logger.info(
            f"Rastreamento concluído — detecções acumuladas: "
            f"jogadores={total_players}, goleiros={total_goalkeepers}, "
            f"árbitros={total_referees}, bola={total_ball}"
        )

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)
            logger.info(f"Stub salvo em: {stub_path}")
        
        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id = None):
        y2 = int(bbox[3])

        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center = (x_center, y2),
            axes = (int(width), int(0.35*width)),
            angle = 0.0,
            startAngle = -45,
            endAngle = 235,
            color = color,
            thickness = 2,
            lineType = cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2) + 15
        y2_rect = (y2 + rectangle_height//2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED
            )
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ =  get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20],
        ])

        cv2.drawContours(
            frame,
            [triangle_points],
            0,
            color,
            cv2.FILLED
        )
        
        cv2.drawContours(
            frame,
            [triangle_points],
            0,
            (0, 0, 0),
            2
        )

        return frame

    def draw_annotations(self, video_frames: list, tracks):
        logger.info(f"Desenhando anotações em {len(video_frames)} frames")
        output_video_frames = []
        players_with_ball = 0

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
 
            player_dict     = tracks["players"][frame_num]
            goalkeeper_dict = tracks["goalkeepers"][frame_num]
            ball_dict       = tracks["ball"][frame_num]
            referee_dict    = tracks["referees"][frame_num]
 
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))
                    players_with_ball += 1
 
            for track_id, goalkeeper in goalkeeper_dict.items():
                frame = self.draw_ellipse(frame, goalkeeper["bbox"], (255, 255, 0), track_id)
 
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255), track_id)
 
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))
 
            output_video_frames.append(frame)
 
        logger.info(f"Anotações concluídas — frames com jogador em posse: {players_with_ball}")
        return output_video_frames