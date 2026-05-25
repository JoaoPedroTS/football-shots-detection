import logging
import cv2
import sys
sys.path.append("../")

from utils import measure_distance, get_foot_position

logger = logging.getLogger(__name__)

class SpeedAndDistanceEstimator():
    def __init__(self):
        self.frame_window = 5
        self.frame_rate = 24
        logger.info(f"SpeedAndDistanceEstimator inicializado (janela={self.frame_window} frames, fps={self.frame_rate})")

    def add_speed_and_distance_to_tracks(self, tracks):
        logger.info("Calculando velocidade e distância dos tracks")
        total_distance = {}
        total_calculated = 0
        max_speed = 0.0
        max_speed_label = ""
        
        for object, object_tracks in tracks.items():
            if object == "referees":
                logger.debug("Pulando árbitros no cálculo de velocidade")
                continue
            
            number_of_frames = len(object_tracks)
            logger.info(f"Processando '{object}': {number_of_frames} frames")

            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames-1)

                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    star_position = object_tracks[frame_num][track_id].get("position_transformed")
                    end_position = object_tracks[last_frame][track_id].get("position_transformed")

                    if star_position is None or end_position is None:
                        continue

                    distance_covered = measure_distance(star_position, end_position)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    speed_meters_per_second =distance_covered / time_elapsed
                    speed_km_per_hour = speed_meters_per_second*3.6

                    if object not in total_distance:
                        total_distance[object] = {}

                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0

                    total_distance[object][track_id] += distance_covered
                    total_calculated += 1

                    if speed_km_per_hour > max_speed:
                        max_speed = speed_km_per_hour
                        max_speed_label = f"{object} id={track_id}"

                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        
                        tracks[object][frame_num_batch][track_id]["speed"] = speed_km_per_hour
                        tracks[object][frame_num_batch][track_id]["distance"] = total_distance[object][track_id]
        logger.info(
            f"Velocidade calculada: {total_calculated} segmentos processados - "
            f"Velocidade máxima: {max_speed:.1f} km/h ({max_speed_label})"
        )                        

    def draw_speed_and_distance(self, frames, tracks):
        logger.info(f"Desenhando velocidade e distância em {len(frames)} frames")
        output_frames = []
        drawn = 0

        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object == "referees":
                    continue
                for _, track_info in object_tracks[frame_num].items():
                    if "speed"  in track_info:
                        speed = track_info.get("speed", None)
                        distance = track_info.get("distance", None)

                        if speed is None or distance is None:
                            continue
                        
                        bbox = track_info["bbox"]
                        position = get_foot_position(bbox)
                        position = list(position)
                        position[1] += 40

                        position = tuple(map(int, position))
                        cv2.putText(
                            frame,
                            f"Speed: {speed:.2f} km/h",
                            position,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 0),
                            2
                        )
                        
                        cv2.putText(
                            frame,
                            f"Distance: {distance:.2f} m",
                            (position[0], position[1]+20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 0),
                            2
                        )
            output_frames.append(frame)
        
        logger.info(f"Anotações de velocidade/distância concluídas: {drawn} rótulos desenhados")
        return output_frames