import os 
import cv2
import logging
import numpy as np 
import sys

sys.path.append("../")

from utils import get_center_of_bbox, measure_distance

logger = logging.getLogger(__name__)

# Zona de finalização em coordenadas reais do campo (metros)

SHOT_ZONE_X_MAX = 23.32 # Até a linha de fundo 
SHOT_ZONE_X_MIN = 10.0 # Distância mínima do gol para considerar chute
SHOT_ZONE_Y_MIN = 10.0 # Faixa lateral
SHOT_ZONE_Y_MAX = 58.0

# Limiar de aceleração da bola para detectar chute (pixels entre frames)
BALL_SPEED_THRESHOLD = 40

class ShotExtractor:
    def __init__(self, frame_window: int = 5):
        self.frame_window = frame_window

    def _ball_positions(self, tracks: dict) -> list:
        """
        Extrai lista de centros da bola por frame (`None` se ausente)
        """
        position = []
        for frame_ball in tracks["ball"]:
            ball = frame_ball.get(1)
            if ball and "bbox" in ball:
                position.append(get_center_of_bbox(ball["bbox"]))
            else:
                position.append(None)
        return position
    
    
    #==========================
    # Detecção de finalizações
    #==========================

    def _ball_speed_pixels(self, position: list, frame_num: int) -> float:
        """
        Velocidade da bola em pixels entre `frame_num-1` e `frame_num`
        """
        if frame_num == 0:
            return 0.0
        
        p_prev = position[frame_num - 1]
        p_curr = position[frame_num]
        
        if p_prev is None or p_curr is None:
            return 0.0
        
        return measure_distance(p_prev, p_curr)
    
    def _in_shot_zone(self, tracks: dict, frame_num: int, player_id: int) -> bool:
        """
        Verifica se o jogador com posse está na zona de finalização (coordenadas reis)
        """
        try:
            player_info = tracks["players"][frame_num][player_id]
            pos = player_info.get("position_transformed")

            if pos is None:
                return False
            
            x, y = pos
            return (
                SHOT_ZONE_X_MIN <= x <= SHOT_ZONE_X_MAX
                and SHOT_ZONE_Y_MIN <= y <= SHOT_ZONE_Y_MAX
            )
        except (KeyError, TypeError):
            return False
        
    def _moving_toward_goal(self, positions: list, frame_num: int) -> bool:
        """
        Verifica se a bola está se movendo em direção ao gol (aumento de x)
        """
        if frame_num < 2:
            return False
        
        p_prev = positions[frame_num - 2]
        p_curr = positions[frame_num]

        if p_prev is None or p_curr is None:
            return False
        
        return p_curr[0] > p_prev[0]
    
    def detect_shot_frames(self, tracks: dict) -> list[int]:
        """
        Retorna lista de frame_num onde uma finalização foi detectada
        Critérios:
        1. Aceleração brusca da bola (pico de velocidade em pixels)
        2. Jogador com posse na zona de finalização
        3. Bola se movendo em direção ao gol
        """
        ball_positions = self._ball_positions(tracks)
        shot_frames = []
        cooldown = 0 # Evita detectar o mesmo chute em frames consecutivos

        num_frames = len(tracks["players"])

        for frame_num in range(1, num_frames):
            if cooldown > 0:
                cooldown -= 1
                continue

            speed = self._ball_speed_pixels(ball_positions, frame_num)
            if speed < BALL_SPEED_THRESHOLD:
                continue

            # Identifica quem tem a pose da bola no frame 
            player_with_ball = None
            for player_id, player_info in tracks["players"][frame_num].items():
                if player_info.get("has_ball", False):
                    player_with_ball = player_id
                    break
            
            if player_with_ball is None:
                continue

            in_zone = self._in_shot_zone(tracks, frame_num, player_with_ball)
            toward_goal = self._moving_toward_goal(ball_positions, frame_num)

            if in_zone and toward_goal:
                logger.info(f"Finalização detectada no frame {frame_num}")
                logger.info(f"(bola: {speed:.1f}px/frame, jogador: {player_with_ball})")
                shot_frames.append(frame_num)
                cooldown = self.frame_window * 2
        
        logger.info(f"Total de finalizações detectadas: {len(shot_frames)}")
        return shot_frames
    
    #=================
    # Salvando frames
    #=================

    def save_shot_frames(self, video_frames: list, shot_frames: list[int], output_dir: str, video_name: str) -> list[str]:
        """
        Salva frame_windows frames antes e depois de cada finalização.
        Retorna lista de caminhos salvos
        """
        raw_dir = os.path.join(output_dir, "shots", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        
        saved_paths = []
        total_frames = len(video_frames)

        for shot_num, shot_frame in enumerate(shot_frames):
            start = max(0, shot_frame - self.frame_window)
            end = min(total_frames -1, shot_frame + self.frame_window)

            for f in range(start, end + 1):
                offset = f - shot_frame # Negativo -> Antes; 0 = Moment; Positivo = Depois
                filename = f"{video_name}_shot{shot_num:03d}_f{offset:+d}.jpg"
                path = os.path.join(raw_dir, filename)
                cv2.imwrite(path, video_frames[f])
                saved_paths.append(path)
        logger.info(f"{len(saved_paths)} frames salvos em {raw_dir}")
        return saved_paths