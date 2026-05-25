import logging
import sys
sys.path.append("../")
from utils import get_center_of_bbox, measure_distance

logger = logging.getLogger(__name__)

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70
        logger.info(f"PlayerBallAssigner inicializado (distância máxima: {self.max_player_ball_distance} px)")

    def assign_ball_to_player(self, players, ball_bbox):
        ball_posistion = get_center_of_bbox(ball_bbox)

        minimum_distance = 99999
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player["bbox"]

            distance_left = measure_distance(
                (player_bbox[0], player_bbox[-1]),
                ball_posistion
            )
            
            distance_right = measure_distance(
                (player_bbox[2], player_bbox[-1]),
                ball_posistion
            )

            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id

        if assigned_player != -1:
            logger.debug(f"Bola atíbuida ao jogador {assigned_player} (distância: {minimum_distance:.1f} px)")
        else:
            logger.debug("Nenhum jogador próximo o suficiente da bola")
        
        return assigned_player