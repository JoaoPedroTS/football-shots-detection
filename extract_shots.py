import argparse
import logging
import glob
import os

from camera_movement_estimator import CameraMovementEstimator
from player_ball_assigner import PlayerBallAssigner
from view_transformer import ViewTransformer
from shot_extractor import ShotExtractor
from trackers import Tracker
from utils import read_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)

def process_video(video_path: str, args):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    logger.info(f"{'='*50}")
    logger.info(f"Processando: {video_name}")
    logger.info(f"{'='*50}")

    # Leitura
    video_frames = read_video(video_path)

    # Tracker
    tracker = Tracker("models/best.pt")

    stub_track = f"stubs/{video_name}_track.pkl" if args.use_stubs else None
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=args.use_stubs,
        stub_path=stub_track
    )

    tracker.add_position_to_tracks(tracks)

    # Movimento de câmera
    camera_estimator = CameraMovementEstimator(video_frames)
    stub_cam = f"stubs/{video_name}_camera.pkl" if args.use_stubs else None
    camera_movement = camera_estimator.get_camera_movement(
        video_frames,
        read_from_stub=args.use_stubs,
        stub_path=stub_cam
    )
    camera_estimator.add_adjust_position_to_tracks(tracks, camera_movement)

    # Transformação de perspectiva
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolação da bola
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Posse de bola
    player_assigner = PlayerBallAssigner()
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num][1]["bbox"]
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks["player"][frame_num][assigned_player]["has_ball"] = True
    
    # Detecção e salvamento de finalizações
    extractor = ShotExtractor(frame_window=args.window)
    shot_frames = extractor.detect_shot_frames(tracks)

    if not shot_frames:
        logger.warning("Nenhuma finalização detectada. Tente ajustar --window ou Revise o vídeo")
        return
    
    saved = extractor.save_shot_frames(
        video_frames=video_frames,
        shot_frames=shot_frames,
        output_dir=args.output_dir,
        video_name=video_name
    )

    logger.info(f"Concluído: {len(shot_frames)} finalizações -> {len(saved)} frames salvos")

def main():
    parser = argparse.ArgumentParser(
        description="Extrai frames de finalizações de vídeos táticos de futebol",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Caminho para um vídeo específico. Se omitido, processa todos em input_videos/"
    )

    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Nº de frames antes e depois da finalização a salvar (padrão: 5)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset",
        help="Pasta raiz de saída (padrão `sataset/`)"
    )

    parser.add_argument(
        "--use-stubs",
        action="store_true",
        help="Usa cache de stubs para tracker e câmera (mais rápido em re-execução)"
    )

    args = parser.parse_args()

    if args.video:
        if not os.path.exists(args.video):
            logger.error(f"Vídeos não encontrado: {args.video}")
            return
        process_video(args.video, args)
    else:
        videos = glob.glob("input_videos/*.mp4")
        if not videos:
            logger.error("Nenhum vídeo encontrado em `input_video/`")
            return
        logger.info(f"{len(videos)} vídeo(s) encontrado(s) para processar")
        for video_path in sorted(videos):
            process_video(video_path, args)

if __name__ == "__main__":
    main()