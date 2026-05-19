import cv2
import os
import glob
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

INPUT_DIR = "input/full_match_videos"
OUTPUT_DIR = "train/football-shots-detection/images"

def extract_frames(video_path: str, n_frames: int, output_dir: str) -> int:
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    save_dir = os.path.join(output_dir, video_name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Não foi possível abrir o vídeo: {video_path}")
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    logger.info(f"vídeo: {video_name}")
    logger.info(f"Frames totais: {total_frames}")
    logger.info(f"FPS: fps{fps:.1f}")
    logger.info(f"Duração: {duration:.1f}s ({duration/60:.1f} min)")
    logger.info(f"A extrair: {n_frames} frames")
    logger.info(f"Saída: {save_dir}")

    if n_frames > total_frames:
        logger.warning(f"Vídeo tem apenas {total_frames} frames - ajustando para {total_frames}")
        n_frames = total_frames

    # Índices uniformemente distribuídos ao longo do vídeo
    step = total_frames / n_frames
    frame_indices = sorted(set(int(i*step) for i in range(n_frames)))

    saved = 0
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Falha ao ler frame {frame_idx}, pulando")
            continue

        filename = f"{video_name}_frame{frame_idx:06d}.jpg"
        cv2.imwrite(os.path.join(save_dir, filename), frame)
        saved += 1

        if saved % 10 == 0 or saved == len(frame_indices):
            logger.info(f"Progresso: {saved}/{len(frame_indices)} frames salvos")

    cap.release()
    logger.info(f"Concluído: {saved} frames salvos em '{save_dir}\n'")
    return saved

def main():
    parser = argparse.ArgumentParser(
        description="Extrai N frames uniformemente distribuídos de vídeos de futebol",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help=f"Caminho para um vídeo específico.\nSe omitido, processa todos os .mp4 em {INPUT_DIR}/"
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Número de frames a extrair por vídeo (padrão: 100)"
    )

    args = parser.parse_args()
 
    videos = [args.video] if args.video else sorted(glob.glob(f"{INPUT_DIR}/*.mp4"))
 
    if not videos:
        logger.error(f"Nenhum vídeo encontrado em {INPUT_DIR}/")
        return
 
    logger.info(f"{'='*50}")
    logger.info(f"{len(videos)} vídeo(s) | {args.frames} frames por vídeo")
    logger.info(f"Saída: {OUTPUT_DIR}/")
    logger.info(f"{'='*50}\n")
 
    total_saved = 0
    for i, video_path in enumerate(videos, 1):
        logger.info(f"[{i}/{len(videos)}] {os.path.basename(video_path)}")
        total_saved += extract_frames(video_path, args.frames, OUTPUT_DIR)
 
    logger.info(f"{'='*50}")
    logger.info(f"Extração finalizada: {total_saved} frames salvos no total")
    logger.info(f"{'='*50}")
 
 
if __name__ == "__main__":
    main()