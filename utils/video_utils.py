import cv2
import logging
import os

logger = logging.getLogger(__name__)

def read_video(video_path: str) -> list:
    logger.info(f"Lendo vídeos: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    logger.info(f"total de frames lidos: {len(frames)}")
    return frames

def save_video(output_video_frames: list, output_video_path: str):
    logger.info(f"Salvando vídeos em: {output_video_path}")

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    if os.path.exists(output_video_path):
        os.remove(output_video_path)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        24,
        (output_video_frames[0].shape[1], output_video_frames[0].shape[0])
    )

    if not out.isOpened():
        logger.error("Falha ao inicializar o VideoWriter. Verifique o caminho e o codec")
        return
    
    for frame in output_video_frames:
        out.write(frame)
    out.release()
    logger.info("Vídeo salvo com sucesso.")