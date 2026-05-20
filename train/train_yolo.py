import argparse
import logging
import shutil
import os
from ultralytics import YOLO
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)
 
# ── Configurações padrão ──────────────────────────────────────────────────────
DATASET_DIR  = "train/football-shots-detection"
DATA_YAML    = os.path.join(DATASET_DIR, "data.yaml")
BASE_MODEL   = "yolov8x.pt"   # modelo base para fine-tuning
 
 
def organize_dataset(dataset_dir: str):
    """
    Garante que train/test/valid estão na estrutura correta.
    Recria a estrutura caso as pastas estejam no nível errado.
    """
    splits = ["train", "test", "valid"]
    moved_any = False
 
    for split in splits:
        src = os.path.join(dataset_dir, split)
        dst = os.path.join(dataset_dir, dataset_dir, split)
 
        if os.path.exists(src) and not os.path.exists(dst):
            logger.info(f"Movendo {src} → {dst}")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            moved_any = True
 
    if not moved_any:
        logger.info("Estrutura do dataset já está correta, nada a mover")
 
 
def train(epochs: int, imgsz: int, model_path: str, data_yaml: str):
    if not os.path.exists(data_yaml):
        logger.error(f"data.yaml não encontrado em: {data_yaml}")
        logger.error("Certifique-se de que o dataset foi exportado corretamente do Roboflow")
        return
 
    logger.info(f"Iniciando treinamento")
    logger.info(f"  Modelo base : {model_path}")
    logger.info(f"  Dataset     : {data_yaml}")
    logger.info(f"  Épocas      : {epochs}")
    logger.info(f"  Imagem size : {imgsz}px")
 
    model = YOLO(model_path)
 
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        project="models",
        name="football-shots",
        exist_ok=True
    )
 
    logger.info("Treinamento concluído!")
    logger.info(f"Modelo salvo em: models/football-shots/weights/best.pt")
    return results
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Treina modelo YOLO para detecção de jogadores/bola",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--epochs",   type=int, default=100,       help="Número de épocas (padrão: 100)")
    parser.add_argument("--imgsz",    type=int, default=640,        help="Tamanho da imagem (padrão: 640)")
    parser.add_argument("--model",    type=str, default=BASE_MODEL, help=f"Modelo base (padrão: {BASE_MODEL})")
    parser.add_argument("--data",     type=str, default=DATA_YAML,  help=f"Caminho do data.yaml (padrão: {DATA_YAML})")
    parser.add_argument("--skip-organize", action="store_true",     help="Pula a reorganização das pastas do dataset")
    args = parser.parse_args()
 
    if not args.skip_organize:
        organize_dataset(DATASET_DIR)
 
    train(
        epochs=args.epochs,
        imgsz=args.imgsz,
        model_path=args.model,
        data_yaml=args.data
    )
 
 
if __name__ == "__main__":
    main()