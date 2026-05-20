import argparse
import logging
import shutil
import os
from ultralytics import YOLO
from roboflow import Roboflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ── Configurações padrão ──────────────────────────────────────────────────────
DATASET_DIR = "train/football-shots-detection"
DATA_YAML   = os.path.join(DATASET_DIR, "data.yaml")
BASE_MODEL  = "yolov8m.pt"


# ── Download ──────────────────────────────────────────────────────────────────
def download_dataset(api_key: str, workspace: str, project: str, version: int, output_dir: str) -> str:
    logger.info(f"Conectando ao Roboflow: {workspace}/{project} v{version}")
    
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download("yolov8", location=output_dir)
    
    logger.info(f"Dataset baixado em: {output_dir}")
    return dataset.location


# ── Organização ───────────────────────────────────────────────────────────────
def organize_dataset(dataset_dir: str):
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
        logger.info("Estrutura do dataset já está correta")


# ── Treinamento ───────────────────────────────────────────────────────────────
def train(epochs: int, imgsz: int, model_path: str, data_yaml: str):
    if not os.path.exists(data_yaml):
        logger.error(f"data.yaml não encontrado em: {data_yaml}")
        logger.error("Use --download para baixar o dataset ou verifique o caminho com --data")
        return

    logger.info("Iniciando treinamento")
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
    logger.info("Modelo salvo em: models/football-shots/weights/best.pt")
    return results


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Treina modelo YOLO para detecção de jogadores/bola",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Dataset
    parser.add_argument("--download",      action="store_true",    help="Baixa o dataset do Roboflow antes de treinar")
    parser.add_argument("--api-key",       type=str, default=None, help="Roboflow API key")
    parser.add_argument("--workspace",     type=str, default=None, help="Roboflow workspace")
    parser.add_argument("--project",       type=str, default=None, help="Roboflow project name")
    parser.add_argument("--version",       type=int, default=1,    help="Versão do dataset (padrão: 1)")
    parser.add_argument("--skip-organize", action="store_true",    help="Pula a reorganização das pastas do dataset")

    # Treinamento
    parser.add_argument("--epochs", type=int, default=100,       help="Número de épocas (padrão: 100)")
    parser.add_argument("--imgsz",  type=int, default=640,       help="Tamanho da imagem (padrão: 640)")
    parser.add_argument("--model",  type=str, default=BASE_MODEL,help=f"Modelo base (padrão: {BASE_MODEL})")
    parser.add_argument("--data",   type=str, default=DATA_YAML, help=f"Caminho do data.yaml (padrão: {DATA_YAML})")

    args = parser.parse_args()

    # Download
    if args.download:
        if not all([args.api_key, args.workspace, args.project]):
            logger.error("Para baixar o dataset informe --api-key, --workspace e --project")
            return
        download_dataset(
            api_key=args.api_key,
            workspace=args.workspace,
            project=args.project,
            version=args.version,
            output_dir=DATASET_DIR
        )

    # Organização
    if not args.skip_organize:
        organize_dataset(DATASET_DIR)

    # Treinamento
    train(
        epochs=args.epochs,
        imgsz=args.imgsz,
        model_path=args.model,
        data_yaml=args.data
    )


if __name__ == "__main__":
    main()