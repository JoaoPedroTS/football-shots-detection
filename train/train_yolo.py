import argparse
import logging
import os
import re
from ultralytics import YOLO
from roboflow import Roboflow
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)
 
# ── Configurações padrão ──────────────────────────────────────────────────────
DATASETS_DIR = "datasets"
BASE_MODEL   = "yolov8x.pt" 
 
 
# ── Download ──────────────────────────────────────────────────────────────────
def download_dataset(api_key: str, workspace: str, project: str, version: int) -> str:
    logger.info(f"Conectando ao Roboflow: {workspace}/{project} v{version}")
    os.makedirs(DATASETS_DIR, exist_ok=True)
 
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download("yolov8", location=DATASETS_DIR)

    location = dataset.location
    yaml_path = os.path.join(location, "data.yaml")

    if not os.path.exists(yaml_path):
        # Tenta encontrar o data.yaml em subpastas
        for root, dirs, files in os.walk(DATASETS_DIR):
            if "data.yaml" in files:
                location = root
                logger.info(f"data.yaml encontrado em: {location}")
                break

    logger.info(f"Dataset location: {location}")
    return location 
 
# ── Corrige data.yaml ─────────────────────────────────────────────────────────
def fix_data_yaml(dataset_location: str) -> str:
    """
    Atualiza os caminhos de train/val no data.yaml para o formato
    esperado pelo ultralytics fora do Colab.
    Equivale ao sed feito no notebook.
    """
    yaml_path = os.path.join(dataset_location, "data.yaml")
    if not os.path.exists(yaml_path):
        logger.error(f"data.yaml não encontrado em: {yaml_path}")
        return yaml_path
 
    with open(yaml_path, "r") as f:
        content = f.read()
 
    content = re.sub(r"(train:\s*).*", r"\1../train/images", content)
    content = re.sub(r"(val:\s*).*",   r"\1../valid/images", content)
 
    with open(yaml_path, "w") as f:
        f.write(content)
 
    logger.info(f"data.yaml atualizado: {yaml_path}")
    return yaml_path
 
 
# ── Treinamento ───────────────────────────────────────────────────────────────
def train(data_yaml: str, epochs: int, imgsz: int, model_path: str):
    if not os.path.exists(data_yaml):
        logger.error(f"data.yaml não encontrado: {data_yaml}")
        logger.error("Use --download para baixar o dataset ou passe o caminho correto com --data")
        return
 
    logger.info("Verificando GPU...")
    import torch
    if torch.cuda.is_available():
        logger.info(f"GPU disponível: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("GPU não encontrada, treinando na CPU (mais lento)")
 
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
        batch=6,          # mesmo valor do notebook (ajustado automaticamente se OOM)
        plots=True,
        project="runs/detect",
        name="train",
        exist_ok=True
    )
 
    logger.info("Treinamento concluído!")
    logger.info("Modelo salvo em: runs/detect/train/weights/best.pt")
    return results
 
 
# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Treina modelo YOLO para detecção de jogadores/bola",
        formatter_class=argparse.RawTextHelpFormatter
    )
 
    # Download
    parser.add_argument("--download",  action="store_true",    help="Baixa o dataset do Roboflow antes de treinar")
    parser.add_argument("--api-key",   type=str, default=None, help="Roboflow API key")
    parser.add_argument("--workspace", type=str, default=None, help="Roboflow workspace")
    parser.add_argument("--project",   type=str, default=None, help="Roboflow project name")
    parser.add_argument("--version",   type=int, default=1,    help="Versão do dataset (padrão: 1)")
 
    # Treinamento
    parser.add_argument("--data",   type=str, default=None,      help="Caminho do data.yaml (quando dataset já baixado)")
    parser.add_argument("--epochs", type=int, default=50,        help="Número de épocas (padrão: 50)")
    parser.add_argument("--imgsz",  type=int, default=1280,      help="Tamanho da imagem (padrão: 1280)")
    parser.add_argument("--model",  type=str, default=BASE_MODEL,help=f"Modelo base (padrão: {BASE_MODEL})")
 
    args = parser.parse_args()
 
    # Download + fix yaml
    dataset_location = None
    if args.download:
        if not all([args.api_key, args.workspace, args.project]):
            logger.error("Para baixar o dataset informe --api-key, --workspace e --project")
            return
        dataset_location = download_dataset(
            api_key=args.api_key,
            workspace=args.workspace,
            project=args.project,
            version=args.version
        )
 
    # Determina o caminho do data.yaml
    if args.data:
        data_yaml = args.data
    elif dataset_location:
        data_yaml = fix_data_yaml(dataset_location)
    else:
        logger.error("Informe --data com o caminho do data.yaml, ou use --download para baixar o dataset")
        return
 
    # Treinamento
    train(
        data_yaml=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        model_path=args.model
    )
 
 
if __name__ == "__main__":
    main()