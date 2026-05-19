import subprocess 
import argparse
import logging 
import json
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def check_yt_dlp():
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        logger.error("yt-dlp não encontrado. Instale com: pip install yt-dlp")
        sys.exit(1)

def get_video_info(url: str) -> dict:
    logger.info("Buscando informações do vídeo ...")
    result = subprocess.run(
        ["yt-dlp", "--dump-json", "--no-download", url],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        logger.error(f"Erro ao buscar informações: {result.stderr}")
        return {}
    return json.loads(result.stdout)

def download_video(url: str, output_dir: str="input/full_match_videos", filename: str=None, resolution: int=720, audio: bool=False):
    os.makedirs(output_dir,exist_ok=True)
    if filename:
        output_template = os.path.join(output_dir, f"{filename}.%(ext)s")
    else:
        output_template = os.path.join(output_dir, "%(title)s.%(ext)s")

    if audio:
        format_str = f"bestvideo[height<={resolution}][ext=mp4][vcodec^=avc]+bestaudio[ext=m4a]/best[height<={resolution}]"
    else:
        format_str = f"bestvideo[height<={resolution}][ext=mp4][vcodec^=avc]/best[height<={resolution}]"

    cmd = [
        "yt-dlp",
        "-f", format_str,
        "-o", output_template,
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--no-overwrites",
        url
    ]

    info = get_video_info(url)
    if info:
        logger.info(f"Título: {info.get('title', 'N/A')}")
        logger.info(f"Duração: {info.get('duration_string', 'N/A')}")
        logger.info(f"Canal: {info.get('channel', 'N/A')}")
    logger.info(f"Baixando em {resolution}p -> {output_dir}/")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        logger.info("Download concluido com sucesso !")
    else:
        logger.error("Falha no download")
        sys.exit(1)

def download_batch(urls_file: str, output_dir: str="input/full_match_videos", resolution: int=720):
    if not os.path.exists(urls_file):
        logger.error(f"Arquivo não encontrado: {urls_file}")
        sys.exit(1)

    with open(urls_file) as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    logger.info(f"{len(urls)} vídeo(s) enconstrado(s) em {urls_file}")

    for i, url in enumerate(urls, 1):
        logger.info(f"[{i}/{len(urls)}] {url}")
        download_video(url, output_dir=output_dir, resolution=resolution)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Download de vídeos do YouTube para dataset de xG",
        formatter_class = argparse.RawTextHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command")

    # Comandos
    ## Vídeo único (single)
    single = subparsers.add_parser("single", help = "Baixa um único vídeo")
    single.add_argument("url", help = "URL do vídeo")
    single.add_argument("-o", "--output-dir", default="input_videos", help="Pasta de destino (padrão: input_videos)")
    single.add_argument("-n", "--name", default=None, help="Nome do arquivo de saída (sem extensão)")
    single.add_argument("-r", "--resolution", type=int, default=720, help="Resolução (padrão: 720)")
    single.add_argument("--no-audio", action="store_true", help="Baixa sem áudio (menor arquivo)")

    ## Vários vídeos (batch)
    batch = subparsers.add_parser("batch", help="Baixa múltiplos vídeos a partir de um arquivo .txt")
    batch.add_argument("file", help="Arquivo .txt com uma URL por linha")
    batch.add_argument("-o", "--output-dir", default="input/full_match_videos", help="Pasta de destino (padrão: input_videos)")
    batch.add_argument("-r", "--resolution", type=int, default=720, help="Resolução (padrão: 720)")

    args = parser.parse_args()

    check_yt_dlp()

    if args.command == "single":
        download_video(
            url = args.url,
            output_dir = args.output_dir,
            filename = args.name,
            resolution = args.resolution,
            audio = not args.no_audio
        )
    elif args.command == "batch":
        download_batch(
            urls_file = args.file,
            output_dir = args.output_dir,
            resolution = args.resolution
        )
    else:
        parser.print_help()