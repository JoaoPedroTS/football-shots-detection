## Executar main
`python3 main.py`

## Baixar vídeos
### Baixar jogo único
`python3 utils/download_videos/download_videos.py single "URL" -n "nome_arquivo" --no-audio`

### Baixar vários vídeos de uma vez (batch) a partir de um arquivo .txt com vídeos listados
`python3 utils/download_videos/download_videos.py batch utils/download_videos/urls.txt`

## Extrair frames para treinar YOLO
### Todos os vídeos da pasta `input/full_match_videos`, 100 frames cada
`python3 utils/extract_frames.py --frames 100`

### Extrair frames de um videos específico
`python3 utils/extract_frames.py --video "caminho do video"`

## Subir frames pro roboflow
`python3 utils/upload_to_roboflow.py`

## Baixar dataset do roboflow com `curl`

## Treinar YOLO
### Padrão (100 épocas, imgsz 640)
`yolo task=detect mode=train model=yolov8x.pt data=train/dataset/data.yaml batch=6 epochs=300 imgsz=1280 plots=True patience=50`

## Executar modulo para extrair finalizações
### Processa todos os vídeos em `input/full_match_videos`
`python3 extract_shots.py`

### Re-execução rápida usando cache
`python3 extract_shots.py --use-stubs`

### Número de frames específico 
`python3 utils/extract_frames.py --frames 200`
