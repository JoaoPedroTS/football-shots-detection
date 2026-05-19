from roboflow import Roboflow
import os

ROBOFLOW_API_KEY = "7C7LbLmEERKtMvATLeGj"

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace("new-workspace").project("football-shots-detection")

images_dir = "train/football-shots-detection/images"

for jogo in os.listdir(images_dir):
    jogo_path = os.path.join(images_dir, jogo)
    if not os.path.isdir(jogo_path):
        continue

    print(f"Uploading {jogo}...")
    project.upload(
        image_path=jogo_path,
        batch_name=jogo,        
        tag_names=[jogo]      
    )