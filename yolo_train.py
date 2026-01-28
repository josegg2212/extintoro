from ultralytics import YOLO
import os
import time 

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Load dataset path
path_dataset = os.getenv('PATH_DATASET')

while True :
    # Comprobar si la ruta existe
    if path_dataset and os.path.exists(path_dataset):
        print(f"Ruta del dataset encontrada en: {path_dataset}")
        print("Iniciando entrenamiento...")
        
        # Train the model
        results = model.train(data=path_dataset, epochs=100, imgsz=640, workers=0, project="/home/code/modelo", name="train")

        break
    else : 
        print("La ruta del dataset no existe o no está definida. Reintentando en 30 segundos...")
        time.sleep(30)