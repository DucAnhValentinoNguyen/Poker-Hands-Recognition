from ultralytics import YOLO
from roboflow import Roboflow

rf = Roboflow(api_key="vlAeyqaYHtL4RhlS7p8H")
project = rf.workspace("augmented-startups").project("playing-cards-ow27d")
version = project.version(4)
# Use 'yolov11' format; YOLO26 reads this exact same folder structure
dataset = version.download("yolo11")

# load base model
model = YOLO("yolo26n.pt")

# fine-tine
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs = 300,
    patience = 50,
    imgsz = 640,
    batch =-1,
    plots = True,
    device = 0 # use device = "cpu" if no GPU available
)
