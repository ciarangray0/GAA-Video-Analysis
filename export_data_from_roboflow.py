from roboflow import Roboflow
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("ROBOFLOW_API_KEY")

rf = Roboflow(api_key=api_key)
project = rf.workspace("FYP").project("Find players")
dataset = project.version(1).download("yolov8")  # version 1, YOLOv8 format

print("Dataset downloaded to:", dataset.location)
