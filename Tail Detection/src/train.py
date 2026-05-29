import os
from roboflow import Roboflow
from ultralytics import YOLO


def download_dataset(api_key: str, workspace: str, project_name: str, version: int = 2):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    dataset = project.version(version).download("yolov11")
    return dataset


def train(data_yaml: str, epochs: int = 200, imgsz: int = 640, project: str = "Tail-clamp-detection-2"):
    model = YOLO("yolo11m.pt")
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        project=project,
    )


if __name__ == "__main__":
    download_dataset(
        api_key="YOUR_API_KEY",
        workspace="mohamed-90rl9",
        project_name="tail-clamp-detection",
    )
    train(data_yaml="Tail-clamp-detection-2/data.yaml")
