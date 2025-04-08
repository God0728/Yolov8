from ultralytics import YOLO

# Load a model
#model = YOLO("yolo11n.pt")  # load an official model
model = YOLO("/root/ultralytics-8.3.27/runs/train/yolov8s_0300_150e/weights/best.pt")  # load a custom trained model

# Export the model
model.export(format="engine", int8=True)