from ultralytics import YOLO

model = YOLO('/scratch/py2050/my_project/YOLOv8-main/ultralytics/cfg/models/v8/yolov8-DSC.yaml') 

results = model.train(data="/scratch/py2050/my_project/YOLOv8-main/ultralytics/cfg/datasets/infrared-augmented.yaml", epochs=200)  
metrics = model.val() 
