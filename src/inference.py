import torch
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import numpy

from src.config import load_config

class YOLOv11Inference:
    def __init__(self, model_name, device="cuda"):
        self.device = device
        self.model=YOLO(model_name)
        self.model.to(device)
        
        config=load_config()
        self.confg_threshold=config["model"]["conf_threshold"]
        self.extention = config["data"]["image_extension"]
        
    
    def process_image(self, image_parh):
        results = self.model.predict(
            source=image_parh,
            conf=self.confg_threshold,
            device=self.device
        )
        
        detetion = []
        class_counts = {}
        
        for result in results:
            for box in result.boxes:
                cls = result.names[int(box.cls)]
                conf = float(box.conf)
                bbox = box.xyxy[0].tolist()
                
                detetion.append({
                    'class':cls,
                    'confidence':conf,
                    'bbox':bbox,
                    'count':1
                })
                
                class_counts[cls] = class_counts.get(cls, 0) + 1
                
        for det in detetion:
            det['class'] = class_counts[det['class']]
                
        
        return {
            'image_path': str(image_parh),
            'detetion':detetion,
            'total_object': len(detetion),
            'unique_class': list(class_counts.keys()),
            'class_count': class_counts
        }
        
        
                
        
    def process_directory(self, directory):
        metadata=[]
        
        patterns = [f"*{ext}" for ext in self.extention]
        image_paths = []
        # D:\sudesh\processed\test_image\metadata.json
        for pattern in patterns:
            image_paths.extend(Path(directory).rglob(pattern))
            
        for image_path in image_paths:
            try:
                metadata.append(self.process_image(image_path))
            except Exception as e:
                print(f"Exception processing {image_path}:{str(e)}")
                continue
            
        return metadata