from pathlib import Path
import json

def ensure_directory(raw_path):
    raw_path = Path(raw_path)
    processed_path = raw_path.parent.parent / "processed" / raw_path.name # this is the processed path
    processed_path.mkdir(parents=True, exist_ok=True)
    return processed_path
    
def save_object(metadata, raw_path):
    processed_path = ensure_directory(raw_path)
    
    output_path = processed_path / "metadata.json"
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f)
        
    return output_path

def load_metadata(metadata_path):
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        
        process_path = metadata_path.parent.parent / "processed" / metadata_path.name / "metadata.json"
        if process_path.exists():
            metadata_path = process_path
        else:
            raise FileNotFoundError(metadata_path)
        
    with open(metadata_path, 'r') as f:
        return json.load(f)
    
def get_uniqueue_counts(metadata):
    
    unique_cls = set()
    count_option = {}
    
    for item in metadata:
        for cls in item['detetion']:
            unique_cls.add(cls['class'])
            if cls['class'] not in count_option:
                count_option[cls['class']] = set()
            count_option[cls['class']].add(cls['count'])
            
    unique_cls = sorted(unique_cls)
    for cls in count_option:
        count_option[cls] = sorted(count_option[cls])
        
    return unique_cls, count_option
        