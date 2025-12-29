import streamlit as st
import sys
import torch
from pathlib import Path
import time
from src.inference import YOLOv11Inference
from pathlib import Path
from src.utils import save_object, load_metadata, get_uniqueue_counts


sys.path.append(str(Path(__file__).parent)) # this is useed to make moduler of code from src. import ---- like this

def initate_session_state():
    session_defaults={
        "metadata":None,
        "unique_classes" : [],
        "count_options" : {} 
    }
    
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
            
initate_session_state()

st.set_page_config(page_title="YOLOv11 Search App", layout="wide")
st.title("Image Annoter Application")

option= st.radio("Choose an option:",
                 ("Processing new image", "loading exsisting data"),
                 horizontal=True)


if option == "Processing new image":
    with st.expander("Processing new image", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            imge_dir = st.text_input("Image dir path", placeholder="path/image/image.jpg")
        with col2:
            model_path= st.text_input("Model weight path:", "yolo11m.pt")
            
        if st.button("Start Inference"):
            if imge_dir:
                try:
                    with st.spinner("Running object detection....."):
                        inference = YOLOv11Inference(model_path)
                        metadata = inference.process_directory(imge_dir) # this provide you the list of items in metadata
                        metadata_path = save_object(metadata, imge_dir)
                        st.success(f"✅ Processed {len(metadata)} image(s) successfully!")
                        st.info(f"📁 Metadata saved at: `{metadata_path}`")
                        st.code(str(metadata))
                        st.session_state.metadata = metadata
                        st.session_state.unique_classes, st.session_state.count_options = get_uniqueue_counts(metadata)
                    
                except Exception as e:
                    st.error(f"Error during Inference:{str(e)}")
            else:
                st.warning("Please enter path")
                
        else:
            st.warning(f"Please enter an image directory path")

else:
    with st.expander("Load Existing Metadata", expanded=True):
        metadata_path=st.text_input("Metadata file path", placeholder="path/to/metadata.json")
        
        if st.button("Load Metadata"):
            if metadata_path:
                try:
                    with st.spinner("RUnning object...."):
                        metadata = load_metadata(metadata_path=metadata_path)
                        st.session_state.metadata = metadata
                        st.session_state.unique_classes, st.session_state.count_options = get_uniqueue_counts(metadata)
                        st.success(f"metadata object no {len(metadata)} loaded successful")
                    
                except Exception as e:
                    st.error("error loading json")
            else:
                st.warning("plese enter image dir")