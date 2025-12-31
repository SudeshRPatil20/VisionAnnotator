import streamlit as st
import sys
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import base64
import time
import io
from src.inference import YOLOv11Inference
from pathlib import Path
from src.utils import save_object, load_metadata, get_uniqueue_counts
import base64
from zipfile import ZipFile  
import os                   

sys.path.append(str(Path(__file__).parent)) 

def img_to_base64(image : Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def initate_session_state():
    session_defaults={
        "metadata":None,
        "unique_classes" : [],
        "count_options" : {},
        "search_results":[],
        "show_boxes":True,
        "grid_columns":3,
        "highlight_matches":True,
        "search_params" : {
            "search_mode" : "Any of selected classes (OR)",
            "selected_classes" : [],
            "thresholds" : {}
        }
    }
    
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
            
initate_session_state()

st.set_page_config(page_title="YOLOv11 Search App", layout="wide")
st.title("Image Annoter Application")

# ⭐ CSS ADDED
st.markdown("""
<style>
.image-card {
    border: 2px solid #e3e3e3;
    border-radius: 12px;
    padding: 10px;
    margin-bottom: 20px;
    background: #fafafa;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
    text-align:center;
}
.image-container img {
    width:100%;
    border-radius:10px;
    cursor:pointer;
    transition:0.25s;
}
.image-container img:hover {
    transform:scale(1.04);
    box-shadow:0 6px 15px rgba(0,0,0,0.20);
}
.meta-overlay {
    margin-top:10px;
    font-size:14px;
    color:#444;
}
.download-btn {
    margin-top:8px;
    padding:6px 12px;
    background:#4CAF50;
    color:white;
    text-decoration:none;
    border-radius:6px;
    font-size:13px;
    display:inline-block;
}
</style>
""", unsafe_allow_html=True)


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
                        metadata = inference.process_directory(imge_dir)
                        metadata_path = save_object(metadata, imge_dir)
                        st.success(f"✅ Processed {len(metadata)} image(s) successfully!")
                        st.info(f"📁 Metadata saved at: `{metadata_path}`")
                        st.session_state.metadata = metadata
                        st.session_state.unique_classes, st.session_state.count_options = get_uniqueue_counts(metadata)
                    
                except Exception as e:
                    st.error(f"Error during Inference:{str(e)}")
            else:
                st.warning("Please enter path")
                

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
                
if st.session_state.metadata:
    st.header("Search Engine")
    
    with st.container():
        st.radio("Search mode:",
                 ("Any of selected classes (or)", "All selected classes (AND)"),
                 horizontal = True
        )
        
        st.session_state.search_params["selected_classes"] = st.multiselect(
            "Classes to search for:",
            options = st.session_state.unique_classes
        )
        
        if st.session_state.search_params["selected_classes"]:
            st.subheader("Count Threshold (optional)")
            cols = st.columns(len(st.session_state.search_params["selected_classes"]))
            for i, cls in enumerate(st.session_state.search_params["selected_classes"]):
                with cols[i]:
                    st.session_state.search_params["thresholds"][cls] = st.selectbox(
                        f"Max count for {cls}",
                        options = ["None"] + st.session_state.count_options[cls]
                    )
        
        if st.button("Search Image", type="primary") and st.session_state.search_params["selected_classes"]:
            result= []
            search_params = st.session_state.search_params
            
            for item in st.session_state.metadata:
                matches = False
                class_matches = {}
                
                for cls in search_params["selected_classes"]:
                    class_detections = [d for d in item['detetion'] if d['class'] == cls]
                    class_count = len(class_detections)
                    class_matches[cls] = False
                    
                    threshold = search_params["thresholds"].get(cls, "None")
                    if threshold == "None":
                        class_matches[cls] = (class_count >= 1)
                    else:
                        class_matches[cls] = (class_count >= 1 and class_count <= int(threshold))
                        
                if search_params["search_mode"] == "Any of selected classes (OR)":
                    matches = any(class_matches.values())
                    
                else:
                    matches = all(class_matches.values())
                    
                if matches:
                    result.append(item)
            st.session_state.search_results = result
            

if st.session_state.search_results:
    results =st.session_state.search_results
    search_params = st.session_state.search_params
    
    st.subheader(f"Result: {(len(results))} matching images")
    
    with st.expander("Display Options", expanded = True):
        cols = st.columns(3)
        with cols[0]:
            st.session_state.show_boxes = st.checkbox(
                "Show bounding boxes",
                value = st.session_state.show_boxes,
            )
        with cols[1]:
            st.session_state.grid_columns = st.slider(
                "Grid columns",
                min_value = 2,
                max_value = 6,
                value = st.session_state.grid_columns,
            )
            
        with cols[2]:
            st.session_state.highlight_matches = st.checkbox(
                "Highlight matches classes",
                value = st.session_state.highlight_matches,
            )
            
    grid_cols = st.columns(st.session_state.grid_columns)
    col_index = 0
    
    for result in results:
        with grid_cols[col_index]:
            try:
                
                img = Image.open(result["image_path"])
                draw= ImageDraw.Draw(img)
                
                try:
                    font = ImageFont.truetype("arial.ttf", 12)
                except:
                    font = ImageFont.load_default()
                if st.session_state.show_boxes:      
                    for det in result['detetion']:
                        cls = det['class']
                        bbox = det['bbox']
                        
                        if cls in search_params["selected_classes"]:
                            color = "#DFDCEC"
                            thickness = 3
                        elif not st.session_state.highlight_matches:
                            color = "#666666"
                            thickness= 1
                        else:
                            continue
                        
                        draw.rectangle(bbox, outline=color, width=thickness)
                        
                        if cls in search_params["selected_classes"] or not st.session_state.highlight_matches:
                            label = f"{cls} {det['confidence']:.2f}"
                            text_box = draw.textbbox((0,0), label, font=font)
                            text_width = text_box[2] - text_box[0]
                            text_height = text_box[3] - text_box[1]
                            
                            draw.rectangle([bbox[0], bbox[1], bbox[0] + text_width + 8, bbox[1] + text_height + 8],
                                        outline=color,
                                        fill = color)
                            
                            draw.text(
                                (bbox[0]+4, bbox[1]+2),
                                label,
                                fill="white",
                                font = font
                            )
            
                meta_items = [f"{k}:{v}" for k, v in result["class_count"].items() if k in search_params["selected_classes"]]
                
                img_b64 = img_to_base64(img)

                # ⭐ CLICK TO PREVIEW + DOWNLOAD
                st.markdown(f"""
                <div class = "image-card">
                    <div class="image-container">
                        <a href="data:image/png;base64,{img_b64}" target="_blank">
                            <img src="data:image/png;base64,{img_b64}">
                        </a>
                    </div>
                    <div class="meta-overlay">
                        <strong>{Path(result['image_path']).name}</strong><br>
                        {", ".join(meta_items) if meta_items else "No matches"}
                    </div>
                    <a download="{Path(result['image_path']).stem}_annotated.png"
                       href="data:image/png;base64,{img_b64}"
                       class="download-btn">⬇️ Download Image</a>
                </div>
                """, unsafe_allow_html = True)

            except Exception as e:
                st.error(f"Error displaying {result['image_path']} : {(e)}")

        # ⭐ FIX (added only - required)
        col_index = (col_index + 1) % st.session_state.grid_columns


# ⭐ DOWNLOAD ALL AS ZIP
if st.session_state.search_results:
    if st.button("📦 Download All Annotated Images as ZIP", type="primary"):
        zip_path = "annotated_results.zip"
        with ZipFile(zip_path, "w") as zipf:
            for item in st.session_state.search_results:
                try:
                    img = Image.open(item["image_path"])
                    draw = ImageDraw.Draw(img)
                    for det in item["detetion"]:
                        bbox = det["bbox"]
                        draw.rectangle(bbox, outline="red", width=3)
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    buf.seek(0)
                    zipf.writestr(Path(item["image_path"]).stem + "_annotated.png", buf.read())
                except:
                    pass

        with open(zip_path, "rb") as f:
            st.download_button("⬇️ Download ZIP File", f, file_name="annotated_images.zip")
