import os
import warnings
import numpy as np
from PIL import Image, ImageFilter
import streamlit as st
from streamlit_drawable_canvas import st_canvas

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
warnings.filterwarnings('ignore', category=UserWarning)

from keras.models import load_model

st.set_page_config(page_title="AI Handwritten Digit Recognition", layout="centered")
st.title("✍️ AI Handwritten Digit Recognition")
st.write("Draw a single digit (0-9) inside the canvas box below and see the prediction real-time!")

@st.cache_resource
def load_digit_model():
    if os.path.exists('digit_cnn_model.keras'):
        return load_model('digit_cnn_model.keras', compile=False)
    elif os.path.exists('mnist.h5'):
        return load_model('mnist.h5', compile=False)
    else:
        st.error("Model file not found! Please run train.py first.")
        return None

model = load_digit_model()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Canvas Layout")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=18, 
        stroke_color="#000000",
        background_color="#FFFFFF",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.subheader("Prediction Metrics")
    
    if canvas_result.image_data is not None and model is not None:
        raw_img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
        gray_img = raw_img.convert('L')
        
        temp_arr = np.array(gray_img)
        total_pixels = np.sum(temp_arr < 235)  
        
        if total_pixels < 30:
            st.info("Draw something on the canvas to trigger estimation parameters.")
        elif total_pixels > 75000:  
            st.warning("🚨 Position contains excessive artifact noise. Clear and redraw.")
        else:
            bbox = gray_img.getbbox() 
            if bbox:
                cropped_img = gray_img.crop(bbox)
                
                w, h = cropped_img.size
                max_dim = max(w, h) + 40 
                centered_canvas = Image.new('L', (max_dim, max_dim), color=255)
                
                paste_x = (max_dim - w) // 2
                paste_y = (max_dim - h) // 2
                centered_canvas.paste(cropped_img, (paste_x, paste_y))
                
                img_resized = centered_canvas.resize((28, 28), resample=Image.LANCZOS)
            else:
                img_resized = gray_img.resize((28, 28), resample=Image.LANCZOS)
            
            img_numpy = np.array(img_resized, dtype=np.float32)
            img_final_inverted = 255.0 - img_numpy  
            
            img_pil_temp = Image.fromarray(img_final_inverted.astype('uint8'))
            img_thickened = img_pil_temp.filter(ImageFilter.MaxFilter(3))
            img_final_array = np.array(img_thickened, dtype=np.float32)
            
            final_input = img_final_array.reshape(1, 28, 28, 1) / 255.0
            
            res = model.predict(final_input, verbose=0)[0]
            sorted_res = np.sort(res)
            margin = sorted_res[-1] - sorted_res[-2]
            
            if max(res) < 0.70 or margin < 0.35:
                st.error("❌ Not a Digit")
                st.caption(f"Margin Delta too narrow to isolate specific configuration context ({max(res)*100:.2f}% target).")
            else:
                digit = np.argmax(res)
                confidence = max(res) * 100
                
                st.metric(label="Predicted Target Digit", value=str(digit))
                st.metric(label="System Class Confidence Value", value=f"{confidence:.2f}%")
                
                st.progress(float(max(res)))