import streamlit as st
import cv2
import numpy as np
from PIL import Image
from predictor import Predictor
import base64

# Initialize the predictor
predictor = Predictor()

def set_bg_hack(main_bg):
    '''
    A function to set the background image.
    '''
    main_bg_ext = "jpg"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/{main_bg_ext};base64,{main_bg});
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def process_frame(frame):
    predictions = predictor.predict(frame)
    for result in predictions:
        x1, y1, x2, y2 = result['position']
        label = result['label']
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame

def main():
    # Load and set the background image
    with open("background.jpg", "rb") as image_file:
        image_bytes = image_file.read()
        encoded_string = base64.b64encode(image_bytes).decode()
        set_bg_hack(encoded_string)
    
    # Centered title
    st.markdown("<h1 style='text-align: center; color: white;'>EMOTION, AGE and GENDER DETECTION</h1>", unsafe_allow_html=True)
    
    # Styling the buttons
    st.markdown("""
        <style>
        .css-1aumxhk {
            background-color: #4CAF50;
            color: white;
            font-size: 20px;
            font-weight: bold;
            padding: 15px 32px;
            border-radius: 12px;
            text-align: center;
            cursor: pointer;
        }
        </style>
        """, unsafe_allow_html=True)

    # Manage state
    if 'option' not in st.session_state:
        st.session_state.option = None
    if 'cap' not in st.session_state:
        st.session_state.cap = None
    if 'stop_capture' not in st.session_state:
        st.session_state.stop_capture = False

    # Select input method
    if st.session_state.option is None:
        st.session_state.option = st.selectbox("Choose Input Method", ["Select", "Upload", "Capture"])
    
    if st.session_state.option == "Upload":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            processed_frame = process_frame(frame)
            st.image(processed_frame, channels="BGR", use_column_width=True)
    
    elif st.session_state.option == "Capture":
        if st.session_state.cap is None:
            st.session_state.cap = cv2.VideoCapture(0)
            st.session_state.stop_capture = False

        stframe = st.empty()
        stop_button = st.button("Stop Capture")
        
        while not st.session_state.stop_capture:
            ret, frame = st.session_state.cap.read()
            if not ret:
                st.write("Error: Could not read frame.")
                break

            frame = process_frame(frame)
            stframe.image(frame, channels="BGR", use_column_width=True)

            if stop_button:
                st.session_state.stop_capture = True

        # Release capture and clean up
        if st.session_state.stop_capture:
            st.session_state.cap.release()
            st.session_state.cap = None
            st.session_state.stop_capture = False

    if st.button("Reset"):
        st.session_state.option = None
        st.session_state.stop_capture = False
        if st.session_state.cap:
            st.session_state.cap.release()
            st.session_state.cap = None

if __name__ == "__main__":
    main()



