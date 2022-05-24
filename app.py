import numpy as np
import cv2
import keras 
import streamlit as st
from tensorflow import keras
from keras.models import load_model
# from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase


# load model
emotion_dict = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]

classifier =load_model('mota.h5')

# load weights into new model
classifier.load_weights("mota.h5")

#load face
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

def main():
    # Face Analysis Application #
    st.title("Real Time Face Emotion Detection Application")
    activiteis = ["Home", "Webcam Face Detection", "About","Contack Us","Error and Solutions"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    # st.sidebar.markdown(""" Developed by Rahul Kumar soni and Ali Asgar lakadwala.""")
    # st.sidebar.markdown(""" LinkedIn profile Links""")
    # st.sidebar.markdown(""">* [Rahul kumar soni] (https://www.linkedin.com/in/rahulsoni1b9757168/)
                             # >* [Ali asgar lakadwala] (https://www.linkedin.com/in/ali-asgar-lakdawala/)""")
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Face Emotion detection application using OpenCV, Custom CNN model and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""The application has two functionalities.""")
        st.write(""">1. Real-time face detection using webcam feed.
                    >2. Real-time face emotion recognization. """)
        
    elif choice == "Webcam Face Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use a webcam and detect your facial emotion")
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
        st.info("If Stuck check Error and Solutions from the sidebar")


    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1= """<div style="background-color:#6D7B8D;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    Real time face emotion detection application using OpenCV, Custom Trained CNN model and Streamlit.</h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

        html_temp4 = """
                                    <div style="background-color:#98AFC7;padding:10px">
                                    <h4 style="color:white;text-align:center;">Rahul Kumar Soni and Ali Asgar Lakdawala created this demo application using the Streamlit Framework, OpenCV, Tensorflow, and Keras libraries. </h4>
                                    <h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                                    </div>
                                    <br></br>
                                    <br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)
    elif choice == "Contack Us":
        st.header("Contact Details")
        st.write(""" LinkedIn profile Links""")
        st.write(""">* [Rahul Kumar Soni] (https://www.linkedin.com/in/rahulsoni1b9757168/)
                           >* [Ali Asgar Lakadwala] (https://www.linkedin.com/in/ali-asgar-lakdawala/)""")
        st.write("""Email Ids""")
        st.write(""">* Rahul Kumar Soni : kr001rahul@gmail.com)
                    >* Ali Asgar Lakadwala : aliasgarlakdawala0209@gmail.com""")


    elif choice == "Error and Solutions":
        st.error('''Could not start video source''')
        st.write('''
                    > * Check for any other application using your camera
                    > * Change the privacy settings of the camera
                    > * Allow browser to access the camera
                 ''')

        
    else:
        pass


if __name__ == "__main__":
    main()