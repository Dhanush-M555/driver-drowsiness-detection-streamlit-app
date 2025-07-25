import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import streamlit as st
from tensorflow import keras
from streamlit_webrtc import webrtc_streamer
from keras.utils import img_to_array
import numpy as np
from threading import Thread
import time
import cv2
import av
import base64

def start_alarm(s):
    # with open(sound, "rb") as f:
    #     data = f.read()
    #     b64 = base64.b64encode(data).decode()
    #     md = f"""
    #         <audio controls autoplay="true">
    #         <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    #         </audio>
    #         """
    #     st.markdown(
    #         md,
    #         unsafe_allow_html=True,
    #     )
    # st.write("# Auto-playing Audio!")
    html_string = """
            <audio controls autoplay>
              <source src="https://www.orangefreesounds.com/wp-content/uploads/2022/04/Small-bell-ringing-short-sound-effect.mp3" type="audio/mp3">
            </audio>
            """
    sound = st.empty()
    sound.markdown(html_string, unsafe_allow_html=True)  # will display a st.audio with the sound you specified in the "src" of the html_string and autoplay it
    time.sleep(1)  # wait for 2 seconds to finish the playing of the audio
    sound.empty()  # optionally delete the element afterwards



classes = ['Closed', 'Open']
face_cascade = cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")
left_eye_cascade = cv2.CascadeClassifier(r"haarcascade_lefteye_2splits.xml")
right_eye_cascade = cv2.CascadeClassifier(r"haarcascade_righteye_2splits.xml")
cap = cv2.VideoCapture(0)
model = keras.models.load_model(r"drowiness_new2.h5")
count = 0
alarm_on = False
alarm_sound = "alarm.mp3"
status1 = ''
status2 = ''
# Initialize the variable corresponding to the FPS calculation
prev_frame_time = 0
new_frame_time = 0


def drowsiness_detection(frame):
    global count, alarm_on
    img = frame.to_ndarray(format="bgr24")
    frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle around the face
        roi_gray = frame_gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        left_eye = left_eye_cascade.detectMultiScale(roi_gray)
        right_eye = right_eye_cascade.detectMultiScale(roi_gray)
        for (x1, y1, w1, h1) in left_eye:
            eye1 = roi_color[y1:y1+h1, x1:x1+w1]
            eye1 = cv2.resize(eye1, (145, 145))
            eye1 = eye1.astype('float') / 255.0
            eye1 = img_to_array(eye1)
            eye1 = np.expand_dims(eye1, axis=0)
            pred1 = model.predict(eye1)
            status1 = np.argmax(pred1)
            if status1 == 2:
                count += 1
                if count >= 10:
                    cv2.putText(img, "Drowsiness Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    st.warning("Drowsiness Alert!!!")
                    if not alarm_on:
                        alarm_on = True
                        t = Thread(target=start_alarm, args=(alarm_sound,))
                        t.daemon = True
                        # st.report_thread.add_report_ctx(t)
                        # st.ReportThread.add_report_ctx(t)
                        # st.script_run_context.add_script_run_ctx(t)
                        t.start()
                        cv2.rectangle(roi_color, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)  # Red rectangle for closed eyes
            else:
                count = 0
                alarm_on = False
                cv2.rectangle(roi_color, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)  # Green rectangle for open eyes
        for (x2, y2, w2, h2) in right_eye:
            eye2 = roi_color[y2:y2 + h2, x2:x2 + w2]
            eye2 = cv2.resize(eye2, (145, 145))
            eye2 = eye2.astype('float') / 255.0
            eye2 = img_to_array(eye2)
            eye2 = np.expand_dims(eye2, axis=0)
            pred2 = model.predict(eye2)
            status2 = np.argmax(pred2)
            if status2 == 2:
                count += 1
                if count >= 10:
                    cv2.putText(img, "Drowsiness Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    st.warning("Drowsiness Alert!!!")
                    if not alarm_on:
                        alarm_on = True
                        t = Thread(target=start_alarm, args=(alarm_sound,))
                        t.daemon = True
                        # st.report_thread.add_report_ctx(t)
                        # st.ReportThread.add_report_ctx(t)
                        # st.script_run_context.add_script_run_ctx(t)
                        t.start()
                        cv2.rectangle(roi_color, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)  # Red rectangle for closed eyes
            else:
                count = 0
                alarm_on = False
                # cv2.rectangle(roi_color, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)  # Green rectangle for open eyes
    return av.VideoFrame.from_ndarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), format="rgb24")

def main():
    st.title("Driver Drowsiness Detector")
    webrtc_ctx = webrtc_streamer(key="example", video_frame_callback=drowsiness_detection,rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })


if __name__ == "__main__":
    main()
