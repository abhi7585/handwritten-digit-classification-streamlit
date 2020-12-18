# from win32com.client import Dispatch
from keras.models import load_model
import cv2
import os
from PIL import Image, ImageEnhance
import numpy as np
import streamlit as st
import warnings
warnings.filterwarnings('ignore')


# def speak(text):
#     speak = Dispatch("SAPI.SpVoice")
#     speak.Speak(text)


model = load_model("model_trained.p")


def preprocessing(img):
    try:
        img = img.astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        img = img/255
        return img
    except Exception as e:
        img = img.astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        img = img/255
        return img


def main():
    st.title("Handwritten Digit Classification Web App")
    st.set_option('deprecation.showfileUploaderEncoding', False)
    activities = ["Program", "Credits"]
    choices = st.sidebar.selectbox("Select Option", activities)

    if choices == "Program":
        st.subheader("Kindly upload file below")
        img_file = st.file_uploader("Upload File", type=['png', 'jpg', 'jpeg'])
        if img_file is not None:
            up_img = Image.open(img_file)
            st.image(up_img)
        if st.button("Predict Now"):
            try:
                img = np.asarray(up_img)
                img = cv2.resize(img, (32, 32))
                img = preprocessing(img)
                img = img.reshape(1, 32, 32, 1)
                prediction = model.predict(img)
                classIndex = model.predict_classes(img)
                probabilityValue = np.amax(prediction)
                if probabilityValue > 0.90:
                    if classIndex == 0:
                        st.success("0")
                        # speak("Predicted Number is Zero")
                    elif classIndex == 1:
                        st.success("1")
                        # speak("Predicted Number is One")
                    elif classIndex == 2:
                        st.success("2")
                        # speak("Predicted Number is Two")
                    elif classIndex == 3:
                        st.success("3")
                        # speak("Predicted Number is Three")
                    elif classIndex == 4:
                        st.success("4")
                        # speak("Predicted Number is Four")
                    elif classIndex == 5:
                        st.success("5")
                        # speak("Predicted Number is Five")
                    elif classIndex == 6:
                        st.success("6")
                        # speak("Predicted Number is Six")
                    elif classIndex == 7:
                        st.success("7")
                        # speak("Predicted Number is Seven")
                    elif classIndex == 8:
                        st.success("8")
                        # speak("Predicted Number is Eight")
                    elif classIndex == 9:
                        st.success("9")
                        # speak("Predicted Number is Nine")
                else:
                    st.success("Invalid input image or Image too large")
            except Exception as e:
                st.error("Connection Error")

    elif choices == 'Credits':
        st.write(
            "Application Developed by Abhishek Tripathi, Aman Verma, Manvendra Pratap Singh.")


if __name__ == '__main__':
    main()
