import tkinter as tk
from tkinter import filedialog, Label, Button
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from PIL import Image, ImageTk
import numpy as np
import cv2
import os

def FacialExpressionModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)
    model.load_weights(weights_file)
    model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=['accuracy'])
    return model

top = tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('Arial', 15, 'bold'))
sign_image = Label(top)

xml_path = 'C:\\Users\\DELL\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml'
if not os.path.isfile(xml_path):
    raise FileNotFoundError(f"The file {xml_path} does not exist. Please download it from the OpenCV repository.")
facec = cv2.CascadeClassifier(xml_path)
if facec.empty():
    raise ValueError(f"Failed to load the cascade classifier from {xml_path}")

model = FacialExpressionModel("model_a.json", "model_weights.weights.h5")
EMOTION_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def Detect(file_path):
    try:
        image = cv2.imread(file_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_image, 1.3, 5)
        if len(faces) == 0:
            raise ValueError("No faces detected")
        for (x, y, w, h) in faces:
            fc = gray_image[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            roi = roi[np.newaxis, :, :, np.newaxis]  # Ensure the shape is correct for prediction
            pred = EMOTION_LIST[np.argmax(model.predict(roi))]
            print("Predicted emotion is " + pred)
            label1.configure(foreground="#011638", text=pred)
    except Exception as e:
        print(e)
        label1.configure(foreground="#011638", text="Unable to detect")

def show_Detect_button(file_path):
    detect_b = Button(top, text="Detect Emotion", command=lambda: Detect(file_path), padx=10, pady=5)
    detect_b.configure(font='bold')
    detect_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        uploaded = Image.open(file_path)
        uploaded.thumbnail((top.winfo_width() / 2.3, top.winfo_height() / 2.3))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        show_Detect_button(file_path)
    except Exception as e:
        print(e)

upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('Arial', 12, 'bold'))
upload.pack(side='bottom')
sign_image.pack(side='bottom', expand=True)
label1.pack(side='bottom', expand=True)
heading = Label(top, text="Facial Emotion Detection", pady=20, font=('Arial', 25, 'bold'), background='#CDCDCD', foreground='#011638')
heading.pack()

top.mainloop()
