import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import threading

# ==============================
# MODELLERİ YÜKLE
# ==============================
eye_model = load_model("D:/Surucu_Projesi/models/eye_model.h5")
yawn_model = load_model("D:/Surucu_Projesi/models/yawn_model.h5")

# ==============================
# HİBRİT KARAR FONKSİYONU
# ==============================
def hibrit_karar(eye_pred, yawn_pred):
    """Eye veya Yawn uykulu ise UYKULU, ikisi de dikkatli ise DİKKATLİ"""
    if eye_pred == 1 or yawn_pred == 1:
        return "UYKULU!", "red"
    return "DİKKATLİ", "green"

# ==============================
# FRAME ÖN İŞLEME
# ==============================
def preprocess_frame(frame, target_size=(224,224)):
    img = cv2.resize(frame, target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# ==============================
# TKINTER ARAYÜZ
# ==============================
class HibritUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Sürücü Yorgunluk Alarmı")
        self.root.geometry("800x600")

        # Alarm Label
        self.alarm_label = Label(self.root, text="Başlatılıyor...", font=("Helvetica", 32), width=20)
        self.alarm_label.pack(pady=10)

        # Video Label
        self.video_label = Label(self.root)
        self.video_label.pack()

        # Kamera aç
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Kamera açılamadı!")
            exit()

        # Thread ile video loop
        threading.Thread(target=self.video_loop, daemon=True).start()
        self.root.mainloop()

    def video_loop(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Ön işlemler
            eye_input = preprocess_frame(frame)
            yawn_input = preprocess_frame(frame)

            # Tahminler
            eye_pred = np.argmax(eye_model.predict(eye_input, verbose=0)[0])
            yawn_pred = np.argmax(yawn_model.predict(yawn_input, verbose=0)[0])

            # Hibrit karar
            status, color = hibrit_karar(eye_pred, yawn_pred)
            self.alarm_label.config(text=status, bg=color)

            # Frame'i Tkinter için dönüştür
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.cap.release()

# ==============================
# PROGRAMI BAŞLAT
# ==============================
if __name__ == "__main__":
    HibritUI()
