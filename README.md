# Sürücü Yorgunluk Tespiti ve Hibrit Uyarı Sistemi

Bu proje, sürücülerin dikkat ve yorgunluk durumlarını gerçek zamanlı olarak
tespit eden bir yapay zeka tabanlı sistemdir.

Göz durumu (açık / kapalı) ve esneme (yawn) analizleri ayrı ayrı
derin öğrenme modelleri ile yapılmış, ardından hibrit bir karar mekanizması
ile birleştirilmiştir.

## Kullanılan Teknolojiler
- Python
- TensorFlow / Keras
- OpenCV
- MobileNetV2
- Tkinter

## Klasör Yapısı

├─ kodlar/
├─ models/
├─ dataset/
├─ processed_dataset/


## Çalıştırma
1. Gerekli kütüphaneleri yükleyin:
pip install tensorflow opencv-python numpy scikit-learn


2. Canlı sistem:
python kodlar/hibrit_uyari_tkinter.py

3. Model performans ölçümü:
python kodlar/performans_olcum.py


## Not
Bu proje akademik amaçlı geliştirilmiştir.







