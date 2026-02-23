import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model import EmotionCNN

# Konfiguracja
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Kolory
colors = {
    'neutral': (128, 128, 128),  # Szary
    'sad':     (255, 0, 0),      # Niebieski
    'fear':    (128, 0, 128),    # Fioletowy
    'happy':   (0, 255, 0),      # Zielony
    'surprise':(255, 255, 255),  # Biały
    'angry':   (0, 0, 255),      # Czerwony
    'disgust': (0, 255, 255)     # Żółty
}

# Wczytanie modelu
cnn = EmotionCNN(num_classes=7)
cnn.to(DEVICE)
try:
    cnn.load_state_dict(torch.load("emotion_model.pth", map_location=DEVICE))
    cnn.eval()
    print("Model wczytany! Naciśnij 'q' aby wyjść.")
except Exception as e:
    print(f"Błąd wczytywania modelu: {e}")
    exit()

# Transformacja
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# Kamerka
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
if not cap.isOpened(): cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Odbicie lustrzane
    frame = cv2.flip(frame, 1)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y+h, x:x+w]
        
        # Przygotowanie obrazu
        roi_pil = Image.fromarray(roi_gray)
        image = transform(roi_pil)
        image = image.unsqueeze(0).to(DEVICE)

        # Predykcja
        with torch.no_grad():
            outputs = cnn(image)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            
            emotion = class_names[pred_class]
            confidence = probs[0, pred_class].item() * 100

        # Wybór koloru
        color = colors.get(emotion, (255, 255, 255))

        # Rysowanie samej ramki i napisu
        
        # 1. Prosta ramka
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # 2. Napis w TYM SAMYM kolorze co ramka, bezpośrednio nad nią
        cv2.putText(frame, f"{emotion} ({int(confidence)}%)", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()