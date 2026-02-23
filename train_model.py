import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from model import EmotionCNN # importowanie modelu z model.py


# 1. USTAWIENIA I HIPERPARAMETRY

BATCH_SIZE = 64 # ilość próbkowanych zdjęć 
learning_rate = 0.001 
EPOCHS = 50 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(f"Uzywane urzadzenie: {DEVICE}")


# 2. PRZYGOTOWANIE DANYCH (TRANSFORMACJE)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# Wczytywanie datasetu z folderów
try:
    train_ds = datasets.ImageFolder(root='train', transform=transform)
    val_ds = datasets.ImageFolder(root='test', transform=val_transform)
except Exception as e:
    print("Blad: Brak folderów 'train' i 'test' w folderze projektu.")
    exit()

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"Klasy: {train_ds.classes}")


# 3. INICJALIZACJA MODELU

cnn = EmotionCNN(num_classes=7)
cnn.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)

# Listy do przechowywania wyników
train_losses = []
best_acc = 0.0


# 4. PĘTLA TRENINGOWA

for epoch in range(EPOCHS):
    total_loss = 0.0
    cnn.train() # Przełączamy model w tryb treningu
    
    for inputs, labels in train_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()           # Reset gradientów
        outputs = cnn(inputs)           # Feed forward
        loss = criterion(outputs, labels.long()) 
        loss.backward()                 # Backpropagation
        optimizer.step()                # Aktualizacja wag
        
        total_loss += loss.item()
    
    # Obliczamy średnią stratę w epoce
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    #  WALIDACJA 
    cnn.eval() # Tryb ewaluacji (wyłącza dropout)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = cnn(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS}: loss = {avg_loss:.4f}, Val Acc = {val_acc:.2f}%")

    # Zapisujemy model tylko w przypadku gdy osiągnał najlepsze val_acc
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(cnn.state_dict(), "emotion_model.pth")

print(f"\nKoniec treningu. Najlepszy wynik: {best_acc:.2f}%")


# 5. EWALUACJA KOŃCOWA I WYKRESY



print("Wczytywanie najlepszego modelu do analizy...")
cnn.load_state_dict(torch.load("emotion_model.pth"))
cnn.eval()

#  A. Zbieranie danych do Macierzy Pomyłek 
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = cnn(inputs)
        _, predicted = torch.max(outputs, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# B. Rysowanie Macierzy Pomyłek 
cm = confusion_matrix(all_labels, all_preds)
class_names = train_ds.classes

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Przewidziana klasa')
plt.ylabel('Prawdziwa klasa')
plt.title(f'Macierz Pomyłek (Best Acc: {best_acc:.2f}%)')
plt.show()

# C. Wizualizacja Predykcji (16 zdjęć) 

dataiter = iter(val_loader)
images, labels = next(dataiter)
images = images.to(DEVICE)
labels = labels.to(DEVICE)

outputs = cnn(images)
_, preds = torch.max(outputs, 1)

fig = plt.figure(figsize=(12, 12))
for idx in range(16):
    ax = fig.add_subplot(4, 4, idx+1, xticks=[], yticks=[])
    
    # Konwersja tensora na obrazek do wyświetlenia
    img = images[idx][0].cpu().numpy()
    ax.imshow(img, cmap='gray')
    
    pred_label = class_names[preds[idx]]
    true_label = class_names[labels[idx]]
    
    color = 'green' if pred_label == true_label else 'red'
    ax.set_title(f"P: {pred_label}\nR: {true_label}", color=color)

plt.tight_layout()
plt.show()

# D. Wykres Straty (Loss) 
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.title('Training Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()