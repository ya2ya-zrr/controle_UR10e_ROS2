#!/usr/bin/env python3
import os
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 1. Architecture CNN PyTorch (Normalisation + Régression)
class BallDetectorCNN(nn.Module):
    def __init__(self):
        super(BallDetectorCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

# 2. Dataset personnalisable
class BallDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        
        possible_img_cols = ['image_name', 'filename', 'image', 'img', 'file']
        self.img_col = next((c for c in possible_img_cols if c in self.df.columns), self.df.columns[0])
        
        possible_u_cols = ['u', 'x', 'u_norm', 'center_x']
        possible_v_cols = ['v', 'y', 'v_norm', 'center_y']
        
        self.u_col = next((c for c in possible_u_cols if c in self.df.columns), self.df.columns[1])
        self.v_col = next((c for c in possible_v_cols if c in self.df.columns), self.df.columns[2])
        
        print(f"📊 Colonnes utilisées -> Image: '{self.img_col}', U: '{self.u_col}', V: '{self.v_col}'")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row[self.img_col])
        img_path = os.path.join(self.img_dir, img_name)
        
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Impossible de lire l'image : {img_path}")
            
        h, w, _ = image.shape
        image_resized = cv2.resize(image, (128, 128))
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_tensor = np.transpose(image_normalized, (2, 0, 1))
        
        u_val = float(row[self.u_col])
        v_val = float(row[self.v_col])
        
        if u_val > 1.0 or v_val > 1.0:
            u_val /= w
            v_val /= h
            
        labels = np.array([u_val, v_val], dtype=np.float32)

        return torch.tensor(image_tensor), torch.tensor(labels)

def train():
    dataset_dir = os.path.expanduser('~/dataset_mujoco')
    csv_file = os.path.join(dataset_dir, 'dataset.csv')
    img_dir = os.path.join(dataset_dir, 'images')
    save_path = os.path.join(dataset_dir, 'ball_cnn.pth')

    print("🚀 Chargement du dataset...")
    dataset = BallDataset(csv_file, img_dir)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Entraînement sur : {device}")

    model = BallDetectorCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)

    epochs = 120  # Augmenté à 120 pour converger vers une précision sub-pixel
    print(f"🏋️ Démarrage de l'entraînement sur {epochs} époques...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataset)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Époque [{epoch+1}/{epochs}] - Loss MSE: {epoch_loss:.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"✅ Entraînement terminé ! Modèle sauvegardé dans : {save_path}")

if __name__ == '__main__':
    train()