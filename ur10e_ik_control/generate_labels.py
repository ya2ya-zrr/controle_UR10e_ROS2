import os
import cv2
import pandas as pd
import numpy as np

img_dir = os.path.expanduser('~/dataset_mujoco/images')
data = []

# Parcourir les images
for filename in sorted(os.listdir(img_dir)):
    if filename.endswith('.jpg'):
        img_path = os.path.join(img_dir, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            continue

        # Masque HSV pour détecter la balle rouge
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(m1, m2)
        
        # Extraire le centre de masse (u, v)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                u = float(M["m10"] / M["m00"])
                v = float(M["m01"] / M["m00"])
                data.append([filename, u, v])

# Sauvegarde dans le bon dossier
df = pd.DataFrame(data, columns=['image', 'u', 'v'])
csv_path = os.path.expanduser('~/dataset_mujoco/dataset.csv')
df.to_csv(csv_path, index=False)
print(f"✅ Fichier créé : {csv_path} avec {len(df)} entrées.")