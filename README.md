cat << 'EOF' > README.md
# UR10e Ball Tracking & Kinematics Control (YOLOv8 + HSV & ROS 2)

Ce projet implémente un système complet de suivi de balle et de commande cinématique pour le bras robotisé **Universal Robots UR10e** sous **ROS 2**.

---

##  Architecture du Projet

Le système repose sur une approche hybride IA / Vision Classique :

### 1.  Pipeline Vision Hybride (`vision_node_yolo.py`)
- **Détection de Forme (YOLOv8)** : Utilise un réseau de neurones pour localiser avec précision la géométrie sphérique de la balle.
- **Filtrage de Couleur (Espace HSV)** : Applique un masque colorimétrique HSV pour cibler spécifiquement la **couleur rouge**.
- **Fusion des données** : Combine la détection de forme et le masque de couleur pour s'assurer que seule la balle rouge est suivie.
- **Publication ROS 2** : Transmet les coordonnées spatiales de la cible sur le réseau ROS 2.

### 2.  Planificateur & Contrôle UR10e (`high_level_planner.py` & `ik_node.py`)
- **Planification de Tâche (High-Level Planner)** : Gère la machine à états de la mission et la trajectoire de suivi globale.
- **Cinématique Inverse (IK Node)** : Calcule en temps réel la configuration des 6 articulations (*joints*) du bras UR10e pour positionner l'outil au-dessus de la balle.

---

###  Installation & Utilisation

### 1. Prérequis Python
```bash
pip install ultralytics opencv-python numpy

### 2. Compilation du workspace ROS 2
cd ~/ros2_ws
colcon build --packages-select ur10e_ik_control
source install/setup.bash

### 3. Exécution
# Lancer ik avec le robot sous MuJoCo
ros2 run ur10e_ik_control ik_node

# Lancer le nœud de vision YOLO + HSV
ros2 run ur10e_ik_control vision_node_yolo.py

# Lancer la cinématique et le planificateur du robot
ros2 run ur10e_ik_control high_level_planner.py


