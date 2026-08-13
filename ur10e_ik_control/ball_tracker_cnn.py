#!/usr/bin/env python3
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

# Architecture CNN identique à train_cnn.py
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

class BallTrackerNode(Node):
    def __init__(self):
        super().__init__('ball_tracker_cnn')
        self.bridge = CvBridge()
        
        model_path = os.path.expanduser('~/dataset_mujoco/ball_cnn.pth')
        self.model = BallDetectorCNN()
        
        if not os.path.exists(model_path):
            self.get_logger().error(f"❌ Fichier modèle introuvable : {model_path}")
            self.model_loaded = False
            return
            
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        self.model_loaded = True
        
        self.fx, self.fy = 554.25, 554.25  
        self.cx, self.cy = 320.0, 240.0   
        self.z_depth = 0.9  
        
        self.sub = self.create_subscription(Image, '/ur10e/camera/image_raw', self.image_callback, 10)
        self.ball_pub = self.create_publisher(Point, '/ur10e/detected_ball', 10)
        self.get_logger().info("🧠 Nœud CNN chargé avec succès ! Tracking 3D actif sur /ur10e/detected_ball")

    def image_callback(self, msg):
        if not self.model_loaded:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w, _ = frame.shape

        img_resized = cv2.resize(frame, (128, 128))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_tensor = np.transpose(img_normalized, (2, 0, 1))
        img_tensor = torch.tensor(img_tensor).unsqueeze(0)

        with torch.no_grad():
            pred = self.model(img_tensor).numpy()[0]

        u_pred = float(np.clip(pred[0], 0, 1) * w)
        v_pred = float(np.clip(pred[1], 0, 1) * h)

        x_3d = (u_pred - self.cx) * self.z_depth / self.fx
        y_3d = (v_pred - self.cy) * self.z_depth / self.fy
        z_3d = self.z_depth

        point_msg = Point()
        point_msg.x, point_msg.y, point_msg.z = x_3d, y_3d, z_3d
        self.ball_pub.publish(point_msg)

        cv2.circle(frame, (int(u_pred), int(v_pred)), 8, (0, 255, 0), -1)
        cv2.putText(frame, f"3D: ({x_3d:.2f}, {y_3d:.2f}, {z_3d:.2f})m", 
                    (int(u_pred) + 10, int(v_pred) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("CNN Ball Tracking 3D", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = BallTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()