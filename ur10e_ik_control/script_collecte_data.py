#!/usr/bin/env python3
import os
import glob
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image, '/ur10e/camera/image_raw', self.callback, 10
        )

        # Dossier où sauvegarder les images
        self.save_dir = os.path.expanduser('~/dataset_mujoco/images')
        os.makedirs(self.save_dir, exist_ok=True)

        # Astuce : Reprendre le compteur automatique sans écraser les anciennes images
        existing_images = glob.glob(os.path.join(self.save_dir, 'ball_*.jpg'))
        if existing_images:
            # Récupère le plus grand numéro existant
            indices = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in existing_images]
            self.img_count = max(indices) + 1
        else:
            self.img_count = 0

        self.get_logger().info(f"📁 Sauvegarde dans : {self.save_dir}")
        self.get_logger().info(f"🔢 Prochain index : ball_{self.img_count:03d}.jpg")
        self.get_logger().info("📸 Appuie sur 'S' pour enregistrer une image | 'Q' pour quitter")

    def callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        display_frame = frame.copy()

        cv2.putText(
            display_frame,
            f"Images capturees : {self.img_count}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        cv2.imshow("Collecte de Data", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Appuie sur 's' pour Sauvegarder
            img_path = os.path.join(self.save_dir, f'ball_{self.img_count:03d}.jpg')
            cv2.imwrite(img_path, frame)
            self.get_logger().info(f"✅ Image enregistrée : ball_{self.img_count:03d}.jpg")
            self.img_count += 1
        elif key == ord('q'):  # Quitter proprement avec 'q'
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = DataCollector()
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