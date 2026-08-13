#!/usr/bin/env python3
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO


class VisionNodeYOLO(Node):

  def __init__(self):
    super().__init__('vision_node_yolo')

    # Bridge ROS 2 <-> OpenCV
    self.bridge = CvBridge()
    self.latest_frame = None

    # Abonnements et Publications
    self.image_sub = self.create_subscription(
        Image, '/ur10e/camera/image_raw', self.image_callback, 10
    )
    self.ball_pub = self.create_publisher(Point, '/ur10e/detected_ball', 10)

    # Paramètres Caméra (Modèle Pinhole 640x480, FOV 60°)
    self.img_width = 640
    self.img_height = 480
    self.cam_x, self.cam_y, self.cam_z = 0.5, 0.0, 1.5
    self.target_z = 0.9

    fov_y_rad = np.radians(60.0)
    self.fy = (self.img_height / 2.0) / np.tan(fov_y_rad / 2.0)
    self.fx = self.fy
    self.cx, self.cy = self.img_width / 2.0, self.img_height / 2.0

    self.display_available = True

    # Chargement YOLOv8
    self.get_logger().info('Chargement du modèle YOLOv8n...')
    self.model = YOLO('yolov8n.pt')

    # Timer à 30 Hz
    self.timer = self.create_timer(0.033, self.process_and_display)
    self.get_logger().info('Nœud Vision Hybride (YOLO + HSV Region) Prêt !')

  def image_callback(self, msg):
    try:
      self.latest_frame = self.bridge.imgmsg_to_cv2(
          msg, desired_encoding='bgr8'
      )
    except Exception as e:
      self.get_logger().error(f'Erreur CvBridge: {e}')

  def is_red_box(self, crop_img):
    """Filtre HSV avec tolérance accrue pour les zones d'ombre."""
    if crop_img.size == 0:
      return False
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

    # Tolérance S=50 et V=30 pour capturer le rouge ombragé sous la balle
    m1 = cv2.inRange(hsv, np.array([0, 50, 30]), np.array([10, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([170, 50, 30]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(m1, m2)

    red_ratio = np.sum(mask > 0) / (crop_img.shape[0] * crop_img.shape[1])
    return red_ratio > 0.05

  def process_and_display(self):
    if self.latest_frame is None:
      return

    cv_image = self.latest_frame.copy()

    # 1. ÉTAPE YOLO : Localisation initiale de la balle
    results = self.model(
        cv_image, conf=0.000038, classes=[32], device='cpu', verbose=False
    )
    ball_found = False

    for result in results:
      for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

        # 2. PADDING LARGE (40px) : Pour ne pas couper les bords/ombres
        padding = 40
        x1_pad, y1_pad = max(0, x1 - padding), max(0, y1 - padding)
        x2_pad, y2_pad = min(self.img_width, x2 + padding), min(
            self.img_height, y2 + padding
        )

        crop = cv_image[y1_pad:y2_pad, x1_pad:x2_pad]

        # 3. ÉTAPE HSV + MORPHOLOGIE : Recadrage sur 100% de la sphère
        if self.is_red_box(crop):
          hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
          m1 = cv2.inRange(
              hsv_crop, np.array([0, 50, 30]), np.array([10, 255, 255])
          )
          m2 = cv2.inRange(
              hsv_crop, np.array([170, 50, 30]), np.array([180, 255, 255])
          )
          mask = cv2.bitwise_or(m1, m2)

          # Fermeture morphologique pour lisser et combler les trous d'éclairage
          kernel = np.ones((3, 3), np.uint8)
          mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

          contours, _ = cv2.findContours(
              mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
          )
          if contours:
            c = max(contours, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(c)

            # Recalcul de la boîte englobante dans le repère global
            real_x1, real_y1 = x1_pad + bx, y1_pad + by
            real_x2, real_y2 = real_x1 + bw, real_y1 + bh

            # Centre (u, v) exact sur la totalité de la balle
            u = float((real_x1 + real_x2) / 2.0)
            v = float((real_y1 + real_y2) / 2.0)

            # 4. PROJECTION 3D
            Z = self.cam_z - self.target_z
            X_cam = (u - self.cx) * Z / self.fx
            Y_cam = (v - self.cy) * Z / self.fy

            world_x = self.cam_x + X_cam
            world_y = self.cam_y - Y_cam
            world_z = self.target_z

            # Publication ROS 2
            self.ball_pub.publish(
                Point(x=float(world_x), y=float(world_y), z=float(world_z))
            )
            ball_found = True

            # Affichage OpenCV
            cv2.rectangle(
                cv_image, (real_x1, real_y1), (real_x2, real_y2), (0, 255, 0), 2
            )
            cv2.circle(cv_image, (int(u), int(v)), 4, (0, 0, 255), -1)
            cv2.putText(
                cv_image,
                f'Red Ball: [{world_x:.2f}, {world_y:.2f}]',
                (real_x1, real_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            break
      if ball_found:
        break

    if not ball_found:
      self.get_logger().warn(
          'Aucune balle rouge détectée.', throttle_duration_sec=3.0
      )

    # Affichage de la fenêtre
    try:
      cv2.imshow('Detection Vision YOLOv8', cv_image)
      cv2.waitKey(1)
    except cv2.error as e:
      if self.display_available:
        self.get_logger().error(f"Erreur d'affichage OpenCV: {e}")
        self.display_available = False


def main(args=None):
  rclpy.init(args=args)
  node = VisionNodeYOLO()
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