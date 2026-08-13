#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')

        self.bridge = CvBridge()
        self.latest_frame = None
        self.frame_received_count = 0

        self.image_sub = self.create_subscription(
            Image,
            '/ur10e/camera/image_raw',
            self.image_callback,
            10
        )

        self.ball_pub = self.create_publisher(
            Point,
            '/ur10e/detected_ball',
            10
        )

        # Paramètres vision / pinhole
        self.img_width = 640
        self.img_height = 480
        self.cam_x = 0.5   # position X de la caméra dans le monde
        self.cam_y = 0.0   # position Y de la caméra dans le monde
        self.cam_z = 1.5
        self.target_z = 0.9

        fov_y_rad = np.radians(60.0)
        self.fy = (self.img_height / 2.0) / np.tan(fov_y_rad / 2.0)
        self.fx = self.fy
        self.cx = self.img_width / 2.0
        self.cy = self.img_height / 2.0

        self.display_available = True

        self.timer = self.create_timer(0.033, self.process_and_display)

        self.get_logger().info("VisionNode prêt et à l'écoute sur /ur10e/camera/image_raw")

    def image_callback(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.frame_received_count += 1

            if self.frame_received_count % 90 == 0:
                self.get_logger().info(
                    f"Frames reçues: {self.frame_received_count}, "
                    f"min={self.latest_frame.min()} max={self.latest_frame.max()}"
                )
        except Exception as e:
            self.get_logger().error(f"Erreur CvBridge: {e}")

    def process_and_display(self):
        if self.latest_frame is None:
            self.get_logger().warn(
                "Aucune frame reçue sur /ur10e/camera/image_raw. "
                "Vérifie que le nœud low_level_ik tourne et publie bien.",
                throttle_duration_sec=3.0
            )
            return

        cv_image = self.latest_frame.copy()
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ball_found = False
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 50:
                ball_found = True
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    u = float(M["m10"] / M["m00"])
                    v = float(M["m01"] / M["m00"])

                    # Distance caméra -> plan de la balle
                    Z = self.cam_z - self.target_z

                    # Offsets en mètres dans le repère caméra (pinhole)
                    X_cam = (u - self.cx) * Z / self.fx   # décalage horizontal (u)
                    Y_cam = (v - self.cy) * Z / self.fy   # décalage vertical (v)

                    # --- FIX: mapping correct selon l'orientation réelle de la caméra ---
                    # x_local caméra = +X monde  -> u croissant = X monde croissant
                    # y_local caméra = +Y monde, mais v croissant = -Y_local = -Y monde
                    world_x = self.cam_x + X_cam
                    world_y = self.cam_y - Y_cam
                    world_z = self.target_z

                    target_msg = Point(x=float(world_x), y=float(world_y), z=float(world_z))
                    self.ball_pub.publish(target_msg)

                    cv2.circle(cv_image, (int(u), int(v)), 8, (0, 255, 0), -1)
                    cv2.putText(cv_image, f"Cible: [{world_x:.2f}, {world_y:.2f}]",
                                (int(u) + 10, int(v)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if not ball_found:
            self.get_logger().warn(
                "Aucune balle rouge détectée dans l'image (mask vide ou trop petit).",
                throttle_duration_sec=3.0
            )

        try:
            cv2.imshow("Detection Vision OpenCV", cv_image)
            cv2.waitKey(1)
        except cv2.error as e:
            if self.display_available:
                self.get_logger().error(
                    f"Impossible d'ouvrir une fenêtre d'affichage (pas de X server ?): {e}"
                )
                self.display_available = False


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()

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