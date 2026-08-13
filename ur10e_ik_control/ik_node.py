#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import mujoco
import mujoco.viewer
import numpy as np
import cv2

class LowLevelIK(Node):
    def __init__(self):
        super().__init__('low_level_ik')

        # Charger le modèle MuJoCo
        xml_path = "/home/localuser/mujoco_ur10e/mujoco_menagerie/universal_robots_ur10e/scene.xml"
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Identification du site
        self.ee_site_name = "attachment_site"
        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name
        )
        
        if self.ee_site_id == -1:
            self.get_logger().error(f"ERREUR : Site '{self.ee_site_name}' introuvable ! Vérifie le nom.")

        # Récupération dynamique des indices de joints
        self.qpos_indices = []
        for i in range(self.model.nu):
            joint_id = self.model.actuator_trnid[i, 0]
            self.qpos_indices.append(self.model.jnt_qposadr[joint_id])
        self.qpos_indices = np.array(self.qpos_indices)

        # Cible initiale (Home) accessible dans le workspace
        self.target = np.array([0.4, 0.2, 0.4])

        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, '/ur10e/camera/image_raw', 10)
        self.target_sub = self.create_subscription(
            Point, '/ur10e/target_position', self.target_callback, 10
        )
        self.dist_pub = self.create_publisher(Float64, '/ur10e/target_distance', 10)

        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.create_timer(0.033, self.publish_camera_frame)

        self.get_logger().info("Nœud LowLevelIK prêt.")

    def target_callback(self, msg: Point):
        self.target = np.array([msg.x, msg.y, msg.z])
        self.get_logger().info(f"Nouvelle cible : X={msg.x:.3f}, Y={msg.y:.3f}, Z={msg.z:.3f}")

    def compute_ik_step(self, dt=0.002):
        if self.target is None or self.ee_site_id == -1:
            print("BLOCAGE : Target non définie ou site ID invalide.")
            return

        # 1. Position actuelle de l'end-effector
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        error = self.target - ee_pos
        dist = np.linalg.norm(error)

        # Zone morte (3 mm) : verrouillage sur cible
        if dist < 0.003:
            self.dist_pub.publish(Float64(data=float(dist)))
            return

        # 2. Jacobienne de position (3 x nv)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)

        # Extraction des colonnes associées aux joints du bras
        J = jacp[:, self.qpos_indices]

        # 3. Damped Least Squares (DLS)
        damping = 1e-2
        JJt = J @ J.T
        lambda_sq = (damping ** 2) * np.eye(3)
        gain = 2.0
        
        dq = J.T @ np.linalg.solve(JJt + lambda_sq, gain * error)

        # 4. Saturation de la vitesse articulaire
        max_joint_vel = 1.0  # rad/s
        dq = np.clip(dq, -max_joint_vel * dt, max_joint_vel * dt)

        # 5. Contrôle cinématique direct (Bypass des moteurs pour éliminer les blocages)
        self.data.qpos[self.qpos_indices] += dq
        if self.model.nu >= len(self.qpos_indices):
            self.data.ctrl[:len(self.qpos_indices)] = self.data.qpos[self.qpos_indices]

        # Re-calcul immédiat de la géométrie
        mujoco.mj_fwdPosition(self.model, self.data)

        self.dist_pub.publish(Float64(data=float(dist)))

    def publish_camera_frame(self):
        try:
            self.renderer.update_scene(self.data, camera="overhead_cam")
            rgb_img = self.renderer.render()
            bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            img_msg = self.bridge.cv2_to_imgmsg(bgr_img, encoding="bgr8")
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = "overhead_cam"
            self.image_pub.publish(img_msg)
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = LowLevelIK()

    with mujoco.viewer.launch_passive(node.model, node.data) as viewer:
        while rclpy.ok() and viewer.is_running():
            node.compute_ik_step(dt=node.model.opt.timestep)
            mujoco.mj_step(node.model, node.data)
            rclpy.spin_once(node, timeout_sec=0.001)
            viewer.sync()

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()