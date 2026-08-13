#!/usr/bin/env python3
import time
from geometry_msgs.msg import Point
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


class TrajectoryPlanner(Node):

  def __init__(self):
    super().__init__('high_level_planner')

    self.target_pub = self.create_publisher(
        Point, '/ur10e/target_position', 10
    )
    self.ball_sub = self.create_subscription(
        Point, '/ur10e/detected_ball', self.ball_callback, 10
    )
    self.dist_sub = self.create_subscription(
        Float64, '/ur10e/target_distance', self.distance_callback, 10
    )

    self.state = 'INIT'
    self.current_distance = 999.0
    self.ball_pos = None

    # Positions configurées (X, Y, Z)
    self.initial_pos = Point(x=0.5, y=0.5, z=0.5)
    self.second_pos = Point(x=0.5, y=0.0, z=0.7)

    self.loop_count = 0
    self.MAX_LOOPS = 4

    # Timestamp pour imposer un délai minimum après publication
    self.last_state_change_time = 0.0
    self.MIN_WAIT_TIME = 1.0  # Attendre au moins 1 seconde avant d'accepter l'arrivée à destination

    self.timer = self.create_timer(0.05, self.step_machine)
    self.get_logger().info('Planner de trajectoire prêt avec verrouillage.')

  def ball_callback(self, msg: Point):
    self.ball_pos = msg

  def distance_callback(self, msg: Float64):
    self.current_distance = msg.data

  def step_machine(self):
    PRECISION_TOUCH = 0.001  # 1 mm
    PRECISION_AUTRE = 0.005  # 5 mm

    now = self.get_clock().now().nanoseconds / 1e9

    if self.ball_pos is None or self.state == 'FINISHED':
      return

    # Bloquer la vérification si 1 seconde ne s'est pas écoulée depuis le dernier changement d'état
    time_since_change = now - self.last_state_change_time

    if self.state == 'INIT':
      self.loop_count += 1
      self.get_logger().info(
          f'=== CYCLE {self.loop_count}/{self.MAX_LOOPS} ==='
      )
      self.get_logger().info('--> Aller vers la balle (Toucher à 1mm)')

      self.current_distance = 999.0
      self.target_pub.publish(self.ball_pos)
      self.last_state_change_time = now
      self.state = 'BALL'

    elif self.state == 'BALL':
      if time_since_change >= self.MIN_WAIT_TIME:
        if self.current_distance <= PRECISION_TOUCH:
          self.get_logger().info(
              '--> Balle touchée ! Aller vers la position (0.5, 0.0, 0.7)'
          )

          self.current_distance = 999.0
          self.target_pub.publish(self.second_pos)
          self.last_state_change_time = now
          self.state = 'GOING_TO_SECOND_POS'

    elif self.state == 'GOING_TO_SECOND_POS':
      if time_since_change >= self.MIN_WAIT_TIME:
        if self.current_distance <= PRECISION_AUTRE:
          if self.loop_count < self.MAX_LOOPS:
            self.get_logger().info(
                '--> Position (0.5, 0.0, 0.7) atteinte. Relance du cycle '
                f' ({self.loop_count}/{self.MAX_LOOPS})'
            )
            self.last_state_change_time = now
            self.state = 'INIT'
          else:
            self.get_logger().info(
                '--> 4 cycles terminés ! Retour final à la position initiale '
                ' (0.5, 0.5, 0.5)'
            )

            self.current_distance = 999.0
            self.target_pub.publish(self.initial_pos)
            self.last_state_change_time = now
            self.state = 'GOING_HOME'

    elif self.state == 'GOING_HOME':
      if time_since_change >= self.MIN_WAIT_TIME:
        if self.current_distance <= PRECISION_AUTRE:
          self.get_logger().info(
              '--> Séquence complète terminée ! Robot revenu en position'
              ' initiale.'
          )
          self.state = 'FINISHED'


def main(args=None):
  rclpy.init(args=args)
  node = TrajectoryPlanner()
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
  main()