import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

import numpy as np


class BarrierDetector (Node):
    def __init__(self):
        super().__init__('pedestrian')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_scan_callback,
            10
        )

    def laser_scan_callback(self, msg):
        cmd_vel_msg = Twist()
        # [164:197]
        center_point = len(msg.ranges)
        #view = np.array(msg.ranges[2:-2])
        view1 = np.array(msg.ranges[center_point - 18: center_point - 1])
        view2 = np.array(msg.ranges[:18])
        view = np.concatenate((view1, view2), axis=0)

        # self.get_logger().info('view1: {}'.format(view1))
        # self.get_logger().info('view2: {}'.format(view2))
        self.get_logger().info('view: {}'.format(view))

        
        # view1 = view1 * np.cos(np.radians(np.abs(np.abs((np.array(range(0, 18))) - 180))))
        # view2 = view2 * np.cos(np.radians(np.abs(np.array(range(center_point - 18, center_point - 1)) - 180)))
        # view = np.concatenate((view1, view2), axis=0)
        if np.all(view > 0.42):
            cmd_vel_msg.linear.x = 0.3
        self.publisher.publish(cmd_vel_msg)
        self.get_logger().info(f'Published cmd_vel: linear.x={cmd_vel_msg.linear.x}, '
                               f'angular.z={cmd_vel_msg.angular.z}')


def main():
    rclpy.init()

    barrier_detector = BarrierDetector ()
    rclpy.spin(barrier_detector)

    barrier_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()