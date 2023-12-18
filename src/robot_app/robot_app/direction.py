import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from std_msgs.msg import String
import time


class Direction(Node):
    def __init__(self):
        super().__init__('direction')
        self.direction_pub = self.create_publisher(String, '/direction', 10)
        self.detection_sub = self.create_subscription(..., '...', self.subs_callback, 10)
        self.update_timer = self.create_timer(0.01, self.update_callback)
        self.direction = "forward"

    def subs_callback(self, msg):
        pass

    def update_callback(self):
        msg = String()
        msg.data = self.direction
        self.direction_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Direction()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

