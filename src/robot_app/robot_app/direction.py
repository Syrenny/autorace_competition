import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Integer
import time


states = {
    "disable": 0,
    "stop": 1,
    "forward": 2,
    "left": 3,
    "right": 4
}

class Head(Node):
    def __init__(self):
        super().__init__('head')
        self.direction_pub = self.create_publisher(Integer, '/state', 10)
        self.detection_sub = self.create_subscription(..., '...', self.subs_callback, 10)
        self.update_timer = self.create_timer(0.01, self.update_callback)
        self.current_state = states["forward"]

    def subs_callback(self, msg):
        pass

    def update_callback(self):
        msg = Integer()
        msg.data = self.current_state
        self.direction_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Head()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

