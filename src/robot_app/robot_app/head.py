import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Integer
import time

signs = {
    "red_light": 0,
    "yellow_light": 1,
    "green_light": 2,
    "intersection": 3,
    "right": 4,
    "left": 5,
    "construction": 6,
    "parking": 7,
    "pedestrian": 8,
    "tunnel": 9
}

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
        self.state_publisher = self.create_publisher(Integer, '/state', 10)
        self.detection_sub = self.create_subscription(..., '...', self.detection_callback, 10)
        self.update_timer = self.create_timer(0.01, self.update_callback)
        self.current_state = None

    def detection_callback(self, msg):
        self.sign_handler(msg.data)

    def update_callback(self):
        if self.current_state is not None:
            msg = Integer()
            msg.data = self.current_state
            self.state_publisher.publish(msg)

    def sign_handler(self, sign):
        if sign == signs["red_light"] or sign == signs["yellow_light"]:
            self.current_state = states["stop"]
        elif sign == signs["green_light"]:
            self.current_state = states["forward"]
        elif sign == signs["intersection"]:
            pass 
        elif sign == signs["right"]:
            self.current_state = states["right"]
        elif sign == signs["left"]:
            self.current_state = states["left"]
        elif sign == signs["construction"]:
            self.current_state = states["right"]
        elif sign == signs["parking"]:
            pass
        elif sign == signs["pedestrian"]:
            pass 
        elif sign == signs["tunnel"]:
            pass
        

def main(args=None):
    rclpy.init(args=args)
    node = Head()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

