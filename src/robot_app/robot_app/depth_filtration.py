from distutils import dep_util
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class DepthFilter(Node):
    def __init__(self):
        super().__init__('depth_filter')

        self.declare_parameters(
            namespace='',
            parameters=[
            ('max_depth', 80),
        ])

        self.img_depth_sub = self.create_subscription(Image, '/depth/image', self.depth_callback, 10)
        self.img_sub = self.create_subscription(Image, '/color/image', self.subs_callback, 10)
        self.img_pub = self.create_publisher(Image, '/color/depth_filtered', 10)
        self.bridge = CvBridge()

        self.depth_mask = None
        self.max_depth = self.get_parameter("max_depth").get_parameter_value().integer_value

    def depth_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        self.depth_mask =  cv2.inRange(image, 
                                       np.array([0], dtype=np.uint8), 
                                       np.array([self.max_depth], dtype=np.uint8))
        
    def subs_callback(self, msg):
        if self.depth_mask is not None:
            self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            filtrated_image = cv2.bitwise_and(self.frame, self.frame, mask=self.depth_mask)
            self.img_pub.publish(self.bridge.cv2_to_imgmsg(filtrated_image, "bgr8"))


def main(args=None):
    rclpy.init(args=args)
    node = DepthFilter()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

