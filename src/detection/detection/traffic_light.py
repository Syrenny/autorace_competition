import rclpy
import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from std_msgs.msg import UInt8
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class TrafficLightDetection(Node):
    def __init__(self):
        super().__init__('traffic_light_detector_node')
        self.current_light = 0

        self.image_camera_subscription = self.create_subscription(
            Image,
            '/color/image',
            self.find_traffic_light,
            10
        )
        self.image_camera_subscription

        self.traffic_light_order_publisher = self.create_publisher(
            UInt8,
            '/sign_detection',
            10
        )
        self.cv_bridge = CvBridge()
        self.frame = None

    def find_traffic_light(self, image_msg):
        # drop the frame to 1/5 (6fps) because of the processing speed. This is up to your computer's operating power.
        image = self.cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
        image = image[:, image.shape[1]//2:]
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        try:
            send_msg = False
            msg = UInt8()

            if self.current_light == 0:
                # lower mask (0-10)
                lower_red = np.array([0,50,50])
                upper_red = np.array([10,255,255])
                mask0 = cv2.inRange(image_hsv, lower_red, upper_red)

                # upper mask (170-180)
                lower_red = np.array([170,50,50])
                upper_red = np.array([180,255,255])
                mask1 = cv2.inRange(image_hsv, lower_red, upper_red)
                # join my masks
                mask = mask0 + mask1
                print(np.sum(mask > 0))
                if np.sum(mask > 0) > 1000:
                    msg.data = self.current_light
                    self.current_light += 1
                    send_msg = True

            elif self.current_light == 1:
                lower_green=np.array([20, 100,100])
                upper_green=np.array([30, 255, 255])
                mask = cv2.inRange(image_hsv, lower_green, upper_green)
                print(np.sum(mask > 0))
                if np.sum(mask > 0) > 1000:
                    msg.data = self.current_light
                    self.current_light += 1
                    send_msg = True

            elif self.current_light == 2:
                lower_green=np.array([50, 100,100])
                upper_green=np.array([70, 255, 255])
                mask = cv2.inRange(image_hsv, lower_green, upper_green)
                print(np.sum(mask > 0))
                if np.sum(mask > 0) > 1000:
                    msg.data = self.current_light
                    self.current_light += 1
                    send_msg = True

            if send_msg:
                self.traffic_light_order_publisher.publish(msg)
                if self.current_light == 3:
                    print('Start moving!!!')
                    sys.exit()
                    
        except Exception:
            pass

        cv2.waitKey(100)


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightDetection()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()