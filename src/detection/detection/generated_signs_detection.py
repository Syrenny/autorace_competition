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

class GenSignDetection(Node):
    def __init__(self):
        super().__init__('denerative_sing_detector_node')
        self.min_match_count = 3
        self.current_sign_number = 0
        self.prepare_detector()

        self.image_camera_subscription = self.create_subscription(
            Image,
            '/color/image',
            self.find_generated_sign,
            10
        )
        self.image_camera_subscription

        self.sign_subscription = self.create_subscription(
            UInt8,
            '/sign_detection',
            self.check_if_intersection,
            10
        )
        self.sign_subscription

        self.direction_signs_publisher = self.create_publisher(
            UInt8,
            '/sign_detection',
            10
        )
        self.cv_bridge = CvBridge()
        self.frame = None
        self.intersection_achieved = False

    def check_if_intersection(self, msg):
        if msg.data == 0:
            print("intersection was found")
            self.intersection_achieved = True

    def detect_and_compute_current(self):
        self.left_sign = cv2.imread(self.signs_path + '/' + 'to_the_left.png', 0)
        self.right_sign = cv2.imread(self.signs_path + '/' + 'to_the_right.png', 0)
        self.left_kp, self.left_des = self.orb.detectAndCompute(self.left_sign, None)
        self.right_kp, self.right_des = self.orb.detectAndCompute(self.right_sign, None)

    def prepare_detector(self):
        self.orb = cv2.ORB_create()
        self.signs_path = os.path.dirname(os.path.realpath(__file__)) + '/generative_signs'
        self.detect_and_compute_current()
        self.brute_force_match = cv2.BFMatcher()

    def find_generated_sign(self, image_msg):
        if self.intersection_achieved:
            cv_image_input = self.cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
            cv_image_input = cv2.cvtColor(cv_image_input, cv2.COLOR_BGR2GRAY)
            cv_image_input = cv_image_input[:cv_image_input.shape[0]//2, 200:cv_image_input.shape[1]-200]

            try:
                input_kp, input_des = self.orb.detectAndCompute(cv_image_input, None)
                left_matches = self.brute_force_match.knnMatch(self.left_des, input_des, k=2)
                good_left = []
                for m, n in left_matches:
                    if m.distance < 0.7 * n.distance:
                        good_left.append(m)

                right_matches = self.brute_force_match.knnMatch(self.right_des, input_des, k=2)
                good_right = []
                for m, n in right_matches:
                    if m.distance < 0.7 * n.distance:
                        good_right.append(m)

                if len(good_left) > len(good_right) and len(good_left) > self.min_match_count:
                    msg_sign_number = UInt8()
                    msg_sign_number.data = 5
                    self.direction_signs_publisher.publish(msg_sign_number)
                    print('found left sign')
                    sys.exit()

                elif len(good_left) < len(good_right) and len(good_right) > self.min_match_count:
                    msg_sign_number = UInt8()
                    msg_sign_number.data = 4
                    self.direction_signs_publisher.publish(msg_sign_number)
                    print('found right sign')
                    sys.exit()
                else:
                    print(len(good_left), len(good_right))
                    
            except Exception:
                pass
        cv2.waitKey(100)


def main(args=None):
    rclpy.init(args=args)
    node = GenSignDetection()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()