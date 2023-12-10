import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import Twist
import time

def keep_white_and_yellow(image):
    # in gbr
    white_mask = cv2.inRange(image, np.array([240, 240, 240], dtype=np.uint8), np.array([255, 255, 255], dtype=np.uint8))

    yellow_mask = cv2.inRange(image, np.array([0, 100, 0], dtype=np.uint8), np.array([90, 255, 255], dtype=np.uint8))

    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    result_image = cv2.bitwise_and(image, image, mask=combined_mask)

    return result_image

def keep_middle_strip(image, strip_width):
    height, width = image.shape
    left_bound = (width - strip_width) // 2
    right_bound = left_bound + strip_width
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[::, left_bound: right_bound] = 255
    result_image = cv2.bitwise_and(image, mask)
    return result_image

class PIDController:
    def __init__(self, kp, ki, kd, setpoint):
        self.kp = kp  # Коэффициент пропорциональности
        self.ki = ki  # Коэффициент интеграции
        self.kd = kd  # Коэффициент дифференциации
        self.setpoint = setpoint  # Заданное значение
        self.prev_error = 0  # Предыдущее значение ошибки
        self.integral = 0  # Сумма значений ошибки для интеграции

    def update(self, current_value):
        # Расчет ошибки
        error = self.setpoint - current_value
        # Пропорциональная составляющая
        proportional = self.kp * error
        # Интегральная составляющая
        self.integral += error
        integral = self.ki * self.integral
        # Дифференциальная составляющая
        derivative = self.kd * (error - self.prev_error)
        # Общий выход PID-регулятора
        output = proportional + integral + derivative
        # Сохранение текущего значения ошибки для использования на следующем шаге
        self.prev_error = error
        return output

pid_params = {
    'kp': 0.08,
    'ki': 0,
    'kd': 0,
    'setpoint': 0,

}

params = {
    "view_part": 1 / 3
}
class LaneFollowing(Node):
    def __init__(self):
        super().__init__('lanefollowing')
        self.img_sub = self.create_subscription(Image, '/color/image', self.subs_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.update_timer = self.create_timer(0.01, self.update_callback)
        self.bridge = CvBridge()

        self.pid_controller = PIDController(**pid_params)

        self.frame = None
        self.gray = None
        self.dst = None
        self.prevpt1 = np.array([280, 60])
        self.prevpt2 = np.array([560, 60])
        self.error = 0

    def subs_callback(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.frame = cv2.GaussianBlur(self.frame, (9, 9), cv2.BORDER_DEFAULT)
        self.frame = keep_white_and_yellow(self.frame)

        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        _, self.gray = cv2.threshold(self.gray, 160, 255, cv2.THRESH_BINARY)

        height, width = self.gray.shape
        left_border = int(0.10 * width)
        right_border = int(0.9 * width)

        self.gray[:, :left_border] = 0
        self.gray[:, right_border:] = 0 
          
        blacked_part_size = int(self.gray.shape[0] * (1 - params['view_part']))

        self.dst = self.gray[blacked_part_size:, :]

        if self.dst.dtype != np.uint8:
            self.dst = self.dst.astype(np.uint8)
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(self.dst)

        if retval > 1:
            mindistance1 = []
            mindistance2 = []

            for p in centroids:
                ptdistance1 = np.abs(p - self.prevpt1)
                ptdistance2 = np.abs(p - self.prevpt2)
                mindistance1.append(ptdistance1[0])
                mindistance2.append(ptdistance2[0])

            threshdistance1 = min(mindistance1)
            threshdistance2 = min(mindistance2)

            minlb1 = np.argmin(mindistance1)
            minlb2 = np.argmin(mindistance2)

            cpt1 = (centroids[minlb1, 0], centroids[minlb1, 1])
            cpt2 = (centroids[minlb2, 0], centroids[minlb2, 1])

            if threshdistance1 > 100:
                cpt1 = self.prevpt1
            if threshdistance2 > 100:
                cpt2 = self.prevpt2

        else:
            cpt1 = self.prevpt1
            cpt2 = self.prevpt2

        self.prevpt1 = np.array(cpt1)
        self.prevpt2 = np.array(cpt2)

        fpt = ((cpt1[0] + cpt2[0]) / 2, (cpt1[1] + cpt2[1]) / 2 + blacked_part_size)
        cv2.cvtColor(self.dst, cv2.COLOR_GRAY2BGR)
        for centroid in centroids:
            cv2.circle(self.frame, (int(centroid[0]), int(centroid[1]) + blacked_part_size), 2, (0, 255, 0), 2)
        cv2.circle(self.frame, (int(fpt[0]), int(fpt[1])), 2, (0, 0, 255), 2)
        cv2.circle(self.dst, (int(cpt1[0]), int(cpt1[1])), 2, (0, 0, 255), 2)
        cv2.circle(self.dst, (int(cpt2[0]), int(cpt2[1])), 2, (0, 0, 255), 2)

        self.error = self.dst.shape[1] // 2 - fpt[0]

        cv2.imshow("camera", self.frame)
        cv2.imshow("gray", self.dst)
        cv2.waitKey(1)

    def update_callback(self):
        cmd_vel = Twist()
        output = self.pid_controller.update(self.error)
        cmd_vel.linear.x = 0.2
        cmd_vel.angular.z = -float(output)

        self.cmd_vel_pub.publish(cmd_vel)


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowing()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

