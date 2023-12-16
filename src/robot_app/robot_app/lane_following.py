import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import Twist
import time


def keep_road(image):
    corners_mask = cv2.inRange(image, 
                               np.array([0, 0, 0], dtype=np.uint8), 
                               np.array([2, 2, 2], dtype=np.uint8))
    road_mask = cv2.inRange(image, 
                            np.array([0, 0, 0], dtype=np.uint8), 
                            np.array([60, 60, 60], dtype=np.uint8))
    road_mask = cv2.bitwise_xor(road_mask, corners_mask)
    result_image = cv2.bitwise_and(np.ones_like(image) * 255, 
                                   np.ones_like(image) * 255, 
                                   mask=road_mask)
    return result_image


class PIDController:
    def __init__(self, kp, ki, kd, setpoint):
        self.kp = kp  # Коэффициент пропорциональности
        self.ki = ki  # Коэффициент интеграции
        self.kd = kd  # Коэффициент дифференциации
        self.setpoint = setpoint  # Заданное значение
        self.prev_error = 0  # Предыдущее значение ошибки
        self.integral_sum = 0  # Сумма значений ошибки для интеграции

    def update(self, current_value):
        # Расчет ошибки
        error = self.setpoint - current_value
        # Пропорциональная составляющая
        proportional = self.kp * error
        # Интегральная составляющая
        self.integral_sum += error
        integral = self.ki * self.integral_sum
        # Дифференциальная составляющая
        derivative = self.kd * (error - self.prev_error)
        # Общий выход PID-регулятора
        output = proportional + integral + derivative
        # Сохранение текущего значения ошибки для использования на следующем шаге
        self.prev_error = error
        return output

pid_params = {
    'kp': 0.004,
    'ki': 0.0,
    'kd': 0.0,
    'setpoint': 0,
}

params = {
    "top_border_crop": 7 / 10,
    "bottom_border_crop": 9 / 10, # top_border_crop < bottom_border_crop
    "left_border_crop": 0.0, 
    "right_border_crop": 1.0, # left_border_crop < right_border_crop 
    "max_velocity": 0.35,
    "min_velocity": 0.05,
    # Степень >= 1.0. Чем больше значение, тем больше линейная скорость зависит от ошибки
    "error_impact_on_linear_vel": 2, 
    "previous_point_impact": 0.0, # 0 <= x < 1.0
    "connectivity": 8,
}

class LaneFollowing(Node):
    def __init__(self):
        super().__init__('lanefollowing')
        self.img_sub = self.create_subscription(Image, '/color/image_projected_compensated', self.subs_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.update_timer = self.create_timer(0.01, self.update_callback)
        self.bridge = CvBridge()

        self.allow_running = False

        self.pid_controller = PIDController(**pid_params)
        self.width = None
        self.depth_mask = None
        self.frame = None
        self.gray = None
        self.dst = None
        self.prevpt = None
        self.error = 0
        self.original_image = None

        rclpy.get_default_context().on_shutdown(self.on_shutdown_method)

    def subs_callback(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.original_image = self.frame
        self.frame = keep_road(self.frame)

        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.gray = cv2.GaussianBlur(self.gray, (9, 9), cv2.BORDER_DEFAULT)
        _, self.gray = cv2.threshold(self.gray, 160, 255, cv2.THRESH_BINARY)
        height, self.width = self.gray.shape

        left_crop = int(params["left_border_crop"] * self.width)
        right_crop = int(params["right_border_crop"] * self.width)
        self.gray[:, :left_crop] = 0
        self.gray[:, right_crop:] = 0


        top_crop = int(height * params["top_border_crop"])
        bottom_crop = int(height * params["bottom_border_crop"])
        window_height = bottom_crop - top_crop
        window_center = top_crop + window_height // 2
        self.dst = self.gray[top_crop: bottom_crop, :].astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.dst, connectivity=params["connectivity"])
        if num_labels > 1: # Не считая фон
            # Индекс области с наибольшей площадью (исключаем фон)
            largest_area_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            # Получаем центроиду для области с наибольшей площадью
            if self.prevpt is not None:
                self.prevpt = self.prevpt * params["previous_point_impact"] + centroids[largest_area_index][0] * (1 - params["previous_point_impact"])
            else:
                self.prevpt = int(centroids[largest_area_index][0])

        fpt = (self.prevpt, window_center)

        self.error = fpt[0] - self.width // 2
        # self.error /= (self.self.width / 2) # нормализация ошибки относительно ширины картинки 
        # Рисование
        cv2.cvtColor(self.dst, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(self.frame, 
                      (left_crop, top_crop), 
                      (right_crop, bottom_crop), 
                      (0, 255, 0), 
                      2, 
                      cv2.LINE_AA)
        cv2.circle(self.frame, ((self.width // 2), window_center), 2, (0, 0, 255), 2)
        cv2.circle(self.frame, (int(fpt[0]), int(fpt[1])), 6, (0, 0, 255), 2)
        cv2.imshow("camera", self.frame)
        # cv2.imshow("gray", self.dst)
        cv2.imshow("original_image", self.original_image)
        cv2.waitKey(1)

    def update_callback(self):
        if self.width is not None:
            cmd_vel = Twist()
            output = self.pid_controller.update(self.error)
            if self.allow_running:
                cmd_vel.linear.x = max(params["max_velocity"] * ((1 - abs(self.error) / (self.width // 2)))**params["error_impact_on_linear_vel"], params['min_velocity'])
                cmd_vel.angular.z = float(output)
            self.cmd_vel_pub.publish(cmd_vel)

    def on_shutdown_method(self):
        cmd_vel = Twist()
        cmd_vel.linear.x = 0
        cmd_vel.linear.y = 0
        cmd_vel.linear.z = 0
        cmd_vel.angular.x = 0
        cmd_vel.angular.y = 0
        cmd_vel.angular.z = 0
        self.cmd_vel_pub.publish(cmd_vel) 


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowing()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

