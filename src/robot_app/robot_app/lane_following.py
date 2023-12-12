import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import Twist
import time

def keep_middle(image, left_border, right_border):
    _, width = image.shape
    left_border = int(left_border * width)
    right_border = int(right_border * width)
    image[:, :left_border] = 0
    image[:, right_border:] = 0
    return image

def keep_road(image):
    corners_mask = cv2.inRange(image, np.array([0, 0, 0], dtype=np.uint8), np.array([2, 2, 2], dtype=np.uint8))
    road_mask = cv2.inRange(image, np.array([0, 0, 0], dtype=np.uint8), np.array([60, 60, 60], dtype=np.uint8))
    road_mask = cv2.bitwise_xor(road_mask, corners_mask)
    result_image = cv2.bitwise_and(np.ones_like(image) * 255, np.ones_like(image) * 255, mask=road_mask)
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
    'kp': 0.01,
    'ki': 0.0,
    'kd': 0.0,
    'setpoint': 0,
}

params = {
    "max_depth": 100,
    "view_part": 1 / 4,
    "left_border_crop": 0.0, 
    "right_border_crop": 1.0,
    "max_velocity": 0.4,
    "min_velocity": 0.05,
    "error_impact_on_linear_vel": 4, # Степень >= 1.0. Чем больше значение, тем больше линейная скорость зависит от ошибки
    "previous_point_impact": 0.6, # 0 <= x < 1.0
    "connectivity": 8
}

class LaneFollowing(Node):
    def __init__(self):
        super().__init__('lanefollowing')
        self.img_sub = self.create_subscription(Image, '/color/image_projected_compensated', self.subs_callback, 10)
        self.img_depth_sub = self.create_subscription(Image, '/depth/image', self.depth_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.update_timer = self.create_timer(0.01, self.update_callback)
        self.bridge = CvBridge()

        self.pid_controller = PIDController(**pid_params)

        self.depth_mask = None
        self.frame = None
        self.gray = None
        self.dst = None
        self.prevpt = None
        self.error = 0
        self.width = None
        self.original_image = None

        rclpy.get_default_context().on_shutdown(self.on_shutdown_method)

    def depth_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        self.depth_mask =  cv2.inRange(image, np.array([0], dtype=np.uint8), np.array([params['max_depth']], dtype=np.uint8))

    def subs_callback(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.original_image = self.frame
        # self.frame = keep_white_and_yellow(self.frame)
        self.frame = keep_road(self.frame)

        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.gray = cv2.GaussianBlur(self.gray, (9, 9), cv2.BORDER_DEFAULT)
        _, self.gray = cv2.threshold(self.gray, 160, 255, cv2.THRESH_BINARY)

        self.gray = keep_middle(self.gray, params["left_border_crop"], params["right_border_crop"])
        self.height, self.width = self.gray.shape
        
        blacked_part_size = int(self.gray.shape[0] * (1 - params['view_part']))
        window_center = blacked_part_size + (self.height - blacked_part_size) // 2
        self.dst = self.gray[blacked_part_size:, :].astype(np.uint8)

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
        # self.error /= (self.width / 2) # нормализация ошибки относительно ширины картинки 
        # Рисование
        cv2.cvtColor(self.dst, cv2.COLOR_GRAY2BGR)
        for (px, py) in centroids:
            cv2.circle(self.frame, (int(px), int(py) + blacked_part_size), 2, (0, 255, 0), 2)
        cv2.circle(self.frame, ((self.width // 2), window_center), 2, (0, 0, 255), 2)
        cv2.circle(self.frame, (int(fpt[0]), int(fpt[1])), 6, (0, 0, 255), 2)
        cv2.imshow("camera", self.frame)
        cv2.imshow("gray", self.dst)
        cv2.imshow("original_image", self.original_image)
        cv2.waitKey(1)

    def update_callback(self):
        if self.width is not None:
            cmd_vel = Twist()
            output = self.pid_controller.update(self.error)
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

