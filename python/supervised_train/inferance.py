import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

import torch
import numpy as np
from collections import deque

from src.models.models import load_cnn_model


class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')

        # --- モデル ---
        self.model = load_cnn_model("TinyLidarNet", input_dim=1080, output_dim=2)
        self.model.load_state_dict(torch.load("ckpts/model_epoch_64_loss_0.0030.pth"))
        
        self.model.eval()

        # --- buffer ---
        self.buffer = deque(maxlen=10)
        # --- subscriber ---
        self.sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # --- publisher ---
        self.pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

    def scan_callback(self, msg):
        scan = np.array(msg.ranges, dtype=np.float32)


        self.buffer.append(scan)

        if len(self.buffer) < 10:
            return

        seq = np.array(self.buffer)
        seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            out = self.model(seq).numpy()[0]

        # 安全対策
        steer = float(np.clip(out[0], -1.0, 1.0))
        speed = float(np.clip(out[1], 0.0, 1.0))

        cmd = AckermannDriveStamped()

        cmd.drive.steering_angle = steer * 0.15  # ←重要（そのままだと曲がりすぎ）
        cmd.drive.speed = speed * 4          # ←適度にスケール

        self.pub.publish(cmd)
        print(out)


def main():
    rclpy.init()
    node = InferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    
