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
        self.model = load_cnn_model(
        "TinyLidarActionLstmNet",
        input_dim=1080,
        output_dim=2
        )

        state = torch.load("ckpts/lstm_real.pth", map_location="cpu")
        self.model.load_state_dict(state)

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

        self.get_logger().info("subscriber created")

    def scan_callback(self, msg):
        scan = np.array(msg.ranges, dtype=np.float32)
        scan[np.isinf(scan)] = 30.0
        scan[np.isnan(scan)] = 30.0
        scan = np.clip(scan, 0.0, 30.0)

        self.buffer.append(scan)

        if len(self.buffer) < 10:
            return

        seq = torch.tensor(np.array(self.buffer), dtype=torch.float32).unsqueeze(0)

        # ★重要：actionは固定（まず安定化）
        pre_action = torch.zeros((1, seq.size(1), 2), dtype=torch.float32)

        with torch.no_grad():
            out, _ = self.model(seq, pre_action)
            out = out.cpu().numpy()[0]

        steer = float(np.clip(out[0], -1.0, 1.0))

        cmd = AckermannDriveStamped()
        cmd.drive.steering_angle = steer * 0.1   # ★弱くする
        cmd.drive.speed = 1.0                   # ★固定

        self.pub.publish(cmd)
        print(out.shape)
        

        print("steer:", steer)


def main():
    rclpy.init()
    node = InferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    
