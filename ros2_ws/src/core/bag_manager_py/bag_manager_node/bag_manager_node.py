#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import subprocess
import datetime
import signal
import os
import threading
import rosbag2_py  # 追加

class RosBagManagerNode(Node):
    def __init__(self):
        super().__init__('ros2_bag_manager_node')

        # パラメータ宣言・取得
        self.declare_parameter('output_dir', 'rosbag2_output')
        self.declare_parameter('all_topics', True)
        self.declare_parameter('topics', ['/rosbag2_recorder/trigger'])
        self.output_dir = self.get_parameter('output_dir').get_parameter_value().string_value
        self.all_topics = self.get_parameter('all_topics').get_parameter_value().bool_value
        self.topics = list(self.get_parameter('topics').get_parameter_value().string_array_value)

        # セッション用ディレクトリを作成（ノード起動時）
        now = datetime.datetime.now()
        session_ts = now.strftime('%Y%m%d_%H%M%S')
        self.session_dir = os.path.join(self.output_dir, session_ts)
        os.makedirs(self.session_dir, exist_ok=True)
        self.get_logger().info(f"セッションディレクトリ作成: {self.session_dir}")

        # トリガー購読
        self.trigger_sub = self.create_subscription(
            Bool,
            '/rosbag2_recorder/trigger',
            self.trigger_callback,
            10
        )

        # カット付き停止 (L1) 用を新規追加
        self.cut_trigger_sub = self.create_subscription(Bool, '/rosbag2_recorder/cut_trigger', self.cut_trigger_callback, 10)

        # プロセスと状態管理
        self.recording_process = None
        self.is_recording = False
        self.current_record_dir = None  # 現在録画中のパスを保持　#追加
        self.lock = threading.Lock()

    def trigger_callback(self, msg: Bool):
        """通常トリガーの処理"""
        with self.lock:
            if msg.data:
                self._start_recording()
            else:
                if self.is_recording:
                    self.get_logger().info("通常停止を実行します")
                    self._cleanup()

    def cut_trigger_callback(self, msg: Bool):
        """カットトリガーの処理 (L1用)"""
        with self.lock:
            if msg.data and self.is_recording:
                self.get_logger().info("★★★ 衝突検知: 5秒カットして停止します ★★★")
                path_to_trim = self.current_record_dir
                self._cleanup()
                
                # 加工処理をバックグラウンドで実行
                if path_to_trim:
                    threading.Thread(target=self.trim_bag_data, args=(path_to_trim,)).start()

    def _start_recording(self):
        """録画開始処理"""
        if not self.is_recording:
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            self.current_record_dir = os.path.join(self.session_dir, ts)

            # rosbag2_pyでの編集を容易にするためsqlite3を使用
            cmd = ['ros2', 'bag', 'record']
            if self.all_topics:
                cmd.append('-a')
            elif self.topics:
                cmd.extend(self.topics)
            
            # 保存先とストレージ形式を指定
            cmd.extend(['-o', self.current_record_dir, '-s', 'sqlite3'])

            self.get_logger().info(f"録画開始: {self.current_record_dir}")
            self.recording_process = subprocess.Popen(cmd, preexec_fn=os.setsid)
            self.is_recording = True
        else:
            self.get_logger().warn("すでに録画中です")

    def trim_bag_data(self, input_path):
        """末尾5秒をカットする実処理"""
        try:
            output_path = input_path + "_trimmed"
            self.get_logger().info(f"編集開始: {input_path}")
            
            reader = rosbag2_py.SequentialReader()
            reader.open(
                rosbag2_py.StorageOptions(uri=input_path, storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', '')
            )
            
            metadata = reader.get_metadata()
            # 終了ナノ秒 - 5秒(5,000,000,000ns)
            cutoff_ts = (metadata.starting_time.nanoseconds + metadata.duration.nanoseconds) - (5 * 10**9)
            
            writer = rosbag2_py.SequentialWriter()
            writer.open(
                rosbag2_py.StorageOptions(uri=output_path, storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', '')
            )
            
            for topic in reader.get_all_topics_and_types():
                writer.create_topic(topic)
            
            count = 0
            while reader.has_next():
                topic, data, t = reader.read_next()
                if t > cutoff_ts:
                    break
                writer.write(topic, data, t)
                count += 1
            
            self.get_logger().info(f"編集完了！ {output_path} に保存しました。")
        except Exception as e:
            self.get_logger().error(f"カット処理失敗: {e}")

    def _cleanup(self):
        """録画プロセスが生きていたら SIGINT で停止し、wait する"""
        if self.recording_process:
            try:
                os.killpg(os.getpgid(self.recording_process.pid), signal.SIGINT)
                self.recording_process.wait(timeout=5)
            except Exception as e:
                self.get_logger().error(f"録画プロセス停止中にエラー: {e}")
            finally:
                self.recording_process = None
                self.is_recording = False
                self.get_logger().info("録画を停止しました")

    def destroy_node(self):
        # ノード終了時のクリーンアップ
        self.get_logger().info("ノードシャットダウン: 録画プロセス停止処理を実行します")
        self._cleanup()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RosBagManagerNode()

    # シグナルハンドラ登録（Ctrl+C や docker stop 等）
    def _signal_handler(signum, frame):
        node.get_logger().info(f"シグナル {signal.Signals(signum).name} 受信 — シャットダウンします")
        rclpy.shutdown()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _signal_handler)

    try:
        rclpy.spin(node)
    except Exception:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
