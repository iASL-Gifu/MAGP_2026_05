from pathlib import Path
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import get_typestore, Stores


def extract_and_save_per_bag(bag_path, output_dir, scan_topic, cmd_topic):
    bag_path = Path(bag_path).expanduser().resolve()
    bag_name = bag_path.name
    out_dir = Path(output_dir) / bag_name
    out_dir.mkdir(parents=True, exist_ok=True)

    MAX_SPEED = 3.0 

    scan_data, scan_times = [], []
    cmd_data, cmd_times = [], []
    
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        connections = [c for c in reader.connections if c.topic in [scan_topic, cmd_topic]]
        for conn, timestamp, raw in reader.messages(connections=connections):
            msg = reader.deserialize(raw, conn.msgtype)

            TARGET_DIM = 1080
            
            if conn.topic == scan_topic and conn.msgtype == 'sensor_msgs/msg/LaserScan':
                scan = np.array(msg.ranges, dtype=np.float32)

                # --- 長さを統一 ---
                if len(scan) > TARGET_DIM:
                    scan = scan[:TARGET_DIM]
                elif len(scan) < TARGET_DIM:
                    scan = np.pad(scan, (0, TARGET_DIM - len(scan)), constant_values=0.0)

                scan_data.append(scan)
                scan_times.append(timestamp)

            elif conn.topic == cmd_topic and conn.msgtype == 'geometry_msgs/msg/Twist':
                steering = msg.angular.z
                speed = msg.linear.x
                
                speed = msg.linear.x / MAX_SPEED
                speed = np.clip(speed, 0.0, 1.0)
                
                cmd_data.append(np.array([steering, speed], dtype=np.float32))
                cmd_times.append(timestamp)  # ← これが無いのが原因

    if len(scan_data) == 0 or len(cmd_data) == 0:
        print(f'[WARN] Skipping {bag_name}: insufficient data')
        return

    # 時刻同期
    scan_data, scan_times = np.array(scan_data), np.array(scan_times)
    cmd_data, cmd_times = np.array(cmd_data), np.array(cmd_times)

    synced_scans, synced_steers, synced_speeds = [], [], []
    for i, stime in enumerate(scan_times):
        idx = np.argmin(np.abs(cmd_times - stime))
        synced_scans.append(scan_data[i])
        synced_steers.append(cmd_data[idx][0])
        synced_speeds.append(cmd_data[idx][1])

    # 保存
    np.save(out_dir / 'scans.npy', np.array(synced_scans))
    np.save(out_dir / 'steers.npy', np.array(synced_steers))
    np.save(out_dir / 'speeds.npy', np.array(synced_speeds))
    print(f'[SAVE] {bag_name}: {len(synced_scans)} samples')


def extract_all_bags_in_dir(bags_dir, output_dir, scan_topic, cmd_topic):
    bags_dir = Path(bags_dir).expanduser().resolve()
    bag_dirs = sorted([p for p in bags_dir.iterdir() if (p / 'metadata.yaml').exists()])

    print(f"[INFO] Found {len(bag_dirs)} rosbag directories.")
    for bag_path in bag_dirs:
        extract_and_save_per_bag(bag_path, output_dir, scan_topic, cmd_topic)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bags_dir', required=True, help='Path to directory containing rosbag folders')
    parser.add_argument('--outdir', required=True, help='Output root directory')
    parser.add_argument('--scan_topic', default='/scan_filtered')
    parser.add_argument('--cmd_topic', default='/jetracer/cmd_drive')
    args = parser.parse_args()

    extract_all_bags_in_dir(args.bags_dir, args.outdir, args.scan_topic, args.cmd_topic)