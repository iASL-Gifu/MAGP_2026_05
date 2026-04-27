from pathlib import Path
import numpy as np
from rosbags.highlevel import AnyReader


def extract_and_save_per_bag(bag_path, output_dir, scan_topic, cmd_topic):
    bag_path = Path(bag_path).expanduser().resolve()
    bag_name = bag_path.name
    out_dir = Path(output_dir) / bag_name
    out_dir.mkdir(parents=True, exist_ok=True)

    scan_data, scan_times = [], []
    cmd_data, cmd_times = [], []

    with AnyReader([bag_path]) as reader:
        connections = [c for c in reader.connections if c.topic in [scan_topic, cmd_topic]]
        for conn, timestamp, raw in reader.messages(connections=connections):
            msg = reader.deserialize(raw, conn.msgtype)

            if conn.topic == scan_topic and conn.msgtype == 'sensor_msgs/msg/LaserScan':
                scan_data.append(np.array(msg.ranges, dtype=np.float32))
                scan_times.append(timestamp)

            elif conn.topic == cmd_topic and conn.msgtype == 'ackermann_msgs/msg/AckermannDrive':
                cmd_data.append(np.array([msg.steering_angle, msg.speed], dtype=np.float32))
                cmd_times.append(timestamp)

    if len(scan_data) == 0 or len(cmd_data) == 0:
        print(f'[WARN] Skipping {bag_name}: insufficient data')
        return

    # 時刻同期
    scan_data, scan_times = np.array(scan_data), np.array(scan_times)
    cmd_data, cmd_times = np.array(cmd_data), np.array(cmd_times)

    synced_scans, synced_steers, synced_speeds = [], [], []
    for i, stime in enumerate(scan_times):
        idx = np.argmin(np.abs(cmd_times - stime))
        
        s = np.nan_to_num(scan_data[i], posinf=10.0, neginf=0.0)
        s = s / 10.0

        synced_scans.append(s)
        synced_steers.append(cmd_data[idx][0])
        synced_speeds.append(cmd_data[idx][1])
        
    final_X = np.array(synced_scans, dtype=np.float32)[:, np.newaxis, :]
    final_y = np.array(synced_steers, dtype=np.float32)

    # 保存
    np.save(out_dir / 'X.npy', final_X)
    np.save(out_dir / 'y.npy', final_y)
    print(f'[SAVE] {bag_name}: {final_X.shape} samples saved for CNN')
    np.save(out_dir / 'scans.npy', np.array(synced_scans))
    np.save(out_dir / 'steers.npy', np.array(synced_steers))
    np.save(out_dir / 'speeds.npy', np.array(synced_speeds))
    print(f'[SAVE] {bag_name}: {len(synced_scans)} samples')


def extract_all_bags_in_dir(bags_dir, output_dir, scan_topic, cmd_topic):
    bags_dir = Path(bags_dir).expanduser().resolve()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    bag_dirs = sorted([p for p in bags_dir.iterdir() if (p / 'metadata.yaml').exists()])
    
    all_x = []
    all_y = []

    print(f"[INFO] Found {len(bag_dirs)} rosbag directories.")
    for bag_path in bag_dirs:
        X, y = extract_data_only(bag_path, scan_topic, cmd_topic)
        if X is not None:
            all_X.append(X)
            all_y.append(y)
        extract_and_save_per_bag(bag_path, output_dir, scan_topic, cmd_topic)
        
    if all_X:
        final_X = np.concatenate(all_X, axis=0)
        final_y = np.concatenate(all_y, axis=0)

        # 全データまとまった状態で1つ保存
        np.save(out_dir / 'X_all.npy', final_X)
        np.save(out_dir / 'y_all.npy', final_y)
        print(f"[SUCCESS] Saved combined dataset: {final_X.shape}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bags_dir', required=True, help='Path to directory containing rosbag folders')
    parser.add_argument('--outdir', required=True, help='Output root directory')
    parser.add_argument('--scan_topic', default='/scan_filtered')
    parser.add_argument('--cmd_topic', default='/jetracer/cmd_drive')
    args = parser.parse_args()

    extract_all_bags_in_dir(args.bags_dir, args.outdir, args.scan_topic, args.cmd_topic)
