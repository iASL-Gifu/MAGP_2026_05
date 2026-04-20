import os
import csv
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
#from f1tenth_gym.maps.map_manager import MapManager
# from f1tenth_gym.maps.map_manager import TEST_MAPS as MAP_DICT
from src.envs.envs import make_env
from src.models.models import load_cnn_model
from src.envs.envs import MapManagerAdapter

@hydra.main(config_path="config", config_name="benchmark_sim", version_base="1.2")
def main(cfg: DictConfig):
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    print('------ Configuration ------')
    print(OmegaConf.to_yaml(cfg))
    print('---------------------------')

    # --- 環境／プランナ／エージェント等の初期化 ---
    map_manager = MapManagerAdapter(
        track_name=cfg.envs.map.name,
        line_type=cfg.envs.map.line_type
    )
    env = make_env(cfg.envs, map_manager, cfg.vehicle)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- モデルの読み込み ---
    model = load_cnn_model(
        model_name=cfg.model_name, 
        input_dim=cfg.input_dim, 
        output_dim=cfg.output_dim
    ).to(device)
    model_path = os.path.join(cfg.ckpt_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 評価モードに設定
    model.to(device)

    is_rnn = "Lstm" in cfg.model_name
    is_use_prev_action = "Action" in cfg.model_name
    
    # --- ベンチマーク結果の保存ディレクトリ ---
    benchmark_dir = cfg.benchmark_dir
    if not os.path.exists(benchmark_dir):
        os.makedirs(benchmark_dir)

    test_maps = [cfg.envs.map.name]

    for map_name in test_maps:
        env.map_manager.update_map(map_name)

        map_dir = os.path.join(cfg.benchmark_dir, map_name)
        os.makedirs(map_dir, exist_ok=True)
        
        # CSVファイルのパスを定義
        csv_file = os.path.join(map_dir, f"{map_name}_trajectory.csv")
        lap_file = os.path.join(map_dir, f"{map_name}_lap_times.csv")
        
        # CSVのヘッダー書き込み（初回のみ）
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["x", "y", "velocity"])

        # --- 推論用変数の初期化 ---
        hidden_state = None  # RNN用
        prev_action = torch.zeros((1, 2), device=device) if is_use_prev_action else None

        obs, info = env.reset()
        done = False

        while not done:
            with torch.no_grad():
                scan = obs['agent_0']['scan']
                scan_tensor = torch.from_numpy(scan).float().to(device) / cfg.envs.max_beam_range
                scan_tensor = scan_tensor.unsqueeze(0).unsqueeze(0) 

                if is_rnn:
                    # 修正： hidden_state を正しく更新・保持する
                    if is_use_prev_action:
                        action, hidden_state = model(scan_tensor, pre_action=prev_action, hidden=hidden_state)
                        prev_action = action
                    else:
                        action, hidden_state = model(scan_tensor, hidden=hidden_state)
                else:
                    if is_use_prev_action:
                        action = model(scan_tensor, prev_action)
                        prev_action = action
                    else:
                        action = model(scan_tensor)

                steer, speed_raw = action[0].tolist()
                speed = speed_raw * cfg.envs.map.speed 
                
                # env.step に渡す形式 [steer, speed]
                step_action = np.array([[steer, speed]])

            # --- 環境のステップ実行 ---
            # info には 'current_pos', 'velocity' などが含まれている想定
            next_obs, reward, terminated_env, truncated_env, info = env.step(step_action)

            # --- 終了判定の整理 ---
            is_collision = next_obs['agent_0']['collision'] > 0
            is_lap_finished = next_obs['agent_0']['lap_count'] >= 1
            
            done = is_collision or is_lap_finished

            # --- データの記録 ---
            # info['current_pos'] が [x, y] であることを確認してください
            curr_x, curr_y = info['current_pos'][0], info['current_pos'][1]
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([curr_x, curr_y, info['velocity']])

            if done:
                if is_collision:
                    print(f"  -> Collision detected on {map_name}!")
                    with open(lap_file, mode='a', newline='') as file:
                        csv.writer(file).writerow(["Crash", 0.0])
                elif is_lap_finished:
                    final_time = next_obs['agent_0']['lap_time']
                    print(f"  -> Lap Complete! Time: {final_time:.2f}s")
                    with open(lap_file, mode='a', newline='') as file:
                        csv.writer(file).writerow([1, final_time])
                break

            if cfg.render:
                env.render()

            obs = next_obs # 次のステップへ
    env.close()

if __name__ == "__main__":
    main()