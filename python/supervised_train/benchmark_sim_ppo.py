import os
import csv
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from src.envs.envs import make_env
from src.envs.wrapper import PPOWrapper
from f1tenth_gym.maps.map_manager import MapManager
from f1tenth_gym.maps.map_manager import TEST_MAPS as MAP_DICT
from stable_baselines3 import PPO

@hydra.main(config_path="config", config_name="benchmark_sim", version_base="1.2")
def main(cfg: DictConfig):
    print('------ PPO Benchmark Configuration ------')
    print(OmegaConf.to_yaml(cfg))
    print('-----------------------------------------')

    map_manager = MapManager(
        map_name=cfg.envs.map.name,
        map_ext=cfg.envs.map.ext,
        line_type=cfg.envs.map.line_type
    )
    base_env = make_env(cfg.envs, map_manager, cfg.vehicle)
    env = PPOWrapper(base_env, map_manager, training=False)
    
    # --- PPOモデルの読み込み ---
    model = PPO.load(cfg.ckpt_path)
    print(f"Loaded model from {cfg.ckpt_path}")
    
    # --- ベンチマーク結果の保存ディレクトリ ---
    benchmark_dir = cfg.benchmark_dir
    os.makedirs(benchmark_dir, exist_ok=True)

    for map_name in MAP_DICT:
        print(f"Evaluating on map: {map_name}")
        map_dir = os.path.join(benchmark_dir, map_name)
        os.makedirs(map_dir, exist_ok=True)
        
        csv_file = os.path.join(map_dir, f"{map_name}_trajectory.csv")
        lap_file = os.path.join(map_dir, f"{map_name}_lap_times.csv")
        
        with open(csv_file, mode='w', newline='') as file:
            csv.writer(file).writerow(["x", "y", "velocity"])

        env.update_map(map_name, cfg.envs.map.ext)

        obs, info = env.reset()
        done = False
        while not done:
            # PPOによる予測 (deterministic=True で学習時の探索をOFFにする)
            action, _states = model.predict(obs, deterministic=True)
            
            # --- 環境のステップ実行 ---
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # --- 終了判定の整理 ---
            is_collision = info.get('collision', False)
            lap_times = info.get('lap_times', 0.0)[0]

            # 終了判定
            done = terminated or truncated or is_collision
            
            is_lap_finished = info.get('lap_counts', [0])[0] >= 1
            
            done = done or is_lap_finished

            # --- データの記録 ---
            curr_x = info.get('current_pos', [0.0])[0]
            curr_y = info.get('current_pos', [0.0])[1]
            vel = info.get('velocity', 0)

            with open(csv_file, mode='a', newline='') as file:
                csv.writer(file).writerow([curr_x, curr_y, vel])

            if done:
                if is_collision:
                    print(f"  -> Collision detected at {lap_times:.2f}s!")
                    with open(lap_file, mode='a', newline='') as file:
                        csv.writer(file).writerow(["Crash", 0.0])
                elif is_lap_finished:
                    final_time = info.get('lap_times', [0.0])[0]
                    print(f"  -> Lap Complete! Time: {final_time:.2f}s")
                    with open(lap_file, mode='a', newline='') as file:
                        csv.writer(file).writerow([1, format(final_time, '.2f')])
                break

            if cfg.render:
                env.render(cfg.render_mode)

            obs = next_obs
    env.close()

if __name__ == "__main__":
    main()