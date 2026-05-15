import os
import csv
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from src.envs.envs import make_ppo_env
from f1tenth_gym.maps.map_manager import MapManager
from f1tenth_gym.maps.map_manager import TEST_MAPS as MAP_DICT
# from f1tenth_gym.maps.map_manager import TRAIN_MAPS as MAP_DICT
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

@hydra.main(config_path="config", config_name="benchmark_sim", version_base="1.2")
def main(cfg: DictConfig):
    print('------ PPO Benchmark Configuration ------')
    print(OmegaConf.to_yaml(cfg))
    print('-----------------------------------------')

    map_manager = MapManager(
        map_name=MAP_DICT[0],
        map_ext=cfg.envs.map.ext,
        speed=cfg.envs.map.speed,
        downsample=cfg.envs.map.downsample,
        use_dynamic_speed=cfg.envs.map.use_dynamic_speed,
        a_lat_max=cfg.envs.map.a_lat_max,
        smooth_sigma=cfg.envs.map.smooth_sigma
    )
    env = make_ppo_env(cfg.envs, map_manager, cfg.vehicle, False)

    # 学習時の統計量の読み込み
    stats_path = cfg.model_stats
    if os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, env)
        env.training = False     # 統計量を更新しない
        env.norm_reward = False  # 報酬の正規化は不要
        print(f"[*] Loaded normalization stats from {stats_path}")
    
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

        env.env_method("update_map", map_name, cfg.envs.map.ext)
        obs = env.reset()
        done = False
        while not done:
            # PPOによる予測 (deterministic=True で学習時の探索をOFFにする)
            action, _states = model.predict(obs, deterministic=True)
            
            # --- 環境のステップ実行 ---
            next_obs, reward,done, info = env.step(action)
            
            # --- 終了判定の整理 ---
            is_collision = info[0].get('collision', False)
            lap_times = info[0].get('lap_times', 0.0)[0]

            # 終了判定
            done = done or is_collision
            
            is_lap_finished = info[0].get('lap_counts', [0])[0] >= 1
            
            done = done or is_lap_finished

            # --- データの記録 ---
            curr_x = info[0].get('current_pos', [0.0])[0]
            curr_y = info[0].get('current_pos', [0.0])[1]
            vel = info[0].get('velocity', 0)

            with open(csv_file, mode='a', newline='') as file:
                csv.writer(file).writerow([curr_x, curr_y, vel])

            if done:
                if is_collision:
                    print(f"  -> Collision detected at {lap_times:.2f}s!")
                    with open(lap_file, mode='a', newline='') as file:
                        csv.writer(file).writerow(["Crash", 0.0])
                elif is_lap_finished:
                    final_time = info[0].get('lap_times', [0.0])[0]
                    print(f"  -> Lap Complete! Time: {final_time:.2f}s")
                    with open(lap_file, mode='a', newline='') as file:
                        csv.writer(file).writerow([1, format(final_time, '.2f')])
                break

            if cfg.render:
                env.unwrapped.render(cfg.render_mode)
                env.env_method("render")

            obs = next_obs
    env.close()

if __name__ == "__main__":
    main()