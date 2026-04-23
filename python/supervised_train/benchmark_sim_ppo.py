import os
import csv
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from src.envs.envs import make_env, MapManagerAdapter
from src.envs.wrapper import PPOWrapper
from stable_baselines3 import PPO

@hydra.main(config_path="config", config_name="benchmark_sim", version_base="1.2")
def main(cfg: DictConfig):
    print('------ PPO Benchmark Configuration ------')
    print(OmegaConf.to_yaml(cfg))
    print('-----------------------------------------')

    # --- 環境の初期化 ---
    map_manager = MapManagerAdapter(
        track_name=cfg.envs.map.name,
        line_type=cfg.envs.map.line_type
    )
    base_env = make_env(cfg.envs, map_manager, cfg.vehicle)
    
    # 学習時と同じラッパーを適用
    env = PPOWrapper(base_env, map_manager)
    
    # --- PPOモデルの読み込み ---
    # .zip ファイルをロード
    model = PPO.load(cfg.ckpt_path)
    print(f"Loaded model from {cfg.ckpt_path}")
    
    # --- ベンチマーク結果の保存ディレクトリ ---
    benchmark_dir = cfg.benchmark_dir
    os.makedirs(benchmark_dir, exist_ok=True)

    test_maps = [cfg.envs.map.name]

    for map_name in test_maps:
        env.map_manager.update_map(map_name)
        map_dir = os.path.join(benchmark_dir, map_name)
        os.makedirs(map_dir, exist_ok=True)
        
        csv_file = os.path.join(map_dir, f"{map_name}_trajectory.csv")
        lap_file = os.path.join(map_dir, f"{map_name}_lap_times.csv")
        
        with open(csv_file, mode='w', newline='') as file:
            csv.writer(file).writerow(["x", "y", "velocity"])

        obs, info = env.reset()
        done = False

        while not done:
            # PPOによる予測 (deterministic=True で学習時の探索をOFFにする)
            action, _states = model.predict(obs, deterministic=True)

            # --- 環境のステップ実行 ---
            # PPOWrapper 内で shape の reshape(1, -1) は行われている想定
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # --- 終了判定の整理 ---
            is_collision = info.get('collision', False)
            
            # 【重要】 lap_count >= 1 かつ、ある程度走っていないとゴールと見なさない
            # 5.0秒経過していない場合は「ゴールしていない」と強制する
            is_lap_finished = (info.get('lap_counts', [0])[0] >= 1) and (info.get('sim_time', 0.0) > 5.0)
            
            done = is_collision or is_lap_finished

            # --- データの記録 ---
            # info の中身に合わせて調整してください
            curr_pos = info.get('current_pos', [0, 0])
            vel = info.get('velocity', 0)
            with open(csv_file, mode='a', newline='') as file:
                csv.writer(file).writerow([curr_pos[0], curr_pos[1], vel])

            if done:
                if is_collision:
                    print(f"  -> Collision detected on {map_name}!")
                    with open(lap_file, mode='a', newline='') as file:
                        csv.writer(file).writerow(["Crash", 0.0])
                else:
                    final_time = info.get('lap_times', 0.0)
                    print(f"  -> Lap Complete! Time: {final_time[0]:.2f}s")
                    with open(lap_file, mode='a', newline='') as file:
                        csv.writer(file).writerow([1, final_time[0]])
                break

            if cfg.get('render', False):
                env.render()

            obs = next_obs
    env.close()

if __name__ == "__main__":
    main()