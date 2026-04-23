import os
import hydra
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from src.envs.envs import make_env, MapManagerAdapter
from src.envs.wrapper import PPOWrapper

@hydra.main(config_path="config", config_name="train_ppo")
def train(cfg: DictConfig):
    # 環境の構築 (benchmark_sim の設定を流用)
    # 実際には既存の benchmark_sim の config を読み込むのがベストです
    map_manager = MapManagerAdapter(track_name="IMS") # マップ名は必要に応じて変更
    # 注意: ここで渡す env_cfg は以前使用した dict を用意してください
    base_env = make_env(cfg.envs, map_manager, cfg.vehicle)
    env = PPOWrapper(base_env, map_manager)

    # モデルの定義 (MlpPolicy は観測空間が1次元配列の時に使用)
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        n_steps=cfg.n_steps,
        tensorboard_log=cfg.log_dir
    )

    # 定期的にモデルを保存するコールバック
    checkpoint_callback = CheckpointCallback(   
        save_freq=10000, 
        save_path=cfg.save_path,
        name_prefix="ppo_f1tenth"
    )

    print("[*] 学習を開始します...")
    model.learn(total_timesteps=cfg.total_timesteps, callback=checkpoint_callback)
    
    # 最終保存
    model.save(os.path.join(cfg.save_path, "final_model"))
    print("[✔] 学習完了！モデルを保存しました。")

if __name__ == "__main__":
    train()