import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from src.envs.envs import make_ppo_env, linear_schedule
from f1tenth_gym.maps.map_manager import MapManager
import hydra
from omegaconf import DictConfig, OmegaConf
import os


@hydra.main(config_path="config", config_name="train_ppo", version_base="1.2")
def test_ppo_learning(cfg: DictConfig):

    initial_learning_rate = 0.00015
    lr_schedule = linear_schedule(initial_learning_rate)
    target_map = "Austin"
    
    # --- 環境の構築 ---
    map_manager = MapManager(
            target_map,
            map_ext=cfg.envs.map.ext,
            line_type=cfg.envs.map.line_type
        )

    # 学習用
    env = make_ppo_env(cfg.envs, map_manager, cfg.vehicle, True)
    
    # 評価用（完全に別のインスタンスを作る）
    eval_env = make_ppo_env(cfg.envs, map_manager, cfg.vehicle, False)
    

    # 評価用コールバックの更新
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=cfg.save_path,
        log_path=cfg.save_path,
        eval_freq=cfg.eval.eval_freq,
        n_eval_episodes=cfg.eval.n_eval_episodes,
        deterministic=True,
        render=False
    )

    # --- モデル作成 ---
    # 既存のモデルがあるなら使う
    if os.path.exists(cfg.model_path):
        model = PPO.load(cfg.model_path, env=env, device="cuda")
    # ないなら新規に作成する
    else:
        model = PPO(
            "MultiInputPolicy", 
            env,
            verbose=1,
            n_steps=cfg.n_steps,
            ent_coef=cfg.ent_coef,
            learning_rate=lr_schedule,
            batch_size=cfg.batch_size,
            gamma=cfg.gamma,
            n_epochs=10,
            clip_range=0.25,
            # gae_lambda=cfg.gae_lambda,
            tensorboard_log=cfg.log_dir,
            device="cuda"
        )

    model.set_env(env)

    # --- Gymnasium 準拠チェック ---
    print("[*] 環境の仕様チェック中...")
    check_env(env)
    print("[✔] Gymnasium仕様チェック通過！")

    # --- 学習の試行 ---
    print("[*] 学習を開始します...")

    '''
    # --- STEP 1: Level 1 (低速・安定走行の学習) ---
    env.level = 1

    print("=== Training Level 1: Stability ===")
    model.learn(total_timesteps=500000, callback=eval_callback)
    '''

    # --- STEP 2: Level 2 (高速化への移行) ---
    print("=== Training Level 2: High Speed ===")
    env.level = 2
    # env.raw_env.params.v_max = 10.0 # 制限解除
    
    # 学習率を少し下げて、これまでの安定走行を壊さないように微調整
    model.learn(total_timesteps=cfg.total_timesteps, callback=eval_callback, reset_num_timesteps=False)
    
    os.makedirs(cfg.save_path, exist_ok=True)
    save_path = os.path.join(cfg.save_path, "final_model")
    model.save(save_path)
    
    print(f"[✔] 学習完了！モデルを {save_path} に保存しました。")
    env.close()

if __name__ == "__main__":
    test_ppo_learning()