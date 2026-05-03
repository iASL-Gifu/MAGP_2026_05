import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from src.envs.envs import make_ppo_env, linear_schedule, find_and_update_map
from src.envs.curriculum import CurriculumMapManager
from f1tenth_gym.maps.map_manager import MapManager
import hydra
from omegaconf import DictConfig, OmegaConf
import os


@hydra.main(config_path="config", config_name="train_ppo", version_base="1.2")
def test_ppo_learning(cfg: DictConfig):

    initial_learning_rate = cfg.learning_rate
    lr_schedule = linear_schedule(initial_learning_rate)

    curriculum = CurriculumMapManager()
    total_timesteps = cfg.total_timesteps
    steps_per_round = cfg.steps_per_round
    total_rounds =    int(total_timesteps / steps_per_round)
    
    # --- 環境の構築 ---
    map_manager = MapManager(
            map_name=cfg.envs.map.name,
            map_ext=cfg.envs.map.ext,
            line_type=cfg.envs.map.line_type
        )

    # 学習用
    env = make_ppo_env(cfg.envs, map_manager, cfg.vehicle, True)
    
    # 評価用（完全に別のインスタンスを作る）
    eval_env = make_ppo_env(cfg.envs, map_manager, cfg.vehicle, True)
    

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

    current_total_steps = 0
    for round in range(total_rounds):
        # 1. 進捗に応じたマップの選択
        target_map = curriculum.get_map_by_progress(current_total_steps, cfg.total_timesteps)
        print(f"Round {round}: Training on {target_map}")
        
        # 2. 環境のマップ更新（貫通型メソッドを使用）
        find_and_update_map(env, target_map, cfg.envs.map.ext)
        find_and_update_map(eval_env, target_map, cfg.envs.map.ext)
        
        # 3. 学習の継続
        model.learn(
            total_timesteps=steps_per_round, 
            callback=eval_callback, 
            reset_num_timesteps=False # 累計ステップ数を維持
        )
        
        current_total_steps += steps_per_round
    
    os.makedirs(cfg.save_path, exist_ok=True)
    save_path = os.path.join(cfg.save_path, "final_model")
    model.save(save_path)
    
    print(f"[✔] 学習完了！モデルを {save_path} に保存しました。")
    env.close()

if __name__ == "__main__":
    test_ppo_learning()