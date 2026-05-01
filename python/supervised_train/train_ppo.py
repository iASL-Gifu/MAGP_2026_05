import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from src.envs.envs import make_env
from src.envs.wrapper import PPOWrapper
from f1tenth_gym.maps.map_manager import MapManager
from f1tenth_gym.maps.map_manager import TRAIN_MAPS as MAP_DICT
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys

@hydra.main(config_path="config", config_name="train_ppo", version_base="1.2")
def test_ppo_learning(cfg: DictConfig):
    # マップを難易度別にリスト化
    curriculum_maps = [
        ['Austin', 'Oschersleben'],               # Level 1
        ['Melbourne', 'Spielberg', 'Sakhir', 'Silverstone'], # Level 2
        ['Budapest', 'Spa', 'Nuerburgring', 'YasMarina'],    # Level 3
        ['Hockenheim', 'MexicoCity', 'MoscowRaceway', 'Sepang', 'Sochi'] # Level 4
    ]
    map_ext  = cfg.envs.map.ext

    # 2. 環境の構築
    map_manager = MapManager(
        map_name=curriculum_maps[0][0],
        map_ext=map_ext,
        line_type=cfg.envs.map.line_type
    )
    
    base_env = make_env(cfg.envs, map_manager, cfg.vehicle)
    
    # 3. PPOWrapper でラップ
    env = PPOWrapper(base_env, map_manager, training=True)

    # 4. Gymnasium 準拠チェック (重要！)
    print("[*] 環境の仕様チェック中...")
    check_env(env)
    print("[✔] Gymnasium仕様チェック通過！")

    # 5. 学習の試行
    print("[*] 学習を開始します...")

    # 評価用環境の作成
    eval_base_env = make_env(cfg.envs, map_manager, cfg.vehicle)
    eval_env = PPOWrapper(eval_base_env, map_manager, training=False)

    # YAMLから評価頻度などを取得
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=cfg.save_path, # ベストモデルが保存される場所
        log_path=cfg.save_path,
        eval_freq=cfg.eval.eval_freq,       # YAMLの値を参照
        n_eval_episodes=cfg.eval.n_eval_episodes,
        deterministic=True,
        render=False
    )

    # 既存のモデルがあるなら使う
    if os.path.exists(cfg.model_path):
        model = PPO.load(cfg.model_path, env=env, device="cuda")
    # ないなら新規に作成する
    else:
        model = PPO(
            "MultiInputPolicy", 
            env,
            verbose=1,
            learning_rate=cfg.learning_rate,
            ent_coef=cfg.ent_coef,
            batch_size=cfg.batch_size,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            n_steps=cfg.n_steps,
            device="cuda",
            tensorboard_log=cfg.log_dir
        )

    '''    
    # 各レベルを回す
    for level_idx, map_group in enumerate(curriculum_maps):
        print(f"\n=== Starting Level {level_idx + 1} ===")
        
        for map_name in map_group:
            sys.stdout.write(f"\x1b]2;Training on: {map_name}\x07")
            sys.stdout.flush()
            print(f"--- Focusing on: {map_name} ---")
            
            # 環境の更新
            new_map_manager = MapManager(map_name, map_ext=map_ext, line_type=cfg.envs.map.line_type)
            new_env = PPOWrapper(make_env(cfg.envs, new_map_manager, cfg.vehicle), new_map_manager, training=True)
            new_eval_env = PPOWrapper(make_env(cfg.envs, new_map_manager, cfg.vehicle), new_map_manager, training=False)
            
            model.set_env(new_env)

            save_dir = os.path.join(cfg.save_path, map_name)
            os.makedirs(save_dir, exist_ok=True)

            # start_steps = model.num_timesteps
            # target_steps = start_steps + cfg.steps_per_map
            
            # コールバック更新
            eval_callback = EvalCallback(
                new_eval_env,
                best_model_save_path=save_dir, # ベストモデルが保存される場所
                log_path=save_dir,
                eval_freq=cfg.eval.eval_freq,       # YAMLの値を参照
                n_eval_episodes=cfg.eval.n_eval_episodes,
                deterministic=True,
                render=False
            )

            
            # このマップで学習
            model.learn(total_timesteps=cfg.steps_per_map, callback=eval_callback, reset_num_timesteps=False)
    '''

    target_map = "Austin"  # ここを学習したいマップ名に変更
    print(f"\n=== Training FIXED on: {target_map} ===")
    
    # 環境の構築（固定）
    new_map_manager = MapManager(target_map, map_ext=map_ext, line_type=cfg.envs.map.line_type)
    new_env = PPOWrapper(make_env(cfg.envs, new_map_manager, cfg.vehicle), new_map_manager, training=True)
    new_eval_env = PPOWrapper(make_env(cfg.envs, new_map_manager, cfg.vehicle), new_map_manager, training=False)
    
    model.set_env(new_env)

    # 評価用コールバックの更新
    save_dir = os.path.join(cfg.save_path)
    eval_callback = EvalCallback(
        new_eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=cfg.eval.eval_freq,
        n_eval_episodes=cfg.eval.n_eval_episodes,
        deterministic=True,
        render=False
    )

    # --- STEP 1: Level 1 (低速・安定走行の学習) ---
    new_env.level = 1

    print("=== Training Level 1: Stability ===")
    model.learn(total_timesteps=500000, callback=eval_callback)

    # --- STEP 2: Level 2 (高速化への移行) ---
    print("=== Training Level 2: High Speed ===")
    new_env.level = 2
    # env.raw_env.params.v_max = 10.0 # 制限解除
    
    # 学習率を少し下げて、これまでの安定走行を壊さないように微調整
    model.learning_rate = 5e-5 
    model.learn(total_timesteps=500000, callback=eval_callback, reset_num_timesteps=False)
    
    os.makedirs(cfg.save_path, exist_ok=True)
    save_path = os.path.join(cfg.save_path, "final_model")
    model.save(save_path)
    
    print(f"[✔] 学習完了！モデルを {save_path} に保存しました。")
    env.close()

if __name__ == "__main__":
    test_ppo_learning()