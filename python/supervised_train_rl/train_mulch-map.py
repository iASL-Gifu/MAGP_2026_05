import torch.nn as nn
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from sb3_contrib import RecurrentPPO
from src.envs.envs import make_ppo_env, linear_schedule
from src.envs.curriculum import CurriculumMapManager
from src.models.ppo import TinyLidarExtractor
from f1tenth_gym.maps.map_manager import MapManager
import hydra
from omegaconf import DictConfig
import os


@hydra.main(config_path="config", config_name="train", version_base="1.2")
def test_ppo_learning(cfg: DictConfig):
    '''
    複数のマップで学習する
    '''

    initial_learning_rate = cfg.learning_rate

    curriculum = CurriculumMapManager()
    total_timesteps = cfg.total_timesteps
    steps_per_round = cfg.steps_per_round
    total_rounds =    int(total_timesteps / steps_per_round)
    
    # --- 環境の構築 ---
    map_manager = MapManager(
        map_name=cfg.envs.map.name,
        map_ext=cfg.envs.map.ext,
        speed=cfg.envs.map.speed,
        downsample=cfg.envs.map.downsample,
        use_dynamic_speed=cfg.envs.map.use_dynamic_speed,
        a_lat_max=cfg.envs.map.a_lat_max,
        smooth_sigma=cfg.envs.map.smooth_sigma
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
        model = RecurrentPPO.load(cfg.model_path, env=env, device="cuda")

        # TensorBoard の保存先を変更
        new_logger = configure(
            folder=cfg.log_dir,
            format_strings=["stdout", "tensorboard"]
        )
        model.set_logger(new_logger)

    # ないなら新規に作成する
    else:
        policy_kwargs = {
            "features_extractor_class": TinyLidarExtractor,
            "features_extractor_kwargs": {"features_dim": 256},
            "lstm_hidden_size": 128,
            "n_lstm_layers": 1,
            "net_arch": dict(pi=[100, 50, 10], vf=[100, 50, 10]),
            "activation_fn": nn.ReLU, # ここで活性化関数を指定可能
        }
        model = RecurrentPPO(
            "MultiInputLstmPolicy", 
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            n_steps=cfg.n_steps,
            ent_coef=cfg.ent_coef,
            learning_rate=initial_learning_rate,
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
    # sb3 の Wrapper を使っているからエラー吐く
    print("[*] 環境の仕様チェック中...")
    # check_env(env)
    print("[✔] Gymnasium仕様チェック通過！")

    # --- 学習の試行 ---
    print("[*] 学習を開始します...")

    current_total_steps = 0
    for round in range(total_rounds):
        # 進捗に応じたマップの選択
        target_map = curriculum.get_map_by_progress(current_total_steps, cfg.total_timesteps)
        print(f"Round {round}: Training on {target_map}")
        
        # 環境のマップ更新
        env.env_method("update_map", target_map, cfg.envs.map.ext)
        eval_env.env_method("update_map", target_map, cfg.envs.map.ext)

        env.reset()
        eval_env.reset()
        model._last_obs = None

        # 学習率の計算
        # 1. 全体の進捗から、このラウンドの「開始時」と「終了時」の学習率を計算
        start_progress = current_total_steps / total_timesteps
        end_progress = (current_total_steps + steps_per_round) / total_timesteps
        
        # 例: 0.00015 から 0 まで直線的に落としたい場合
        round_start_lr = initial_learning_rate * (1.0 - start_progress)
        round_end_lr = initial_learning_rate * (1.0 - end_progress)
        
        model.lr_schedule = linear_schedule(round_start_lr, round_end_lr)
        
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