import torch.nn as nn
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize
from sb3_contrib import RecurrentPPO
from src.envs.envs import make_ppo_env, linear_schedule
from src.models.ppo import TinyLidarExtractor
from f1tenth_gym.maps.map_manager import MapManager
import hydra
from omegaconf import DictConfig
import os


@hydra.main(config_path="config", config_name="train", version_base="1.2")
def test_ppo_learning(cfg: DictConfig):
    '''
    単体のマップで学習する
    '''

    initial_learning_rate = cfg.initial_learning_rate
    final_learniing_rate = cfg.final_learning_rate
    map_name = cfg.envs.map.name
    num_envs = 8
    
    # --- 環境の構築 ---
    map_manager = MapManager(
        map_name=map_name,
        map_ext=cfg.envs.map.ext,
        speed=cfg.envs.map.speed,
        downsample=cfg.envs.map.downsample,
        use_dynamic_speed=cfg.envs.map.use_dynamic_speed,
        a_lat_max=cfg.envs.map.a_lat_max,
        smooth_sigma=cfg.envs.map.smooth_sigma
        )
    
    eval_map_manager = MapManager(
        map_name=map_name,
        map_ext=cfg.envs.map.ext,
        speed=cfg.envs.map.speed,
        downsample=cfg.envs.map.downsample,
        use_dynamic_speed=cfg.envs.map.use_dynamic_speed,
        a_lat_max=cfg.envs.map.a_lat_max,
        smooth_sigma=cfg.envs.map.smooth_sigma
        )

    # 学習用
    env = make_ppo_env(cfg.envs, map_manager, cfg.vehicle, True, num_envs=num_envs)
    
    # 評価用
    eval_env = make_ppo_env(cfg.envs, eval_map_manager, cfg.vehicle, True, num_envs=1)
    
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
    save_stats = os.path.join(cfg.save_path, "vec_normalize.pkl")

    # --- モデル作成 ---
    # 既存のモデルがあるなら使う
    if os.path.exists(cfg.model_path):

        if os.path.exists(cfg.model_stats):
            env = VecNormalize.load(cfg.model_stats, env)

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
            learning_rate=linear_schedule(initial_learning_rate, final_learniing_rate),
            batch_size=cfg.batch_size,
            gamma=cfg.gamma,
            n_epochs=10,
            clip_range=0.25,
            # gae_lambda=cfg.gae_lambda,
            tensorboard_log=cfg.log_dir,
            device="cuda"
        )

    # ★重要：評価環境に学習環境の統計量をコピーし、更新をオフにする
    # これをしないと、eval_env での評価がデタラメになります
    eval_env.obs_rms = env.obs_rms
    eval_env.training = False # 評価中に平均・分散を更新しない
    eval_env.norm_reward = False # 評価に報酬正規化は不要

    # --- 学習の試行 ---
    print("[*] 学習を開始します...")
    print(f"Training on: {map_name}")

    env.env_method("update_map", map_name, cfg.envs.map.ext)
    eval_env.env_method("update_map", map_name, cfg.envs.map.ext)

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=eval_callback, 
        reset_num_timesteps=False # 累計ステップ数を維持
    )
    
    os.makedirs(cfg.save_path, exist_ok=True)
    save_path = os.path.join(cfg.save_path, "final_model")
    model.save(save_path)
    env.save(save_stats)
    
    print(f"[✔] 学習完了！モデルを {save_path} に保存しました。")
    env.close()
    eval_env.close()

if __name__ == "__main__":
    test_ppo_learning()