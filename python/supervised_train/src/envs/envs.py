from typing import Dict
from omegaconf import DictConfig
from f1tenth_gym.f110_env import F110Env
from gymnasium.wrappers import TimeLimit
from gymnasium.wrappers import RescaleAction
from .wrapper import F110Wrapper, PPOWrapper
from f1tenth_gym.maps.map_manager import MapManager
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env(env_cfg: DictConfig, map_manager: MapManager,  param: Dict):
    map_name = map_manager.map_path ## マップの名前 
    max_ext = map_manager.map_ext ## マップの拡張子
    num_beams = env_cfg.num_beams ## 2d lidar のビーム数
    num_agents = env_cfg.num_agents ## エージェントの数 CPUの数みたいなイメージ

    ## 公式のベース環境
    env = F110Env(map=map_name, map_ext=max_ext, num_beams=num_beams, num_agents=num_agents, params=param)

    ## 自作のラッパー
    env = F110Wrapper(env, map_manager=map_manager)

    return env

def make_ppo_env(env_cfg, map_manager, param, training):
    """
    PPO学習専用の環境構築フロー。
    F110Wrapper を使わず、新しく作った PPOWrapper を使用します。
    """
    # 1. 公式のベース環境
    env = F110Env(
        map=map_manager.map_path, 
        map_ext=env_cfg.map.ext, 
        num_beams=env_cfg.num_beams, 
        num_agents=env_cfg.num_agents, 
        params=param
    )

    # 自作ラッパー
    env = PPOWrapper(env, map_manager=map_manager, training=training)

    # ステップ制限
    if training:
        env = TimeLimit(env, max_episode_steps=10000)

    # アクションの正規化 ([-1, 1])
    env = RescaleAction(env, min_action=-1.0, max_action=1.0)

    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    return env

def linear_schedule(start_lr: float, end_lr: float):
    """
    開始時の学習率と終了時の学習率を受け取り、その間を直線でつなぐ関数を返す。
    """
    def schedule(progress_remaining: float):
        # progress_remaining: 1.0 (ラウンド開始) -> 0.0 (ラウンド終了)
        # 線形補間: end + (start - end) * progress
        return end_lr + (start_lr - end_lr) * progress_remaining
    return schedule

def find_and_update_map(env, map_name, map_ext):
    """
    環境の階層を外側から内側へ探索し、
    update_map メソッドを最初に持っているラッパー（PPOWrapper等）のメソッドを実行する。
    """
    curr = env
    while curr is not None:
        # 現在の層が update_map を持っているか確認
        # ただし、TimeLimit や RescaleAction は持っていないので自動的にスキップされます
        if hasattr(curr, 'update_map'):
            print(f"  -> Found update_map in {type(curr).__name__}")
            curr.update_map(map_name, map_ext)
            return True
        
        # 次の層（内側の env）へ進む
        curr = getattr(curr, 'env', None)
    
    print("  -> Warning: update_map not found in any wrapper. Falling back to unwrapped.")
    env.unwrapped.update_map(map_name, map_ext)
    return False