import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from src.envs.envs import make_env, MapManagerAdapter
from src.envs.wrapper import PPOWrapper

def test_ppo_learning():
    # 1. コンフィグ設定（本来は hydra で読み込むべきですが、テスト用に直打ち）
    # ※あなたの benchmark_sim.py の設定に合わせて調整してください
    class Config:
        def __init__(self):
            self.envs = type('obj', (object,), {'map': type('obj', (object,), {'name': 'IMS', 'line_type': 'center'}), 
                                               'timestep': 0.01, 'num_beams': 1080, 
                                               'beam_fov': 4.7, 'max_beam_range': 30.0, 
                                               'num_agents': 1, 'render_mode': 'human_fast'})
            self.vehicle = {} # 車両設定
    
    cfg = Config()

    # 2. 環境の構築
    map_manager = MapManagerAdapter(track_name=cfg.envs.map.name)
    base_env = make_env(cfg.envs, map_manager, cfg.vehicle)
    
    # 3. PPOWrapper でラップ
    env = PPOWrapper(base_env, map_manager)

    # 4. Gymnasium 準拠チェック (重要！)
    print("[*] 環境の仕様チェック中...")
    check_env(env)
    print("[✔] Gymnasium仕様チェック通過！")

    # 5. 学習の試行
    print("[*] 学習を開始します...")
    # CnnPolicy: あなたの1080次元LiDARを入力とするためのポリシー
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
    model.learn(total_timesteps=1000)
    
    print("[✔] 学習の試行成功！")
    env.close()

if __name__ == "__main__":
    test_ppo_learning()