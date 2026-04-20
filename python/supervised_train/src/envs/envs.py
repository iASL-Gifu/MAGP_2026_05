import os
import numpy as np
from typing import Dict
from dataclasses import replace
from omegaconf import DictConfig
from f1tenth_gym.envs.f110_env import F110Env
from f1tenth_gym.envs.env_config import EnvConfig
from .wrapper import F110Wrapper
from f1tenth_gym.envs.track import Track as F110Track

# --- 最新の Track を旧 MapManager 互換にするためのアダプター ---
class MapManagerAdapter:
    def __init__(self, track_name: str, line_type: str = "race"):
        self.name = track_name
        self.line_type = line_type
        self.track = None
        self.line_type = line_type

        try:
            # ライブラリ内の maps フォルダから track_name を探す
            self.track = F110Track.from_track_name(track_name)
            print(f"[ADAPTER INIT] SUCCESS: Track '{track_name}' loaded from library.")
        except Exception as e:
            print(f"[ADAPTER INIT] ERROR: Could not load track '{track_name}': {e}")

        map_dir = os.path.dirname(self.track.filepath)

        if self.line_type == "center":
            # 1. センターライン座標の読み込み (カンマ区切り)
            path_cl = os.path.join(map_dir, f"{track_name}_centerline.csv")
            data_cl = np.loadtxt(path_cl, delimiter=',', skiprows=1)
            xs, ys = data_cl[:, 0], data_cl[:, 1]

            dx = np.diff(xs, append=xs[0])
            dy = np.diff(ys, append=ys[0])
            yaws = np.arctan2(dy, dx)
            
            # 2. レースライン速度の読み込みと補間
            path_rl = os.path.join(map_dir, f"{track_name}_raceline.csv")
            
            if os.path.exists(path_rl):
                # セミコロン区切りで読み込み。ヘッダー（#）は自動で飛ばされます
                data_rl = np.loadtxt(path_rl, delimiter=';')
                
                # CSVの列定義に合わせて抽出
                # 2列目:x, 3列目:y, 6列目:velocity (インデックスは 1, 2, 5)
                rl_xs = data_rl[:, 1]
                rl_ys = data_rl[:, 2]
                rl_vs = data_rl[:, 5] 

                # --- 累積距離 (s) を計算して、センターラインの各点に速度を割り当てる ---
                rl_s = np.cumsum(np.sqrt(np.diff(rl_xs, prepend=rl_xs[0])**2 + np.diff(rl_ys, prepend=rl_ys[0])**2))
                cl_s = np.cumsum(np.sqrt(np.diff(xs, prepend=xs[0])**2 + np.diff(ys, prepend=ys[0])**2))
                
                # 補間実行
                vs = np.interp(cl_s, rl_s, rl_vs)
                
                yaw_diffs = np.abs(np.diff(yaws, append=yaws[0]))

                # --- 曲率ベースの減速係数を計算 ---
                # 曲がりが急なほど小さな値（0.6〜0.9）になるように設計
                slowdown_factor = 1.0 - np.clip(yaw_diffs * 15.0, 0.0, 0.4) 

                # レースラインの速度にこの係数を掛ける
                vs = vs * slowdown_factor

                # さらに全体の上限を 5.0m/s 程度に抑えて様子を見る
                vs = np.clip(vs, 2.0, 5.0)

                print(f"  - SUCCESS: Loaded dynamic velocity from {track_name}_raceline.csv")
            else:
                print(f"  - WARNING: {path_rl} not found. Using constant speed.")
                vs = np.full_like(xs, fill_value=4.0)

        else:
            # 3. レースライン読み込み (ライブラリの自動ロード機能を利用)
            # 以前のコードが動いていたのは、ここが self.track.raceline を見ていたからです
            if hasattr(self.track, 'raceline') and self.track.raceline is not None:
                rl = self.track.raceline
                xs, ys, vs, yaws = rl.xs, rl.ys, rl.vxs, rl.yaws
            else:
                xs = ys = vs = yaws = None

        # 4. 最終的なウェイポイントの生成
        if xs is not None:
            self.waypoints = np.stack([xs, ys, vs, yaws], axis=1).astype(np.float64)
            print(f"  - SUCCESS: {self.line_type.upper()} waypoints initialized.")


    def get_future_waypoints(self, current_pos, num_points=10):
        """wrapper.py が step() 内で呼んでいるメソッド"""
        if self.waypoints is None:
            return np.zeros((num_points, 2))
        
        # 現在地から最も近いウェイポイントを探す (L2ノルム)
        dists = np.sum((self.waypoints[:, :2] - current_pos)**2, axis=1)
        idx = np.argmin(dists)
        
        # 未来のポイントを num_points 分抽出（周回対応）
        indices = (np.arange(num_points) + idx) % len(self.waypoints)
        return self.waypoints[indices, :]

    def update_map(self, map_name):
        # リセット時は何もしない（初期化時のマップを使い続ける）
        pass



def make_env(env_cfg: DictConfig, map_manager: MapManagerAdapter,  param: Dict):

    # 1. EnvConfig を作成し、config ファイル(YAML)の値を反映させる
    # デフォルト値を使って初期化し、必要な箇所だけ上書きします
    conf = EnvConfig()

    # ここで渡すパスを明示
    target_map_name = env_cfg.map.name 
    print(f"[MAKE_ENV] Passing name to Env: {target_map_name}")
    
    conf = replace(conf,
        map_name=target_map_name,
        num_agents=env_cfg.num_agents,
        render_enabled=True,
        # シミュレーション設定を新しく作って差し替え
        simulation_config=replace(conf.simulation_config,
            timestep=env_cfg.timestep
        ),
        # LiDAR設定を新しく作って差し替え
        lidar_config=replace(conf.lidar_config,
            num_beams=env_cfg.num_beams,
            field_of_view=env_cfg.beam_fov,
            range_max=env_cfg.max_beam_range
        )
    )

    # 2. 公式のベース環境を config オブジェクトを渡して初期化
    # inspect の結果通り、引数は (config, render_mode) です
    env = F110Env(config=conf, render_mode=env_cfg.render_mode)

    ## 自作のラッパー
    env = F110Wrapper(env, map_manager=map_manager)

    return env