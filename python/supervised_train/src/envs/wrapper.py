import gymnasium as gym
import numpy as np
import math
from pyglet.text import Label


# from f1tenth_gym.maps.map_manager import MapManager

class F110Wrapper(gym.Wrapper):
    """
    F110Env のラッパークラス。
    環境のインターフェースを簡略化し、追加機能を提供する。

    - step() の出力を整形
    - reset() のオプション処理を強化
    - レンダリング処理の簡略化
    - ラップタイム取得機能の追加
    """

    def __init__(self, env, map_manager):
        super().__init__(env)
        self.ego_idx = env.ego_idx
        self.map_manager = map_manager

        self._waypoint_vlists = []  # ← 追加：ウェイポイント用のVertexListを保持
        self.speed = 0.0

        self.env.add_render_callback(self.render_callback)
        # Waypoint描画機能をレンダリングコールバックとして追加
        self.env.add_render_callback(self.render_waypoints)

        self.dt = 0.01  # リワードのあれ

    def step(self, action, num_points=10):
        """
        環境を1ステップ進める。

        返り値:
        - observation: 観測データ
        - reward: ステップごとの報酬
        - terminated: エピソード終了フラグ
        - truncated: エピソード途中終了フラグ
        - info: 追加情報（ラップタイムなど）
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = 0.0

        ## waypointを作成
        ego_state = obs['agent_0']['state']
        frenet = obs['agent_0']['frenet_pose']

        current_pos = ego_state[:2]   # [x, y]
        current_theta = frenet[2]  # theta
        current_vel = ego_state[3]    # velocity

        waypoint = self.map_manager.get_future_waypoints(current_pos, num_points=num_points)
        info['waypoint'] = waypoint
        info['current_pos'] = current_pos
        info['velocity'] = current_vel
        self.speed = current_vel

        # spin
        if abs(current_theta) > 100.0:
            truncated = True
            
        # 1 lap 終了
        if obs['agent_0']['lap_count'] >= 1:
            terminated = True

        ## 報酬
        ### 時間経過で報酬を減らす
        reward -= self.dt

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None, index: int=0):
        """
        環境をリセットする。

        返り値:
        - observation: 初期観測データ
        - info: 追加情報（オプションで初期状態指定可能）
        """

        if options is not None:
            obs, info = self.env.reset(seed=seed, options=options)
            return obs, info

        if options is None:
            positions = []
            if self.map_manager.waypoints is not None:
                num_waypoints = len(self.map_manager.waypoints)
                num_agents = self.env.num_agents

                # 各エージェントに対して均等にWaypointを割り当て
                index_increment = num_waypoints / num_agents

                for i in range(num_agents):
                    # 浮動小数点のインデックスを整数インデックスに変換
                    waypoint_index = int(i * index_increment + index) % num_waypoints 
                    next_waypoint_index = (waypoint_index + 1) % num_waypoints

                    x, y = self.map_manager.waypoints[waypoint_index][:2]

                    next_x, next_y = self.map_manager.waypoints[next_waypoint_index][:2]
                
                    # 角度を計算（ラジアン）
                    dx = next_x - x
                    dy = next_y - y
                    t = math.atan2(dy, dx)

                    positions.append([x, y, t])

            else:
                # Waypointsが存在しない場合のデフォルトの位置
                positions = [[0, 0, 0] for _ in range(self.env.num_agents)]

            options = {
                "poses": np.array(positions)
            }

        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info
    
    def update_map(self, map_name, map_ext):
        """
        環境のマップを更新する。

        引数:
        - map_name: 新しいマップの名前
        - map_ext: 新しいマップの拡張子
        """
        self.map_manager.update_map(map_name)

    def render(self, mode="human"):
        """
        環境のレンダリングを簡単に呼び出せるようにする。
        """
        return self.env.render(mode=mode)

    def get_lap_time(self):
        """
        Ego車両の現在のラップタイムを取得する。

        返り値:
        - float: 現在のラップタイム
        """
        return self.env.lap_times[self.ego_idx]

    def close(self):
        """
        環境を閉じる（リソース解放）。
        """
        self.env.close()

    def render_callback(self, env_renderer):
        # custom extra drawing function for camera update
        e = env_renderer

        try:
            poses = self.env.unwrapped.poses
            car_x = poses[0, 0]
            car_y = poses[0, 1]
        except  Exception:
            return

        car_w = 0.5  # マシンの幅（目安）
        car_h = 0.3  # マシンの高さ（目安）

        top, bottom = car_y + car_h, car_y - car_h
        left, right = car_x - car_w, car_x + car_w

        l = 15.0
        e.left = left - l
        e.right = right + l
        e.top = top + l
        e.bottom = bottom - l

        # 初回のみ Label を生成する        
        if not hasattr(self, 'speed_label'):
            self.speed_label = Label('Speed: 0.00 m/s',
                                    font_name='Times New Roman',
                                    font_size=14,
                                    x=left, y=top + 2,
                                    anchor_x='left', anchor_y='top',
                                    color=(255, 255, 255, 255),
                                    batch=e.batch)

        # テキストの内容だけ更新する
        self.speed_label.text = f'Speed: {self.speed:.2f} m/s'

        # 座標も更新する（カメラが動く場合）
        self.speed_label.x = left
        self.speed_label.y = top + 2
        

    def render_waypoints(self, renderer):
        if self.map_manager.waypoints is None:
            return

        # ウェイポイントの (x, y) 座標
        pts = self.map_manager.waypoints[:, :2].astype(np.float32)
        
        try:
            # エラーメッセージに基づき、pointsを引数として渡す
            # 1451点は重いのでスライスで間引く（重要）
            draw_pts = pts[::10]
            
            # LineRendererオブジェクトを取得（ここでポイントを渡す）
            line_renderer = renderer.get_lines_renderer(draw_pts)
            
            # 色の設定（赤色）
            line_renderer.color = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
            
            # これで画面に赤い線が描画されるはずです
        except Exception as e:
            if not hasattr(self, '_render_err'):
                print(f"[DEBUG] Render Attempt Failed Again: {e}")
                self._render_err = True


class PPOWrapper(F110Wrapper):
    """
    強化学習 (Stable Baselines3) のために Gymnasium 仕様に準拠させたラッパー。
    """
    def __init__(self, env, map_manager):
        super().__init__(env, map_manager)
        
        # PPOが観測データとアクションの範囲を理解するために必須の定義
        # LiDARスキャンは 1080次元、距離は 0.0〜30.0m
        self.observation_space = gym.spaces.Box(
            low=0.0, high=30.0, shape=(1080,), dtype=np.float32
        )
        # アクションは ステアリング(-0.4~0.4) と 速度(0.0~10.0) の2次元
        self.action_space = gym.spaces.Box(
            low=np.array([-0.4, 0.0], dtype=np.float32), 
            high=np.array([0.4, 10.0], dtype=np.float32), 
            dtype=np.float32
        )

    def step(self, action):
        # 1. PPOから来る action は shape=(2,) です。
        # シミュレータが期待する shape=(1, 2) に変換します。
        if action.ndim == 1:
            action = action.reshape(1, -1)

        # 2. 親クラスの step を実行して辞書形式の obs を取得
        obs, reward, terminated, truncated, info = super().step(action)
        
        # 3. 強化学習用に obs を配列に変換 (LiDARデータのみ抽出)
        lidar_obs = obs['agent_0']['scan'].astype(np.float32)

        # 衝突判定
        collision_val = obs['agent_0'].get('collision', 0)
        # 1.0 または True なら衝突とみなす
        info['collision'] = bool(collision_val > 0)
        
        # --- 報酬関数 ---
        if info.get('collision', False):
            reward = -1000.0  # 元の関数に合わせ、強力なペナルティ
        else:
            # 1. 速度成分（vs: 接線方向速度, vd: 法線方向速度）
            # velocity は linear_vels_x です。
            vs = info.get('velocity', 0.0) 
            # 速度が遅すぎる場合（停滞）へのペナルティ
            reward = 0.01
            if abs(vs) <= 0.25:
                reward -= 2.0
            
            # 2. 進捗報酬 (vsを最大化)
            reward += 1.0 * vs
            
            # 3. センター維持 (d: center line からの距離)
            # frenet_pose[1] が center line からの lateral error (d)
            d = obs['agent_0']['frenet_pose'][1]
            reward -= 0.05 * abs(d)
            
            # 4. 滑らかさ (w: 角速度)
            w = info.get('angular_vel', 0.0)
            reward -= 0.05 * abs(w)
            
            # 5. 壁への接近回避 (scans)
            scans = obs['agent_0']['scan']
            min_distance = np.min(scans)
            distance_threshold = 0.5
            if min_distance < distance_threshold:
                reward -= 0.01 * (distance_threshold - min_distance)

        return lidar_obs, float(reward), terminated, truncated, info

    def reset(self, **kwargs):
        # 1. 親クラスの reset を実行
        obs, info = super().reset(**kwargs)
        # 2. LiDAR配列を返却
        return obs['agent_0']['scan'].astype(np.float32), info