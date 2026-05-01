import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from pyglet.gl import GL_POINTS
from pyglet.text import Label


from f1tenth_gym.maps.map_manager import MapManager

class F110Wrapper(gym.Wrapper):
    """
    F110Env のラッパークラス。
    環境のインターフェースを簡略化し、追加機能を提供する。

    - step() の出力を整形
    - reset() のオプション処理を強化
    - レンダリング処理の簡略化
    - ラップタイム取得機能の追加
    """

    def __init__(self, env, map_manager: MapManager):
        super().__init__(env)
        self.ego_idx = env.ego_idx
        self.map_manager = map_manager

        self._waypoint_vlists = []  # ← 追加：ウェイポイント用のVertexListを保持
        self.needs_waypoint_refresh = True
        self.speed = 0.0

        self.env.add_render_callback(self.render_callback)
        # Waypoint描画機能をレンダリングコールバックとして追加
        self.env.add_render_callback(self.render_waypoints)

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
        current_pos = np.array([obs['poses_x'][0], obs['poses_y'][0]])
        waypoint = self.map_manager.get_future_waypoints(current_pos, num_points=num_points)
        info['waypoint'] = waypoint
        info['current_pos'] = current_pos
        vel_x = obs['linear_vels_x'][0]
        vel_y = obs['linear_vels_y'][0]
        vel = np.sqrt(vel_x**2 + vel_y**2)
        info['velocity'] = vel

        self.speed = vel

        # spin
        if abs(obs['poses_theta'][0]) > 100.0:
            truncated = True
            
        # 1 lap 終了
        if obs['lap_counts'][0] == 1:
            terminated = True

        ## 報酬
        ### 時間経過で報酬を減らす
        reward -= self.env.timestep

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None, index: int=0):
        """
        環境をリセットする。

        返り値:
        - observation: 初期観測データ
        - info: 追加情報（オプションで初期状態指定可能）
        """
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
        map_path = self.map_manager.map_yaml_path
        max_ext = self.map_manager.map_ext

        self.needs_waypoint_refresh = True
        self.env.update_map(map_path, max_ext)

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

        # update camera to follow car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)

        l = 800
        e.left = left - l
        e.right = right + l
        e.top = top + l
        e.bottom = bottom - l

        # 初回のみ Label を生成する
        if not hasattr(self, 'speed_label'):
            self.speed_label = Label('Speed: 0.00 m/s',
                                    font_name='Times New Roman',
                                    font_size=14,
                                    x=left, y=top - 30,
                                    anchor_x='left', anchor_y='top',
                                    color=(255, 255, 255, 255),
                                    batch=e.batch)

        # テキストの内容だけ更新する
        self.speed_label.text = f'Speed: {self.speed:.2f} m/s'

        # 座標も更新する（カメラが動く場合）
        self.speed_label.x = left
        self.speed_label.y = top - 30


    def render_waypoints(self, renderer):
        """
        Waypoint を描画するコールバック。
        古い頂点リストは毎回削除してから、新しいものを登録します。
        """
        # 1) Waypoints 未設定時は何もしない
        if self.map_manager.waypoints is None:
            return

        # 2) 更新が必要な場合のみ再生成処理を実行
        if self.needs_waypoint_refresh:
            print(f"[DEBUG] Waypoints refreshing: clearing {len(self._waypoint_vlists)} old points.")
            
            # 古いものを削除
            for vlist in self._waypoint_vlists:
                vlist.delete()
            self._waypoint_vlists.clear()

            # 新しいウェイポイントを登録
            points = np.vstack((
                self.map_manager.waypoints[:, 0],
                self.map_manager.waypoints[:, 1]
            )).T
            scaled = 50. * points

            for i, (x, y) in enumerate(scaled):
                color = [255, 0, 0] if i == 0 else [200, 200, 200]
                v = renderer.batch.add(
                    1, GL_POINTS, None,
                    ('v3f/stream', [x, y, 0.]),
                    ('c3B/stream', color)
                )
                self._waypoint_vlists.append(v)
            
            self.needs_waypoint_refresh = False
            


class PPOWrapper(F110Wrapper):
    """
    強化学習 (Stable Baselines3) のために Gymnasium 仕様に準拠させたラッパー。
    """
    def __init__(self, env, map_manager, training=True):
        self.raw_env = env
        while hasattr(self.raw_env, 'env'):
            self.raw_env = self.raw_env.env
        super().__init__(self.raw_env, map_manager)
        self.training = training

        self.prev_steering = 0.0
        self.noise_std = 0.05  # 学習時のノイズ
        self.debug_count = 0
        self.max_waypoint_idx = 0
        self.level = 1  # カリキュラムレベル
        
        # PPOが観測データとアクションの範囲を理解するために必須の定義
        # LiDARスキャンは 1080次元、距離は 0.0〜30.0m
        self.observation_space = spaces.Dict({
            "scans": spaces.Box(low=0.0, high=35.0, shape=(1080,), dtype=np.float32),
            "state": spaces.Box(low=-20.0, high=20.0, shape=(3,), dtype=np.float32) # vs, vd, steering
        })
        # アクションは ステアリング(-0.4~0.4) と 速度(0.0~10.0) の2次元
        self.action_space = gym.spaces.Box(
            low=np.array([-0.4, 0.0], dtype=np.float32), 
            high=np.array([0.4, 10.0], dtype=np.float32), 
            dtype=np.float32
        )

    def _get_frenet_state(self, current_pos: np.ndarray, velocity: float, theta: float, waypoints: np.ndarray):
        """
        車両の現在状態（デカルト座標）をコース基準（Frenet座標）に変換します。

        Args:
            current_pos (np.ndarray): 現在の [x, y] 座標
            velocity (float): 車両の進行速度 (linear_vels_x)
            theta (float): 車両のヨー角 (poses_theta)
            waypoints (np.ndarray): コースの [N, 3] 形状のウェイポイント配列

        Returns:
            d (float): コース中央線からの横ズレ距離
            vs (float): コース接線方向の速度 (コースに沿った速さ)
            vd (float): コース法線方向の速度 (コースアウトする速さ)
        """

        if waypoints is None or len(waypoints) == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        # 1. 現在地から最も近いウェイポイントのインデックスを特定
        # 距離の二乗和で比較（x, yの2列のみを使用）
        dists = np.sum((waypoints[:, :2] - current_pos)**2, axis=1)
        idx = np.argmin(dists)
        
        # 2. コースの向き (psi) を計算
        # 近傍のポイントの差分から、その地点でのコースの接線方向角度を算出
        next_idx = (idx + 1) % len(waypoints)
        dx = waypoints[next_idx, 0] - waypoints[idx, 0]
        dy = waypoints[next_idx, 1] - waypoints[idx, 1]
        psi_rad = np.arctan2(dy, dx)
        
        # 3. 横ズレ距離 (d) の計算
        # 現在地から最寄りポイントへのベクトルを、コースの法線ベクトルに投影
        dx_to_wp = current_pos[0] - waypoints[idx, 0]
        dy_to_wp = current_pos[1] - waypoints[idx, 1]
        
        # d = -dx*sin(psi) + dy*cos(psi) の導出
        d = -dx_to_wp * np.sin(psi_rad) + dy_to_wp * np.cos(psi_rad)
        
        # 4. Frenet速度 (vs, vd) の計算
        angle_diff = theta - psi_rad
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        vs = velocity * np.cos(angle_diff)
        vd = velocity * np.sin(angle_diff)
        
        return d, vs, vd, idx

    def set_training_mode(self, mode: bool):
        """学習・評価の切り替え用メソッド"""
        self.training = mode

    def compute_reward(self, d, vs, vd, idx, obs, info, action):
        """ステージに応じた報酬計算の分岐"""
        # 共通の衝突判定
        terminated = False
        if np.any(info.get('collision', 0) > 0):
            return -1000.0, True
        


        if self.level == 1:
            return self._reward_level_1(d, vs, vd, idx, action), terminated
        else:
            return self._reward_level_2(d, vs, vd, idx, action), terminated

    def _reward_level_1(self, d, vs, vd, idx, action):
        """Level 1: 完走重視 (速度報酬を抑え、生存とコース維持を優先)"""
        reward = 0.1  # 高めの生存報酬
        reward += 0.8 * min(vs, 5.0) # 速度報酬は控えめ
        reward -= 0.01 * abs(vd)
        reward -= 0.03 * abs(d) # センター維持は適度
        return reward

    def _reward_level_2(self, d, vs, vd, idx, action):
        """Level 2: 高速化 (速度報酬を強化、ライン取りを厳格化)"""
        reward = 0.01 # 生存報酬を削る
        reward += 0.05 * vs
        reward -= 0.05 * abs(d) # ライン外れを厳しく罰する

        # スピンへの警告
        if abs(vd) > abs(vs):
            reward -= 1.0

        return reward

    def step(self, action):
        # 1. PPOから来る action は shape=(2,) です。
        # シミュレータが期待する shape=(1, 2) に変換します。
        if action.ndim == 1:
            action = action.reshape(1, -1)

        # 2. 親クラスの step を実行して辞書形式の obs を取得
        obs, reward, terminated, truncated, info = super().step(action)

        '''
        # 2. 学習時のみノイズを付与
        if self.training:
            # list の可能性を考慮して numpy 配列に変換
            scans = np.array(obs['scans'])
            
            # ノイズを適用
            noise = np.random.normal(0, self.noise_std, scans.shape)
            scans = np.clip(scans + noise, 0.0, 35.0)
            
            # ドロップアウト処理
            if np.random.rand() < 0.01:
                mask = np.random.rand(*scans.shape) < 0.05
                scans[mask] = 35.0
            
            # 辞書に戻す
            obs['scans'] = scans
        '''

        # 必要な情報を obs から info に移す
        info['collision'] = obs.get('collisions')
        info['lap_times'] = obs.get('lap_times')
        info['lap_counts'] = obs.get('lap_counts')

        current_pos = info.get('current_pos')
        theta = obs['poses_theta'][0]
        velocity = info.get('velocity')
        waypoints = info.get('waypoint')
        d, vs, vd, idx = self._get_frenet_state(current_pos, velocity, theta, waypoints)

        reward, terminated = self.compute_reward(d, vs, vd, idx, obs, info, action)

        '''
        # 1. 区間進捗報酬 (Progress Reward)
        # 過去の最高地点を超えた場合のみ、進んだWP数に応じてボーナス
        if idx > self.max_waypoint_idx:
            steps_forward = idx - self.max_waypoint_idx
            # 周回遅れ判定（リセット直後などの大きなジャンプを抑制）
            if steps_forward < 100: 
                reward += 2.0 * steps_forward
            self.max_waypoint_idx = idx
        
        # コースが一周してインデックスが 0 に戻る場合の処理
        if idx < self.max_waypoint_idx - 200:
            self.max_waypoint_idx = 0
        '''
        
        # 4. 滑らかさ (w: 角速度)
        w = obs['ang_vels_z'][0]
        reward -= 0.05 * abs(w)

        
        # 急な舵角変更の抑制           
        current_steering = action[0][0]
        # steering_diff = abs(current_steering - self.prev_steering)
        # reward -= 1.0 * steering_diff # 急な舵角変更を厳しく制限
        self.prev_steering = current_steering

        # 5. 壁への接近回避 (scans)
        scans = obs['scans'][0]
        min_distance = np.min(scans)
        distance_threshold = 0.5
        if min_distance < distance_threshold:
            reward -= 0.01 * (distance_threshold - min_distance)


        # --- 強制 TimeLimit ロジック ---
        if self.debug_count >= 10000:
            truncated = True
        # -----------------------------

        
        # --- ログ表示 ---
        self.debug_count += 1
        if self.debug_count % 100 == 0:
            if self.training:
                # 表示用にレベルに応じた報酬内訳を再計算
                if self.level == 1:
                    r_speed = 0.5 * min(vs, 5.0)
                    r_dist  = -0.1 * abs(d)
                    mode_str = "LV1 (Stable)"
                else:
                    r_speed = 1.0 * vs
                    r_dist  = -0.5 * abs(d)
                    mode_str = "LV2 (HighSpeed)"

                print(f"\n--- [DEBUG] Step: {self.debug_count} | Mode: {mode_str} ---")
                print(f"  Inputs  | vs: {vs:6.2f}, vd: {vd:6.2f}, d: {d:6.2f}, vel: {velocity:5.2f}, WP_idx: {idx}")
                print(f"  Rewards | Total: {reward:6.2f}")
                print(f"    (Detail) Speed: {r_speed:5.2f}, Dist_Pen: {r_dist:5.2f}, WP_max: {self.max_waypoint_idx}")
                
                if reward < -10:
                    print(f"  !!! WARNING: Negative Reward Spike ({reward:.2f}) !!!")
            else:
                print(f"\r[EVAL] Steps: {self.debug_count}/10000 | Speed: {vs:.2f}", end="")
        

        # 強化学習用に obs を配列に変換
        lidar_obs = obs['scans'][0].astype(np.float32)
        obs_dict = {
            "scans": lidar_obs, # 既存の1080次元LiDAR
            "state": np.array([vs, vd, self.prev_steering], dtype=np.float32)
        }


        return obs_dict, float(reward), terminated, truncated, info

    def reset(self, **kwargs):
        # 1. 親クラスの reset を実行
        obs, info = super().reset(**kwargs)

        self.prev_steering = 0.0
        self.max_waypoint_idx =0
        self.debug_count = 0

        lidar_obs = np.array(obs['scans'][0], dtype=np.float32)

        info['collision'] = obs.get('collisions')
        info['lap_times'] = obs.get('lap_times')
        info['lap_counts'] = obs.get('lap_counts')

        current_pos = info.get('current_pos')
        theta = obs['poses_theta'][0]
        velocity = info.get('velocity')
        waypoints = info.get('waypoint')

        d, vs, vd, idx = self._get_frenet_state(current_pos, velocity, theta, waypoints)

        obs_dict = {
            "scans": lidar_obs, # 既存の1080次元LiDAR
            "state": np.array([vs, vd, self.prev_steering], dtype=np.float32)
        }
        
        # 2. LiDAR配列を返却
        return obs_dict, info