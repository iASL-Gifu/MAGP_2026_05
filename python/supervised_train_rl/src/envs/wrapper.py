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
    
class PPOWrapper(gym.Wrapper):
    """
    強化学習 (Stable Baselines3) 専用の独立したラッパークラス。
    F110Wrapper を継承せず、多層ラッパー構造（TimeLimit等）に強い設計。
    """
    def __init__(self, env, map_manager: MapManager, training=False):
        # 親クラス (gym.Wrapper) を初期化。これにより self.env が設定されます。
        super().__init__(env)
        
        # 【重要】芯 (F110Env) を特定して物理パラメータを安全に取得
        # 間に TimeLimit 等が挟まっていても、unwrapped は一番奥まで貫通します
        inner = self.unwrapped 
        
        self.ego_idx = inner.ego_idx
        self.map_manager = map_manager
        self.training = training

        self._waypoint_vlists = []  # ← 追加：ウェイポイント用のVertexListを保持
        self.needs_waypoint_refresh = True

        self.prev_steering = 0.0
        self.prev_idx = 0
        self.noise_std = 0.05
        self.debug_count = 0
        self.speed = 0.0

        # --- 観測・アクション空間の定義 ---
        self.observation_space = spaces.Dict({
            "scans": spaces.Box(low=0.0, high=35.0, shape=(inner.num_beams,), dtype=np.float32),
            "state": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        })
        
        self.action_space = spaces.Box(
            low=np.array([inner.params['s_min'], inner.params['v_min']], dtype=np.float32), 
            high=np.array([inner.params['s_max'], 10.0], dtype=np.float32), 
            dtype=np.float32
        )

        # レンダリングコールバックの登録（これも芯に対して行う）
        inner.add_render_callback(self.render_callback)
        inner.add_render_callback(self.render_waypoints)

    def render(self, mode="human"):
        """
        環境のレンダリングを簡単に呼び出せるようにする。
        """
        return self.env.render(mode=mode)

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

    def get_lap_time(self):
        """
        Ego車両の現在のラップタイムを取得する。

        返り値:
        - float: 現在のラップタイム
        """
        return self.unwrapped.lap_times[self.ego_idx]

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

    def compute_reward(self, d, vs, vd, idx, obs, info, action):
        # 生存報酬
        reward = 0.05
        terminated = False

        # 衝突ペナルティ
        if np.any(info.get('collision', 0) > 0):
            return -1000.0, True
        
        # 速度報酬
        reward += 1.0 * min(8.0, vs)
        reward -= 0.05 * abs(vd)

        # センターラインから逸れたら減点
        reward -= 0.1 * abs(d)
        
        # 停止ペナルティ 
        if abs(obs["linear_vels_x"][0]) <= 0.25: 
            reward -= 2.0

        # 向心加速度ペナルティ
        kappa = abs(self.map_manager.curvatures[idx])
        lateral_g = (vs ** 2) * kappa
        g_threshold = 6.0
        
        if lateral_g > g_threshold:
            reward -= 2.0 * (lateral_g - g_threshold)

        # ターゲット速度ペナルティ
        target_v = self.map_manager.waypoints[idx, 2] + 1.0
        if vs > target_v:
            # 目標速度を超えた分だけ罰則を与える
            reward -= 1.0 * (vs - target_v)

        # 5. 壁への接近ペナルティ
        scans = obs['scans'][0]
        min_distance = np.min(scans)
        distance_threshold = 0.5
        if min_distance < distance_threshold:
            reward -= 0.01 * (distance_threshold - min_distance)

        # 100ステップに1回、計算結果をプリント
        if self.debug_count % 100 == 0:
            target_v = self.map_manager.waypoints[idx, 2]
            p_accel = max(0.0, 2.0 * (lateral_g - 6.0))
            p_target = max(0.0, 1.0 * (vs - target_v))
            
            # 全ての情報を1行にまとめ、先頭に \r を付与
            log_text = (
                f"\r[Debug] idx:{idx:3} | vx:{obs['linear_vels_x'][0]:.2f} vs:{vs:.2f} tv:{target_v:.2f} | "
                f"k:{kappa:.4f} | P_Acc:-{p_accel:.2f} P_Tgt:-{p_target:.2f} | Rew:{reward:.2f}" 
            )
            # 改行せずに上書き出力
            print(log_text, end="", flush=True)

        return reward, terminated


    def set_training_mode(self, mode: bool):
        """学習・評価の切り替え用メソッド"""
        self.training = mode

    def update_map(self, map_name, map_ext):
        self.map_manager.update_map(map_name)
        map_path = self.map_manager.map_yaml_path
        max_ext = self.map_manager.map_ext

        self.needs_waypoint_refresh = True
        self.env.update_map(map_path, max_ext)

    def step(self, action):
        # 1. PPOから来る action の形状調整
        if action.ndim == 1:
            action = action.reshape(1, -1)

        # 2. 直下の環境 (self.env) の step を実行
        # ここで TimeLimit があれば truncated=True が正しく返ってきます
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 3. waypoint作成と情報の抽出 (F110Wrapper の step 内にあった処理)
        current_pos = np.array([obs['poses_x'][0], obs['poses_y'][0]])
        waypoint, base_idx = self.map_manager.get_future_waypoints(current_pos, num_points=10)
        
        info.update({
            'waypoint': waypoint,
            'current_pos': current_pos,
            'collision': obs.get('collisions'),
            'lap_times': obs.get('lap_times'),
            'lap_counts': obs.get('lap_counts')
        })
        
        vel_x, vel_y = obs['linear_vels_x'][0], obs['linear_vels_y'][0]
        self.speed = np.sqrt(vel_x**2 + vel_y**2)
        info['velocity'] = self.speed

        # 4. 終了判定 (F110Wrapper のロジック)
        if abs(obs['poses_theta'][0]) > 100.0: truncated = True
        if obs['lap_counts'][0] == 1: terminated = True

        # 5. 報酬計算
        d, vs, vd, local_idx = self._get_frenet_state(current_pos, self.speed, obs['poses_theta'][0], waypoint)
        global_idx = (base_idx + local_idx) % len(self.map_manager.waypoints)
        reward, terminated = self.compute_reward(d, vs, vd, global_idx, obs, info, action)

        # 6. 観測データの整形
        self.last_action = np.array(action, dtype=np.float32).flatten()
        obs_dict = {
            "scans": obs['scans'][0].astype(np.float32),
            "state": self.last_action
        }

        # --- ログ表示 ---
        self.debug_count += 1
        '''
        if self.debug_count % 100 == 0:
            if self.training:

                print(f"\r--- [TRAIN] Step: {self.debug_count:5} | vs: {vs:5.2f} | Rew: {reward:6.2f} ---", end="")
                
                if reward < -10:
                    print(f"  !!! WARNING: Negative Reward Spike ({reward:.2f}) !!!")
            else:
                print(f"\r[EVAL] Steps: {self.debug_count}/10000 | Speed: {vs:.2f}", end="")
        '''

        return obs_dict, float(reward), terminated, truncated, info

    def reset(self, seed=None, options=None, index: int=0):
        # --- 初期ポーズ(poses)の自動生成ロジック ---
        if options is None or "poses" not in options:
            positions = []
            if self.map_manager.waypoints is not None:
                num_waypoints = len(self.map_manager.waypoints)
                num_agents = self.unwrapped.num_agents # 芯の環境からエージェント数を取得

                # エージェントごとに開始地点を分散（単独ならindex番目から）
                index_increment = num_waypoints / num_agents

                for i in range(num_agents):
                    waypoint_index = int(i * index_increment + index) % num_waypoints 
                    next_waypoint_index = (waypoint_index + 1) % num_waypoints

                    x, y = self.map_manager.waypoints[waypoint_index][:2]
                    next_x, next_y = self.map_manager.waypoints[next_waypoint_index][:2]
                
                    # 進行方向から角度(yaw)を計算
                    dx = next_x - x
                    dy = next_y - y
                    t = np.arctan2(dy, dx) # math.atan2 の代わりに np を使用

                    positions.append([x, y, t])
            else:
                # ウェイポイントがない場合のフォールバック
                positions = [[0, 0, 0] for _ in range(self.unwrapped.num_agents)]

            # optionsを再構築
            options = options or {}
            options["poses"] = np.array(positions)

        # 直下の環境の reset を実行
        obs, info = self.env.reset(seed=seed, options=options)
        
        self.prev_steering = 0.0
        self.prev_idx = 0
        self.debug_count = 0

        # 初期状態の情報を取得
        current_pos = np.array([obs['poses_x'][0], obs['poses_y'][0]])
        waypoint, base_idx = self.map_manager.get_future_waypoints(current_pos, num_points=10)
        d, vs, vd, local_idx = self._get_frenet_state(current_pos, 0.0, obs['poses_theta'][0], waypoint)

        # 観測データの整形
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)
        obs_dict = {
            "scans": obs['scans'][0].astype(np.float32),
            "state": self.last_action
        }
        return obs_dict, info

    def close(self):
        """リソース解放"""
        self.env.close()