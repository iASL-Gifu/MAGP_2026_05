import numpy as np
from numba import njit

"""
Planner Helpers
"""
@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    """
    diffs = trajectory[1:,:] - trajectory[:-1,:]
    l2s   = diffs[:,0]**2 + diffs[:,1]**2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1,:] + (t*diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    """
    starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i,:]
        end = trajectory[i+1,:]+1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V,V)
        b = 2.0*np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
        discriminant = b*b-4*a*c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0*a)
        t2 = (-b + discriminant) / (2.0*a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0],:]
            end = trajectory[(i+1) % trajectory.shape[0],:]+1e-6
            V = end - start

            a = np.dot(V,V)
            b = 2.0*np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
            discriminant = b*b-4*a*c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0*a)
            t2 = (-b + discriminant) / (2.0*a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t

@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    # ターゲットまでの相対距離
    dx = lookahead_point[0] - position[0]
    dy = lookahead_point[1] - position[1]

    # ターゲットへの絶対角度
    target_angle = np.arctan2(dy, dx)
    alpha = target_angle - pose_theta
    
    # 正規化
    if alpha > np.pi:
        alpha -= 2.0 * np.pi
    elif alpha < -np.pi:
        alpha += 2.0 * np.pi
    
    # 車体座標系への変換
    rel_y = lookahead_distance * np.sin(alpha)
    
    speed = lookahead_point[2]
    if np.abs(rel_y) < 1e-6:
        return speed, 0.0, alpha
    
    # 曲率半径 R = L^2 / 2y
    steering_angle = np.arctan(2.0 * wheelbase * rel_y / (lookahead_distance**2))
    
    return speed, steering_angle, alpha

class PurePursuitPlanner:
    """
    Example Planner
    """
    def __init__(self, wheelbase, map_manager, lookahead=2.0, gain=0.2, max_reacquire=20.):
        
        self.wheelbase = wheelbase
        self.map_manager = map_manager
        self.wpt_xind = 0
        self.wpt_yind = 1
        self.wpt_vind = 2
        self.max_reacquire = max_reacquire
        self.lookahead = lookahead
        self.gain = gain
        self.prev_steer = 0.0
    
    def update_map(self, map_manager):
        self.map_manager = map_manager


    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        """
        gets the current waypoint to follow
        """
        wpts = np.vstack((
            self.map_manager.waypoints[:, self.wpt_xind],
            self.map_manager.waypoints[:, self.wpt_yind]
        )).T.astype(np.float64)
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        if nearest_dist < lookahead_distance:

            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(
                position,
                float(lookahead_distance),
                wpts,
                float(i+t),
                wrap=True
            )

            if i2 is None:
                return None

            current_waypoint = np.empty((3, ), dtype=np.float64)
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            speed_lookahead_idx = (i2 + 10) % len(waypoints)  ## ハンドル目標点から更に先の速度を見る
            current_waypoint[2] = waypoints[speed_lookahead_idx, self.wpt_vind]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], waypoints[i, self.wpt_vind])
        else:
            return None

    def plan(self, obs, id=0):
        """
        gives actuation given observation
        """
        agent_id = f'agent_{id}'

        ego_state = obs[agent_id]['state']

        # Pure_pursuit.py の plan メソッド内
        ego_state = obs[agent_id]['state']

        position = np.array(ego_state[:2], dtype=np.float64)
        theta = float(ego_state[4])
        current_velocity = float(ego_state[3])

        lookahead_distance = float(self.lookahead + self.gain * current_velocity)

        all_waypoints = self.map_manager.waypoints.astype(np.float64)

        # 最小距離のガード（0除算防止）
        if lookahead_distance < 0.01:
            lookahead_distance = 0.5
        
        lookahead_point = self._get_current_waypoint(all_waypoints, lookahead_distance, position, theta)

        if lookahead_point is not None:
            lookahead_point = lookahead_point.astype(np.float64)

        if lookahead_point is None:
            return 0.0, 2.0

        lookahead_point = lookahead_point.astype(np.float64)
        wheelbase = float(self.wheelbase)

        speed, steering_angle, alpha = get_actuation(
            theta,
            lookahead_point,
            position,
            lookahead_distance,
            wheelbase
        )

        # --- ステアリングのスムージング---
        # 0.7 と 0.3 の比率は調整可能です。を大きくするほど動きがマイルドになります。
        smoothing_alpha = 0.4
        smoothed_steer = (1.0 - smoothing_alpha) * self.prev_steer + smoothing_alpha * steering_angle
        
        # --- 速度に応じた Pゲインの抑制 ---
        # 速度が速いときに、ステアリングの反応を少し鈍くします
        if speed > 3.5:
            # 速度が上がるほど 0.8倍、0.7倍...と反応を抑える
            speed_factor = np.clip(1.0 - (speed - 4.0) * 0.05, 0.6, 1.0)
            smoothed_steer *= speed_factor

        # --- デッドバンドの導入 ---
        # 直線での微小なガタつきを無視する
        if np.abs(smoothed_steer) < 0.005:
            smoothed_steer = 0.0

        # 次回の計算のために保存
        self.prev_steer = smoothed_steer

        if np.abs(steering_angle) > 0.35: # ハンドルを大きく切っている時
            speed *= 0.8 # さらに20%減速して曲がりやすくする


        # 最終的な速度調整
        vgain = np.float32(0.85)
        speed = vgain * speed
        

        return float(steering_angle), float(speed)