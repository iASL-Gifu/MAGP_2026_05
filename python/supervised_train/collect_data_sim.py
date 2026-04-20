import os
import numpy as np
import hydra
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from src.envs.envs import make_env, MapManagerAdapter
from src.planner.purePursuit import PurePursuitPlanner

@hydra.main(config_path="config", config_name="collect_data_sim", version_base="1.2")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # 実行ごとに固有のランIDディレクトリを作成
    base_out = cfg.output_dir
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(base_out, run_id)
    os.makedirs(out_root, exist_ok=True)
    MAP_DICT = [cfg.envs.map.name]  # Configから取得するようにすると柔軟です

    # マップごとのディレクトリを事前に作成 (これは残しておくと全体構造が分かりやすい)
    for map_name in MAP_DICT:
        os.makedirs(os.path.join(out_root, map_name), exist_ok=True)

    # 環境とプランナーの初期化
    map_cfg     = cfg.envs.map
    map_manager = MapManagerAdapter(
        track_name=MAP_DICT[0], 
        line_type=cfg.envs.map.line_type  # YAMLの envs: map: line_type を渡す
    )
    env = make_env(env_cfg=cfg.envs, map_manager=map_manager, param=cfg.vehicle)
    # print(f"DEBUG: Searching for map yaml near: {os.path.abspath(cfg.envs.map.name + '.yaml')}")

    wheelbase = cfg.planner.wheelbase
    lookahead = cfg.planner.lookahead
    planner = PurePursuitPlanner(
        wheelbase=wheelbase,
        map_manager=map_manager,
        lookahead=lookahead,
        gain=cfg.planner.gain,
        max_reacquire=cfg.planner.max_reacquire,
    )

    render_flag = cfg.render
    render_mode = cfg.render_mode
    num_sets      = cfg.num_sets # 新しい設定項目
    num_steps     = cfg.num_steps
    num_waypoints = cfg.get('num_waypoints', 10)

    map_counters = {m: 0 for m in MAP_DICT}

    # 各マップを num_sets 回繰り返すループ構造に変更
    for set_idx in range(num_sets): # セット数のループ
        print(f"\n--- Data Collection Set: {set_idx + 1}/{num_sets} ---")
        for map_id, name in enumerate(MAP_DICT): # 各マップのループ
            env.update_map(map_name=name, map_ext=map_cfg.ext)
            obs, info = env.reset()

            count = map_counters[name]
            map_counters[name] += 1

            # --- 保存先パスを更新：マップ名の下にさらにランIDのディレクトリを作成 ---
            # 例: output_dir/20240101_123456/map_name/run0/
            run_output_dir = os.path.join(out_root, name, f"run{count}")
            os.makedirs(run_output_dir, exist_ok=True)

            # --- データの初期化 ---
            positions = []
            scans = []
            waypoints = []
            prev_actions = []
            actions = []

            prev_action = np.zeros((1, 2), dtype='float32')
            current_pos = info.get('current_pos', np.array([0.0, 0.0], dtype='float32'))
            truncated = False

            print(f"Collecting data for Map: {name}, Run: {count + 1}...")

            for step in range(num_steps):
                steer, speed = planner.plan(obs)

                ### デバッグ用 ###
                # speed = 2.0
                #################

                action = np.array([steer, speed], dtype='float32').reshape(1, 2)
                scan = obs['agent_0']['scan'].astype('float32')

                wpts = map_manager.get_future_waypoints(
                    current_pos, num_points=num_waypoints
                ).astype('float32')

                wpts = wpts[:, :3]
                if wpts.shape[0] < num_waypoints:
                    pad = np.repeat(wpts[-1][None, :], num_waypoints - wpts.shape[0], axis=0)
                    wpts = np.vstack([wpts, pad])
                wpts = wpts.reshape(1, num_waypoints, 3)

                # --- データの保存 ---
                positions.append(current_pos)
                scans.append(scan)
                waypoints.append(wpts)
                prev_actions.append(prev_action)
                actions.append(action)

                next_obs, reward, terminated, truncated, info = env.step(action)

                '''
                # --- 速度デバッグログを追加 ---
                if step % 20 == 0:
                    # シミュレータ上の実際の速度を取得
                    actual_speed = next_obs['agent_0']['state'][3] # state[3] は通常 vx (速度)
                    
                    print(f"[{name}] Step:{step:03d} | Target:{speed:5.2f}m/s | Actual:{actual_speed:5.2f}m/s")
                    
                    # 速度に変化があるかチェックするための補助表示
                    if speed < 7.0:
                        print(f"  >>> Slowing down for curve: {speed:.2f}m/s")
                # ------------------------------
                '''

                if truncated:
                    print(f"Episode terminated or truncated at step {step + 1}.")
                    break
                obs = next_obs
                prev_action = action
                current_pos = next_obs['agent_0']['state'][:2]

                if render_flag:
                    env.render(mode=render_mode) if render_mode else env.render()

            # `terminated` または `truncated` で早期終了しなかった場合のみデータを保存
            if not truncated:
                # --- データの書き込み ---
                np.save(os.path.join(run_output_dir, "scans.npy"), np.array(scans))

                # actions を steers と speeds に分割して保存
                all_actions = np.array(actions)
                all_steers = all_actions[:, 0, 0] # actions の最初の要素がsteer
                all_speeds = all_actions[:, 0, 1] # actions の二番目の要素がspeed

                # 速度を正規化
                max_speed = map_cfg.speed
                if max_speed != 0:
                    all_speeds_normalized = all_speeds / max_speed
                else:
                    print("[WARN] map_cfg.speed is 0, cannot normalize speed. Saving unnormalized speed.")
                    all_speeds_normalized = all_speeds

                np.save(os.path.join(run_output_dir, "steers.npy"), all_steers)
                np.save(os.path.join(run_output_dir, "speeds.npy"), all_speeds_normalized) 

                # 必要であれば、positions, waypoints, prev_actions も保存
                # np.save(os.path.join(run_output_dir, "positions.npy"), np.array(positions))
                # np.save(os.path.join(run_output_dir, "waypoints.npy"), np.array(waypoints))
                # np.save(os.path.join(run_output_dir, "prev_actions.npy"), np.array(prev_actions))
                
                print(f"Data for Map: {name}, Run: {count + 1} saved to: {run_output_dir}")
            else:
                print(f"Data for Map: {name}, Run: {count + 1} not saved due to early termination or truncation.")

if __name__ == '__main__':
    main()