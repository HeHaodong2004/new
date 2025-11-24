# worker.py  (full, old-style, repo-compatible, NEW near/far + near equal partition + far rdv plan)

import os
import time
import math
from copy import deepcopy
from collections import deque

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

from env import Env
from agent import Agent
from utils import *  # MapInfo, Dijkstra, get_Dijkstra_path_and_dist, get_cell_position_from_coords, ...
from node_manager import NodeManager
from ground_truth_node_manager import GroundTruthNodeManager
from parameter import *


# -------------------------- IO utils --------------------------
if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)

def make_gif_safe(frame_paths, out_path, duration_ms=120):
    frame_paths = [p for p in frame_paths if os.path.exists(p)]
    frame_paths.sort()
    if not frame_paths:
        return
    frames, base_size = [], None
    for p in frame_paths:
        try:
            im = Image.open(p).convert("RGB")
            if base_size is None:
                base_size = im.size
            elif im.size != base_size:
                im = im.resize(base_size, Image.BILINEAR)
            frames.append(im)
        except Exception:
            continue
    if not frames:
        return
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=False)


# -------------------------- Rendezvous Contract --------------------------
class Contract:
    def __init__(self, P_list, r, t_min_list, t_max_list, participants, created_t):
        self.P_list = [np.array(P, dtype=float) for P in P_list]
        self.r = float(r)
        self.t_min_list = [int(x) for x in t_min_list]
        self.t_max_list = [int(x) for x in t_max_list]
        self.participants = set(participants)
        self.created_t = int(created_t)

        # status: 'armed' waiting split | 'active' executing | 'done' success | 'failed' give up
        self.status = 'armed'
        self.target_idx = 0
        self.meta = {}

    @property
    def P(self):
        return self.P_list[self.target_idx]

    @property
    def t_min(self):
        return self.t_min_list[self.target_idx]

    @property
    def t_max(self):
        return self.t_max_list[self.target_idx]

    def within_region(self, pos_xy):
        return np.linalg.norm(np.asarray(pos_xy, dtype=float) - self.P) <= self.r

    def has_backup(self):
        return len(self.P_list) >= 2

    def switch_to_backup(self):
        if self.has_backup() and self.target_idx == 0:
            self.target_idx = 1
            self.meta = {}
            return True
        return False


# --------------------------------- Worker ---------------------------------
class Worker:
    def __init__(self, meta_agent_id, policy_net, predictor, global_step,
                 device='cpu', save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device

        self.env = Env(global_step, plot=self.save_image)
        self.node_manager = NodeManager(plot=self.save_image)
        self.robots = [
            Agent(i, policy_net, predictor, self.node_manager, device=device, plot=save_image)
            for i in range(N_AGENTS)
        ]
        self.gtnm = GroundTruthNodeManager(
            self.node_manager, self.env.ground_truth_info, device=device, plot=save_image
        )

        self.episode_buffer = [[] for _ in range(27)]
        self.perf_metrics = {}

        self.last_known_locations = [self.env.robot_locations.copy() for _ in range(N_AGENTS)]
        self.last_known_intents = [{aid: [] for aid in range(N_AGENTS)} for _ in range(N_AGENTS)]

        self.run_dir = os.path.join(
            gifs_path, f"run_g{self.global_step}_w{self.meta_agent_id}_{os.getpid()}_{int(time.time()*1000)}"
        )
        if self.save_image:
            os.makedirs(self.run_dir, exist_ok=True)
        self.env.frame_files = []

        # Rendezvous / negotiation state
        self.contract: Contract = None
        self.candidate_buffer = []  # kept for old backup logic (optional)
        self.cand_last_update_t = -1
        self.was_fully_connected = False
        self.negotiated_plan = None

        gt_map = self.env.ground_truth_info.map
        self._gt_free_total = int(np.count_nonzero(gt_map == FREE))
        self._t = 0

        # optional global viz fields
        self.assign_masks = None          # (N_AGENTS,H,W) bool, near partitions
        self.far_uncertain_mask = None
        self.nonexist_mask = None
        self.rdv_point = None


    # padding if ckpt older/newer feature dim
    def _match_intent_channels(self, obs_pack):
        n, m, e, ci, ce, ep = obs_pack
        need, got = NODE_INPUT_DIM, n.shape[-1]
        if got < need:
            pad = torch.zeros((n.shape[0], n.shape[1], need - got),
                              dtype=n.dtype, device=n.device)
            n = torch.cat([n, pad], dim=-1)
        return [n, m, e, ci, ce, ep]


    # ===================== NEW PLAN: near/far split + near equal partition + far rdv =====================

    def _compute_near_far_masks(self, global_info: MapInfo, trav_mask):
        """
        near: predicted-traversable cells within near_dist of ANY agent (reliable prediction zone)
        far : remaining predicted-traversable cells (weak prediction zone)

        near_dist_m default uses SENSOR_RANGE (override by RDV_NEAR_DIST_M in parameter.py)
        """
        H, W = trav_mask.shape
        cell_size = float(global_info.cell_size)

        near_dist_m = globals().get('RDV_NEAR_DIST_M', 0.9 * SENSOR_RANGE)
        near_pix = max(1, int(round(near_dist_m / cell_size)))

        near_mask = np.zeros((H, W), dtype=bool)

        for agent in self.robots:
            rr, cc = self._world_to_cell_rc(agent.location, global_info)
            rr = int(np.clip(rr, 0, H - 1))
            cc = int(np.clip(cc, 0, W - 1))

            r0 = max(0, rr - near_pix); r1 = min(H, rr + near_pix + 1)
            c0 = max(0, cc - near_pix); c1 = min(W, cc + near_pix + 1)
            for r in range(r0, r1):
                dr = r - rr
                for c in range(c0, c1):
                    dc = c - cc
                    if dr * dr + dc * dc <= near_pix * near_pix:
                        near_mask[r, c] = True

        near_mask = near_mask & trav_mask
        far_mask = trav_mask & (~near_mask)
        return near_mask, far_mask


    def _balanced_partition_near(self, near_mask, global_info: MapInfo):
        """
        Balanced multi-source BFS partition of near_mask into N_AGENTS contiguous regions
        with roughly equal cell counts.

        Returns:
            assign_masks: (N_AGENTS,H,W) bool
            owner_map: (H,W) int32, -1 for unassigned
        """
        H, W = near_mask.shape
        total = int(near_mask.sum())
        if total == 0:
            assign_masks = np.zeros((N_AGENTS, H, W), dtype=bool)
            owner_map = -np.ones((H, W), dtype=np.int32)
            return assign_masks, owner_map

        quota = int(math.ceil(total / float(N_AGENTS)))
        owner_map = -np.ones((H, W), dtype=np.int32)
        frontiers = [deque() for _ in range(N_AGENTS)]
        counts = [0 for _ in range(N_AGENTS)]
        dirs = [(1,0), (-1,0), (0,1), (0,-1)]

        # seed each agent
        for aid, agent in enumerate(self.robots):
            rr, cc = self._world_to_cell_rc(agent.location, global_info)
            rr = int(np.clip(rr, 0, H - 1))
            cc = int(np.clip(cc, 0, W - 1))

            if not near_mask[rr, cc]:
                seed = self._find_nearest_valid_cell(near_mask, np.array([rr, cc], dtype=int))
                rr, cc = int(seed[0]), int(seed[1])

            if near_mask[rr, cc] and owner_map[rr, cc] == -1:
                owner_map[rr, cc] = aid
                counts[aid] += 1
                frontiers[aid].append((rr, cc))

        active = deque([aid for aid in range(N_AGENTS) if len(frontiers[aid]) > 0])

        # if no seeds landed inside near, fall back to Voronoi on near
        if len(active) == 0:
            owner_voro, _ = self._compute_owner_map_voronoi(global_info, trav_mask=near_mask)
            assign_masks = np.zeros((N_AGENTS, H, W), dtype=bool)
            for aid in range(N_AGENTS):
                assign_masks[aid] = (owner_voro == aid) & near_mask
            return assign_masks, owner_voro

        # balanced growth
        while True:
            unassigned = (owner_map == -1) & near_mask
            if not unassigned.any():
                break
            if len(active) == 0:
                break

            aid = active.popleft()

            if counts[aid] >= quota:
                continue
            if not frontiers[aid]:
                continue

            r, c = frontiers[aid].popleft()
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if (0 <= nr < H and 0 <= nc < W
                        and near_mask[nr, nc]
                        and owner_map[nr, nc] == -1):
                    owner_map[nr, nc] = aid
                    counts[aid] += 1
                    frontiers[aid].append((nr, nc))
                    if counts[aid] >= quota:
                        break

            if counts[aid] < quota and frontiers[aid]:
                active.append(aid)

        # second pass: assign leftover near cells to neighbor/nearest agent
        leftover = (owner_map == -1) & near_mask
        if leftover.any():
            rr_idx, cc_idx = np.where(leftover)
            for rr, cc in zip(rr_idx, cc_idx):
                best_a = None
                # 4-neighbor owner
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and owner_map[nr, nc] >= 0:
                        best_a = int(owner_map[nr, nc])
                        break
                if best_a is None:
                    best_d = 1e18
                    for aid, agent in enumerate(self.robots):
                        ar, ac = self._world_to_cell_rc(agent.location, global_info)
                        d = (rr - ar)**2 + (cc - ac)**2
                        if d < best_d:
                            best_d = d
                            best_a = aid
                owner_map[rr, cc] = int(best_a)

        assign_masks = np.zeros((N_AGENTS, H, W), dtype=bool)
        for aid in range(N_AGENTS):
            assign_masks[aid] = (owner_map == aid) & near_mask
        return assign_masks, owner_map


    def _select_rdv_in_far(self, far_mask, global_info: MapInfo,
                           trav_mask, unknown_mask, p_free, t_now: int):
        """
        Pick a rendezvous point inside far_mask.
        Preference: far unknown density high, p_free high, dispersion low.
        """
        H, W = far_mask.shape
        cell_size = float(global_info.cell_size)
        if int(far_mask.sum()) == 0:
            return None

        rr_idx, cc_idx = np.where(far_mask & unknown_mask)
        if rr_idx.size == 0:
            rr_idx, cc_idx = np.where(far_mask)
        if rr_idx.size == 0:
            return None

        n_cand = min(int(rr_idx.size), globals().get('RDV_CAND_K', 256))
        sel = np.random.choice(rr_idx.size, n_cand, replace=False)

        dist_maps = []
        for agent in self.robots:
            s_rc = self._world_to_cell_rc(agent.location, global_info)
            if not trav_mask[s_rc[0], s_rc[1]]:
                s_rc = self._find_nearest_valid_cell(trav_mask, np.array(s_rc))
            dist_maps.append(self._bfs_dist_map(trav_mask, tuple(s_rc)))

        R_info_pix = max(1, int(round(RDV_INFO_RADIUS_M / cell_size)))

        best_score = -1e18
        best_pack = None

        for k in sel:
            rr, cc = int(rr_idx[k]), int(cc_idx[k])

            etas = []
            feasible = True
            for j in range(N_AGENTS):
                d_steps = dist_maps[j][rr, cc]
                if not np.isfinite(d_steps):
                    feasible = False
                    break
                etas.append(float(d_steps / max(NODE_RESOLUTION, 1e-6)))
            if not feasible:
                continue

            disp = float(max(etas) - min(etas))
            risk = 1.0 - float(p_free[rr, cc])

            r0 = max(0, rr - R_info_pix); r1 = min(H, rr + R_info_pix + 1)
            c0 = max(0, cc - R_info_pix); c1 = min(W, cc + R_info_pix + 1)
            local_unk = float(unknown_mask[r0:r1, c0:c1].mean())

            score = (1.4 * local_unk
                     + 0.6 * float(p_free[rr, cc])
                     - 0.7 * disp
                     - 0.8 * risk)

            if score > best_score:
                best_score = score
                best_pack = (rr, cc, etas, risk, local_unk)

        if best_pack is None:
            return None

        rr, cc, etas, risk, local_unk = best_pack

        eta_max = max(etas)
        t_mid = t_now + int(round(eta_max))
        t_min = t_mid - int(round(RDV_WINDOW_ALPHA_EARLY * eta_max + RDV_WINDOW_BETA_EARLY))
        t_max = t_mid + int(round(RDV_WINDOW_ALPHA_LATE  * eta_max + RDV_WINDOW_BETA_EARLY))

        P_world = np.array([
            global_info.map_origin_x + cc * cell_size,
            global_info.map_origin_y + rr * cell_size
        ], dtype=float)

        return {
            'P': P_world,
            'score': float(best_score),
            'etas': etas,
            'risk': float(risk),
            'ig_total': float(local_unk),
            't_min': int(t_min),
            't_max': int(t_max),
            'ctype': 'far'
        }


    def _update_plan_when_connected(self, global_info: MapInfo,
                                    trav_mask, unknown_mask, p_free, t_now: int):
        """
        Required by you:
          1) split predicted traversable into near vs far
          2) partition near into equal regions => self.assign_masks
          3) pick rdv point in far => self.negotiated_plan / self.rdv_point
        """
        near_mask, far_mask = self._compute_near_far_masks(global_info, trav_mask)
        self.assign_masks, _ = self._balanced_partition_near(near_mask, global_info)
        cand = self._select_rdv_in_far(far_mask, global_info,
                                       trav_mask, unknown_mask, p_free, t_now)
        self.negotiated_plan = cand
        self.rdv_point = cand['P'] if cand is not None else None


    # --------------------------------- Main Loop ---------------------------------
    def run_episode(self):
        done = False

        # init graph / prediction / intent
        for i, r in enumerate(self.robots):
            r.update_graph(self.env.get_agent_map(i), self.env.robot_locations[i])
        for r in self.robots:
            r.update_predict_map()
        for i, r in enumerate(self.robots):
            r.update_planning_state(self.env.robot_locations)
            self.last_known_intents[r.id][r.id] = deepcopy(r.intent_seq)

        groups0 = self._compute_groups_from_positions(self.env.robot_locations)
        self.was_fully_connected = (len(groups0) == 1 and len(groups0[0]) == N_AGENTS)

        if self.save_image:
            self.plot_env(step=0)

        for t in range(MAX_EPISODE_STEP):
            self._t = t

            # global map / masks
            global_map_info = MapInfo(
                self.env.global_belief,
                self.env.belief_origin_x,
                self.env.belief_origin_y,
                self.env.cell_size
            )
            belief_map = global_map_info.map

            if self.robots[0].pred_mean_map_info is not None:
                p_free = self.robots[0].pred_mean_map_info.map.astype(np.float32) / 255.0
            else:
                p_free = (belief_map == FREE).astype(np.float32)

            trav_mask = (((p_free) >= RDV_TAU_FREE) | (belief_map == FREE)) & (belief_map != OCCUPIED)
            unknown_mask = (belief_map == UNKNOWN)

            # current comm connectivity
            groups = self._compute_groups_from_positions(self.env.robot_locations)
            is_fully_connected = (len(groups) == 1 and len(groups[0]) == N_AGENTS)

            # ---------- fully connected: plan near partition + far rdv ----------
            if is_fully_connected:
                if self.contract is not None and self.contract.status == 'active':
                    if RDV_VERBOSE:
                        print(f"[RDV] All-connected @t={t}: cancel active contract.")
                    self.contract = None

                if (t % RDV_CAND_UPDATE_EVERY == 0):
                    try:
                        self._update_plan_when_connected(
                            global_map_info, trav_mask, unknown_mask, p_free, t
                        )
                        if RDV_VERBOSE and self.rdv_point is not None:
                            print(f"[PLAN] connected @t={t}: rdv_point={np.round(self.rdv_point,2)}")
                    except Exception as e:
                        print(f"[PLAN] connected update failed @t={t}: {e}")

            # ---------- split: activate NEW-plan contract ----------
            if (not is_fully_connected) and self.was_fully_connected and (self.contract is None):
                cand = self.negotiated_plan
                if cand is not None:
                    P_c = cand['P']
                    r_meet = RDV_REGION_FRAC * COMMS_RANGE
                    sched = self._chance_constrained_schedule(
                        P_c, r_meet, t, global_map_info, belief_map, p_free
                    )
                    if sched is not None:
                        T_tar, dep_times, q_i, eps_used = sched
                        self.contract = Contract(
                            P_list=[P_c], r=r_meet,
                            t_min_list=[cand['t_min']], t_max_list=[cand['t_max']],
                            participants=set(range(N_AGENTS)), created_t=t
                        )
                        self.contract.meta = {
                            'score': cand['score'],
                            'ig': cand['ig_total'],
                            'risk': cand['risk'],
                            'T_tar': T_tar,
                            't_dep': dep_times,
                            'q_i': q_i,
                            'eps': eps_used
                        }
                        self.contract.status = 'active'
                        if RDV_VERBOSE:
                            print(f"[RDV] Activate NEW-plan rdv @t={t} center={np.round(P_c,2)}, "
                                  f"window=({cand['t_min']},{cand['t_max']}), "
                                  f"T_tar={T_tar}, t_dep={dep_times}")

            self.was_fully_connected = is_fully_connected

            # ---------- local exploration picks ----------
            picks_raw, dists = [], []
            for i, r in enumerate(self.robots):
                obs = r.get_observation(self.last_known_locations[i], self.last_known_intents[i])
                c_obs, _ = self.gtnm.get_ground_truth_observation(
                    r.location, r.pred_mean_map_info, self.env.robot_locations
                )
                r.save_observation(obs, self._match_intent_channels(c_obs))

                nxt, _, act = r.select_next_waypoint(obs)
                r.save_action(act)
                picks_raw.append(nxt)
                dists.append(np.linalg.norm(nxt - r.location))

            # ---------- execute: free before dep_time, forced after ----------
            picks = []
            for i, r in enumerate(self.robots):
                if self.contract is None or self.contract.status != 'active':
                    picks.append(picks_raw[i]); continue

                if self.contract.within_region(r.location):
                    picks.append(self._in_zone_patrol_step(i, r, global_map_info)); continue

                t_dep_i = int(self.contract.meta['t_dep'][i])
                if t < t_dep_i:
                    picks.append(picks_raw[i]); continue

                nxt_xy = self._forced_dijkstra_next_xy(
                    aid=i, agent=r, picks_raw_i=picks_raw[i],
                    global_map_info=global_map_info
                )
                picks.append(nxt_xy)

            # ---------- conflict resolution & env step ----------
            picks = self.resolve_conflicts(picks, dists)

            prev_max = self.env.get_agent_travel().max()
            prev_total = self.env.get_total_travel()
            for rbt, loc in zip(self.robots, picks):
                self.env.step(loc, rbt.id)
            self.env.max_travel_dist = self.env.get_agent_travel().max()

            delta_max = self.env.max_travel_dist - prev_max
            delta_total = self.env.get_total_travel() - prev_total

            # ---------- comm sync after moving ----------
            groups_after_move = self._compute_groups_from_positions(self.env.robot_locations)
            for g in groups_after_move:
                for i in g:
                    for j in g:
                        self.last_known_locations[i][j] = self.env.robot_locations[j].copy()
                        self.last_known_intents[i][j] = deepcopy(self.robots[j].intent_seq)

            # ---------- update graph/pred/intent ----------
            for i, r in enumerate(self.robots):
                r.update_graph(self.env.get_agent_map(i), self.env.robot_locations[i])
                r.update_predict_map()
                r.update_planning_state(self.last_known_locations[i], self.last_known_intents[i])
                self.last_known_intents[i][r.id] = deepcopy(r.intent_seq)

            # ---------- contract progress / fallback ----------
            if self.contract is not None and self.contract.status == 'active':
                try:
                    self._contract_progress_and_fallback(t, global_map_info, trav_mask,
                                                         unknown_mask, p_free)
                except Exception as e:
                    print(f"[RDV] progress/fallback error @t={t}: {e}")

            # ---------- rewards & termination ----------
            team_reward_env, per_agent_obs_rewards = self.env.calculate_reward()
            team_reward = (team_reward_env
                           - ((delta_max / UPDATING_MAP_SIZE) * MAX_TRAVEL_COEF)
                           - ((delta_total / UPDATING_MAP_SIZE) * TOTAL_TRAVEL_COEF))

            utilities_empty = all((r.utility <= 0).all() for r in self.robots)

            if self._gt_free_total > 0:
                per_agent_free_counts = [int(np.count_nonzero(r.map_info.map == FREE)) for r in self.robots]
                per_agent_cov = [c / self._gt_free_total for c in per_agent_free_counts]
                coverage_ok = all(c >= 0.995 for c in per_agent_cov)
            else:
                coverage_ok = False

            done = utilities_empty or coverage_ok
            if done:
                team_reward += 10.0

            for i, r in enumerate(self.robots):
                indiv_total = team_reward + per_agent_obs_rewards[i]
                r.save_reward(indiv_total)
                r.save_done(done)

                next_obs = r.get_observation(self.last_known_locations[i], self.last_known_intents[i])
                c_next_obs, _ = self.gtnm.get_ground_truth_observation(
                    r.location, r.pred_mean_map_info, self.env.robot_locations
                )
                r.save_next_observations(next_obs, self._match_intent_channels(c_next_obs))

            if self.save_image:
                self.plot_env(step=t + 1)

            if done:
                break

        # ---------- summarize metrics ----------
        self.perf_metrics.update({
            'travel_dist': self.env.get_total_travel(),
            'max_travel': self.env.get_max_travel(),
            'explored_rate': self.env.explored_rate,
            'success_rate': done
        })

        free_m2, occ_m2 = self.env.get_discovered_area()
        area_per_agent_m2 = free_m2 + occ_m2
        self.perf_metrics['area_std_m2'] = float(np.std(area_per_agent_m2, ddof=0))

        if self.save_image:
            make_gif_safe(self.env.frame_files,
                          os.path.join(self.run_dir, f"ep_{self.global_step}.gif"))

        for r in self.robots:
            for k in range(len(self.episode_buffer)):
                self.episode_buffer[k] += r.episode_buffer[k]


    # ------------------------------- Conflicts -------------------------------
    def resolve_conflicts(self, picks, dists):
        picks = np.array(picks).reshape(-1, 2)

        forced_idx, free_idx = [], []
        for i in range(len(self.robots)):
            if self.contract is not None and self.contract.status == 'active':
                in_zone = self.contract.within_region(self.robots[i].location)
                dep_list = self.contract.meta.get('t_dep', [0] * N_AGENTS)
                t_dep_i = int(dep_list[i]) if i < len(dep_list) else 0
                in_forced = (not in_zone) and (self._t >= t_dep_i)
            else:
                in_forced = False
            (forced_idx if in_forced else free_idx).append(i)

        order = forced_idx + free_idx
        chosen_complex, resolved = set(), [None] * len(self.robots)

        for rid in order:
            robot = self.robots[rid]
            try:
                loc_key = np.around(robot.location, 1).tolist()
                neighbor_coords = sorted(
                    list(self.node_manager.nodes_dict.find(loc_key).data.neighbor_set),
                    key=lambda c: np.linalg.norm(np.array(c) - picks[rid])
                )
            except Exception:
                neighbor_coords = [robot.location.copy()]

            picked = None
            for cand in neighbor_coords:
                key = complex(cand[0], cand[1])
                if key not in chosen_complex:
                    picked = np.array(cand)
                    break

            resolved[rid] = picked if picked is not None else robot.location.copy()
            chosen_complex.add(complex(resolved[rid][0], resolved[rid][1]))

        return np.array(resolved).reshape(-1, 2)


    # ----------------------- Chance-Constrained Scheduling -----------------------
    def _chance_constrained_schedule(self, P, r, t_now, map_info,
                                     belief_map, p_free):
        H, W = belief_map.shape
        cell_size = float(map_info.cell_size)

        def nearest_goal_rc(trav_mask_s):
            r_pix = int(max(1, round(r / cell_size)))
            c_rc = self._world_to_cell_rc(P, map_info)

            best, bestd = None, float('inf')
            r0 = max(0, c_rc[0] - r_pix); r1 = min(H, c_rc[0] + r_pix + 1)
            c0 = max(0, c_rc[1] - r_pix); c1 = min(W, c_rc[1] + r_pix + 1)

            for rr in range(r0, r1):
                for cc in range(c0, c1):
                    if trav_mask_s[rr, cc]:
                        d = (rr - c_rc[0]) ** 2 + (cc - c_rc[1]) ** 2
                        if d < bestd:
                            bestd, best = d, (rr, cc)
            return best

        samples = []
        for _ in range(RDV_TT_N_SAMPLES):
            rand = np.random.rand(H, W).astype(np.float32)
            trav_s = (((rand < p_free) | (belief_map == FREE)) & (belief_map != OCCUPIED))
            samples.append(trav_s)

        T_samples = {aid: [] for aid in range(N_AGENTS)}

        for s in range(RDV_TT_N_SAMPLES):
            trav_s = samples[s]
            goal_rc = nearest_goal_rc(trav_s)

            if goal_rc is None:
                for aid in range(N_AGENTS):
                    T_samples[aid].append(float('inf'))
                continue

            dist_maps = []
            for aid in range(N_AGENTS):
                start_rc = self._world_to_cell_rc(self.robots[aid].location, map_info)
                if not trav_s[start_rc[0], start_rc[1]]:
                    start_rc = self._find_nearest_valid_cell(trav_s, np.array(start_rc))
                dist_maps.append(self._bfs_dist_map(trav_s, tuple(start_rc)))

            for aid in range(N_AGENTS):
                d = dist_maps[aid][goal_rc[0], goal_rc[1]]
                T_samples[aid].append(float(d / max(NODE_RESOLUTION, 1e-6))
                                      if np.isfinite(d) else float('inf'))

        q_i = []
        for aid in range(N_AGENTS):
            arr = np.array(T_samples[aid], dtype=np.float32)
            feasible = np.isfinite(arr)
            if feasible.sum() < max(1, int((1.0 - RDV_EPSILON) * RDV_TT_N_SAMPLES)):
                return None
            vals = np.sort(arr[feasible])
            k = max(0, int(math.ceil((1.0 - RDV_EPSILON) * len(vals))) - 1)
            q_i.append(float(vals[k]))

        T_tar = max([t_now + qi for qi in q_i])
        dep_times = [int(max(t_now, math.floor(T_tar - qi))) for qi in q_i]
        return int(T_tar), dep_times, q_i, RDV_EPSILON


    # ----------------------- Graph-based forced navigation -----------------------
    def _graph_key(self, xy):
        return tuple(np.around(np.asarray(xy, dtype=float), 1).tolist())

    def _nearest_graph_key(self, xy):
        key = self._graph_key(xy)
        try:
            _ = self.node_manager.nodes_dict.find(list(key))
            return key
        except Exception:
            best_key, best_d = None, float('inf')
            for nd in self.node_manager.nodes_dict.__iter__():
                c = np.asarray(nd.data.coords, dtype=float)
                d = np.linalg.norm(c - xy)
                if d < best_d:
                    best_d, best_key = d, self._graph_key(c)
            return best_key

    def _region_goal_keys(self, P, r):
        keys = []
        for nd in self.node_manager.nodes_dict.__iter__():
            c = np.asarray(nd.data.coords, dtype=float)
            if np.linalg.norm(c - P) <= r:
                keys.append(self._graph_key(c))
        return keys

    def _forced_dijkstra_next_xy(self, aid, agent, picks_raw_i, global_map_info):
        if self.contract is not None and self.contract.within_region(agent.location):
            return self._in_zone_patrol_step(aid, agent, global_map_info)

        start_key = self._nearest_graph_key(agent.location)
        if start_key is None:
            return picks_raw_i

        goal_keys = self._region_goal_keys(self.contract.P, self.contract.r)
        if not goal_keys:
            return picks_raw_i

        dist_dict, prev_dict = Dijkstra(self.node_manager.nodes_dict, start_key)
        feas_goals = [g for g in goal_keys if dist_dict.get(g, 1e8) < 1e8 - 1]
        if not feas_goals:
            return picks_raw_i

        end_key = min(feas_goals, key=lambda g: dist_dict[g])
        path, dist = get_Dijkstra_path_and_dist(dist_dict, prev_dict, end_key)

        if dist >= 1e7 or not path:
            return picks_raw_i

        nxt_xy = np.array(path[0], dtype=float)
        return nxt_xy


    # ----------------------- In-zone patrol -----------------------
    def _in_zone_patrol_step(self, aid, agent, map_info):
        try:
            loc_key = np.around(agent.location, 1).tolist()
            node = self.node_manager.nodes_dict.find(loc_key).data
            neighbor_coords = list(node.neighbor_set)
        except Exception:
            neighbor_coords = [agent.location.copy()]

        def inside(xy): return self.contract.within_region(xy)

        best, best_score = agent.location.copy(), -1e18
        for nb in neighbor_coords:
            if not inside(nb):
                continue

            s_frontier = 0.0
            try:
                nd = self.node_manager.nodes_dict.find(np.around(nb, 1).tolist()).data
                s_frontier = float(max(nd.utility, 0.0))
            except Exception:
                pass

            rr, cc = self._world_to_cell_rc(nb, map_info)

            risk = 1.0
            if self.robots[0].pred_mean_map_info is not None:
                pmap = self.robots[0].pred_mean_map_info.map.astype(np.float32) / 255.0
                if 0 <= rr < pmap.shape[0] and 0 <= cc < pmap.shape[1]:
                    risk = 1.0 - float(pmap[rr, cc])

            score = s_frontier - RDV_RISK_LAMBDA * risk
            if score > best_score:
                best_score, best = score, np.array(nb, dtype=float)

        return best


    # ----------------------- Contract progress & fallback -----------------------
    def _contract_progress_and_fallback(self, t, map_info, trav_mask,
                                        unknown_mask, p_free):
        if self.contract is None or self.contract.status != 'active':
            return

        # 1) all in zone -> success
        all_in = all(self.contract.within_region(r.location) for r in self.robots)
        if all_in:
            if RDV_VERBOSE:
                print(f"[RDV] success rendezvous @t={t}")
            self.contract.status = 'done'
            self.contract = None
            return

        # 2) timeout -> backup or fail (backup optional; candidate_buffer may be empty in NEW plan)
        if t > int(self.contract.t_max):
            if RDV_VERBOSE:
                print(f"[RDV] timeout @t={t}")

            if self.contract.has_backup() and self.contract.target_idx == 0:
                if self.contract.switch_to_backup():
                    ok = self._activate_contract_with_schedule(
                        self.contract, target_idx=1, t_now=t,
                        map_info=map_info, belief_map=map_info.map, p_free=p_free
                    )
                    if ok:
                        if RDV_VERBOSE:
                            print(f"[RDV] switch to backup target @t={t}")
                        return

            backup_added = self._try_add_backup_and_activate(t, map_info, p_free)
            if backup_added:
                return

            self.contract.status = 'failed'
            self.contract = None
            return

        # 3) no graph node in region -> wait/try backup
        if len(self._region_goal_keys(self.contract.P, self.contract.r)) == 0:
            if RDV_VERBOSE:
                print(f"[RDV] region has no graph node; try backup @t={t}")

            if self.contract.has_backup() and self.contract.target_idx == 0:
                if self.contract.switch_to_backup():
                    ok = self._activate_contract_with_schedule(
                        self.contract, target_idx=1, t_now=t,
                        map_info=map_info, belief_map=map_info.map, p_free=p_free
                    )
                    if ok:
                        return

            _ = self._try_add_backup_and_activate(t, map_info, p_free)
            return


    def _activate_contract_with_schedule(self, contract: Contract,
                                         target_idx, t_now,
                                         map_info, belief_map, p_free):
        P = contract.P_list[target_idx]
        r = contract.r

        sched = self._chance_constrained_schedule(P, r, t_now, map_info,
                                                  belief_map, p_free)
        if sched is None:
            return False

        T_tar, dep_times, q_i, eps_used = sched
        contract.status = 'active'
        contract.target_idx = target_idx
        contract.meta = {
            'T_tar': int(T_tar),
            't_dep': [int(x) for x in dep_times],
            'q_i': [float(x) for x in q_i],
            'eps': float(eps_used)
        }
        return True


    def _try_add_backup_and_activate(self, t_now, map_info, p_free):
        if not self.candidate_buffer:
            return False

        primary = {'P': self.contract.P_list[0]}
        cand_backup = self._select_best_candidate_from_buffer(
            idx=1,
            min_dist=max(RDV_BACKUP_MIN_DIST, 0.8 * self.contract.r),
            ref=primary
        )
        if cand_backup is None:
            return False

        self.contract.P_list.append(np.array(cand_backup['P'], dtype=float))
        self.contract.t_min_list.append(int(cand_backup['t_min']))
        self.contract.t_max_list.append(int(cand_backup['t_max']))
        self.contract.target_idx = 1

        ok = self._activate_contract_with_schedule(
            self.contract, target_idx=1, t_now=t_now,
            map_info=map_info, belief_map=map_info.map, p_free=p_free
        )
        if ok and RDV_VERBOSE:
            print(f"[RDV] add & activate backup center={cand_backup['P']} @t={t_now}")

        return ok


    # ----------------------- Candidate Buffer (kept for backup, optional) -----------------------
    def _select_best_candidate_from_buffer(self, idx=0, min_dist=0.0, ref=None):
        if not self.candidate_buffer:
            return None
        if idx == 0:
            return self.candidate_buffer[0]

        taken = 0
        for c in self.candidate_buffer:
            if ref is not None:
                if np.linalg.norm(np.asarray(c['P']) - np.asarray(ref['P'])) < max(1e-6, min_dist):
                    continue
            if taken == idx - 1:
                return c
            taken += 1
        return None


    # ------------------------------- Utilities -------------------------------
    def _compute_groups_from_positions(self, positions):
        n = len(positions)
        if n == 0:
            return []
        used = [False] * n
        groups = []
        for i in range(n):
            if used[i]:
                continue
            comp, q = [], [i]
            used[i] = True
            while q:
                u = q.pop()
                comp.append(u)
                for v in range(n):
                    if used[v]:
                        continue
                    if np.linalg.norm(np.asarray(positions[u]) - np.asarray(positions[v])) <= COMMS_RANGE + 1e-6:
                        used[v] = True
                        q.append(v)
            groups.append(tuple(sorted(comp)))
        return groups

    def _world_to_cell_rc(self, world_xy, map_info):
        cell = get_cell_position_from_coords(np.array(world_xy, dtype=float), map_info)
        return int(cell[1]), int(cell[0])  # (r,c)

    def _find_nearest_valid_cell(self, mask, start_rc):
        q = deque([tuple(start_rc)])
        visited = {tuple(start_rc)}
        H, W = mask.shape
        while q:
            r, c = q.popleft()
            if 0 <= r < H and 0 <= c < W and mask[r, c]:
                return np.array([r, c])
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited:
                    q.append((nr, nc))
                    visited.add((nr, nc))
        return np.asarray(start_rc)

    def _bfs_dist_map(self, trav_mask, start_rc):
        H, W = trav_mask.shape
        dist_map = np.full((H, W), np.inf, dtype=np.float32)
        q = deque([start_rc])
        if (0 <= start_rc[0] < H and 0 <= start_rc[1] < W
                and trav_mask[start_rc]):
            dist_map[start_rc[0], start_rc[1]] = 0.0
        while q:
            r, c = q.popleft()
            base = dist_map[r, c]
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < H and 0 <= nc < W
                        and trav_mask[nr, nc]
                        and np.isinf(dist_map[nr, nc])):
                    dist_map[nr, nc] = base + 1.0
                    q.append((nr, nc))
        return dist_map

    def _bfs_path_rc(self, trav_mask, start_rc, goal_rc):
        H, W = trav_mask.shape
        parent = {tuple(start_rc): None}
        q = deque([tuple(start_rc)])
        if not (0 <= start_rc[0] < H and 0 <= start_rc[1] < W
                and trav_mask[start_rc[0], start_rc[1]]):
            return []
        while q:
            r, c = q.popleft()
            if (r, c) == tuple(goal_rc):
                path, cur = [], (r, c)
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < H and 0 <= nc < W
                        and trav_mask[nr, nc]
                        and (nr, nc) not in parent):
                    parent[(nr, nc)] = (r, c)
                    q.append((nr, nc))
        return []

    def _bresenham_line_rc(self, r0, c0, r1, c1):
        points = []
        dr, dc = abs(r1 - r0), abs(c1 - c0)
        sr, sc = (1 if r1 >= r0 else -1), (1 if c1 >= c0 else -1)
        if dc > dr:
            err, r = dc // 2, r0
            for c in range(c0, c1 + sc, sc):
                points.append((r, c))
                err -= dr
                if err < 0:
                    r += sr
                    err += dc
        else:
            err, c = dr // 2, c0
            for r in range(r0, r1 + sr, sr):
                points.append((r, c))
                err -= dc
                if err < 0:
                    c += sc
                    err += dr
        return points

    def _compute_owner_map_voronoi(self, global_info, trav_mask=None):
        """
        owner_map[r, c] = agent id who 'owns' this cell.
        Default: nearest-agent Voronoi partition over traversable cells.
        """
        belief = global_info.map
        H, W = belief.shape

        if trav_mask is None:
            trav_mask = (belief != OCCUPIED)

        agent_rc = []
        for rbt in self.robots:
            rr, cc = self._world_to_cell_rc(rbt.location, global_info)
            rr = int(np.clip(rr, 0, H - 1))
            cc = int(np.clip(cc, 0, W - 1))
            agent_rc.append((rr, cc))

        rr_grid, cc_grid = np.indices((H, W))
        dist_stack = []
        for (r_a, c_a) in agent_rc:
            dist2 = (rr_grid - r_a) ** 2 + (cc_grid - c_a) ** 2
            dist_stack.append(dist2)

        dist_stack = np.stack(dist_stack, axis=0)
        owner = np.argmin(dist_stack, axis=0).astype(np.int32)

        owner_map = -np.ones((H, W), dtype=np.int32)
        owner_map[trav_mask] = owner[trav_mask]
        return owner_map, trav_mask


    # --------------------------------- Plotting ---------------------------------
    def plot_env(self, step):
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(18, 10), dpi=110)
        gs = GridSpec(N_AGENTS, 3, figure=fig, width_ratios=[2.5, 1.2, 1.2], wspace=0.15, hspace=0.1)
        ax_global = fig.add_subplot(gs[:, 0])
        ax_locals_obs = [fig.add_subplot(gs[i, 1]) for i in range(N_AGENTS)]
        ax_locals_pred = [fig.add_subplot(gs[i, 2]) for i in range(N_AGENTS)]
        agent_colors = plt.cm.get_cmap('cool', N_AGENTS)

        global_info = MapInfo(self.env.global_belief, self.env.belief_origin_x, self.env.belief_origin_y,
                              self.env.cell_size)
        ax_global.set_title(f"Global View | Step {step}/{MAX_EPISODE_STEP}", fontsize=14, pad=10)
        ax_global.imshow(global_info.map, cmap='gray', origin='lower')
        ax_global.set_aspect('equal', adjustable='box')
        ax_global.set_axis_off()

        # ----------------- GLOBAL PARTITION OVERLAY -----------------
        try:
            H, W = global_info.map.shape

            if (self.assign_masks is not None
                and isinstance(self.assign_masks, np.ndarray)
                and self.assign_masks.shape[0] == N_AGENTS
                and self.assign_masks.shape[1] == H
                and self.assign_masks.shape[2] == W):

                cmap_part = plt.cm.get_cmap('tab10', N_AGENTS)
                colored = np.zeros((H, W, 4), dtype=np.float32)

                for aid in range(N_AGENTS):
                    colored[self.assign_masks[aid]] = cmap_part(aid)

                colored[..., 3] = 0.22
                ax_global.imshow(colored, origin='lower', zorder=2)

                for aid in range(N_AGENTS):
                    ax_global.text(
                        5, 15 + 12 * aid,
                        f"Near region {aid}",
                        color=cmap_part(aid), fontsize=9, ha='left', va='center',
                        bbox=dict(facecolor='black', alpha=0.35, edgecolor='none', pad=1)
                    )
            else:
                owner_map, mask_used = self._compute_owner_map_voronoi(global_info)
                cmap_part = plt.cm.get_cmap('tab10', N_AGENTS)
                colored = np.zeros((H, W, 4), dtype=np.float32)
                for aid in range(N_AGENTS):
                    colored[owner_map == aid] = cmap_part(aid)
                colored[..., 3] = 0.18
                colored[~mask_used, 3] = 0.0
                ax_global.imshow(colored, origin='lower', zorder=2)

        except Exception as e:
            ax_global.text(5, 5, f"Partition overlay err: {e}",
                           fontsize=8, color='red', ha='left', va='top')
        # ---------------------------------------------------------------------


        # predicted heat (unchanged)
        if self.robots and self.robots[0].pred_mean_map_info is not None:
            pred_mean = self.robots[0].pred_mean_map_info.map.astype(np.float32) / 255.0
            belief = global_info.map
            unknown_mask = (belief == UNKNOWN)
            prob = np.zeros_like(pred_mean)
            prob[unknown_mask] = pred_mean[unknown_mask]
            ax_global.imshow(prob, cmap='magma', origin='lower', alpha=0.35)

        # connectivity lines (unchanged)
        groups = self._compute_groups_from_positions(self.env.robot_locations)
        for group in groups:
            for i_idx, i in enumerate(list(group)):
                for j in list(group)[i_idx + 1:]:
                    p1 = self._world_to_cell_rc(self.robots[i].location, global_info)
                    p2 = self._world_to_cell_rc(self.robots[j].location, global_info)
                    ax_global.plot([p1[1], p2[1]], [p1[0], p2[0]], color="#33ff88", lw=2, alpha=0.8, zorder=5)

        # trajectories / comm circles / positions (unchanged)
        for i, r in enumerate(self.robots):
            pos_cell = self._world_to_cell_rc(r.location, global_info)
            if hasattr(r, 'trajectory_x') and r.trajectory_x:
                traj_cells = [self._world_to_cell_rc(np.array([x, y]), global_info)
                              for x, y in zip(r.trajectory_x, r.trajectory_y)]
                ax_global.plot([c for rr, c in traj_cells], [rr for rr, c in traj_cells],
                               color=agent_colors(i), lw=1.5, zorder=3)
            comms_radius = patches.Circle((pos_cell[1], pos_cell[0]), COMMS_RANGE / CELL_SIZE,
                                          fc=(0, 1, 0, 0.05), ec=(0, 1, 0, 0.4), ls='--', lw=1.5, zorder=4)
            ax_global.add_patch(comms_radius)
            ax_global.plot(pos_cell[1], pos_cell[0], 'o', ms=10, mfc=agent_colors(i),
                           mec='white', mew=1.5, zorder=10)

        # RDV circle (unchanged)
        if self.contract is not None:
            for k, Pk in enumerate(self.contract.P_list):
                p_cell = self._world_to_cell_rc(Pk, global_info)
                if k == self.contract.target_idx and self.contract.status == 'active':
                    ax_global.plot(p_cell[1], p_cell[0], '*', ms=22, mfc='yellow', mec='white', mew=2, zorder=12)
                else:
                    ax_global.plot(p_cell[1], p_cell[0], '+', ms=20, c=('yellow' if k == 0 else 'orange'),
                                   mew=2, zorder=11)
                radius = patches.Circle(
                    (p_cell[1], p_cell[0]),
                    self.contract.r / CELL_SIZE,
                    fc=(1, 1, 0, 0.07 if k == self.contract.target_idx else 0.04),
                    ec=('yellow' if k == 0 else 'orange'),
                    ls='--', lw=1.8, zorder=11
                )
                ax_global.add_patch(radius)

            if self.contract.status == 'active':
                ax_global.text(5, 5,
                               f"RDV(active) target={self.contract.target_idx} "
                               f"T_tar: {self.contract.meta.get('T_tar', '-')}",
                               fontsize=10, color='yellow', ha='left', va='top')
            elif self.contract.status == 'armed':
                ax_global.text(5, 5, f"RDV(armed) waiting for disconnect",
                               fontsize=10, color='yellow', ha='left', va='top')

        # locals: obs / intent / pred (unchanged)
        for i, r in enumerate(self.robots):
            ax_obs = ax_locals_obs[i]
            local_map_info = r.map_info
            ax_obs.set_title(f"Agent {i} View", fontsize=10, pad=5)
            ax_obs.imshow(local_map_info.map, cmap='gray', origin='lower')
            ax_obs.set_aspect('equal', adjustable='box')
            pos_cell_local = self._world_to_cell_rc(r.location, local_map_info)
            ax_obs.plot(pos_cell_local[1], pos_cell_local[0], 'o', ms=8, mfc=agent_colors(i),
                        mec='white', mew=1.5, zorder=10)
            if r.intent_seq:
                intent_world = [r.location] + r.intent_seq
                intent_cells = [self._world_to_cell_rc(pos, local_map_info) for pos in intent_world]
                ax_obs.plot([c for rr, c in intent_cells], [rr for rr, c in intent_cells],
                            'x--', c=agent_colors(i), lw=2, ms=6, zorder=8)
            ax_obs.set_axis_off()

            ax_pred = ax_locals_pred[i]
            ax_pred.set_title(f"Agent {i} Predicted (local)", fontsize=10, pad=5)
            ax_pred.set_aspect('equal', adjustable='box')
            ax_pred.set_axis_off()

            try:
                if r.pred_mean_map_info is not None or r.pred_max_map_info is not None:
                    pred_info = r.pred_mean_map_info if r.pred_mean_map_info is not None else r.pred_max_map_info
                    pred_local = r.get_updating_map(r.location, base=pred_info)
                    belief_local = r.get_updating_map(r.location, base=r.map_info)
                    ax_pred.imshow(pred_local.map, cmap='gray', origin='lower', vmin=0, vmax=255)
                    alpha_mask = (belief_local.map == FREE) * 0.45
                    ax_pred.imshow(belief_local.map, cmap='Blues', origin='lower', alpha=alpha_mask)
                    rc = get_cell_position_from_coords(r.location, pred_local)
                    ax_pred.plot(rc[0], rc[1], 'mo', markersize=8, zorder=6)
                else:
                    ax_pred.text(0.5, 0.5, 'No prediction', ha='center', va='center', fontsize=9)
            except Exception as e:
                ax_pred.text(0.5, 0.5, f'Pred plot err:\n{e}', ha='center', va='center', fontsize=8)

        out_path = os.path.join(self.run_dir, f"t{step:04d}.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        self.env.frame_files.append(out_path)
