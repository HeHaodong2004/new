# worker.py  (full, old-style, repo-compatible, no-learning local planner)

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

        # Agent 不使用 policy_net 做 local planning（inside Agent we won't call it）
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
        self.candidate_buffer = []  # list of dict candidates
        self.cand_last_update_t = -1
        self.was_fully_connected = False
        self.negotiated_plan = None

        gt_map = self.env.ground_truth_info.map
        self._gt_free_total = int(np.count_nonzero(gt_map == FREE))
        self._t = 0

        # global viz fields
        self.assign_masks = None           # (N,H,W) near partitions
        self.far_uncertain_mask = None     # (H,W) far zone
        self.rdv_point = None              # world xy


    # padding if ckpt older/newer feature dim (critic obs)
    def _match_intent_channels(self, obs_pack):
        n, m, e, ci, ce, ep = obs_pack
        need, got = NODE_INPUT_DIM, n.shape[-1]
        if got < need:
            pad = torch.zeros((n.shape[0], n.shape[1], need - got),
                              dtype=n.dtype, device=n.device)
            n = torch.cat([n, pad], dim=-1)
        return [n, m, e, ci, ce, ep]


    # --------------------------------- Main Loop ---------------------------------
    def run_episode(self):
        done = False

        # init graph / prediction
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

            # predicted free prob
            if self.robots[0].pred_mean_map_info is not None:
                p_free = self.robots[0].pred_mean_map_info.map.astype(np.float32) / 255.0
            else:
                p_free = (belief_map == FREE).astype(np.float32)

            tau_free = float(getattr(__import__('parameter'), "RDV_TAU_FREE", 0.55))
            trav_mask = (((p_free) >= tau_free) | (belief_map == FREE)) & (belief_map != OCCUPIED)
            unknown_mask = (belief_map == UNKNOWN)

            # current comm connectivity
            groups = self._compute_groups_from_positions(self.env.robot_locations)
            is_fully_connected = (len(groups) == 1 and len(groups[0]) == N_AGENTS)

            # ---------- fully connected: do global split + near partition + far rdv selection ----------
            if is_fully_connected:
                if self.contract is not None and self.contract.status == 'active':
                    # if all connected again, cancel active contract
                    self.contract = None

                # update global assignment periodically or first time
                update_every = int(getattr(__import__('parameter'), "GLOBAL_PARTITION_UPDATE_EVERY", 6))
                if (t % update_every == 0) or (self.assign_masks is None) or (self.rdv_point is None):

                    self._global_partition_and_rdv(
                        global_map_info=global_map_info,
                        trav_mask=trav_mask,
                        unknown_mask=unknown_mask,
                        p_free=p_free,
                        t_now=t
                    )

            # ---------- split: activate negotiated contract ----------
            if (not is_fully_connected) and self.was_fully_connected and (self.contract is None):
                cand = self.negotiated_plan
                if cand is not None:
                    P_c = cand['P']
                    r_meet = float(getattr(__import__('parameter'), "RDV_REGION_FRAC", 0.45)) * COMMS_RANGE
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

            self.was_fully_connected = is_fully_connected

            # ---------- local picks: nearest frontier within my near region ----------
            picks_raw, dists = [], []

            # 哪些 agent 的 near 区域已经没有 frontier
            region_empty = [False] * N_AGENTS
            for aid in range(N_AGENTS):
                region_empty[aid] = (not self._agent_region_has_frontier(aid, global_map_info))

            for i, r in enumerate(self.robots):
                assign_mask_i = None
                if self.assign_masks is not None:
                    assign_mask_i = self.assign_masks[i].astype(bool)

                # ✅ repo-compatible call: Agent now accepts these kwargs
                nxt_xy = r.select_next_waypoint(
                    global_map_info=global_map_info,
                    assign_mask=assign_mask_i,
                    robot_locations=self.last_known_locations[i],
                    global_intents=self.last_known_intents[i],
                )
                if nxt_xy is None:
                    nxt_xy = r.location.copy()

                picks_raw.append(np.array(nxt_xy, dtype=float))
                dists.append(float(np.linalg.norm(picks_raw[-1] - r.location)))

            # ---------- execute: if region empty -> FORCE go RDV immediately ----------
            picks = []
            for i, r in enumerate(self.robots):

                # 1) 区域探索完了：无 frontier -> 直接去 RDV（不等 t_dep）
                if region_empty[i]:
                    if self.contract is not None and self.contract.status == 'active':
                        nxt_xy = self._forced_dijkstra_next_xy(
                            aid=i, agent=r, picks_raw_i=picks_raw[i],
                            global_map_info=global_map_info
                        )
                        picks.append(nxt_xy)
                        continue
                    elif self.rdv_point is not None:
                        nxt_xy = self._forced_dijkstra_to_point(
                            aid=i, agent=r, target_xy=self.rdv_point,
                            global_map_info=global_map_info
                        )
                        picks.append(nxt_xy)
                        continue

                # 2) 没探索完：走正常 forced/free 规则
                if self.contract is None or self.contract.status != 'active':
                    picks.append(picks_raw[i])
                    continue

                if self.contract.within_region(r.location):
                    picks.append(self._in_zone_patrol_step(i, r, global_map_info))
                    continue

                t_dep_i = int(self.contract.meta['t_dep'][i])
                if t < t_dep_i:
                    picks.append(picks_raw[i])
                    continue

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

            # ---------- update graph/pred ----------
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

            # safe utilities_empty
            utilities_empty = True
            for rb in self.robots:
                if rb.utility is None:
                    utilities_empty = False
                else:
                    utilities_empty = utilities_empty and bool((rb.utility <= 0).all())

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


    # ----------------------- Global partition + Far RDV selection -----------------------
    def _global_partition_and_rdv(self, global_map_info, trav_mask, unknown_mask, p_free, t_now):
        """
        While fully connected:
          1) split trav area into near(strong & close) and far(weak & far)
          2) partition near into N contiguous balanced regions
          3) pick RDV from far simultaneously
        """
        belief = global_map_info.map
        H, W = belief.shape
        cell_size = float(global_map_info.cell_size)

        # ---- compute distance-to-nearest-agent for each cell (euclidean in cell space) ----
        rr_grid, cc_grid = np.indices((H, W))
        agent_rc = []
        for rbt in self.robots:
            rr, cc = self._world_to_cell_rc(rbt.location, global_map_info)
            rr = int(np.clip(rr, 0, H - 1))
            cc = int(np.clip(cc, 0, W - 1))
            agent_rc.append((rr, cc))

        dist2_stack = []
        for (ra, ca) in agent_rc:
            dist2_stack.append((rr_grid - ra)**2 + (cc_grid - ca)**2)
        dist2_stack = np.stack(dist2_stack, axis=0)
        min_dist_pix = np.sqrt(np.min(dist2_stack, axis=0)).astype(np.float32)
        min_dist_m = min_dist_pix * cell_size

        # ---- near/far split ----
        near_radius_frac = float(getattr(__import__('parameter'), "NEAR_RADIUS_FRAC", 1.2))
        near_radius_m = near_radius_frac * COMMS_RANGE

        tau_near = float(getattr(__import__('parameter'), "TAU_NEAR_FREE",
                                 getattr(__import__('parameter'), "RDV_TAU_FREE", 0.55)))
        tau_far  = float(getattr(__import__('parameter'), "TAU_FAR_FREE", 0.4))

        near_mask = trav_mask & (min_dist_m <= near_radius_m) & (p_free >= tau_near)

        # far = traversable but either far away or weak prediction
        far_mask = trav_mask & (~near_mask) & (p_free >= tau_far)
        self.far_uncertain_mask = far_mask.astype(np.uint8)

        # ---- partition near into N contiguous balanced regions via capacity-limited multi-source BFS ----
        assign_masks = self._balanced_region_growing_partition(
            near_mask=near_mask,
            agent_rc=agent_rc
        )
        self.assign_masks = assign_masks

        # ---- pick RDV from far_mask (simultaneous) ----
        best = self._pick_rdv_from_far(
            global_map_info=global_map_info,
            trav_mask=trav_mask,
            far_mask=far_mask,
            unknown_mask=unknown_mask,
            p_free=p_free,
            t_now=t_now
        )

        if best is not None:
            self.negotiated_plan = best
            self.rdv_point = best['P'].copy()


    def _balanced_region_growing_partition(self, near_mask, agent_rc):
        """
        Return assign_masks: (N,H,W) contiguous regions with approx equal workload.
        Workload weight = 1 per cell in near_mask.
        """
        H, W = near_mask.shape
        N = N_AGENTS
        owner = -np.ones((H, W), dtype=np.int32)

        # total workload
        total = float(np.sum(near_mask))
        if total <= 1:
            return np.zeros((N, H, W), dtype=np.uint8)

        target = total / float(N)
        tol = float(getattr(__import__('parameter'), "PARTITION_TOL", 0.10))

        # seeds: nearest valid near cell around each agent
        seeds = []
        for (rr, cc) in agent_rc:
            if near_mask[rr, cc]:
                seeds.append((rr, cc))
            else:
                seeds.append(tuple(self._find_nearest_valid_cell(near_mask, np.array([rr, cc]))))

        queues = [deque([s]) for s in seeds]
        sizes = [0.0 for _ in range(N)]

        active = True
        while active:
            active = False
            for aid in range(N):
                if not queues[aid]:
                    continue
                active = True
                r, c = queues[aid].popleft()
                if not (0 <= r < H and 0 <= c < W):
                    continue
                if not near_mask[r, c]:
                    continue
                if owner[r, c] != -1:
                    continue

                if sizes[aid] <= target * (1.0 + tol) or all(s >= target for s in sizes):
                    owner[r, c] = aid
                    sizes[aid] += 1.0
                    for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and near_mask[nr, nc] and owner[nr, nc] == -1:
                            queues[aid].append((nr, nc))
                else:
                    queues[aid].append((r, c))

        unassigned = np.argwhere(near_mask & (owner == -1))
        if unassigned.size > 0:
            for (r, c) in unassigned:
                neigh_owners = []
                for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and owner[nr, nc] != -1:
                        neigh_owners.append(owner[nr, nc])
                if neigh_owners:
                    best_a = min(neigh_owners, key=lambda a: sizes[a])
                    owner[r, c] = best_a
                    sizes[best_a] += 1.0

        assign_masks = np.zeros((N, H, W), dtype=np.uint8)
        for aid in range(N):
            assign_masks[aid] = (owner == aid).astype(np.uint8)
        return assign_masks


    def _pick_rdv_from_far(self, global_map_info, trav_mask, far_mask, unknown_mask, p_free, t_now):
        """
        RDV candidate only from far_mask.
        Score = IG - beta*disp - gamma*risk
        """
        H, W = trav_mask.shape
        cell_size = float(global_map_info.cell_size)

        r_idx, c_idx = np.where(far_mask)
        if r_idx.size == 0:
            return None

        RDV_CAND_K_local = int(getattr(__import__('parameter'), "RDV_CAND_K", 240))
        n_cand = min(r_idx.size, RDV_CAND_K_local)
        sel = np.random.choice(r_idx.size, n_cand, replace=False)
        rc = np.stack([r_idx[sel], c_idx[sel]], axis=1)

        stride = int(getattr(__import__('parameter'), "RDV_CAND_STRIDE", 2))
        if stride > 1:
            keep, seen = [], set()
            for r, c in rc:
                key = (int(r // stride), int(c // stride))
                if key in seen:
                    continue
                seen.add(key)
                keep.append((r, c))
            rc = np.array(keep, dtype=int)

        dist_maps, starts_rc = [], []
        for agent in self.robots:
            start_rc = self._world_to_cell_rc(agent.location, global_map_info)
            starts_rc.append(start_rc)
            if not trav_mask[start_rc[0], start_rc[1]]:
                start_rc = self._find_nearest_valid_cell(trav_mask, np.array(start_rc))
            dist_maps.append(self._bfs_dist_map(trav_mask, tuple(start_rc)))

        candidates = []
        R_info_pix = int(getattr(__import__('parameter'), "RDV_INFO_RADIUS_M", 2.0) / cell_size)
        RDV_ALPHA_local = float(getattr(__import__('parameter'), "RDV_ALPHA", 1.0))
        RDV_BETA_local  = float(getattr(__import__('parameter'), "RDV_BETA", 1.0))
        RDV_GAMMA_local = float(getattr(__import__('parameter'), "RDV_GAMMA", 1.0))
        NODE_RES = max(float(NODE_RESOLUTION), 1e-6)

        for (r_c, c_c) in rc:
            r0 = max(0, r_c - R_info_pix); r1 = min(H, r_c + R_info_pix + 1)
            c0 = max(0, c_c - R_info_pix); c1 = min(W, c_c + R_info_pix + 1)
            local_ig = int(unknown_mask[r0:r1, c0:c1].sum())

            etas, risks, path_ig, feasible = [], [], [], True
            for j, agent in enumerate(self.robots):
                d_steps = dist_maps[j][r_c, c_c]
                if not np.isfinite(d_steps):
                    feasible = False; break
                eta_j = d_steps / NODE_RES
                etas.append(float(eta_j))

                r_s, c_s = self._world_to_cell_rc(agent.location, global_map_info)
                line = self._bresenham_line_rc(r_s, c_s, r_c, c_c)

                line_risk, line_ig = 0.0, 0
                for (rr, cc) in line:
                    if 0 <= rr < H and 0 <= cc < W:
                        line_risk += float(1.0 - p_free[rr, cc])
                        if unknown_mask[rr, cc]:
                            line_ig += 1

                risks.append(line_risk / max(len(line), 1))
                path_ig.append(line_ig)

            if not feasible:
                continue

            ig_total = float(RDV_ALPHA_local * (sum(path_ig) + local_ig))
            disp = float(max(etas) - min(etas))
            risk_total = float(sum(risks))

            score = ig_total - RDV_BETA_local * disp - RDV_GAMMA_local * risk_total

            eta_max = max(etas)
            wa_e = float(getattr(__import__('parameter'), "RDV_WINDOW_ALPHA_EARLY", 0.15))
            wb_e = float(getattr(__import__('parameter'), "RDV_WINDOW_BETA_EARLY", 4.0))
            wa_l = float(getattr(__import__('parameter'), "RDV_WINDOW_ALPHA_LATE",  0.25))
            wb_l = float(getattr(__import__('parameter'), "RDV_WINDOW_BETA_LATE",  6.0))

            t_mid = t_now + int(round(eta_max))
            t_min = t_mid - int(round(wa_e * eta_max + wb_e))
            t_max = t_mid + int(round(wa_l * eta_max + wb_l))

            P_world = np.array([
                global_map_info.map_origin_x + c_c * cell_size,
                global_map_info.map_origin_y + r_c * cell_size
            ], dtype=float)

            candidates.append({
                'P': P_world,
                'score': float(score),
                'etas': etas,
                'risk': float(risk_total),
                'ig_total': float(ig_total),
                't_min': int(t_min),
                't_max': int(t_max),
                'ctype': 'far'
            })

        if not candidates:
            return None
        candidates.sort(key=lambda d: d['score'], reverse=True)
        return candidates[0]


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

        RDV_TT_N_SAMPLES_local = int(getattr(__import__('parameter'), "RDV_TT_N_SAMPLES", 12))
        RDV_EPS_local = float(getattr(__import__('parameter'), "RDV_EPSILON", 0.2))
        NODE_RES = max(float(NODE_RESOLUTION), 1e-6)

        samples = []
        for _ in range(RDV_TT_N_SAMPLES_local):
            rand = np.random.rand(H, W).astype(np.float32)
            trav_s = (((rand < p_free) | (belief_map == FREE)) & (belief_map != OCCUPIED))
            samples.append(trav_s)

        T_samples = {aid: [] for aid in range(N_AGENTS)}

        for s in range(RDV_TT_N_SAMPLES_local):
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
                T_samples[aid].append(float(d / NODE_RES)
                                      if np.isfinite(d) else float('inf'))

        q_i = []
        for aid in range(N_AGENTS):
            arr = np.array(T_samples[aid], dtype=np.float32)
            feasible = np.isfinite(arr)
            if feasible.sum() < max(1, int((1.0 - RDV_EPS_local) * RDV_TT_N_SAMPLES_local)):
                return None
            vals = np.sort(arr[feasible])
            k = max(0, int(math.ceil((1.0 - RDV_EPS_local) * len(vals))) - 1)
            q_i.append(float(vals[k]))

        T_tar = max([t_now + qi for qi in q_i])
        dep_times = [int(max(t_now, math.floor(T_tar - qi))) for qi in q_i]
        return int(T_tar), dep_times, q_i, RDV_EPS_local


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


    def _forced_dijkstra_to_point(self, aid, agent, target_xy, global_map_info, r_goal_m=None):
        if r_goal_m is None:
            r_goal_m = 1.5 * CELL_SIZE

        start_key = self._nearest_graph_key(agent.location)
        if start_key is None:
            return agent.location.copy()

        goal_keys = []
        for nd in self.node_manager.nodes_dict.__iter__():
            c = np.asarray(nd.data.coords, dtype=float)
            if np.linalg.norm(c - np.asarray(target_xy, dtype=float)) <= r_goal_m:
                goal_keys.append(self._graph_key(c))
        if not goal_keys:
            gk = self._nearest_graph_key(target_xy)
            if gk is not None:
                goal_keys = [gk]

        if not goal_keys:
            return agent.location.copy()

        dist_dict, prev_dict = Dijkstra(self.node_manager.nodes_dict, start_key)
        feas_goals = [g for g in goal_keys if dist_dict.get(g, 1e8) < 1e8 - 1]
        if not feas_goals:
            return agent.location.copy()

        end_key = min(feas_goals, key=lambda g: dist_dict[g])
        path, dist = get_Dijkstra_path_and_dist(dist_dict, prev_dict, end_key)

        if dist >= 1e7 or not path:
            return agent.location.copy()

        return np.array(path[0], dtype=float)


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

            lamb = float(getattr(__import__('parameter'), "RDV_RISK_LAMBDA", 0.8))
            score = s_frontier - lamb * risk
            if score > best_score:
                best_score, best = score, np.array(nb, dtype=float)

        return best


    # ----------------------- Contract progress & fallback -----------------------
    def _contract_progress_and_fallback(self, t, map_info, trav_mask,
                                        unknown_mask, p_free):
        if self.contract is None or self.contract.status != 'active':
            return

        all_in = all(self.contract.within_region(r.location) for r in self.robots)
        if all_in:
            self.contract.status = 'done'
            self.contract = None
            return

        if t > int(self.contract.t_max):
            self.contract.status = 'failed'
            self.contract = None
            return

        if len(self._region_goal_keys(self.contract.P, self.contract.r)) == 0:
            self.contract.status = 'failed'
            self.contract = None
            return


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


    def _agent_region_has_frontier(self, aid, global_info):
        if self.assign_masks is None:
            return True
        H, W = global_info.map.shape
        m = self.assign_masks[aid].astype(bool)
        if m.sum() == 0:
            return True

        fr_set = getattr(self.robots[aid], "frontier", set())
        if fr_set is None or len(fr_set) == 0:
            return False

        fr_world = np.array(list(fr_set), dtype=float).reshape(-1, 2)
        fr_rc = np.array([self._world_to_cell_rc(p, global_info) for p in fr_world], dtype=int)

        ok = ((fr_rc[:, 0] >= 0) & (fr_rc[:, 0] < H) &
              (fr_rc[:, 1] >= 0) & (fr_rc[:, 1] < W))
        fr_rc = fr_rc[ok]
        if fr_rc.shape[0] == 0:
            return False

        return bool(np.any(m[fr_rc[:, 0], fr_rc[:, 1]]))


    # ------------------------------- Plotting -------------------------------
    def plot_env(self, step):
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(18, 10), dpi=110)
        gs = GridSpec(N_AGENTS, 3, figure=fig,
                      width_ratios=[2.5, 1.2, 1.2], wspace=0.15, hspace=0.1)
        ax_global = fig.add_subplot(gs[:, 0])
        ax_locals_obs = [fig.add_subplot(gs[i, 1]) for i in range(N_AGENTS)]
        ax_locals_pred = [fig.add_subplot(gs[i, 2]) for i in range(N_AGENTS)]
        agent_colors = plt.cm.get_cmap('cool', N_AGENTS)

        global_info = MapInfo(self.env.global_belief,
                              self.env.belief_origin_x,
                              self.env.belief_origin_y,
                              self.env.cell_size)
        H, W = global_info.map.shape

        ax_global.set_title(f"Global View | Step {step}/{MAX_EPISODE_STEP}",
                            fontsize=14, pad=10)
        ax_global.imshow(global_info.map, cmap='gray',
                         origin='lower', alpha=0.55)
        ax_global.set_aspect('equal', adjustable='box')
        ax_global.set_axis_off()

        try:
            if self.far_uncertain_mask is not None:
                far_mask = self.far_uncertain_mask.astype(bool)
                far_rgba = np.zeros((H, W, 4), dtype=np.float32)
                far_rgba[far_mask] = np.array([0.0, 1.0, 1.0, 0.18], dtype=np.float32)
                ax_global.imshow(far_rgba, origin='lower', zorder=1)
        except Exception as e:
            ax_global.text(5, 30, f"far overlay err: {e}",
                           fontsize=8, color='red', ha='left', va='top')

        try:
            if (self.assign_masks is not None
                    and isinstance(self.assign_masks, np.ndarray)
                    and self.assign_masks.shape == (N_AGENTS, H, W)):

                cmap_part = plt.cm.get_cmap('tab10', N_AGENTS)
                colored = np.zeros((H, W, 4), dtype=np.float32)

                for aid in range(N_AGENTS):
                    m = self.assign_masks[aid].astype(bool)
                    colored[m] = cmap_part(aid)

                colored[..., 3] = 0.45
                ax_global.imshow(colored, origin='lower', zorder=2)

                for aid in range(N_AGENTS):
                    m = self.assign_masks[aid].astype(bool)
                    if m.sum() == 0:
                        continue

                    ax_global.contour(
                        m.astype(np.float32),
                        levels=[0.5],
                        colors=[cmap_part(aid)],
                        linewidths=2.6,
                        origin='lower',
                        zorder=4
                    )

                    rr, cc = np.where(m)
                    cy, cx = float(np.mean(rr)), float(np.mean(cc))
                    ax_global.text(
                        cx, cy, f"A{aid}",
                        color='white', fontsize=16, weight='bold',
                        ha='center', va='center', zorder=5,
                        bbox=dict(facecolor=cmap_part(aid),
                                  alpha=0.85, edgecolor='white',
                                  boxstyle='round,pad=0.2')
                    )

                for aid in range(N_AGENTS):
                    ax_global.text(
                        5, 15 + 14 * aid,
                        f"Near region A{aid}",
                        color=cmap_part(aid), fontsize=10,
                        ha='left', va='center',
                        bbox=dict(facecolor='black', alpha=0.55,
                                  edgecolor='none', pad=1.5)
                    )
        except Exception as e:
            ax_global.text(5, 5, f"Partition overlay err: {e}",
                           fontsize=8, color='red', ha='left', va='top')

        if self.robots and self.robots[0].pred_mean_map_info is not None:
            pred_mean = self.robots[0].pred_mean_map_info.map.astype(np.float32) / 255.0
            belief = global_info.map
            unk = (belief == UNKNOWN)
            prob = np.zeros_like(pred_mean)
            prob[unk] = pred_mean[unk]
            ax_global.imshow(prob, cmap='magma', origin='lower', alpha=0.35, zorder=3)

        groups = self._compute_groups_from_positions(self.env.robot_locations)
        for group in groups:
            for i_idx, i in enumerate(list(group)):
                for j in list(group)[i_idx + 1:]:
                    p1 = self._world_to_cell_rc(self.robots[i].location, global_info)
                    p2 = self._world_to_cell_rc(self.robots[j].location, global_info)
                    ax_global.plot([p1[1], p2[1]], [p1[0], p2[0]],
                                   color="#33ff88", lw=2, alpha=0.9, zorder=6)

        for i, r in enumerate(self.robots):
            pos_cell = self._world_to_cell_rc(r.location, global_info)
            if hasattr(r, 'trajectory_x') and r.trajectory_x:
                traj_cells = [self._world_to_cell_rc(np.array([x, y]), global_info)
                              for x, y in zip(r.trajectory_x, r.trajectory_y)]
                ax_global.plot([c for rr, c in traj_cells], [rr for rr, c in traj_cells],
                               color=agent_colors(i), lw=1.8, zorder=5)
            comms_radius = patches.Circle((pos_cell[1], pos_cell[0]),
                                          COMMS_RANGE / CELL_SIZE,
                                          fc=(0, 1, 0, 0.05),
                                          ec=(0, 1, 0, 0.5),
                                          ls='--', lw=1.6, zorder=5)
            ax_global.add_patch(comms_radius)
            ax_global.plot(pos_cell[1], pos_cell[0], 'o', ms=11, mfc=agent_colors(i),
                           mec='white', mew=1.8, zorder=10)

        if self.rdv_point is not None:
            p_cell = self._world_to_cell_rc(self.rdv_point, global_info)
            ax_global.plot(p_cell[1], p_cell[0], '*', ms=22,
                           mfc='yellow', mec='white', mew=2.0, zorder=12)

        if self.contract is not None:
            for k, Pk in enumerate(self.contract.P_list):
                p_cell = self._world_to_cell_rc(Pk, global_info)
                if k == self.contract.target_idx and self.contract.status == 'active':
                    ax_global.plot(p_cell[1], p_cell[0], '*', ms=24,
                                   mfc='yellow', mec='white', mew=2.2, zorder=12)
                else:
                    ax_global.plot(p_cell[1], p_cell[0], '+', ms=20,
                                   c=('yellow' if k == 0 else 'orange'),
                                   mew=2, zorder=11)
                radius = patches.Circle(
                    (p_cell[1], p_cell[0]),
                    self.contract.r / CELL_SIZE,
                    fc=(1, 1, 0, 0.10 if k == self.contract.target_idx else 0.06),
                    ec=('yellow' if k == 0 else 'orange'),
                    ls='--', lw=2.0, zorder=11
                )
                ax_global.add_patch(radius)

        for i, r in enumerate(self.robots):
            ax_obs = ax_locals_obs[i]
            local_map_info = r.map_info
            ax_obs.set_title(f"Agent {i} View", fontsize=10, pad=5)
            ax_obs.imshow(local_map_info.map, cmap='gray', origin='lower')
            ax_obs.set_aspect('equal', adjustable='box')
            pos_cell_local = self._world_to_cell_rc(r.location, local_map_info)
            ax_obs.plot(pos_cell_local[1], pos_cell_local[0], 'o', ms=8,
                        mfc=agent_colors(i), mec='white', mew=1.5, zorder=10)
            if r.intent_seq:
                intent_world = [r.location] + r.intent_seq
                intent_cells = [self._world_to_cell_rc(pos, local_map_info)
                                for pos in intent_world]
                ax_obs.plot([c for rr, c in intent_cells],
                            [rr for rr, c in intent_cells],
                            'x--', c=agent_colors(i), lw=2, ms=6, zorder=8)
            ax_obs.set_axis_off()

            ax_pred = ax_locals_pred[i]
            ax_pred.set_title(f"Agent {i} Predicted (local)", fontsize=10, pad=5)
            ax_pred.set_aspect('equal', adjustable='box')
            ax_pred.set_axis_off()

            try:
                if r.pred_mean_map_info is not None or r.pred_max_map_info is not None:
                    pred_info = (r.pred_mean_map_info
                                 if r.pred_mean_map_info is not None else r.pred_max_map_info)
                    pred_local = r.get_updating_map(r.location, base=pred_info)
                    belief_local = r.get_updating_map(r.location, base=r.map_info)
                    ax_pred.imshow(pred_local.map, cmap='gray',
                                   origin='lower', vmin=0, vmax=255)
                    alpha_mask = (belief_local.map == FREE) * 0.45
                    ax_pred.imshow(belief_local.map, cmap='Blues',
                                   origin='lower', alpha=alpha_mask)
                    rc = get_cell_position_from_coords(r.location, pred_local)
                    ax_pred.plot(rc[0], rc[1], 'mo', markersize=8, zorder=6)
                else:
                    ax_pred.text(0.5, 0.5, 'No prediction',
                                 ha='center', va='center', fontsize=9)
            except Exception as e:
                ax_pred.text(0.5, 0.5, f'Pred plot err:\n{e}',
                             ha='center', va='center', fontsize=8)

        out_path = os.path.join(self.run_dir, f"t{step:04d}.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        self.env.frame_files.append(out_path)
