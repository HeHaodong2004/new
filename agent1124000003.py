import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

from utils import *
from parameter import *
from node_manager import NodeManager
from ground_truth_node_manager import GroundTruthNodeManager  # 保留引用，当前实现未直接用它组装观测


class Agent:
    def __init__(self, id, policy_net, predictor, node_manager, device='cpu', plot=False):
        self.id = id
        self.device = device
        self.policy_net = policy_net
        self.predictor = predictor
        self.plot = plot

        # map-related
        self.location = None
        self.map_info = None
        self.cell_size = CELL_SIZE
        self.node_resolution = NODE_RESOLUTION
        self.updating_map_size = UPDATING_MAP_SIZE

        self.updating_map_info = None
        self.frontier = set()

        self.node_manager = node_manager

        self.pred_mean_map_info, self.pred_max_map_info = None, None

        self.node_coords = None
        self.utility = None
        self.guidepost = None           
        self.explored_sign = None
        self.adjacent_matrix = None
        self.neighbor_indices = None

        self.episode_buffer = [[] for _ in range(27)]

        if plot:
            self.trajectory_x, self.trajectory_y = [], []
            
        self.intent_seq = []

        self.rdv_path_nodes_set = set()
        
        self.guidepost_nodes_set = set()

        self.time_left_norm = 0.0        

        # ---------------- Heuristic local planner ----------------
        # 开关：如果 parameter.py 里有 USE_HEURISTIC_LOCAL_PLANNER 就用它；否则默认 True
        try:
            from parameter import USE_HEURISTIC_LOCAL_PLANNER
            self.use_heuristic = bool(USE_HEURISTIC_LOCAL_PLANNER)
        except Exception:
            self.use_heuristic = True

        # worker 每步会塞进来：shape = (H,W) bool，True 表示“我负责的 near 区域”
        self.assigned_mask = None

    def set_assigned_mask(self, mask_hw):
        """mask_hw: (H,W) bool or None"""
        self.assigned_mask = mask_hw

    def _node_in_mask(self, node_xy, map_info, mask_hw):
        """判断某个 graph node 是否落在 assigned mask 内"""
        if mask_hw is None:
            return True
        try:
            rr, cc = get_cell_position_from_coords(np.asarray(node_xy, dtype=float), map_info)
            rr = int(np.clip(rr, 0, mask_hw.shape[1]-1))
            cc = int(np.clip(cc, 0, mask_hw.shape[0]-1))
            # 注意 get_cell_position_from_coords 返回 (x,y) = (c,r)
            return bool(mask_hw[cc, rr])  # mask[row, col]
        except Exception:
            return True

    def _heuristic_next_waypoint(self):
        """
        最近 frontier heuristic：
        1) goals = utility>0 的节点
        2) 如果 assigned_mask 存在，先只在 mask 内找 goals
        3) Dijkstra 找最近 goal，走第一步
        """
        nds = self.node_manager.nodes_dict
        if nds.__len__() == 0 or self.node_coords is None or len(self.node_coords) == 0:
            return self.location.copy(), 0  # stay

        # current graph key
        start_key = tuple(np.around(self.location, 1).tolist())
        try:
            _ = nds.find(list(start_key))
        except Exception:
            # fallback: nearest node
            nn = nds.nearest_neighbors(self.location.tolist(), 1)[0].data.coords
            start_key = tuple(np.around(nn, 1).tolist())

        # collect frontier-like goals (utility>0)
        goals_all = []
        for n in nds.__iter__():
            if getattr(n.data, "utility", 0.0) > 0:
                goals_all.append(tuple(np.around(n.data.coords, 1).tolist()))

        if not goals_all:
            return self.location.copy(), 0

        # prefer goals inside assigned region
        goals_region = [g for g in goals_all
                        if self._node_in_mask(g, self.map_info, self.assigned_mask)]
        goals = goals_region if len(goals_region) > 0 else goals_all  # region empty -> relax

        # Dijkstra from start
        dist_dict, prev_dict = Dijkstra(nds, start_key)

        best_goal, best_dist = None, 1e18
        for g in goals:
            d = dist_dict.get(g, 1e18)
            if d < best_dist:
                best_dist, best_goal = d, g

        if best_goal is None or best_dist >= 1e17:
            return self.location.copy(), 0

        path_nodes, _ = get_Dijkstra_path_and_dist(dist_dict, prev_dict, best_goal)
        if not path_nodes:
            return self.location.copy(), 0

        nxt_xy = np.array(path_nodes[0], dtype=float)
        # find node index in self.node_coords
        key_all = self.node_coords[:, 0] + 1j * self.node_coords[:, 1]
        nxt_key = complex(round(float(nxt_xy[0]),1), round(float(nxt_xy[1]),1))
        idx_arr = np.argwhere(key_all == nxt_key)
        nxt_idx = int(idx_arr[0][0]) if idx_arr.size > 0 else 0
        return nxt_xy, nxt_idx


    def update_map(self, map_info):
        self.map_info = map_info

    def update_location(self, location):
        self.location = np.around(location, 1)
        if self.plot:
            self.trajectory_x.append(self.location[0])
            self.trajectory_y.append(self.location[1])
        if self.node_manager.nodes_dict.__len__() != 0:
            node = self.node_manager.nodes_dict.find(self.location.tolist())
            if node is not None:
                node.data.set_visited()

    def get_updating_map(self, location, base=None):
        base = base if base is not None else self.map_info
        H, W = base.map.shape
        cx, cy = get_cell_position_from_coords(np.array(location, dtype=float), base)
        cx, cy = int(cx), int(cy)
        half_cells = int(round(self.updating_map_size / (2.0 * self.cell_size)))
        x0 = max(0, cx - half_cells)
        y0 = max(0, cy - half_cells)
        x1 = min(W - 1, cx + half_cells)
        y1 = min(H - 1, cy + half_cells)
        sub = base.map[y0:y1+1, x0:x1+1]
        ox = base.map_origin_x + x0 * self.cell_size
        oy = base.map_origin_y + y0 * self.cell_size
        return MapInfo(sub, np.round(ox, 1), np.round(oy, 1), self.cell_size)

    def update_frontiers(self):
        self.frontier = get_frontier_in_map(self.updating_map_info)

    def update_graph(self, map_info, location):
        self.update_map(map_info)
        self.update_location(location)
        self.updating_map_info = self.get_updating_map(self.location, base=self.map_info)
        self.update_frontiers()
        self.node_manager.update_graph(self.location, self.frontier, self.updating_map_info, self.map_info)

    def update_predict_map(self):
        x_belief, mask, x_raw = self.pre_process_input()
        onehots = torch.tensor(
            [[0.333, 0.333, 0.333], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]],
            device=x_belief.device
        ).unsqueeze(1).float()
        preds = []
        for i in range(self.predictor.nsample):
            _, x_inpaint = self.predictor.eval_step(x_belief, mask, onehots[i], self.map_info.map.shape)
            x_proc = self.predictor.post_process(x_inpaint, x_raw, kernel_size=5)
            x_proc = np.where(x_proc > 0, FREE, OCCUPIED)
            preds.append(x_proc)
        self.pred_mean_map_info = MapInfo(
            np.mean(preds, axis=0), self.map_info.map_origin_x, self.map_info.map_origin_y, self.cell_size
        )
        self.pred_max_map_info = MapInfo(
            np.max(preds, axis=0), self.map_info.map_origin_x, self.map_info.map_origin_y, self.cell_size
        )

    def update_planning_state(self, robot_locations, intents_view=None):
        try:
            self.intent_seq = self.predict_intent_path(
                k=INTENT_HORIZON,
                robot_locations_view=robot_locations,
                global_intents_view=intents_view
            )
        except Exception:
            self.intent_seq = []
        self._update_guidepost_path()

    def predict_intent_path(self, k, robot_locations_view=None, global_intents_view=None):
        device = next(self.policy_net.parameters()).device
        intents_for_rollout = {}
        if isinstance(global_intents_view, dict):
            intents_for_rollout = {aid: list(path) for aid, path in global_intents_view.items()}
            intents_for_rollout[self.id] = []
        else:
            intents_for_rollout = {}
        obs = self.get_observation(
            robot_locations=robot_locations_view,
            global_intents=intents_for_rollout
        )
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = [
            t.to(device) for t in obs
        ]
        intent = []
        for _ in range(k):
            with torch.no_grad():
                logp = self.policy_net(node_inputs, node_padding_mask, edge_mask,
                                    current_index, current_edge, edge_padding_mask)
            a = int(torch.argmax(logp, dim=1).item())
            next_idx = int(current_edge[0, a, 0].item())
            intent.append(self.node_coords[next_idx].tolist())
            current_index = torch.tensor([[[next_idx]]], device=device, dtype=torch.long)
            neigh = np.argwhere(self.adjacent_matrix[next_idx] == 0).reshape(-1)
            ce = torch.tensor(neigh, dtype=torch.long, device=device).view(1, -1, 1)
            pad = K_SIZE - ce.size(1)
            if pad > 0:
                ce = torch.nn.functional.pad(ce, (0, 0, 0, pad), value=0)
            current_edge = ce
            ep = torch.cat([
                torch.zeros(1, 1, ce.size(1) - pad, dtype=torch.int16, device=device),
                torch.ones (1, 1, pad,              dtype=torch.int16, device=device)
            ], dim=-1)
            edge_padding_mask = ep
        return intent

    def pre_process_input(self):
        width_in, height_in, _ = self.predictor.config['image_shape']
        width_map, height_map = self.map_info.map.shape
        pad = width_map < width_in and height_map < height_in
        if pad:
            pad_left = (width_in - width_map) // 2
            pad_top = (height_in - height_map) // 2
            pad_right = width_in - width_map - pad_left
            pad_bottom = height_in - height_map - pad_top
            belief = np.pad(self.map_info.map, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')
        else:
            belief = self.map_info.map
        mask = belief.copy()
        mask[mask != UNKNOWN] = 0
        mask[mask == UNKNOWN] = FREE
        x_raw = Image.fromarray(self.map_info.map).convert('L')
        x_belief = Image.fromarray(belief).convert('L')
        mask = Image.fromarray(mask).convert('1')
        if not pad:
            x_belief = transforms.Resize((width_in, height_in))(x_belief)
            mask = transforms.Resize((width_in, height_in))(mask)
        x_belief = transforms.ToTensor()(x_belief).unsqueeze(0).to(self.predictor.device).mul_(2).add_(-1)
        x_raw = transforms.ToTensor()(x_raw).unsqueeze(0).to(self.predictor.device).mul_(2).add_(-1)
        mask = transforms.ToTensor()(mask).unsqueeze(0).to(self.predictor.device)
        return x_belief, mask, x_raw

    def _update_guidepost_path(self):
        try:
            nds = self.node_manager.nodes_dict
            start = nds.nearest_neighbors(self.location.tolist(), 1)[0].data.coords
            frontier_nodes = []
            for n in nds.__iter__():
                if n.data.utility > 0:
                    p = n.data.coords
                    frontier_nodes.append((p[0], p[1]))
            if len(frontier_nodes) == 0:
                self.guidepost_nodes_set = set()
                return
            dist_dict, prev_dict = Dijkstra(nds, (start[0], start[1]))
            best_goal, best_dist = None, 1e18
            for g in frontier_nodes:
                if g in dist_dict and dist_dict[g] < best_dist:
                    best_dist, best_goal = dist_dict[g], g
            if best_goal is None or best_dist >= 1e17:
                self.guidepost_nodes_set = set()
                return
            path_nodes, _ = get_Dijkstra_path_and_dist(dist_dict, prev_dict, best_goal)
            self.guidepost_nodes_set = { (round(float(x),1), round(float(y),1)) for (x,y) in path_nodes }
        except Exception:
            self.guidepost_nodes_set = set()


    def _assemble_observation_from_shared_graph(self, robot_locations=None, global_intents=None):
        """
        节点特征顺序（逐节点拼接）——固定 11 维：
        1) rel_xy(2)                # 相对当前节点的偏移（归一化）
        2) utility(1)               # 归一化 utility
        3) pred_prob(1)             # 预测 free 概率（来自 pred_mean）
        4) known(1)                 # 是否已知（explored_sign）
        5) guidepost(1)             # 最近 frontier 的最短路节点=1
        6) occupancy(1)             # -1 自身，1 其他机器人，0 其它
        7) intent_any(1)            # （新）把通信范围内队友的未来意图节点标 1
        8) conn_frac(1)             # （新）当前与“我”可直连通信的队友占比（不含自己）
        9) rdv_path(1)  deleted            # rendezvous 最短路节点=1 #delete in this code for ablation study
        10) time_left_norm(1)  deleted       # 会合紧迫度，所有节点同一常数 #delete in this code for ablation study
        """
        all_node_coords = []
        for n in self.node_manager.nodes_dict.__iter__():
            all_node_coords.append(np.around(n.data.coords, 1))
        all_node_coords = np.array(all_node_coords).reshape(-1, 2)

        n_nodes = all_node_coords.shape[0]
        if n_nodes == 0:
            node_inputs = torch.zeros((1, NODE_PADDING_SIZE, NODE_INPUT_DIM), dtype=torch.float32, device=self.device)
            node_padding_mask = torch.ones((1, 1, NODE_PADDING_SIZE), dtype=torch.int16, device=self.device)
            edge_mask = torch.ones((1, NODE_PADDING_SIZE, NODE_PADDING_SIZE), dtype=torch.float32, device=self.device)
            current_index_t = torch.tensor([0], dtype=torch.long, device=self.device).reshape(1, 1, 1)
            current_edge = torch.zeros((1, K_SIZE, 1), dtype=torch.long, device=self.device)
            edge_padding_mask = torch.ones((1, 1, K_SIZE), dtype=torch.int16, device=self.device)
            pack = [node_inputs, node_padding_mask, edge_mask, current_index_t, current_edge, edge_padding_mask]
            # meta: coords, utility, guidepost(mask), explored_sign, adj, neighbor_indices
            meta = [np.zeros((0,2)), np.zeros((0,)), np.zeros((0,)), np.zeros((0,)), np.ones((0,0), dtype=int), np.zeros((0,), dtype=int)]
            return pack, meta

        utility = []
        adjacent_matrix = np.ones((n_nodes, n_nodes), dtype=int)
        key_all = all_node_coords[:, 0] + 1j * all_node_coords[:, 1]
        for i, coords in enumerate(all_node_coords):
            nd = self.node_manager.nodes_dict.find(coords.tolist()).data
            utility.append(nd.utility)
            for nb in nd.neighbor_set:
                idx = np.argwhere(key_all == (nb[0] + 1j*nb[1]))
                if idx.size > 0:
                    adjacent_matrix[i, idx[0][0]] = 0
        utility = np.array(utility, dtype=np.float32)

        # ---------- 3) explored_sign（known） ----------
        explored_sign = []
        for coords in all_node_coords:
            cell = get_cell_position_from_coords(coords, self.map_info)
            known = (self.map_info.map[cell[1], cell[0]] != UNKNOWN)
            explored_sign.append(1 if known else 0)
        explored_sign = np.array(explored_sign, dtype=np.float32)

        # ---------- 4) pred_prob ----------
        if self.pred_mean_map_info is None:
            pred_prob = np.zeros((n_nodes,), dtype=np.float32)
        else:
            pred_prob = []
            for coords in all_node_coords:
                cell = get_cell_position_from_coords(coords, self.pred_mean_map_info)
                pred_prob.append(float(self.pred_mean_map_info.map[cell[1], cell[0]]))
            pred_prob = np.array(pred_prob, dtype=np.float32)

        # ---------- 5) current_index 与邻居 ----------
        mykey = self.location[0] + 1j * self.location[1]
        idx_arr = np.argwhere(key_all == mykey)
        if idx_arr.size == 0:
            nn = self.node_manager.nodes_dict.nearest_neighbors(self.location.tolist(), 1)[0].data.coords
            mykey = nn[0] + 1j * nn[1]
            idx_arr = np.argwhere(key_all == mykey)
        current_index = int(idx_arr[0][0])

        curr_node = self.node_manager.nodes_dict.find(all_node_coords[current_index].tolist()).data
        neighbor_indices = []
        for nb in curr_node.neighbor_set:
            idx = np.argwhere(key_all == (nb[0] + 1j*nb[1]))
            if idx.size > 0:
                neighbor_indices.append(int(idx[0][0]))
        neighbor_indices = np.unique(np.array(neighbor_indices, dtype=np.int64))

        # ---------- 6) occupancy ----------
        occupancy = np.zeros((n_nodes,), dtype=np.float32)
        occupancy[current_index] = -1.0
        if robot_locations is not None:
            for pos in robot_locations:
                if np.allclose(pos, self.location):
                    continue
                try:
                    nn = self.node_manager.nodes_dict.nearest_neighbors(pos.tolist(), 1)[0].data.coords
                    jarr = np.argwhere(key_all == (nn[0] + 1j*nn[1]))
                    if jarr.size > 0:
                        j = int(jarr[0][0])
                        if j != current_index:
                            occupancy[j] = 1.0
                except Exception:
                    pass

        # ---------- 7) intent_any ----------
        def to_key_xy(p):
            return (round(float(p[0]), 1), round(float(p[1]), 1))

        intent_any = np.zeros((n_nodes, 1), dtype=np.float32)
        if (global_intents is not None and isinstance(global_intents, dict)
            and robot_locations is not None and len(robot_locations) > 0):
            # 只聚合与“我”可通信的队友
            me = np.array(self.location, dtype=float)
            visible_ids = []
            for aid, pos in enumerate(robot_locations):
                if np.allclose(pos, self.location):
                    continue
                if np.linalg.norm(np.array(pos, dtype=float) - me) <= float(COMMS_RANGE) + 1e-6:
                    visible_ids.append(aid)

            node_keys = [(round(float(c[0]), 1), round(float(c[1]), 1)) for c in all_node_coords]
            key2idx = {k: i for i, k in enumerate(node_keys)}

            # 标记这些队友的前 INTENT_HORIZON 步意图
            H = int(getattr(__import__('parameter'), 'INTENT_HORIZON', 3))
            for aid in visible_ids:
                path = global_intents.get(aid, [])
                if not path: 
                    continue
                for p in path[:H]:
                    k = to_key_xy(p)
                    if k in key2idx:
                        intent_any[key2idx[k], 0] = 1.0

        # ---------- 8) conn_frac ----------
        if robot_locations is None or len(robot_locations) <= 1:
            conn_frac_col = np.zeros((n_nodes, 1), dtype=np.float32)
        else:
            me = np.array(self.location, dtype=float)
            total, cnt = 0, 0
            for pos in robot_locations:
                if np.allclose(pos, self.location):
                    continue
                total += 1
                if np.linalg.norm(np.array(pos, dtype=float) - me) <= float(COMMS_RANGE) + 1e-6:
                    cnt += 1
            frac = (cnt / total) if total > 0 else 0.0
            conn_frac_col = np.full((n_nodes, 1), float(frac), dtype=np.float32)

        # ---------- 9) rdv_path mask ----------
        rdv_path = np.zeros((n_nodes, 1), dtype=np.float32)
        if isinstance(self.rdv_path_nodes_set, set) and len(self.rdv_path_nodes_set) > 0:
            path_keys = self.rdv_path_nodes_set
            for i, c in enumerate(all_node_coords):
                key = (round(float(c[0]), 1), round(float(c[1]), 1))
                if key in path_keys:
                    rdv_path[i, 0] = 1.0

        # ---------- 10) guidepost mask ----------
        guidepost_mask = np.zeros((n_nodes, 1), dtype=np.float32)
        if isinstance(self.guidepost_nodes_set, set) and len(self.guidepost_nodes_set) > 0:
            gp = self.guidepost_nodes_set
            for i, c in enumerate(all_node_coords):
                key = (round(float(c[0]), 1), round(float(c[1]), 1))
                if key in gp:
                    guidepost_mask[i, 0] = 1.0

        # ---------- 11) 组装特征 ----------
        node_coords = all_node_coords
        current_node_coords = node_coords[current_index]
        rel_xy = np.concatenate(
            (node_coords[:, [0]] - current_node_coords[0], node_coords[:, [1]] - current_node_coords[1]),
            axis=-1
        ) / UPDATING_MAP_SIZE / 2.0

        node_utility   = utility.reshape(-1, 1) / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)
        node_predprob  = pred_prob.reshape(-1, 1) / float(FREE)
        node_known     = explored_sign.reshape(-1, 1)
        node_occupancy = occupancy.reshape(-1, 1)

        # 常数列：会合紧迫度
        time_left_col  = np.full((n_nodes, 1), float(getattr(self, "time_left_norm", 0.0)), dtype=np.float32)

        feats = np.concatenate(
            (
                rel_xy,            # 2
                node_utility,      # 1
                node_predprob,     # 1
                node_known,        # 1
                guidepost_mask,    # 1
                node_occupancy,    # 1
                intent_any,        # 1  
                conn_frac_col,     # 1  
            ),
            axis=1
        )
        assert feats.shape[1] == NODE_INPUT_DIM, f"NODE_INPUT_DIM({NODE_INPUT_DIM}) != feats({feats.shape[1]})"
        node_inputs = torch.as_tensor(feats, dtype=torch.float32, device=self.device).unsqueeze(0)

        # ---------- 12) padding 与 mask ----------
        n_node = node_coords.shape[0]
        assert n_node < NODE_PADDING_SIZE, f"{n_node} >= {NODE_PADDING_SIZE}"
        node_inputs = torch.nn.ZeroPad2d((0, 0, 0, NODE_PADDING_SIZE - n_node))(node_inputs)

        node_padding_mask = torch.zeros((1, 1, n_node), dtype=torch.int16, device=self.device)
        node_padding = torch.ones((1, 1, NODE_PADDING_SIZE - n_node), dtype=torch.int16, device=self.device)
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

        edge_mask = torch.as_tensor(adjacent_matrix, dtype=torch.float32, device=self.device).unsqueeze(0)
        edge_mask = torch.nn.ConstantPad2d((0, NODE_PADDING_SIZE - n_node, 0, NODE_PADDING_SIZE - n_node), 1)(edge_mask)

        if current_index not in neighbor_indices.tolist():
            neighbor_indices = np.unique(np.concatenate([neighbor_indices, np.array([current_index], dtype=np.int64)]))

        current_edge = torch.as_tensor(neighbor_indices, dtype=torch.long, device=self.device).unsqueeze(0)
        k_size = current_edge.size(-1)
        try:
            current_in_edge = int(np.argwhere(neighbor_indices == current_index)[0][0])
        except Exception:
            current_in_edge = 0
        current_edge = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 0)(current_edge).unsqueeze(-1)

        edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16, device=self.device)
        if k_size > 0 and 0 <= current_in_edge < k_size:
            edge_padding_mask[0, 0, current_in_edge] = 1
        edge_padding_mask = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 1)(edge_padding_mask)

        current_index_t = torch.as_tensor([current_index], dtype=torch.long, device=self.device).reshape(1, 1, 1)

        # meta: coords, utility, guidepost(mask), explored_sign, adj, neighbor_indices
        pack = [node_inputs, node_padding_mask, edge_mask, current_index_t, current_edge, edge_padding_mask]
        meta = [node_coords, utility, guidepost_mask.squeeze(1), explored_sign, adjacent_matrix, neighbor_indices]
        return pack, meta

    def get_observation(self, robot_locations=None, global_intents=None):
        pack, meta = self._assemble_observation_from_shared_graph(
            robot_locations=robot_locations,
            global_intents=global_intents
        )
        [self.node_coords, self.utility, self.guidepost, self.explored_sign,
         self.adjacent_matrix, self.neighbor_indices] = meta
        return pack

    def _select_next_waypoint_from_obs(self, observation, greedy=False):
        _, _, _, _, current_edge, _ = observation

        # ------------------ Heuristic mode ------------------
        if self.use_heuristic or greedy == "heuristic":
            nxt_pos, nxt_idx = self._heuristic_next_waypoint()

            # 把 nxt_idx 映射回 current_edge 里的 action index，保证 buffer 不炸
            action_index = 0
            try:
                ce_np = current_edge[0, :, 0].detach().cpu().numpy().astype(int)
                hit = np.argwhere(ce_np == nxt_idx)
                if hit.size > 0:
                    action_index = int(hit[0][0])
            except Exception:
                action_index = 0

            act_t = torch.as_tensor(action_index, dtype=torch.long, device=self.device)
            return nxt_pos, nxt_idx, act_t

        # ------------------ RL mode (original) ------------------
        with torch.no_grad():
            logp = self.policy_net(*observation)
        action_index = (torch.argmax(logp, dim=1).long()
                        if greedy else torch.multinomial(logp.exp(), 1).long().squeeze(1))
        next_node_index = int(current_edge[0, action_index.item(), 0].item())
        next_position = self.node_coords[next_node_index]
        return next_position, next_node_index, action_index


    # 新 wrapper：兼容 worker 新旧调用方式
    def select_next_waypoint(self, observation=None, greedy=False,
                             global_map_info=None, assign_mask=None,
                             robot_locations=None, global_intents=None):
        """
        兼容两种调用：

        1) 老式（training/RL）:
           nxt_pos, nxt_idx, act = select_next_waypoint(observation, greedy)

        2) 新式（你现在的 worker）:
           nxt_pos = select_next_waypoint(global_map_info=..., assign_mask=...)
        """

        # worker 传进来的 assigned_mask
        if assign_mask is not None:
            self.set_assigned_mask(assign_mask.astype(bool))

        # --- 新式调用：没有 observation ---
        if observation is None:
            obs = self.get_observation(
                robot_locations=robot_locations,
                global_intents=global_intents
            )
            nxt_pos, _, _ = self._select_next_waypoint_from_obs(obs, greedy=greedy)
            return nxt_pos

        # --- 老式调用：正常返回三元组 ---
        return self._select_next_waypoint_from_obs(observation, greedy=greedy)

    def save_observation(self, observation, critic_observation):
        n, m, e, ci, ce, ep = observation
        self.episode_buffer[0] += n
        self.episode_buffer[1] += m.bool()
        self.episode_buffer[2] += e.bool()
        self.episode_buffer[3] += ci
        self.episode_buffer[4] += ce
        self.episode_buffer[5] += ep.bool()

        cn, cm, cee, cci, cce, cep = critic_observation
        self.episode_buffer[15] += cn
        self.episode_buffer[16] += cm.bool()
        self.episode_buffer[17] += cee.bool()
        self.episode_buffer[18] += cci
        self.episode_buffer[19] += cce
        self.episode_buffer[20] += cep.bool()

    def save_action(self, action_index):
        self.episode_buffer[6] += action_index.reshape(1, 1, 1)

    def save_reward(self, reward):
        self.episode_buffer[7] += torch.as_tensor([reward], dtype=torch.float32, device=self.device).reshape(1, 1, 1)

    def save_done(self, done):
        self.episode_buffer[8] += torch.as_tensor([int(done)], dtype=torch.long, device=self.device).reshape(1, 1, 1)

    def save_next_observations(self, observation, critic_observation):
        n, m, e, ci, ce, ep = observation
        self.episode_buffer[9] += n
        self.episode_buffer[10] += m.bool()
        self.episode_buffer[11] += e.bool()
        self.episode_buffer[12] += ci
        self.episode_buffer[13] += ce
        self.episode_buffer[14] += ep.bool()

        cn, cm, cee, cci, cce, cep = critic_observation
        self.episode_buffer[21] += cn
        self.episode_buffer[22] += cm.bool()
        self.episode_buffer[23] += cee.bool()
        self.episode_buffer[24] += cci
        self.episode_buffer[25] += cce
        self.episode_buffer[26] += cep.bool()

    def plot_env(self):
        if not self.plot:
            return
        plt.switch_backend('agg')
        plt.figure(figsize=(19, 5))
        plt.subplot(1, 4, 2)
        plt.axis('off')

        nodes = get_cell_position_from_coords(self.node_coords, self.map_info)
        if len(self.frontier) > 0:
            fr = get_cell_position_from_coords(np.array(list(self.frontier)), self.map_info).reshape(-1, 2)
            plt.scatter(fr[:, 0], fr[:, 1], c='r', s=2, zorder=4)

        robot = get_cell_position_from_coords(self.location, self.map_info)

        if self.pred_max_map_info is not None:
            plt.imshow(self.pred_max_map_info.map, cmap='gray', vmin=0, vmax=255)
        alpha_mask = (self.map_info.map == FREE) * 0.5
        plt.imshow(self.map_info.map, cmap='Blues', alpha=alpha_mask)

        util_vis = np.where(self.utility > 0, self.utility, 0).astype(np.uint8)
        plt.scatter(nodes[:, 0], nodes[:, 1], c=util_vis, zorder=2)
        for c, u in zip(nodes, util_vis):
            if u > 0:
                plt.text(c[0], c[1], str(u), fontsize=8, zorder=3)
        plt.plot(robot[0], robot[1], 'mo', markersize=16, zorder=5)

        for coords in self.node_coords:
            n = self.node_manager.nodes_dict.find(coords.tolist())
            if n is None:
                continue
            node = n.data
            for neigh in node.neighbor_set:
                end = (np.array(neigh) - coords) / 2 + coords
                plt.plot((np.array([coords[0], end[0]]) - self.map_info.map_origin_x) / self.cell_size,
                         (np.array([coords[1], end[1]]) - self.map_info.map_origin_y) / self.cell_size,
                         'tan', linewidth=1, zorder=1)
