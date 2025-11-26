import torch
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from planner.my_flash_att.flash_model import create_model
from typing import Dict, Tuple
from imputation.history_avg_at_time_interval import get_travel_time_tensor, get_real_travel_time_tensor


def build_seg_table(segment_mapping, num_nodes):
    seg_table = torch.full((num_nodes, num_nodes), -1, dtype=torch.long)
    for (u, v), seg_id in segment_mapping.items():
        seg_table[u, v] = seg_id
    return seg_table

class TransPlanner(nn.Module):
    def __init__(
            self,
            G: nx.Graph,
            A: torch.Tensor,
            device: torch.device,
            x_emb_dim: int,
            segment_mapping: Dict[Tuple[int, int], int],
            time_dist_seg: np.ndarray,
            pretrain_path=None
    ):
        super().__init__()
        self.lambda_tt = 0.3
        self.segment_mapping = build_seg_table(segment_mapping, A.shape[0]).to(device)
        # self.time_dist_seg = time_dist_seg
        self.time_dist_seg = torch.from_numpy(time_dist_seg).float().to(device)
        self.device = device
        # find max degree and build mask
        self.max_deg = int(A.long().sum(1).max())




        # 结点数量
        self.n_vertex = A.shape[0]
        self.mask = torch.zeros(self.n_vertex, self.max_deg + 1).long().to(self.device)
        self.mask[torch.arange(self.n_vertex), A.sum(1, keepdim=False).long()] = 1
        self.mask.cumsum_(dim=-1)
        self.mask = self.mask[:, :-1].bool()

        self.locations = torch.zeros([self.n_vertex, 2]).to(self.device)
        for k in range(self.n_vertex):
            self.locations[k, 0], self.locations[k, 1] = G.nodes[k]["lng"], G.nodes[k]["lat"]

        self.v_to_ord = dict()  # v : dict from v to ord
        self.ord_to_v = dict()  # v : [] list of vertices
        val, ind = A.long().topk(self.max_deg, dim=1)
        for i in range(self.n_vertex):
            valid_ind = ind[i][val[i] == 1].cpu().tolist()
            self.v_to_ord[i] = dict(zip(valid_ind, list(range(len(valid_ind)))))
            self.ord_to_v[i] = valid_ind

        # two vertex abs direction
        # (xb - xa)/ \sqrt{(y_b - y_a)^2 + (xb - xa)^2}, (yb - ya)/ \sqrt{(y_b - y_a)^2 + (xb - xa)^2}
        self.tv_dir = torch.zeros([self.n_vertex, self.n_vertex, 2]).to(self.device)
        self.adj_dir = torch.zeros(self.n_vertex, self.max_deg, 2).to(self.device)
        for k in range(self.n_vertex):
            xb_m_xa = self.locations[:, 0] - self.locations[k, 0]
            yb_m_ya = self.locations[:, 1] - self.locations[k, 1]
            denom = (xb_m_xa.square() + yb_m_ya.square()).sqrt()
            self.tv_dir[k, denom > 0, 0], self.tv_dir[k, denom > 0, 1] = (xb_m_xa / denom)[denom > 0], \
                (yb_m_ya / denom)[denom > 0]
            # each vertex adjacent direction
            self.adj_dir[k, torch.arange(len(self.ord_to_v[k])), :] = self.tv_dir[k, self.ord_to_v[k], :]
        self.flash_att = create_model().to(device)

        distance_dim = 50
        direction_dim = 50
        travel_time_dim = 50
        hidden_dim = 100
        self.distance_mlp = nn.Linear(1, distance_dim).to(self.device)
        self.direction_mlp = nn.Linear(2, direction_dim).to(self.device)
        self.travel_time_mlp = nn.Linear(1, travel_time_dim).to(self.device)
        self.out_bias_mlp = nn.Sequential(
            nn.Linear(travel_time_dim, travel_time_dim // 2),
            nn.ReLU(),
            nn.Linear(travel_time_dim // 2, 1)
        ).to(self.device)

        self.out_mlp = nn.Sequential(
           # hidden from gpt, distance, direction, destination
           nn.Linear(x_emb_dim  + direction_dim + distance_dim + x_emb_dim, hidden_dim).to(self.device),
           nn.ReLU(),
           nn.Linear(hidden_dim, int(0.5 * hidden_dim)).to(self.device),
           nn.ReLU(),
           nn.Linear(int(0.5 * hidden_dim), 1).to(self.device)
        )

        self.x_embedding = nn.Embedding(self.n_vertex + 2, x_emb_dim, padding_idx=self.n_vertex, device=device).to(
            self.device)
        torch.nn.init.normal_(self.x_embedding.weight, mean=0.0, std=0.01)

    def forward(self, xs, start_time):
        lengths = [x.shape[0] for x in xs]
        xs_padded = pad_sequence(xs, batch_first=True, padding_value=0).to(self.device)
        xs_actions = [
            torch.Tensor([self.v_to_ord[a.item()][b.item()] for a, b in zip(x, x[1:])]).long().to(self.device)
            for x in xs]
        batch_size, horizon = xs_padded.shape
        xs_padded_emb = self.x_embedding(xs_padded)

        # get neighbors of route
        route_neighbors = torch.zeros(batch_size, horizon, self.max_deg, dtype=torch.long, device=self.device)
        for b in range(batch_size):
            for t in range(lengths[b] - 1):
                v_t = xs_padded[b, t].item()
                if v_t in self.ord_to_v:
                    neighbor_ids = self.ord_to_v[v_t]
                    for j, u in enumerate(neighbor_ids):
                        route_neighbors[b, t, j] = u

        # destination embedding (batch_size, max_len, max_deg, x_emb_dim)
        destinations = torch.Tensor([x[-1] for x in xs]).long().to(self.device)
        dest_emb = self.x_embedding(destinations)
        dest_emb_expand = dest_emb[:, None, None, :].repeat(1, horizon, self.max_deg, 1)

        # prefix embedding (batch_size, max_len, max_deg, x_emb_dim)
        attention_mask = torch.zeros_like(xs_padded,dtype=torch.bool)
        for k in range(batch_size):
            attention_mask[k, lengths[k]:] = True
        attn_mask = torch.triu(torch.ones(horizon,horizon), diagonal=1).bool().to(self.device)
        hidden_attention = self.flash_att(xs_padded_emb,attention_mask,attn_mask)
        hidden_attention_expand = hidden_attention[:, :, None, :].repeat(1, 1, self.max_deg, 1)

        # distances embedding (batch_size, max_len, max_deg, distance_dim)
        dest_coords = self.locations[destinations].unsqueeze(1).unsqueeze(2)
        neighbor_coords = self.locations[route_neighbors]
        distances = (neighbor_coords - dest_coords).abs().sum(dim=-1, keepdim=True) * 100
        distances_feature = self.distance_mlp(distances)

        # directions embedding (batch_size, max_len, max_deg, direction_dim)
        current_coords = self.locations[xs_padded]
        current_coords = current_coords.unsqueeze(2)
        neighbor_coords = self.locations[route_neighbors]
        vec = neighbor_coords - current_coords
        norm = vec.norm(dim=-1, keepdim=True) + 1e-8
        unit_vec = vec / norm
        directions_feature = self.direction_mlp(unit_vec)

        # neighbor traffic embedding (batch_size, max_len, max_deg, travel_time_dim)
        neighbor_mask = (route_neighbors != 0)
        neighbor_travel_time = get_real_travel_time_tensor(
            route_neighbors,
            xs_padded,
            start_time,
            self.segment_mapping,
            self.time_dist_seg
        ).to(self.device)
        valid_tt = neighbor_travel_time[neighbor_mask]
        mean = valid_tt.mean()
        std = valid_tt.std() + 1e-6
        neighbor_travel_time = (neighbor_travel_time - mean) / std
        travel_time_feature = self.travel_time_mlp(neighbor_travel_time.unsqueeze(-1))
        travel_time_feature = travel_time_feature * neighbor_mask.unsqueeze(-1).float()

        feed = torch.concat(
            [
                hidden_attention_expand,
                distances_feature,
                directions_feature,
                dest_emb_expand,
            ],
            dim=-1
        )

        out_bias = self.out_bias_mlp(travel_time_feature).squeeze(-1)
        out_logits = self.out_mlp(feed).squeeze(-1) + out_bias
        #out_logits = self.out_mlp(feed).squeeze(-1)

        loss = sum([
            F.cross_entropy(out_logits[k][:lengths[k] - 1], xs_actions[k], reduction="mean")
            for k in range(batch_size)
        ])

        return loss

    def plan(self, origs, dests, start_time, eval_nll=False):
        with torch.no_grad():
            if isinstance(origs, list):
                origs = torch.tensor(origs).long().to(self.device)
            if isinstance(dests, list):
                dests = torch.tensor(dests).long().to(self.device)

            batch_size = origs.size(0)
            max_len = 50
            xs = torch.zeros([batch_size, max_len], dtype=torch.long).to(self.device)
            xs[:, 0] = origs
            stop = torch.zeros(batch_size, dtype=torch.bool).to(self.device)
            actual_length = torch.ones(batch_size, dtype=torch.long).to(self.device) * max_len

            if eval_nll:
                nlls = torch.zeros(batch_size).to(self.device)

            for i in range(1, max_len):
                prefix = xs[:, :i]
                lengths = [prefix.shape[1]] * batch_size
                last_nodes = prefix[:, -1]  # (B,)

                # get neighbor
                route_neighbors = torch.zeros(batch_size, 1, self.max_deg, dtype=torch.long, device=self.device)
                for b in range(batch_size):
                    v_t = last_nodes[b].item()
                    if v_t in self.ord_to_v:
                        neighbors = self.ord_to_v[v_t]
                        for j, u in enumerate(neighbors):
                            route_neighbors[b, 0, j] = u

                # prefix
                prefix_emb = self.x_embedding(prefix)
                attn_mask = torch.triu(torch.ones(prefix.shape[1], prefix.shape[1]), diagonal=1).bool().to(self.device)
                attention_mask = torch.zeros_like(prefix, dtype=torch.bool)
                hidden = self.flash_att(prefix_emb, attention_mask, attn_mask)
                hidden = hidden[:, -1, :]
                hidden_expand = hidden[:, None, :].repeat(1, self.max_deg, 1)

                # destination
                dests_emb = self.x_embedding(dests)
                dests_expand = dests_emb[:, None, :].repeat(1, self.max_deg, 1)

                # distance
                dest_coords = self.locations[dests].unsqueeze(1)
                neighbor_coords = self.locations[route_neighbors.squeeze(1)]
                distances = (neighbor_coords - dest_coords).abs().sum(dim=-1, keepdim=True) * 100
                distances_feature = self.distance_mlp(distances)

                # direction
                curr_coords = self.locations[last_nodes].unsqueeze(1)
                vec = neighbor_coords - curr_coords
                norm = vec.norm(dim=-1, keepdim=True) + 1e-8
                unit_vec = vec / norm
                directions_feature = self.direction_mlp(unit_vec)

                pad_prefix = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
                pad_prefix[:, :i] = prefix  # 保留已走过的轨迹，其余补 0

                # travel time
                travel_time = get_real_travel_time_tensor(
                    route_neighbors,
                    pad_prefix,
                    start_time,
                    self.segment_mapping,
                    self.time_dist_seg
                ).squeeze(1)  # (B, D)

                neighbor_mask = (route_neighbors.squeeze(1) != 0)  # (B, D)
                valid_tt = travel_time[neighbor_mask]
                mean = valid_tt.mean()
                std = valid_tt.std() + 1e-6
                travel_time = (travel_time - mean) / std  # (B, D)

                travel_time_feature = self.travel_time_mlp(travel_time.unsqueeze(-1))  # (B, D, time_dim)
                travel_time_feature = travel_time_feature * neighbor_mask.unsqueeze(-1).float()

                # 拼接所有特征
                feed = torch.cat([
                    hidden_expand,
                    distances_feature,
                    directions_feature,
                    dests_expand
                ], dim=-1)  # (B, D, feat_dim)

                logits = self.out_mlp(feed).squeeze(-1) + self.out_bias_mlp(travel_time_feature).squeeze(-1)
                #logits = self.out_mlp(feed).squeeze(-1)

                # 蒙版非法邻居
                mask = torch.zeros_like(logits).bool()
                for b in range(batch_size):
                    v_t = last_nodes[b].item()
                    invalid = self.mask[v_t]
                    mask[b, invalid[:self.max_deg]] = True
                logits = torch.masked_fill(logits, mask, -1e20)

                probs = torch.softmax(logits, dim=-1)
                actions = torch.argmax(probs, dim=-1)  # shape: (B,)

                # 用预测的邻居填入下一位置
                next_nodes = torch.tensor([
                    self.ord_to_v[last_nodes[k].item()][actions[k].item()]
                    for k in range(batch_size)
                ], dtype=torch.long).to(self.device)
                xs[:, i] = next_nodes

                if eval_nll:
                    nlls[~stop] -= (probs[~stop, actions[~stop]] + 1e-8).log()

                actual_length[xs[:, i] == dests] = i + 1
                stop = stop | (xs[:, i] == dests)
                if stop.all():
                    break

            xs_list = [xs[k, :actual_length[k]].cpu().tolist() for k in range(batch_size)]
            xs_list_refined = self.refine(xs_list, dests.cpu().tolist())

            if eval_nll:
                return xs_list_refined, nlls
            return xs_list_refined

    def refine(self, paths, dests):
        # two things: 1) cut the recursive
        refined_paths = []
        for k, path in enumerate(paths):
            # 1) if one step close, directly cut
            destination = dests[k]
            for i, v in enumerate(path):
                if destination in self.v_to_ord[v]:
                    cutted_path = path[:i] + [destination]
                    break
            else:
                cutted_path = path
            # 2) cut the recursive
            showup = set()
            points_filtered = []
            for _, v in enumerate(cutted_path):
                if v not in showup:
                    showup.add(v)
                    points_filtered.append(v)
                else:
                    while points_filtered[-1] != v:
                        showup.discard(points_filtered[-1])
                        points_filtered.pop()
            refined_paths.append(points_filtered)
        return refined_paths
