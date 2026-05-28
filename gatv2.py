import torch
from torch import nn


class GATv2Layer(nn.Module):
    def __init__(
        self,
        dmodel,
        n_heads=8,
        dropout=0.1,
        leaky_relu=0.2,
        num_types=4,
        num_relations=6,
        rel_dim=8,
        gate_bias_init=-2.0,
        gate_tau=0.5,
        gate_strength=0.2,
        gate_min=0.6,
    ):
        super().__init__()
        if dmodel % n_heads != 0:
            raise ValueError("dmodel must be divisible by n_heads")
        self.dmodel = dmodel
        self.n_heads = n_heads
        self.d_k = dmodel // n_heads
        self.num_types = num_types
        self.num_relations = num_relations
        self.gate_bias_init = gate_bias_init
        self.gate_tau = gate_tau
        self.gate_strength = gate_strength
        self.gate_min = gate_min

        self.lin_msg = nn.Linear(dmodel, n_heads * self.d_k, bias=False)
        self.lin_att = nn.Linear(2 * dmodel, n_heads * self.d_k, bias=False)
        self.att_vec = nn.Parameter(torch.empty(n_heads, self.d_k))
        self.out_lin = nn.Linear(n_heads * self.d_k, dmodel, bias=False)
        self.rel_emb = nn.Embedding(num_relations, rel_dim)
        self.gate_mlp = nn.Linear(2 * dmodel + rel_dim, 1, bias=True)
        self.leaky_relu = nn.LeakyReLU(leaky_relu)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("rel_map", self._build_rel_map(num_types))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_msg.weight)
        nn.init.xavier_uniform_(self.lin_att.weight)
        nn.init.xavier_uniform_(self.att_vec)
        nn.init.xavier_uniform_(self.out_lin.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)
        nn.init.xavier_uniform_(self.gate_mlp.weight)
        if self.gate_mlp.bias is not None:
            nn.init.constant_(self.gate_mlp.bias, self.gate_bias_init)

    def _build_rel_map(self, num_types):
        rel_map = torch.full((num_types, num_types), -1, dtype=torch.long)
        # Default type ids: 0=Method, 1=Test, 2=RTest, 3=Line
        if num_types >= 4:
            method, test, rtest, line = 0, 1, 2, 3
            rel_map[method, line] = 0
            rel_map[line, method] = 1
            rel_map[test, line] = 2
            rel_map[line, test] = 3
            rel_map[rtest, line] = 4
            rel_map[line, rtest] = 5
        return rel_map

    def _edge_index_from_adj(self, adj):
        if adj.is_sparse:
            adj = adj.coalesce()
            idx = adj._indices()  # [3, E]
            val = adj._values()
            return idx, val
        idx = adj.nonzero(as_tuple=False).t()
        val = adj[idx[0], idx[1], idx[2]]
        return idx, val

    def _segment_softmax(self, logits, dst_index, num_nodes):
        # logits: [E, H], dst_index: [E]
        E, H = logits.shape
        device = logits.device
        idx = dst_index.view(-1, 1).expand(-1, H)
        max_per = torch.full((num_nodes, H), -1e9, device=device)
        if hasattr(max_per, "scatter_reduce"):
            max_per = max_per.scatter_reduce(0, idx, logits, reduce="amax", include_self=True)
            exp = torch.exp(logits - max_per[dst_index])
            sum_per = torch.zeros((num_nodes, H), device=device)
            sum_per = sum_per.scatter_reduce(0, idx, exp, reduce="sum", include_self=True)
            return exp / (sum_per[dst_index] + 1e-9)

        # Fallback for older torch versions
        max_per = torch.full((num_nodes, H), -1e9, device=device)
        dst_list = dst_index.tolist()
        for i in range(E):
            d = dst_list[i]
            max_per[d] = torch.maximum(max_per[d], logits[i])
        exp = torch.exp(logits - max_per[dst_index])
        sum_per = torch.zeros((num_nodes, H), device=device)
        for i in range(E):
            d = dst_list[i]
            sum_per[d] += exp[i]
        return exp / (sum_per[dst_index] + 1e-9)

    def forward(self, x, adj, node_type=None):
        # x: [B, N, D], adj: [B, N, N] (sparse or dense), node_type: [B, N]
        B, N, _ = x.size()
        device = x.device
        # store attention for visualization (only last forward)
        self.last_attn = []
        # Keep full edge index/weight on CPU to avoid large GPU memory spikes on dense graphs.
        edge_index, edge_weight = self._edge_index_from_adj(adj)
        edge_index = edge_index.cpu()
        edge_weight_cpu = edge_weight.cpu() if edge_weight is not None else None
        rel_map_cpu = self.rel_map.cpu()

        out = torch.zeros_like(x)
        gate_l1_list = []
        for b in range(B):
            xb = x[b]
            msg = self.lin_msg(xb).view(N, self.n_heads, self.d_k)

            if edge_index.numel() == 0:
                out[b] = xb
                gate_l1_list.append(torch.tensor(0.0, device=device))
                continue

            batch_mask = edge_index[0] == b
            if batch_mask.sum() == 0:
                out[b] = xb
                gate_l1_list.append(torch.tensor(0.0, device=device))
                continue

            # adj indices: [batch, row, col] => row is dst, col is src
            dst_all = edge_index[1][batch_mask]
            src_all = edge_index[2][batch_mask]
            ew_all = edge_weight_cpu[batch_mask] if edge_weight_cpu is not None else None

            # Hard edge cap to avoid OOM on very dense graphs (e.g., Time)
            max_edges = 50000
            if src_all.numel() > max_edges:
                keep_idx = torch.arange(max_edges)
                src_all = src_all[:max_edges]
                dst_all = dst_all[:max_edges]
                if ew_all is not None:
                    ew_all = ew_all[:max_edges]

            if dst_all.numel() == 0:
                out[b] = xb
                gate_l1_list.append(torch.tensor(0.0, device=device))
                continue

            if node_type is not None:
                nt_cpu = node_type[b].long().cpu()
                src_t = nt_cpu.index_select(0, src_all)
                dst_t = nt_cpu.index_select(0, dst_all)
                rel_all = rel_map_cpu[src_t, dst_t]
                valid_all = rel_all >= 0
                if valid_all.sum() == 0:
                    out[b] = xb
                    gate_l1_list.append(torch.tensor(0.0, device=device))
                    continue
                src_all = src_all[valid_all]
                dst_all = dst_all[valid_all]
                rel_all = rel_all[valid_all]
                if ew_all is not None:
                    ew_all = ew_all[valid_all]
                if src_all.numel() == 0:
                    out[b] = xb
                    gate_l1_list.append(torch.tensor(0.0, device=device))
                    continue
            else:
                rel_all = None

            # Accumulators on device
            agg = torch.zeros((N, self.n_heads, self.d_k), device=device)
            gate_raw_accum = []
            attn_rec = []

            # Process each relation separately to keep GPU memory low
            rel_list = rel_all.unique().tolist() if rel_all is not None else [None]
            for r in rel_list:
                if rel_all is not None:
                    rel_mask = rel_all == r
                    if rel_mask.sum() == 0:
                        continue
                    src = src_all[rel_mask].to(device)
                    dst = dst_all[rel_mask].to(device)
                    rel_feat = self.rel_emb(torch.full((src.size(0),), r, device=device, dtype=torch.long))
                    ew = ew_all[rel_mask].to(device) if ew_all is not None else None
                else:
                    src = src_all.to(device)
                    dst = dst_all.to(device)
                    rel_feat = torch.zeros((src.size(0), self.rel_emb.embedding_dim), device=device)
                    ew = ew_all.to(device) if ew_all is not None else None

                if src.numel() == 0:
                    continue

                # Chunk edges to control peak memory; for very dense graphs (e.g., Time) use smaller chunks
                num_edges = src.size(0)
                if num_edges > 200000:
                    chunk_size = 100
                elif num_edges > 100000:
                    chunk_size = 250
                else:
                    chunk_size = 1000
                for start in range(0, num_edges, chunk_size):
                    end = min(start + chunk_size, num_edges)
                    src_c = src[start:end]
                    dst_c = dst[start:end]
                    rel_feat_c = rel_feat[start:end]
                    ew_c = ew[start:end] if ew is not None else None

                    h_src = msg.index_select(0, src_c)
                    cat = torch.cat([xb.index_select(0, dst_c), xb.index_select(0, src_c)], dim=-1)
                    att_h = self.lin_att(cat).view(-1, self.n_heads, self.d_k)
                    e = (self.leaky_relu(att_h) * self.att_vec).sum(-1)
                    alpha = self._segment_softmax(e, dst_c, N)
                    if ew_c is not None:
                        alpha = alpha * ew_c.unsqueeze(-1)
                    gate_in = torch.cat([xb.index_select(0, dst_c), xb.index_select(0, src_c), rel_feat_c], dim=-1)
                    gate_raw = torch.sigmoid(self.gate_mlp(gate_in) / self.gate_tau).squeeze(-1)
                    gate = self.gate_min + (1.0 - self.gate_min) * gate_raw
                    alpha = alpha * (1.0 - self.gate_strength + self.gate_strength * gate).unsqueeze(-1)
                    alpha = self.dropout(alpha)

                    agg.index_add_(0, dst_c, h_src * alpha.unsqueeze(-1))
                    gate_raw_accum.append(gate_raw.mean())
                    # record attention (average over heads)
                    attn_rec.append(
                        (
                            dst_c.detach().cpu(),
                            src_c.detach().cpu(),
                            alpha.mean(dim=1).detach().cpu(),  # [edges]
                        )
                    )

            agg = agg.reshape(N, self.n_heads * self.d_k)
            out[b] = self.out_lin(agg)
            if gate_raw_accum:
                gate_l1_list.append(torch.stack(gate_raw_accum).mean())
            else:
                gate_l1_list.append(torch.tensor(0.0, device=device))
            # save per-batch attention
            self.last_attn.append(
                {
                    "N": N,
                    "records": attn_rec,
                }
            )
        gate_l1 = torch.stack(gate_l1_list, dim=0)
        return out, gate_l1
