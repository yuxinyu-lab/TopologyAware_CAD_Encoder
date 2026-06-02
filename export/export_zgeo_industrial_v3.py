
"""
export_zgeo_industrial_v3.py

用途：
读取 industrial_cad_encoder_v3_fp16fix.pth
导出：
zgeo_industrial_v3.csv

不需要重新训练。
"""

import os
import csv
import math
import argparse
import numpy as np

import torch
import torch.nn as nn

LATENT_DIM = 64
NODE_DIM = 128
N_LAYERS = 6
MAX_FACES = 256
FACE_SURF_TYPES = 7

# ============================================================
# UTILS
# ============================================================

def sanitize(x):
    x = np.asarray(x, dtype=np.float32)
    return np.nan_to_num(x)

def collect_npz_files(data_dir):
    return sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".npz")
    ])

def load_npz(path):

    d = np.load(path, allow_pickle=False)

    edge_to_faces = d["edge_to_faces"]

    if edge_to_faces.size == 0:
        edge_to_faces = np.zeros((0,2), dtype=np.int64)

    return {
        "file": os.path.basename(path),
        "faces_uv": sanitize(d["faces_uv"]),
        "faces_type": sanitize(d["faces_type"]),
        "edge_to_faces": edge_to_faces.astype(np.int64),
        "n_faces": int(d["faces_uv"].shape[0]),
    }

# ============================================================
# FACE CNN
# ============================================================

class FaceCNN(nn.Module):

    def __init__(self):
        super().__init__()

        in_ch = 7 + FACE_SURF_TYPES

        self.net = nn.Sequential(

            nn.Conv2d(in_ch,64,3,padding=1),
            nn.GroupNorm(8,64),
            nn.GELU(),

            nn.Conv2d(64,128,3,padding=1),
            nn.GroupNorm(8,128),
            nn.GELU(),

            nn.Conv2d(128,256,3,padding=1),
            nn.GroupNorm(16,256),
            nn.GELU(),

            nn.AdaptiveAvgPool2d(1),
        )

        self.proj = nn.Sequential(
            nn.Linear(256, NODE_DIM),
            nn.LayerNorm(NODE_DIM),
        )

    def forward(self, uv, surf):

        n_faces, _, n, _ = uv.shape

        surf = surf.unsqueeze(-1).unsqueeze(-1)
        surf = surf.expand(n_faces,-1,n,n)

        x = torch.cat([uv, surf], dim=1)

        x = self.net(x).flatten(1)

        return self.proj(x)

# ============================================================
# GRAPH ATTENTION
# ============================================================

class GraphAttentionBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.q = nn.Linear(NODE_DIM, NODE_DIM)
        self.k = nn.Linear(NODE_DIM, NODE_DIM)
        self.v = nn.Linear(NODE_DIM, NODE_DIM)

        self.out = nn.Linear(NODE_DIM, NODE_DIM)

        self.ffn = nn.Sequential(
            nn.Linear(NODE_DIM, NODE_DIM*4),
            nn.GELU(),
            nn.Linear(NODE_DIM*4, NODE_DIM),
        )

        self.norm1 = nn.LayerNorm(NODE_DIM)
        self.norm2 = nn.LayerNorm(NODE_DIM)

    def forward(self, x, adj):

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        scores = q @ k.T
        scores = scores / math.sqrt(q.shape[-1])

        mask = (adj == 0)

        scores = scores.masked_fill(
            mask,
            -1e4
        )

        attn = torch.softmax(scores, dim=-1)

        h = attn @ v

        x = self.norm1(
            x + self.out(h)
        )

        x = self.norm2(
            x + self.ffn(x)
        )

        return x

# ============================================================
# HIERARCHICAL POOLING
# ============================================================

class HierarchicalPooling(nn.Module):

    def __init__(self):
        super().__init__()

        self.local_score = nn.Sequential(
            nn.Linear(NODE_DIM, NODE_DIM),
            nn.GELU(),
            nn.Linear(NODE_DIM,1)
        )

        self.out = nn.Sequential(
            nn.LayerNorm(NODE_DIM*2),
            nn.Linear(NODE_DIM*2, LATENT_DIM),
        )

    def forward(self, x):

        local_attn = torch.softmax(
            self.local_score(x),
            dim=0
        )

        local_feat = (x * local_attn).sum(dim=0)

        global_feat = x.mean(dim=0)

        feat = torch.cat([
            local_feat,
            global_feat
        ], dim=-1)

        return self.out(feat)

# ============================================================
# ENCODER
# ============================================================

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.face_cnn = FaceCNN()

        self.blocks = nn.ModuleList([
            GraphAttentionBlock()
            for _ in range(N_LAYERS)
        ])

        self.pool = HierarchicalPooling()

    def build_adj(self, edge_to_faces, n_faces, device):

        adj = torch.eye(
            n_faces,
            dtype=torch.float32,
            device=device
        )

        if len(edge_to_faces) > 0:

            etf = torch.tensor(
                edge_to_faces,
                dtype=torch.long,
                device=device
            )

            i = etf[:,0]
            j = etf[:,1]

            adj[i,j] = 1
            adj[j,i] = 1

        return adj

    def encode_single(self, data, device):

        uv = torch.tensor(
            data["faces_uv"].transpose(0,3,1,2),
            dtype=torch.float32,
            device=device
        )

        surf = torch.tensor(
            data["faces_type"],
            dtype=torch.float32,
            device=device
        )

        x = self.face_cnn(uv, surf)

        adj = self.build_adj(
            data["edge_to_faces"],
            data["n_faces"],
            device
        )

        for blk in self.blocks:
            x = blk(x, adj)

        z = self.pool(x)

        return z

# ============================================================
# EXPORT
# ============================================================

@torch.no_grad()
def export_zgeo(model, files, device, out_csv):

    model.eval()

    rows = []

    for idx, p in enumerate(files):

        d = load_npz(p)

        if d["n_faces"] > MAX_FACES:
            continue

        z = model.encode_single(d, device)

        z = z.detach().cpu().numpy()

        row = {
            "file": d["file"],
            "n_faces": d["n_faces"]
        }

        for i,v in enumerate(z):
            row[f"d{i}"] = float(v)

        rows.append(row)

        if idx % 50 == 0:
            print(f"{idx}/{len(files)}")

    fields = (
        ["file","n_faces"] +
        [f"d{i}" for i in range(LATENT_DIM)]
    )

    with open(out_csv,"w",newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nsaved: {out_csv}")

# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    files = collect_npz_files(args.data_dir)

    model = Encoder().to(device)

    print("loading model...")

    state = torch.load(
        args.model_path,
        map_location=device
    )

    model.load_state_dict(state, strict=False)

    print("exporting zgeo...")

    export_zgeo(
        model,
        files,
        device,
        args.out_csv
    )

    print("\n完成")

if __name__ == "__main__":
    main()
