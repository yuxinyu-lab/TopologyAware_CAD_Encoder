import os
import csv
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

LATENT_DIM = 64
NODE_DIM = 128
N_HEADS = 4
N_LAYERS = 6
MAX_FACES = 256
FACE_SURF_TYPES = 7

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

class BRepDataset(Dataset):

    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def random_aug(self, data):

        x = dict(data)

        uv = x["faces_uv"].copy()

        noise = np.random.normal(
            0,
            0.01,
            size=uv[..., :6].shape
        )

        uv[..., :6] += noise.astype(np.float32)

        x["faces_uv"] = uv

        return x

    def __getitem__(self, idx):

        data = load_npz(self.files[idx])

        if data["n_faces"] > MAX_FACES:

            return self.__getitem__(
                random.randint(
                    0,
                    len(self.files)-1
                )
            )

        view1 = self.random_aug(data)
        view2 = self.random_aug(data)

        return {
            "view1": view1,
            "view2": view2,
        }

def collate_fn(batch):
    return batch

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

        # FP16 SAFE FIX
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

        return self.pool(x)

    def forward(self, batch, device):

        zs = []

        for item in batch:
            zs.append(
                self.encode_single(item, device)
            )

        return torch.stack(zs)

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()

    def forward(self, batch, device):

        v1 = [x["view1"] for x in batch]
        v2 = [x["view2"] for x in batch]

        z1 = self.encoder(v1, device)
        z2 = self.encoder(v2, device)

        return z1, z2

def contrastive_loss(z1, z2, temp=0.1):

    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    logits = z1 @ z2.T
    logits = logits / temp

    labels = torch.arange(
        z1.shape[0],
        device=z1.device
    )

    return 0.5 * (
        F.cross_entropy(logits, labels) +
        F.cross_entropy(logits.T, labels)
    )

def train(model, loader, device, epochs, lr):

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5
    )

    scaler = torch.cuda.amp.GradScaler()

    model.train()

    for epoch in range(epochs):

        total = 0.0

        for batch in loader:

            opt.zero_grad()

            with torch.cuda.amp.autocast():

                z1, z2 = model(batch, device)

                loss = contrastive_loss(z1, z2)

            scaler.scale(loss).backward()

            scaler.unscale_(opt)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            scaler.step(opt)
            scaler.update()

            total += float(loss.detach().cpu())

        avg = total / max(1, len(loader))

        print(
            f"Epoch {epoch+1:03d} "
            f"| loss={avg:.6f}"
        )

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--out_dir", type=str)

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    files = collect_npz_files(args.data_dir)

    dataset = BRepDataset(files)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = Model().to(device)

    print(f"files: {len(files)}")
    print(f"device: {device}")
    print(f"MAX_FACES: {MAX_FACES}")
    print("FP16 SAFE GRAPH ATTENTION ENABLED")

    train(
        model,
        loader,
        device,
        args.epochs,
        args.lr
    )

    os.makedirs(args.out_dir, exist_ok=True)

    torch.save(
        model.state_dict(),
        os.path.join(
            args.out_dir,
            "cad_encoder.pth"
        )
    )

    print("\\n完成")

if __name__ == "__main__":
    main()
