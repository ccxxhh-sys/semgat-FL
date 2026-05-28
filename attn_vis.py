import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import torch

from Dataset import SumDataset
from Model import NlEncoder


def build_args(proj):
    # minimal args loader (mirrors run.py defaults)
    NlLen_map = {"Time": 3900, "Math": 3000, "Lang": 350, "Chart": 2350, "Mockito": 1370, "Codec": 160, "Compress": 1000, "Gson": 1500, "Cli": 1000, "Jsoup": 2000, "Csv": 1000, "JacksonCore": 1000, "JacksonXml": 500, "Collections": 500}
    CodeLen_map = {"Time": 1300, "Math": 1000, "Lang": 350, "Chart": 5250, "Mockito": 280, "Codec": 190, "Compress": 1500, "Gson": 1500, "Cli": 1000, "Jsoup": 2000, "Csv": 1000, "JacksonCore": 1000, "JacksonXml": 500, "Collections": 500}
    class dotdict(dict):
        __getattr__ = dict.get
    args = dotdict(
        {
            "NlLen": NlLen_map[proj],
            "CodeLen": CodeLen_map[proj],
            "SentenceLen": 10,
            "batch_size": 1,
            "embedding_size": 32,
            "WoLen": 15,
            "Vocsize": 100,
            "Nl_Vocsize": 100,
            "max_step": 3,
            "margin": 0.5,
            "poolsize": 50,
            "Code_Vocsize": 100,
            "seed": 0,
            "lr": 1e-3,
            "UseCompact": False,
            "CompactMapPath": "",
            "CompactEmbPath": "",
            "EdgeGateL1": 0.0,
            "EdgeGateWarmup": 0,
            "UseGCNNBaseline": False,
        }
    )
    return args


def load_sample(args, proj, split="test", idx=0):
    ds = SumDataset(args, split, proj, testid=idx)
    sample = [torch.from_numpy(x).unsqueeze(0) for x in ds.data]  # batch dim=1
    return sample


def run_once(args, proj, idx, checkpoint=None, device="cuda"):
    sample = load_sample(args, proj, "test", idx)
    args.Code_Vocsize = len(sample[4][0]) if hasattr(args, "Code_Vocsize") else args.Code_Vocsize
    model = NlEncoder(args)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state)
    model.eval()
    model.to(device)
    sample = [x.to(device) for x in sample]
    with torch.no_grad():
        _, _, _ = model(*sample)
    # attention from first block
    attn = model.transformerBlocks[0].Tconv_forward.last_attn[0]
    N = attn["N"]
    mat = torch.zeros((N, N))
    for dst, src, alpha in attn["records"]:
        mat[dst, src] = alpha
    return mat


def plot_heat(mat, out_path):
    plt.figure(figsize=(6, 6))
    plt.imshow(mat, cmap="viridis", vmin=0, vmax=float(mat.max().item()) if mat.max() > 0 else 0.6)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    p = argparse.ArgumentParser(description="Export attention heatmap for one sample.")
    p.add_argument("--proj", default="Lang")
    p.add_argument("--idx", type=int, default=0, help="test sample id")
    p.add_argument("--ckpt", default=None, help="path to model checkpoint (optional)")
    p.add_argument("--out", default="attn_heat.png")
    p.add_argument("--cpu", action="store_true")
    args_cli = p.parse_args()

    args = build_args(args_cli.proj)
    device = "cpu" if args_cli.cpu or not torch.cuda.is_available() else "cuda"
    mat = run_once(args, args_cli.proj, args_cli.idx, args_cli.ckpt, device=device)
    plot_heat(mat.cpu(), args_cli.out)
    print(f"saved {args_cli.out}, max attn={mat.max().item():.4f}")


if __name__ == "__main__":
    main()
