"""Visualize oracle regions and zoom crops for one packaged episode."""
import pickle, sys
import blosc, numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from partzoom.model import PartZoomDiffuserActor

path, out = sys.argv[1], sys.argv[2]
src = int(sys.argv[3]) if len(sys.argv) > 3 else 6
ep = pickle.loads(blosc.decompress(open(path, "rb").read()))
cams = list(ep[3][0].keys())
T = len(ep[0])
fig, axes = plt.subplots(T, 4 + 8, figsize=(2 * 12, 2 * T))
axes = np.atleast_2d(axes)
for t in range(T):
    st = torch.as_tensor(ep[1][t])  # ncam, 2, 3, H, W
    rgb = (st[:, 0] / 2 + 0.5).clamp(0, 1)
    pcd = st[:, 1]
    reg = torch.as_tensor(ep[src][t]).float()
    for c in range(4):
        img = rgb[c].permute(1, 2, 0).numpy().copy()
        p = pcd[c].permute(1, 2, 0)
        for r, col in enumerate([(1, 0, 0), (0, 0, 1)]):
            m = ((p >= reg[r, 0] - 0.01) & (p <= reg[r, 1] + 0.01)).all(-1).numpy()
            img[m] = 0.5 * img[m] + 0.5 * np.array(col)
        axes[t, c].imshow(img); axes[t, c].set_title(f"kf{t} {cams[c]}", fontsize=7)
    # crops through the model's own crop code (identity normalization)
    m = PartZoomDiffuserActor.__new__(PartZoomDiffuserActor)
    torch.nn.Module.__init__(m)
    m.crop_margin, m.min_crop_px, m.crop_size = 0.25, 24, 128
    r = reg.clone(); r[:, 0] -= 0.01; r[:, 1] += 0.01
    m._regions = r[None]
    theta, valid, zoom = m.crop_boxes(pcd[None])
    th = theta.reshape(-1, 2, 3)
    grid = torch.nn.functional.affine_grid(th, (th.shape[0], 3, 128, 128), align_corners=False)
    crops = torch.nn.functional.grid_sample(rgb[None].expand(2, -1, -1, -1, -1).reshape(-1, 3, 256, 256), grid, align_corners=False)
    for k in range(8):
        axes[t, 4 + k].imshow(crops[k].permute(1, 2, 0).clamp(0, 1).numpy())
        axes[t, 4 + k].set_title(f"r{k//4} {cams[k%4][:5]} z{zoom.flatten()[k]:.1f} {'v' if valid.flatten()[k] else 'X'}", fontsize=7)
for a in axes.flat: a.axis("off")
plt.tight_layout(); plt.savefig(out, dpi=70)
print("regions kf0", ep[src][0])
