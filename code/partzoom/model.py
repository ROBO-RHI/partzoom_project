"""Part-guided visual token selection and part zoom for 3D Diffuser Actor.

The diffusion head, its losses and the full-context cross-attention are
unchanged. The only thing that changes is which visual tokens fill the fixed
budget S = N / fps_subsampling_factor that the trajectory tokens attend to in
the relative self-attention layers (819 of 4096 tokens in the PerAct setup).

token_mode
  'fps'  : upstream behaviour, farthest point sampling in feature space.
  'part' : part-aware sampling. Coarse tokens whose 3D position falls in an
           oracle part region are kept first (up to rel_cap * budget, by FPS
           within the region if more), the rest of the budget is filled by FPS
           over the remaining scene tokens (global context).
zoom
  If True, K_f of the S budget slots are replaced by fine tokens. For each part
  region and each camera the RGB and aligned point cloud are cropped around the
  region's pixels, resized to crop_size, re-encoded with the same backbone +
  FPN, and the resulting tokens keep their own 3D positions (same world frame).
  Fine tokens inside the region are preferred; others fill remaining slots.
  The coarse budget is reduced to S - K_f, so the total stays S.
"""
import dgl.geometry as dgl_geo
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffuser_actor import DiffuserActor


def _fps_subset(feats, subset_mask, k):
    """Batched FPS restricted to a subset, by feature distance.

    feats: (B, N, C); subset_mask: (B, N) bool; returns (B, k) indices into N,
    ordered by FPS pick order. For sample b only the first
    min(k, subset_mask[b].sum()) picks are guaranteed to be in the subset.
    """
    B, N, C = feats.shape
    # put subset members first so start index 0 is a member
    order = torch.argsort((~subset_mask).float(), dim=1, stable=True)
    f = torch.gather(feats, 1, order[..., None].expand(-1, -1, C))
    m = torch.gather(subset_mask, 1, order)
    # non-members become copies of the first point: distance 0 to the
    # selected set after the first pick, so FPS never prefers them
    f = torch.where(m[..., None], f, f[:, :1].expand(-1, N, -1))
    picks = dgl_geo.farthest_point_sampler(f.double(), k, 0).long()
    return torch.gather(order, 1, picks)


def _merge(first, n_first, second, total):
    """Per sample, take first[:n_first[b]] then second[:total - n_first[b]]."""
    j = torch.arange(total, device=first.device)[None]
    use_first = j < n_first[:, None]
    i1 = j.clamp(max=first.shape[1] - 1).expand(first.shape[0], -1)
    i2 = (j - n_first[:, None]).clamp(min=0, max=second.shape[1] - 1)
    return torch.where(use_first, torch.gather(first, 1, i1),
                       torch.gather(second, 1, i2))


def _in_boxes(pts, boxes):
    """pts (B, N, 3), boxes (B, R, 2, 3) -> (B, N) bool, inside any box."""
    lo = boxes[:, None, :, 0]
    hi = boxes[:, None, :, 1]
    p = pts[:, :, None]
    return ((p >= lo) & (p <= hi)).all(-1).any(-1)


class PartZoomDiffuserActor(DiffuserActor):

    def __init__(self, *args, token_mode='fps', zoom=False, zoom_tokens=256,
                 rel_cap=0.5, crop_size=128, crop_margin=0.25,
                 min_crop_px=24, region_pad=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        assert token_mode in ('fps', 'part')
        self.token_mode = token_mode
        self.zoom = zoom
        self.zoom_tokens = zoom_tokens
        self.rel_cap = rel_cap
        self.crop_size = crop_size
        self.crop_margin = crop_margin
        self.min_crop_px = min_crop_px
        self.region_pad = region_pad
        C = self.encoder.instruction_encoder.out_features
        # zero-init level embedding so fine tokens start as plain FPN features
        self.fine_level_embed = nn.Parameter(torch.zeros(C))
        self._regions = None
        self._rgb_full = None
        self.last_stats = {}

    # ------------------------------------------------------------------ API
    def forward(self, gt_trajectory, trajectory_mask, rgb_obs, pcd_obs,
                instruction, curr_gripper, run_inference=False, regions=None):
        """regions: (B, R, 2, 3) world-frame AABBs (min, max), or None."""
        assert not self._relative
        if regions is not None:
            r = regions.clone().float()
            r[:, :, 0] -= self.region_pad
            r[:, :, 1] += self.region_pad
            self._regions = self.normalize_pos(r)
        else:
            self._regions = None
            assert self.token_mode == 'fps' and not self.zoom
        return super().forward(gt_trajectory, trajectory_mask, rgb_obs,
                               pcd_obs, instruction, curr_gripper,
                               run_inference=run_inference)

    # ------------------------------------------------------------- encoding
    def encode_inputs(self, visible_rgb, visible_pcd, instruction,
                      curr_gripper):
        enc = self.encoder
        rgb_feats_pyramid, pcd_pyramid = enc.encode_images(
            visible_rgb, visible_pcd)
        context_feats = einops.rearrange(
            rgb_feats_pyramid[0], "b ncam c h w -> b (ncam h w) c")
        context = pcd_pyramid[0]

        instr_feats = None
        if self.use_instruction:
            instr_feats, _ = enc.encode_instruction(instruction)
            context_feats = enc.vision_language_attention(
                context_feats, instr_feats)

        adaln_gripper_feats, _ = enc.encode_curr_gripper(
            curr_gripper, context_feats, context)

        B, N, C = context_feats.shape
        S = max(N // enc.fps_subsampling_factor, 1)
        K_f = self.zoom_tokens if self.zoom else 0
        S_c = S - K_f

        if self._regions is not None:
            rel = _in_boxes(context, self._regions)  # (B, N)
        else:
            rel = torch.zeros(B, N, dtype=torch.bool, device=context.device)
        all_mask = torch.ones_like(rel)

        if self.token_mode == 'fps':
            idx_c = _fps_subset(context_feats, all_mask, S_c)
        else:
            cap = max(int(self.rel_cap * S_c), 1)
            n_rel = rel.sum(1).clamp(max=cap)
            idx_rel = _fps_subset(context_feats, rel, cap)
            idx_rest = _fps_subset(context_feats, ~rel, S_c)
            idx_c = _merge(idx_rel, n_rel, idx_rest, S_c)

        sel_feats = torch.gather(context_feats, 1,
                                 idx_c[..., None].expand(-1, -1, C))
        sel_pos = torch.gather(context, 1, idx_c[..., None].expand(-1, -1, 3))

        # stats: how many in-region coarse tokens survive selection
        with torch.no_grad():
            n_rel_all = rel.sum(1).float()
            kept_rel = torch.gather(rel, 1, idx_c).sum(1).float()
            self.last_stats = {
                'coarse_rel_tokens': n_rel_all.mean().item(),
                'coarse_rel_kept': kept_rel.mean().item(),
                'coarse_rel_retention': (
                    kept_rel / n_rel_all.clamp(min=1)).mean().item(),
                'frac_budget_rel': (kept_rel / S).mean().item(),
            }

        if self.zoom:
            f_feats, f_pos, crop_info = self.encode_crops(
                visible_rgb, visible_pcd, instr_feats)
            f_rel = _in_boxes(f_pos, self._regions)
            n_frel = f_rel.sum(1).clamp(max=K_f)
            idx_frel = _fps_subset(f_feats, f_rel, K_f)
            idx_fall = _fps_subset(f_feats, ~f_rel, K_f)
            idx_f = _merge(idx_frel, n_frel, idx_fall, K_f)
            Cf = f_feats.shape[-1]
            sel_feats = torch.cat([sel_feats, torch.gather(
                f_feats, 1, idx_f[..., None].expand(-1, -1, Cf))], 1)
            sel_pos = torch.cat([sel_pos, torch.gather(
                f_pos, 1, idx_f[..., None].expand(-1, -1, 3))], 1)
            with torch.no_grad():
                self.last_stats.update({
                    'fine_rel_tokens': f_rel.sum(1).float().mean().item(),
                    'fine_rel_kept': n_frel.float().mean().item(),
                    'crop_valid_frac': crop_info['valid'].float().mean().item(),
                    'crop_zoom': crop_info['zoom'].mean().item(),
                })

        fps_feats = einops.rearrange(sel_feats, "b n c -> n b c")
        fps_pos = enc.relative_pe_layer(sel_pos)
        return (context_feats, context, instr_feats, adaln_gripper_feats,
                fps_feats, fps_pos)

    def crop_boxes(self, pcd):
        """Square pixel crops per (region, camera) from the in-region pixels.

        pcd: (B, ncam, 3, H, W) normalized. Returns theta (B, R, ncam, 2, 3)
        for affine_grid, valid (B, R, ncam) and zoom factor.
        """
        B, ncam, _, H, W = pcd.shape
        R = self._regions.shape[1]
        p = einops.rearrange(pcd, "b n c h w -> b 1 n h w c")
        lo = self._regions[:, :, 0][:, :, None, None, None]
        hi = self._regions[:, :, 1][:, :, None, None, None]
        mask = ((p >= lo) & (p <= hi)).all(-1)  # (B, R, ncam, H, W)
        count = mask.flatten(-2).sum(-1)
        rows = mask.any(-1).float()  # (B,R,ncam,H)
        cols = mask.any(-2).float()  # (B,R,ncam,W)
        y0 = rows.argmax(-1).float()
        y1 = (H - 1 - rows.flip(-1).argmax(-1)).float()
        x0 = cols.argmax(-1).float()
        x1 = (W - 1 - cols.flip(-1).argmax(-1)).float()
        valid = count >= 4
        side = torch.maximum(x1 - x0 + 1, y1 - y0 + 1) * (1 + 2 * self.crop_margin)
        side = side.clamp(min=self.min_crop_px, max=min(H, W))
        cx = (x0 + x1 + 1) / 2
        cy = (y0 + y1 + 1) / 2
        side = torch.where(valid, side, torch.full_like(side, min(H, W)))
        cx = torch.where(valid, cx, torch.full_like(cx, W / 2))
        cy = torch.where(valid, cy, torch.full_like(cy, H / 2))
        cx = torch.minimum(torch.maximum(cx, side / 2), W - side / 2)
        cy = torch.minimum(torch.maximum(cy, side / 2), H - side / 2)
        theta = torch.zeros(B, R, ncam, 2, 3, device=pcd.device)
        theta[..., 0, 0] = side / W
        theta[..., 1, 1] = side / H
        theta[..., 0, 2] = cx / W * 2 - 1
        theta[..., 1, 2] = cy / H * 2 - 1
        zoom = self.crop_size / side * (W / 256.0)
        return theta, valid, zoom

    def encode_crops(self, rgb, pcd, instr_feats):
        B, ncam, _, H, W = rgb.shape
        theta, valid, zoom = self.crop_boxes(pcd)
        R = theta.shape[1]
        th = theta.reshape(-1, 2, 3)
        cs = self.crop_size
        grid = F.affine_grid(th, (th.shape[0], 3, cs, cs), align_corners=False)
        rgb_r = rgb[:, None].expand(-1, R, -1, -1, -1, -1).reshape(-1, 3, H, W)
        pcd_r = pcd[:, None].expand(-1, R, -1, -1, -1, -1).reshape(-1, 3, H, W)
        rgb_c = F.grid_sample(rgb_r, grid, mode='bilinear', align_corners=False)
        pcd_c = F.grid_sample(pcd_r, grid, mode='nearest', align_corners=False)
        rgb_c = rgb_c.reshape(B, R * ncam, 3, cs, cs)
        pcd_c = pcd_c.reshape(B, R * ncam, 3, cs, cs)
        feats_pyr, pcd_pyr = self.encoder.encode_images(rgb_c, pcd_c)
        f_feats = einops.rearrange(feats_pyr[0], "b n c h w -> b (n h w) c")
        f_feats = f_feats + self.fine_level_embed
        if self.use_instruction:
            f_feats = self.encoder.vision_language_attention(f_feats, instr_feats)
        return f_feats, pcd_pyr[0], {'valid': valid, 'zoom': zoom,
                                     'rgb_crops': rgb_c}


def load_released_weights(model, path):
    sd = torch.load(path, map_location='cpu')
    sd = sd.get('weight', sd)
    sd = {k[len('module.'):] if k.startswith('module.') else k: v
          for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected
