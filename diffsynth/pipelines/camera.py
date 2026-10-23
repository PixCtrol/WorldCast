import einops
import torch
import torch.nn.functional as F


@torch.amp.autocast("cuda", enabled=False)
def batch_sample_rays(intrinsic, extrinsic, image_h=None, image_w=None):
    ''' get rays
    Args:
        intrinsic: [BF, 3, 3],
        extrinsic: [BF, 4, 4],
        h, w: int
        # normalize: let the first camera R=I
    Returns:
        rays_o, rays_d: [BF, N, 3]
    '''

    # FIXME: PPU does not support inverse in GPU
    device = intrinsic.device
    B = intrinsic.shape[0]

    c2w = torch.inverse(extrinsic)[:, :3, :4].to(device)  # [BF,3,4]
    x = torch.arange(image_w, device=device).float() - 0.5
    y = torch.arange(image_h, device=device).float() + 0.5
    points = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
    points = einops.repeat(points, 'w h c -> b (h w) c', b=B)
    points = torch.cat([points, torch.ones_like(points)[:, :, 0:1]], dim=-1)
    directions = points @ intrinsic.inverse().to(device).transpose(-1, -2) * 1  # depth is 1

    rays_d = F.normalize(directions @ c2w[:, :3, :3].transpose(-1, -2), dim=-1)  # [BF,N,3]
    rays_o = c2w[..., :3, 3]  # [BF, 3]

    rays_o = rays_o[:, None, :].expand_as(rays_d)  # [BF, N, 3]

    return rays_o, rays_d


@torch.amp.autocast("cuda", enabled=False)
def embed_rays(rays_o, rays_d, nframe):
    if len(rays_o.shape) == 4:  # [b,f,n,3]
        rays_o = einops.rearrange(rays_o, "b f n c -> (b f) n c")
        rays_d = einops.rearrange(rays_d, "b f n c -> (b f) n c")
    cross_od = torch.cross(rays_o, rays_d, dim=-1)
    cam_emb = torch.cat([rays_d, cross_od], dim=-1)
    cam_emb = einops.rearrange(cam_emb, "(b f) n c -> b f n c", f=nframe)
    return cam_emb


@torch.amp.autocast("cuda", enabled=False)
def camera_center_normalization(w2c, nframe, camera_scale=2.0):
    # copy from SEVA, w2c: [BF, 4, 4]
    # ensure the first view is eye matrix
    c2w_view0 = w2c[::nframe].inverse()  # [B,4,4]
    c2w_view0 = c2w_view0.repeat_interleave(nframe, dim=0)  # [BF,4,4]
    w2c = c2w_view0 @ w2c

    # camera centering
    c2w = torch.linalg.inv(w2c)
    camera_dist_2med = torch.norm(c2w[:, :3, 3] - c2w[:, :3, 3].median(0, keepdim=True).values, dim=-1)
    valid_mask = camera_dist_2med <= torch.clamp(torch.quantile(camera_dist_2med, 0.97) * 10, max=1e6)
    c2w[:, :3, 3] -= c2w[valid_mask, :3, 3].mean(0, keepdim=True)
    w2c = torch.linalg.inv(c2w)

    # camera normalization
    camera_dists = c2w[:, :3, 3].clone()
    translation_scaling_factor = (
        camera_scale
        if torch.isclose(
            torch.norm(camera_dists[0]),
            torch.zeros(1, dtype=camera_dists.dtype, device=camera_dists.device),
            atol=1e-5,
        ).any()
        else (camera_scale / torch.norm(camera_dists[0]))
    )
    w2c[:, :3, 3] *= translation_scaling_factor
    c2w[:, :3, 3] *= translation_scaling_factor

    return w2c


@torch.amp.autocast("cuda", enabled=False)
def camera_center_normalization_params(w2c, nframe, camera_scale=2.0):
    """Derive the normalization camera_center_normalization applies, WITHOUT
    discarding the transform: returns (gauge, center_offset, scale) so the exact
    same normalized frame can be imposed on a DIFFERENT camera set (anchor view
    cameras) that must share the target trajectory's frame.

    Standalone copy of camera_center_normalization's math — keep the two in
    sync. w2c: [BF, 4, 4] in the cam_info "as stored" convention, exactly the
    matrices fed to get_camera_embedding(normalize=True).

    Returns:
      gauge         [B,4,4]  inv(E_0) per batch element (first-view gauge)
      center_offset [1,3]    valid-mean camera-center offset subtracted AFTER
                             the gauge (computed over ALL BF cameras, matching
                             the original)
      scale         scalar   translation scaling factor camera_scale/|C_0|
    """
    c2w_view0 = w2c[::nframe].inverse()  # [B,4,4]
    gauge = c2w_view0.clone()
    w2c = c2w_view0.repeat_interleave(nframe, dim=0) @ w2c

    c2w = torch.linalg.inv(w2c)
    camera_dist_2med = torch.norm(c2w[:, :3, 3] - c2w[:, :3, 3].median(0, keepdim=True).values, dim=-1)
    valid_mask = camera_dist_2med <= torch.clamp(torch.quantile(camera_dist_2med, 0.97) * 10, max=1e6)
    center_offset = c2w[valid_mask, :3, 3].mean(0, keepdim=True)  # [1,3]
    c2w[:, :3, 3] -= center_offset

    camera_dists = c2w[:, :3, 3].clone()
    scale = (
        camera_scale
        if torch.isclose(
            torch.norm(camera_dists[0]),
            torch.zeros(1, dtype=camera_dists.dtype, device=camera_dists.device),
            atol=1e-5,
        ).any()
        else (camera_scale / torch.norm(camera_dists[0]))
    )
    return gauge, center_offset, scale


@torch.amp.autocast("cuda", enabled=False)
def apply_camera_normalization(w2c, gauge, center_offset, scale):
    """Apply a normalization derived from a reference camera set (the target
    trajectory) to these cameras, so both end up in the SAME normalized frame.

    w2c: [N,4,4] cameras in the same raw "as stored" convention AND world frame
    as the reference set (i.e. aligned anchor cameras already inverted to the
    cam_info stored convention). gauge [1,4,4] / center_offset [1,3] / scale
    scalar from camera_center_normalization_params (single batch element).
    Returns the normalized [N,4,4], matching what camera_center_normalization
    returns for the reference set."""
    c2w = torch.linalg.inv(gauge[0] @ w2c)  # first-view gauge from the reference
    c2w[:, :3, 3] -= center_offset[0]
    c2w[:, :3, 3] *= scale
    return torch.linalg.inv(c2w)


def get_anchor_camera_embedding_shared(anchor_intrinsic, anchor_extrinsic, anchor_n,
                                       target_extrinsic, target_nframe,
                                       h, w, camera_scale=2.0):
    """Anchor Plücker embedding in the SAME normalized frame as the target path.

    The normalization (first-view gauge, valid-mean centering, camera_scale/
    |C_0|) is derived from the TARGET frames only and then applied unchanged to
    the anchors — the target's own embedding (get_camera_embedding with
    normalize=True over the same selected target_extrinsic) is bit-unaffected.

    Args:
      anchor_intrinsic: [A,3,3] anchor intrinsics at (h, w) pixel resolution.
      anchor_extrinsic: [A,4,4] anchor cameras in the cam_info "as stored"
        convention (aligned_camera_info.json stores w2c in the target world
        frame — invert before passing, i.e. torch.linalg.inv(w2c)).
      anchor_n: number of anchor views (= A).
      target_extrinsic: [T,4,4] target cameras EXACTLY as fed to the target's
        get_camera_embedding(normalize=True) call (same stored convention, same
        frame selection).
      target_nframe: T.
      h, w: pixel resolution to sample rays at.
    Returns:
      [1, 6, A, h, w] anchor camera embedding.
    """
    gauge, center_offset, scale = camera_center_normalization_params(
        target_extrinsic, nframe=target_nframe, camera_scale=camera_scale)
    anc_w2c = apply_camera_normalization(anchor_extrinsic, gauge, center_offset, scale)
    rays_o, rays_d = batch_sample_rays(anchor_intrinsic, anc_w2c, image_h=h, image_w=w)
    camera_embedding = embed_rays(rays_o, rays_d, nframe=anchor_n)
    camera_embedding = einops.rearrange(camera_embedding, "b f (h w) c -> b c f h w", h=h, w=w)
    return camera_embedding


def get_camera_embedding(intrinsic, extrinsic, f, h, w, normalize=True):
    if normalize:
        extrinsic = camera_center_normalization(extrinsic, nframe=f)

    rays_o, rays_d = batch_sample_rays(intrinsic, extrinsic, image_h=h, image_w=w)
    camera_embedding = embed_rays(rays_o, rays_d, nframe=f)
    camera_embedding = einops.rearrange(camera_embedding, "b f (h w) c -> b c f h w", h=h, w=w)

    return camera_embedding
