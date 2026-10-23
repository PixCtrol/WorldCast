#!/usr/bin/env python3
"""WorldCast trajectory processing — build target-trajectory conditioning dirs.

Given a BASE case render dir (an existing render conditioning dir whose
cam_info describes an estimated/original camera trajectory), this module
constructs a new, inference-ready conditioning dir for a NAMED trajectory:

  preset                   semantics (first pose E0 pinned bit-exact)
  -----------------------  ------------------------------------------------
  flatring                 horizontal 360-deg ring at f0's height about the
                           subject's vertical axis (constant subject
                           distance); CLOSED (f80 = f0)
  lemniscate               Gerono figure-8 in f0's screen plane (SEVA
                           semantics, lobe aperture 60 deg); CLOSED
  smooth_flatring_tilt20   constant-radius ring in the f0-containing plane,
                           pivoted to a 20-deg tilt; CLOSED
  180_push                 180-deg horizontal arc about the subject with a
                           sinusoidal push-pull radius ripple (amp 25% of r0,
                           2 waves, starts pulling in); OPEN (f80 = arc end)
  zoom_out                 one-way geometric dolly out along the f0->aim ray
                           (1.5x the original distance); OPEN (f80 = far end)

Parameterized families are accepted as names too: `smooth_flatring_tilt<N>`
(any tilt 1..89) and `<N>_push` (any sweep 30..360), e.g.
`smooth_flatring_tilt35` or `360_push`.

Construction semantics (ported from the internal data pipeline's
make_smooth_ring_f0 builder):
  - the base trajectory defines the world frame, ring plane/direction and
    the aim target T = least-squares convergence of its forward rays;
  - every pose orients with look_at(T) composed with the constant f0 bias
    D = R0 @ look_at(c0, T)^-1, so frame 0 is EXACTLY the base E0;
  - the world frame is untouched -> anchor cameras stay valid and
    no_padding_black/ is copied verbatim (first-frame anchor + the aligned
    anchor cameras consumed by --use_anchor_plucker);
  - render.mp4 / render_mask.mp4 are rendered from the FIRST-FRAME depth
    point cloud (mask: 0 = covered, 255 = hole) — pulling-back / sliding
    trajectories reveal periphery the cloud never saw (growing holes are
    geometric, not a bug).

Base render dir requirements:
  cam_info.json                {height, width, intrinsic, extrinsic[N,4,4]
                                c2w (OpenCV convention, Z-forward), N >= 2}
  depth/first_frame_depth.npy  (H, W) float depth at the first pose
  first_frame.png              reference frame (point-cloud colors)
  no_padding_black/            optional; copied verbatim when present
  camera_motion.json           optional; copied verbatim when present

Each generated dir contains the full render_path layout: cam_info.json
(extrinsic replaced, trajectory provenance recorded), render.mp4,
render_mask.mp4, first_frame.png, and the no_padding_black/ copy.

Standalone usage (repo root):
  python scripts/data/trajectories.py \
      --render_dir <base_case_render_dir> \
      --traj flatring,lemniscate,smooth_flatring_tilt20,180_push,zoom_out \
      [--out_root <dir>] [--nframes 81] [--force]

From the inference script this is driven by `--traj` (see
scripts/infer/infer_worldcast.py), which expands every manifest row into one
row per requested trajectory before generating.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

import cv2
import numpy as np
import torch

Y_UP = np.array([0.0, 1.0, 0.0])
RENDER_POINT_SIZE = 2     # source pipeline DEPTH_RENDER["point_size"]
RENDER_BG = 0             # source pipeline DEPTH_RENDER["bg_color"]
RENDER_FPS = 24           # source pipeline DEPTH_RENDER["render_fps"]

MIN_DEPTH_THRESHOLD = 0.1
DEPTH_EPSILON = 1e-4
_splat_offset_cache = {}


# ── camera math (ported from the internal trajectory builders) ──

def look_at(center, target):
    """Z-FORWARD c2w (this tree's convention: +Z = look dir), gravity up."""
    fwd = target - center
    fwd = fwd / np.linalg.norm(fwd)
    X = np.cross(Y_UP, fwd); X = X / np.linalg.norm(X)
    Y = np.cross(fwd, X)
    E = np.eye(4); E[:3, 0], E[:3, 1], E[:3, 2] = X, Y, fwd
    E[:3, 3] = center
    return E


def ray_convergence(c, fwd):
    """Least-squares point closest to all rays c_i + t*fwd_i (aim target)."""
    A = np.eye(3) * len(c)
    for d in fwd:
        A -= np.outer(d, d)
    b = c.copy()
    for i, d in enumerate(fwd):
        b[i] = c[i] - d * (d @ c[i])
    return np.linalg.solve(A, b.sum(0))


# ── first-frame-depth cloud render (ported verbatim from the internal
#    DA3 render utilities — torch projection + splat z-buffer + ffmpeg) ──

def depth_to_pointcloud(depth_map, intrinsic, frame_img, extrinsic_c2w):
    """Convert a depth map to a global 3D point cloud.

    Returns points (N,3) global float32, colors (N,3) float32 RGB in [0,1].
    """
    h, w = depth_map.shape
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_map
    valid = z > 0.1
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    pts_local = np.stack([x, y, z], axis=-1)
    pts_local = pts_local[valid]

    R = extrinsic_c2w[:3, :3]
    t = extrinsic_c2w[:3, 3]
    pts_global = (pts_local @ R.T + t).astype(np.float32)

    colors = frame_img[valid].astype(np.float32) / 255.0
    colors = colors[:, ::-1]  # BGR to RGB

    return pts_global, colors


def render_batch(points, colors, c2w, K, width, height, point_size=2,
                 ss_ratio=2.0, bg_color=0):
    """Render point cloud from given camera pose with supersampling.

    Returns (rgb_bgr, mask) as numpy arrays (mask 255 = covered).
    """
    if points is None or colors is None:
        return (np.zeros((height, width, 3), dtype=np.uint8),
                np.zeros((height, width), dtype=np.uint8))

    H_high = int(height * ss_ratio)
    W_high = int(width * ss_ratio)

    K_high = K.clone()
    K_high[0, :] *= ss_ratio
    K_high[1, :] *= ss_ratio

    p_size_high = int(point_size * ss_ratio)
    if p_size_high % 2 == 0:
        p_size_high += 1

    w2c = torch.linalg.inv(c2w)
    N = points.shape[0]
    points_h = torch.cat([points, torch.ones((N, 1), device=points.device)], dim=1)
    cam_xyz = (points_h @ w2c.T)[:, :3]
    z = cam_xyz[:, 2]

    mask_z = z > MIN_DEPTH_THRESHOLD
    if mask_z.sum() == 0:
        return (np.zeros((height, width, 3), dtype=np.uint8),
                np.zeros((height, width), dtype=np.uint8))

    xyz = cam_xyz[mask_z]
    rgb = colors[mask_z]
    z = z[mask_z]

    u_float = (K_high[0, 0] * xyz[:, 0] / z) + K_high[0, 2]
    v_float = (K_high[1, 1] * xyz[:, 1] / z) + K_high[1, 2]

    if p_size_high > 1:
        cache_key = (p_size_high, points.device)
        if cache_key not in _splat_offset_cache:
            radius = p_size_high // 2
            offset_range = torch.arange(-radius, radius + 1, device=points.device)
            dy, dx = torch.meshgrid(offset_range, offset_range, indexing='ij')
            _splat_offset_cache[cache_key] = (dx.flatten(), dy.flatten())
        dx, dy = _splat_offset_cache[cache_key]

        u_final = torch.round(u_float.unsqueeze(1) + dx.unsqueeze(0)).long().view(-1)
        v_final = torch.round(v_float.unsqueeze(1) + dy.unsqueeze(0)).long().view(-1)
        z_final = z.unsqueeze(1).expand(-1, dx.shape[0]).reshape(-1)
        rgb_final = rgb.unsqueeze(1).expand(-1, dx.shape[0], 3).reshape(-1, 3)
    else:
        u_final = torch.round(u_float).long()
        v_final = torch.round(v_float).long()
        z_final = z
        rgb_final = rgb

    valid = (u_final >= 0) & (u_final < W_high) & (v_final >= 0) & (v_final < H_high)
    u = u_final[valid]
    v = v_final[valid]
    rgb = rgb_final[valid]
    z = z_final[valid]

    indices = v * W_high + u
    depth_buffer = torch.full((H_high * W_high,), float('inf'), device=points.device)
    depth_buffer.scatter_reduce_(0, indices, z, reduce='min', include_self=True)
    is_closest = z <= depth_buffer[indices] + DEPTH_EPSILON

    final_u = u[is_closest]
    final_v = v[is_closest]
    final_rgb = rgb[is_closest]

    canvas = torch.full((3, H_high, W_high), bg_color / 255.0, device=points.device)
    canvas[:, final_v, final_u] = final_rgb.permute(1, 0)

    mask_canvas = torch.zeros((H_high, W_high), dtype=torch.uint8, device=points.device)
    mask_canvas[final_v, final_u] = 255

    img_high = (canvas.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img_final = cv2.resize(img_high, (width, height), interpolation=cv2.INTER_AREA)
    mask_final = cv2.resize(mask_canvas.cpu().numpy(), (width, height),
                            interpolation=cv2.INTER_NEAREST)

    img_bgr = cv2.cvtColor(img_final, cv2.COLOR_RGB2BGR)
    return img_bgr, mask_final


def open_ffmpeg_writer(output_path, width, height, fps=24):
    """Open ffmpeg subprocess for streaming raw RGB24 frames to mp4."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-loglevel", "warning",
        output_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


# ── trajectory presets ──
# kind -> cam_info provenance key (matches the internal builder's layout)
PROV_KEY = {
    "flat": "flat_ring",
    "smooth_tilt": "smooth_ring",
    "lemniscate": "lemniscate",
    "arc_pushpull": "arc_pushpull",
    "zoom_out": "zoom",
}

TRAJECTORIES = {
    "flatring": dict(kind="flat", flat_center="person"),
    "lemniscate": dict(kind="lemniscate", degree=60.0),
    "smooth_flatring_tilt20": dict(kind="smooth_tilt", tilt_deg=20.0),
    "180_push": dict(kind="arc_pushpull", sweep_deg=180.0,
                     amp_frac=0.25, waves=2),
    "zoom_out": dict(kind="zoom_out", end_dist_frac=1.5),
}


def parse_traj_name(name):
    """Resolve a trajectory name to a spec dict.

    Named presets plus the parameterized families `smooth_flatring_tilt<N>`
    and `<N>_push` (e.g. smooth_flatring_tilt35, 360_push).
    """
    if name in TRAJECTORIES:
        return dict(TRAJECTORIES[name])
    m = re.fullmatch(r"smooth_flatring_tilt(\d+(?:\.\d+)?)", name)
    if m:
        tilt = float(m.group(1))
        if not 0.0 < tilt < 90.0:
            raise ValueError(f"{name}: tilt must be in (0, 90)")
        return dict(kind="smooth_tilt", tilt_deg=tilt)
    m = re.fullmatch(r"(\d+(?:\.\d+)?)_push", name)
    if m:
        sweep = float(m.group(1))
        if not 0.0 < sweep <= 360.0:
            raise ValueError(f"{name}: sweep must be in (0, 360]")
        return dict(kind="arc_pushpull", sweep_deg=sweep,
                    amp_frac=0.25, waves=2)
    raise ValueError(
        f"unknown trajectory '{name}' — expected one of "
        f"{', '.join(sorted(TRAJECTORIES))} (or smooth_flatring_tilt<N> / <N>_push)")


def traj_tag(spec):
    """Source-builder-style dir tag for the provenance record."""
    kind = spec["kind"]
    if kind == "flat":
        return "flatring_person" if spec["flat_center"] == "person" else "flatring"
    if kind == "smooth_tilt":
        return f"smoothring_t{int(round(spec['tilt_deg']))}"
    if kind == "lemniscate":
        return "lemniscate"
    if kind == "arc_pushpull":
        amp = spec["amp_frac"]
        amp_tag = "" if abs(amp - 0.25) < 1e-9 else f"_amp{int(round(amp * 100))}"
        return (f"arc_pushpull_s{int(round(spec['sweep_deg']))}"
                f"_w{spec['waves']}{amp_tag}")
    if kind == "zoom_out":
        return "zoom_out"
    raise ValueError(f"unknown kind {kind}")


def build_poses(cam, spec, nframes):
    """Construct the trajectory poses from a base cam_info (c2w, N >= 2).

    Returns (poses, prov): poses is a list of nframes 4x4 c2w matrices with
    poses[0] bit-exact the base E0; prov is the provenance dict recorded in
    the output cam_info.json.
    """
    E = np.array(cam["extrinsic"], dtype=np.float64)     # c2w, Z-forward
    if len(E) < 2:
        raise ValueError(
            "base cam_info needs >= 2 extrinsic frames (ring fit + aim "
            "target come from the base trajectory)")
    c = E[:, :3, 3]
    O = c.mean(0)
    _, _, Vt = np.linalg.svd(c - O, full_matrices=False)
    n = Vt[2] * (1.0 if Vt[2][1] >= 0 else -1.0)         # up-ish plane normal
    tau0 = float(np.degrees(np.arccos(abs(n[1]))))

    E0 = E[0]                                            # pinned bit-exact
    c0 = E0[:3, 3]
    t0 = c[1] - c[0]                                     # original direction
    T = ray_convergence(c, E[:, :3, 2])                  # original aim point
    kind = spec["kind"]

    theta = np.linspace(0.0, 2.0 * np.pi, nframes, endpoint=True)

    if kind == "flat":
        # HORIZONTAL ring at f0's height; radius = f0's horizontal distance
        # to the chosen center axis (person -> constant subject distance).
        y_h = float(c0[1])
        ax_xz = np.array([T[0], T[2]]) if spec["flat_center"] == "person" \
            else np.array([O[0], O[2]])
        dxz = c0[[0, 2]] - ax_xz
        r0 = float(np.linalg.norm(dxz))
        u = np.zeros(3); u[[0, 2]] = dxz / r0            # f0 horizontal radial
        v = np.array([u[2], 0.0, -u[0]])                 # horizontal in-plane
        if float(t0 @ v) < 0:                            # keep original direction
            v = -v
        axis = np.array([ax_xz[0], y_h, ax_xz[1]])
        tau1 = 0.0
        ctrs = axis + r0 * (np.cos(theta)[:, None] * u
                            + np.sin(theta)[:, None] * v)
        closed = True
        orient_txt = ("const-height const-radius const-speed HORIZONTAL "
                      "circle at f0's height about the "
                      f"{spec['flat_center']} center axis; look_at(T) with "
                      "f0 bias D; f0 = original pose bit-exact; f80 = f0 "
                      "closed")

    elif kind == "smooth_tilt":
        # Ring in the f0-containing plane, pivoted about the f0 radial until
        # the ring tilt reaches the target (smallest rotation = continuing
        # the original pivot sense; the plane keeps containing the center
        # and c0 -> f0 stays pinned).
        r0 = float(np.linalg.norm(c0 - O))
        u = (c0 - O) / r0                                # f0 radial (theta=0)
        n1 = n - (n @ u) * u
        n1 = n1 / np.linalg.norm(n1)

        def tilt_of(nv):
            return float(np.degrees(np.arccos(np.clip(abs(nv[1]), -1, 1))))

        target = float(spec["tilt_deg"])
        w = np.cross(u, n1)
        cands = []
        for sgn in (1.0, -1.0):
            for deg in np.arange(0.05, 180.0, 0.05):
                phi = np.radians(deg)
                nv = n1 * np.cos(phi) + sgn * w * np.sin(phi)
                if tilt_of(nv) >= target - 1e-9:
                    cands.append((float(deg), nv))
                    break
        if not cands:
            raise ValueError(
                f"target tilt {target}deg unreachable about the f0 radial")
        extra, n1 = min(cands, key=lambda x: x[0])
        tau1 = float(np.degrees(np.arccos(abs(n1[1]))))
        v = np.cross(n1, u)
        if float(t0 @ v) < 0:                            # keep original direction
            v = -v
        axis = O
        ctrs = axis + r0 * (np.cos(theta)[:, None] * u
                            + np.sin(theta)[:, None] * v)
        closed = True
        orient_txt = (f"const-radius const-speed circle in the f0-containing "
                      f"plane pivoted to a {target:g}-deg tilt (pivot "
                      f"{extra:.2f}deg about the f0 radial, smallest "
                      "rotation); look_at(T) with f0 bias D; f0 = original "
                      "pose bit-exact; f80 = f0 closed")

    elif kind == "lemniscate":
        # Gerono figure-8 in the plane through c0 spanned by f0's camera
        # right/up axes (the 'screen' plane), starting AND ending at the
        # crossing point c0; lobe half-width a = |c0-T|*tan(degree/2).
        right = E0[:3, 0].copy()                         # f0 camera right
        up_ax = E0[:3, 1].copy()                         # f0 camera up
        d0 = float(np.linalg.norm(c0 - T))
        a_lem = d0 * float(np.tan(np.radians(spec["degree"]) / 2.0))
        tt = theta + np.pi / 2.0                         # start at crossing
        denom = 1.0 + np.sin(tt) ** 2
        lx = a_lem * np.cos(tt) / denom
        ly = a_lem * np.sin(tt) * np.cos(tt) / denom
        ctrs = (c0[None, :] + lx[:, None] * right[None, :]
                + ly[:, None] * up_ax[None, :])
        axis = T.copy()
        r0, tau1 = None, None
        closed = True
        orient_txt = (f"Gerono figure-8 (SEVA semantics, degree "
                      f"{spec['degree']:g}) in f0's screen plane through c0; "
                      "both lobes traced, start = end = crossing point c0; "
                      "lobe half-width a = |c0-T|*tan(degree/2); look_at(T) "
                      "with f0 bias D; f0 = original pose bit-exact; "
                      "f80 = f0 closed")

    elif kind == "arc_pushpull":
        # Horizontal arc about the subject's vertical axis (T) at constant
        # angular speed with a sinusoidal push-pull radius ripple: starts
        # pulling IN, one full in/out cycle per wave, integer waves end
        # back at r0. OPEN (f80 = arc end at +sweep deg).
        y_h = float(c0[1])
        ax_xz = np.array([T[0], T[2]])
        dxz = c0[[0, 2]] - ax_xz
        r0 = float(np.linalg.norm(dxz))
        u = np.zeros(3); u[[0, 2]] = dxz / r0
        v = np.array([u[2], 0.0, -u[0]])
        if float(t0 @ v) < 0:
            v = -v
        axis = np.array([ax_xz[0], y_h, ax_xz[1]])
        t_n = np.linspace(0.0, 1.0, nframes)
        th_sp = np.radians(spec["sweep_deg"]) * t_n
        s_n = np.sin(2.0 * np.pi * spec["waves"] * t_n)
        radii = r0 * (1.0 - spec["amp_frac"] * s_n)
        ctrs = (axis[None, :]
                + radii[:, None] * (np.cos(th_sp)[:, None] * u
                                    + np.sin(th_sp)[:, None] * v))
        tau1 = None
        closed = False
        orient_txt = (f"const angular speed {spec['sweep_deg']:g}-deg "
                      "horizontal ARC about the subject T axis with "
                      f"SINUSOIDAL push-pull radius r0*(1-amp*sin(2*pi*"
                      f"{spec['waves']}*t)), amp {spec['amp_frac']:.0%} of "
                      "r0, starts pulling in, integer waves end back at r0; "
                      "look_at(T) with f0 bias D; f0 = original pose "
                      "bit-exact; OPEN (f80 = arc end)")

    elif kind == "zoom_out":
        # One-way dolly out along the f0->T ray: camera center stays on the
        # ray T->c0 while the distance grows d0 -> d0*end_frac with a
        # GEOMETRIC profile (constant ratio per frame = perceptually
        # uniform pull-back). Orientation EXACTLY R0 every frame (pure
        # dolly at fixed focal length). OPEN (f80 = far end). Pulling back
        # reveals periphery the first-frame cloud never saw.
        d0_z = float(np.linalg.norm(c0 - T))
        w_dir = (c0 - T) / d0_z                          # T -> c0 unit ray
        d_end_z = spec["end_dist_frac"] * d0_z
        t_n = np.linspace(0.0, 1.0, nframes)
        dists = d0_z * (spec["end_dist_frac"]) ** t_n
        ctrs = T[None, :] + dists[:, None] * w_dir[None, :]
        axis = T.copy()
        r0, tau1 = d0_z, None
        closed = False
        orient_txt = (f"ONE-WAY DOLLY OUT along the f0->T ray: camera center "
                      f"on the ray T->c0, distance GEOMETRIC d0={d0_z:.3f} -> "
                      f"d_end={d_end_z:.3f} ({spec['end_dist_frac']:.0%} of "
                      "d0, constant ratio per frame = perceptually uniform "
                      "pull-back); direction to T constant -> orientation "
                      "EXACTLY R0 every frame (pure dolly at FIXED focal "
                      "length); f0 = original pose bit-exact; OPEN (f80 = "
                      "far end)")

    else:
        raise ValueError(f"unknown trajectory kind '{kind}'")

    # Orientations: look_at(T) with the constant f0 bias D -> frame 0 is
    # exactly the base pose; pure-dolly trajectories degenerate to R0.
    R0 = E0[:3, :3]
    D = R0 @ np.linalg.inv(look_at(c0, T)[:3, :3])
    poses = []
    for i in range(nframes):
        P = look_at(ctrs[i], T)
        P[:3, :3] = D @ P[:3, :3]
        poses.append(P)
    poses[0] = E0                                       # f0 bit-exact
    if closed:
        poses[-1] = E0                                 # f80 = f0 (closed)

    # Verification numbers (informational, recorded in the provenance).
    cc = np.array([p[:3, 3] for p in poses])
    az = np.degrees(np.unwrap(np.arctan2(cc[:, 0] - axis[0], cc[:, 2] - axis[2])))
    sweep = float(az[-1] - az[0])
    r_new = np.linalg.norm(cc - axis, axis=1)

    prov = {
        "tag": traj_tag(spec),
        "kind": kind,
        "orig_tilt_deg": tau0,
        "tilt_deg": tau1,
        "radius": r0,
        "center": axis.tolist(),
        "aim_target": T.tolist(),
        "flat_center": spec.get("flat_center") if kind == "flat" else None,
        "target_tilt_deg": spec["tilt_deg"] if kind == "smooth_tilt" else None,
        "lemniscate_degree": spec["degree"] if kind == "lemniscate" else None,
        "lemniscate_lobe_a": float(a_lem) if kind == "lemniscate" else None,
        "arc_sweep_deg": spec["sweep_deg"] if kind == "arc_pushpull" else None,
        "arc_waves": spec["waves"] if kind == "arc_pushpull" else None,
        "arc_amp_frac": spec["amp_frac"] if kind == "arc_pushpull" else None,
        "zoom_end_dist_frac": (spec["end_dist_frac"] if kind == "zoom_out"
                               else None),
        "zoom_d0": float(d0_z) if kind == "zoom_out" else None,
        "zoom_d_end": float(d_end_z) if kind == "zoom_out" else None,
        "closed": closed,
        "radius_range": [float(r_new.min()), float(r_new.max())],
        "direction": "original (f0->f1 sign)",
        "sweep_deg": sweep,
        "orientation": orient_txt,
    }
    return poses, prov


# ── conditioning-dir builder ──

def _dir_is_complete(out):
    return all(os.path.isfile(os.path.join(out, f))
               for f in ("cam_info.json", "render.mp4", "render_mask.mp4"))


def build_trajectory(render_dir, traj, out_root=None, out_dir=None,
                     nframes=81, force=False, device=None):
    """Build one trajectory conditioning dir from a base case render dir.

    Returns the output dir path. Idempotent: a complete dir is skipped
    unless force=True (then it is rebuilt in place).
    """
    rp = render_dir.rstrip("/")
    case = os.path.basename(rp)
    spec = parse_traj_name(traj) if isinstance(traj, str) else dict(traj)

    cam_path = os.path.join(rp, "cam_info.json")
    if not os.path.isfile(cam_path):
        raise SystemExit(f"[traj] base render dir {rp} has no cam_info.json")
    if out_dir is None:
        root = out_root if out_root else os.path.dirname(rp)
        out_dir = os.path.join(root, f"{case}__{traj}")
    if _dir_is_complete(out_dir) and not force:
        print(f"[traj] {traj}: already complete, skip ({out_dir})", flush=True)
        return out_dir

    cam = json.load(open(cam_path))
    poses, prov = build_poses(cam, spec, nframes)

    os.makedirs(out_dir, exist_ok=True)
    new_cam = dict(cam)
    new_cam["extrinsic"] = [p.tolist() for p in poses]
    new_cam["num_frames"] = len(poses)
    new_cam[PROV_KEY[spec["kind"]]] = prov
    json.dump(new_cam, open(os.path.join(out_dir, "cam_info.json"), "w"),
              indent=2)
    shutil.copy2(os.path.join(rp, "first_frame.png"),
                 os.path.join(out_dir, "first_frame.png"))
    if os.path.isfile(os.path.join(rp, "camera_motion.json")):
        shutil.copy2(os.path.join(rp, "camera_motion.json"),
                     os.path.join(out_dir, "camera_motion.json"))
    if os.path.isdir(os.path.join(rp, "no_padding_black")):
        # World frame untouched -> anchor cameras stay valid; copied verbatim
        # (first-frame anchor + anchors_aligned_camera_info.json for
        # --use_anchor_plucker + person_mask/).
        shutil.copytree(os.path.join(rp, "no_padding_black"),
                        os.path.join(out_dir, "no_padding_black"),
                        dirs_exist_ok=True)

    # ---- cloud render from the base first-frame depth at pose E0 ----
    depth_path = os.path.join(rp, "depth", "first_frame_depth.npy")
    if not os.path.isfile(depth_path):
        raise SystemExit(
            f"[traj] {rp} is missing depth/first_frame_depth.npy — the "
            "trajectory render is built from the base first-frame depth "
            "(see README 'Trajectory generation')")
    Hc, Wc = cam["height"], cam["width"]
    K = np.array(cam["intrinsic"], dtype=np.float64)[:3, :3]
    depth = np.load(depth_path)
    frame = cv2.imread(os.path.join(rp, "first_frame.png"))
    if depth.shape != (Hc, Wc) or frame.shape[:2] != (Hc, Wc):
        raise SystemExit(
            f"[traj] depth {depth.shape} / first_frame {frame.shape[:2]} do "
            f"not match cam_info {Hc}x{Wc} in {rp}")
    pts, cols = depth_to_pointcloud(depth, K, frame, poses[0])
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pts_t = torch.from_numpy(pts.copy()).to(device)
    cols_t = torch.from_numpy(np.ascontiguousarray(cols)).to(device)
    K_t = torch.tensor(K, device=device, dtype=torch.float32)

    video_proc = open_ffmpeg_writer(os.path.join(out_dir, "render.mp4"),
                                    Wc, Hc, RENDER_FPS)
    mask_proc = open_ffmpeg_writer(os.path.join(out_dir, "render_mask.mp4"),
                                   Wc, Hc, RENDER_FPS)
    stats = []
    for i, P in enumerate(poses):
        rgb_bgr, mask = render_batch(pts_t, cols_t,
                                     torch.tensor(P, device=device,
                                                  dtype=torch.float32),
                                     K_t, Wc, Hc,
                                     point_size=RENDER_POINT_SIZE,
                                     bg_color=RENDER_BG)
        video_proc.stdin.write(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).tobytes())
        mask_proc.stdin.write(cv2.cvtColor(255 - mask,
                                           cv2.COLOR_GRAY2RGB).tobytes())
        if i % 20 == 0 or i == len(poses) - 1:
            g = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
            stats.append(f"f{i:02d}: hole={(g == 0).mean():.0%}")
    video_proc.stdin.close(); video_proc.wait()
    mask_proc.stdin.close(); mask_proc.wait()
    print(f"[traj] {case}__{traj}: {traj_tag(spec)} — cloud {len(pts)} pts "
          f"from first-frame depth @E0, {nframes} poses -> {out_dir} "
          f"({' '.join(stats)})", flush=True)
    return out_dir


def expand_samples(samples, trajs, out_root="", nframes=81, force=False):
    """Expand manifest rows into per-trajectory rows (inference --traj path).

    Each row's render_path is used as the BASE case dir; the row is replaced
    by one row per trajectory whose render_path points at the generated
    conditioning dir. Output filenames follow the generated dir name
    (<case>__<preset>_seed<seed>.mp4) because the video field is cleared.
    """
    expanded = []
    for row in samples:
        rp = (row.get("render_path") or "").rstrip("/")
        if not rp or not os.path.isdir(rp):
            raise SystemExit(f"[traj] --traj needs a render_path directory "
                             f"per manifest row, got: {rp!r}")
        for traj in trajs:
            out_dir = build_trajectory(rp, traj, out_root=out_root or None,
                                       nframes=nframes, force=force)
            new_row = dict(row)
            new_row["render_path"] = out_dir
            new_row["video"] = ""      # name outputs after the traj dir
            new_row["trajectory"] = traj
            expanded.append(new_row)
    return expanded


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render_dir", required=True,
                    help="base case render dir (cam_info.json c2w + "
                         "depth/first_frame_depth.npy + first_frame.png)")
    ap.add_argument("--traj", required=True,
                    help="comma-separated trajectories: "
                         + ", ".join(sorted(TRAJECTORIES))
                         + " (or smooth_flatring_tilt<N> / <N>_push)")
    ap.add_argument("--out_root", default="",
                    help="output root for the generated dirs (default: "
                         "sibling of --render_dir; dirs named "
                         "<case>__<preset>)")
    ap.add_argument("--nframes", type=int, default=81)
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if the output dir is complete")
    args = ap.parse_args()

    trajs = [t.strip() for t in args.traj.split(",") if t.strip()]
    for traj in trajs:
        try:
            parse_traj_name(traj)
        except ValueError as e:
            raise SystemExit(f"[traj] {e}")
    rows = [dict(prompt="", reference_image_path="",
                 render_path=args.render_dir)]
    expanded = expand_samples(rows, trajs, out_root=args.out_root,
                              nframes=args.nframes, force=args.force)
    for row in expanded:
        print(f"  -> {row['render_path']}", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
