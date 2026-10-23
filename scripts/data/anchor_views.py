import json
import os


DEFAULT_BODY_VIEWS = ("front", "left_side", "right_side", "back")


def resolve_semantic_anchor_paths(anchor_dir, prefer_nopad=False, max_views=4):
    """Return anchor image paths and view names in metadata semantic order."""
    if not isinstance(anchor_dir, str):
        return [], []
    if ";" in anchor_dir:
        paths = [path.strip() for path in anchor_dir.split(";") if os.path.isfile(path.strip())]
        if max_views and max_views > 0:
            paths = paths[:max_views]
        names = [
            os.path.splitext(os.path.basename(path))[0].replace("body_view_", "")
            for path in paths
        ]
        return paths, names
    if not os.path.isdir(anchor_dir):
        return [], []

    selected_views = []
    meta_path = os.path.join(anchor_dir, "body_views_meta.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                selected_views = json.load(f).get("selected_views") or []
        except (OSError, ValueError, TypeError):
            selected_views = []
    if not selected_views:
        selected_views = list(DEFAULT_BODY_VIEWS)

    paths, view_names = [], []
    for view_name in selected_views:
        filename = f"body_view_{view_name}.png"
        candidates = [
            os.path.join(anchor_dir, "nopad", filename),
            os.path.join(anchor_dir, filename),
        ]
        if not prefer_nopad:
            candidates.reverse()
        path = next((candidate for candidate in candidates if os.path.isfile(candidate)), None)
        if path is None:
            continue
        paths.append(path)
        view_names.append(view_name)
        if max_views and len(paths) >= max_views:
            break

    # Legacy directories may contain arbitrary image names rather than the
    # body_view_<semantic>.png convention. Preserve the old sorted fallback.
    if not paths:
        image_paths = [
            os.path.join(anchor_dir, name)
            for name in sorted(os.listdir(anchor_dir))
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]
        if max_views and max_views > 0:
            image_paths = image_paths[:max_views]
        paths = image_paths
        view_names = [
            os.path.splitext(os.path.basename(path))[0].replace("body_view_", "")
            for path in paths
        ]
    return paths, view_names


def resolve_gen3r_anchor_paths(anchor_dir, render_path, variant=None, max_views=4,
                               include_first_frame=False):
    """Image list fed to the frozen Gen3R geometry stack (train workers + infer).

    variant=None keeps the legacy behavior EXACTLY: the same files and order
    resolve_semantic_anchor_paths(prefer_nopad=True) returns (nopad/ then the
    padded root, 4 body views).

    variant="black"/"gray"/"crop"/"blank"/"scale" resolves each body view through
    that variant's sibling chain (same _NOPAD_VARIANT_SUBS chain the VAE anchor
    encode uses), so the Gen3R block sees the same images the anchor prefix
    encodes. include_first_frame=True prepends the first-frame anchor
    (<render_path>/<variant>/first_frame.png, falling back through the crop and
    black siblings) FIRST — mirroring the anchor prefix's [first_frame | body]
    order. Used by the scale line, whose 5-anchor canvas includes first_frame.

    A view that falls back OFF the requested variant's directory gets a LOUD
    warning (never a silent distribution shift); a missing first-frame with
    include_first_frame=True is an error, not a skip.

    Returns (paths, view_names); view_names[0] == "first_frame" when prepended.
    """
    if variant is None:
        return resolve_semantic_anchor_paths(anchor_dir, prefer_nopad=True,
                                             max_views=max_views)

    from diffsynth.pipelines.uni3c_pipeline import _NOPAD_VARIANT_SUBS, _resolve_nopad_view_file
    chain = _NOPAD_VARIANT_SUBS.get(variant)
    if chain is None:
        raise ValueError(f"unknown anchor variant for Gen3R images: {variant!r}")

    # Semantic order from the shared resolver (root-first so the variant chain
    # decides which sibling actually gets read), then map through the chain.
    root_paths, view_names = resolve_semantic_anchor_paths(
        anchor_dir, prefer_nopad=False, max_views=max_views)
    paths = []
    for path in root_paths:
        resolved = _resolve_nopad_view_file(path, variant)
        if resolved is None:
            continue
        if os.path.dirname(resolved) != os.path.join(os.path.dirname(path), chain[0]):
            print(f"[gen3r-anchor] WARN: {os.path.basename(path)} fell back off the "
                  f"{chain[0]}/ variant ({variant}) to {resolved} — check the variant "
                  f"data was generated for this case", flush=True)
        paths.append(resolved)
    if not paths:
        return [], []

    if include_first_frame:
        # Variant first, then the crop and black siblings (every line's
        # first-frame data chain ends at no_padding_black).
        candidates = [s for s in chain if s.startswith("no_padding")]
        if "no_padding_black" not in candidates:
            candidates.append("no_padding_black")
        ff_path = None
        for sub in candidates:
            cand = os.path.join(render_path, sub, "first_frame.png")
            if os.path.isfile(cand):
                ff_path = cand
                if sub != candidates[0]:
                    print(f"[gen3r-anchor] WARN: first_frame fell back off the "
                          f"{candidates[0]}/ variant to {cand}", flush=True)
                break
        if ff_path is None:
            raise FileNotFoundError(
                f"gen3r first-frame anchor requested but none of {candidates} "
                f"exists under {render_path}")
        paths = [ff_path] + paths
        view_names = ["first_frame"] + view_names
    return paths, view_names
