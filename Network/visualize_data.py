import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


WORKSPACE_ROOT = Path(__file__).resolve().parent
RUNS_ROOT = WORKSPACE_ROOT / "runs"
VISUALS_ROOT = WORKSPACE_ROOT / "visualizations"


def to_2d_image(data):
    array = np.asarray(data)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (2, 3):
        array = array[0]
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return array


def split_visibility_channels(visibility):
    array = np.asarray(visibility)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] == 2:
        real_part = array[0]
        imag_part = array[1]
    elif array.ndim == 3 and array.shape[-1] == 2:
        real_part = array[..., 0]
        imag_part = array[..., 1]
    else:
        real_part = to_2d_image(array)
        imag_part = None
    return real_part, imag_part


def plot_scalar(ax, data, title, cmap):
    image = ax.imshow(data, cmap=cmap)
    ax.set_title(f"{title}\n{data.shape}", fontsize=10)
    ax.axis("off")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def extract_panels_from_npz(loaded):
    panels = []
    ordered_keys = (
        "scene",
        "observe",
        "ifft",
        "outputs",
        "predicted",
        "noise",
        "xt",
    )
    for key in ordered_keys:
        if key in loaded:
            panels.append((key, to_2d_image(loaded[key])))

    if "visibility" in loaded:
        vis_real, vis_imag = split_visibility_channels(loaded["visibility"])
        panels.append(("visibility_real", vis_real))
        if vis_imag is not None:
            panels.append(("visibility_imag", vis_imag))

    if not panels:
        for key in loaded.keys():
            if key.startswith("__"):
                continue
            value = loaded[key]
            if np.asarray(value).ndim >= 2:
                panels.append((key, to_2d_image(value)))

    return panels


def infer_save_path(file_path, input_root, save_root):
    if input_root is None:
        return Path(save_root) / f"{file_path.stem}.png"

    relative_parent = file_path.parent.relative_to(input_root)
    destination_dir = Path(save_root) / relative_parent
    destination_dir.mkdir(parents=True, exist_ok=True)
    return destination_dir / f"{file_path.stem}.png"


def visualize_npz_file(file_path, save_path, cmap="jet"):
    loaded = np.load(file_path)
    panels = extract_panels_from_npz(loaded)

    if not panels:
        print(f"Skipping {file_path.name}: no visualizable arrays found")
        return

    timestep_value = None
    if "timestep" in loaded:
        timestep_value = int(np.asarray(loaded["timestep"]).reshape(-1)[0])

    sample_id = None
    if "sample_id" in loaded:
        sample_id = np.asarray(loaded["sample_id"]).reshape(-1)[0]

    n_panels = len(panels)
    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (title, data) in zip(axes, panels):
        plot_scalar(ax, data, title, cmap)

    for ax in axes[len(panels):]:
        ax.axis("off")

    header = file_path.stem
    if sample_id is not None:
        header += f" | sample={sample_id}"
    if timestep_value is not None:
        header += f" | t={timestep_value}"
    fig.suptitle(header, fontsize=14)
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_npy_file(file_path, save_path, cmap="jet"):
    data = np.load(file_path)
    image_2d = to_2d_image(data)

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_scalar(ax, image_2d, file_path.stem, cmap)
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def collect_target_files(target_path, recursive=True):
    target = resolve_target_path(target_path)
    if not target.exists():
        raise FileNotFoundError(f"Path not found: {target}")

    if target.is_file():
        if target.suffix.lower() not in (".npy", ".npz"):
            raise ValueError(f"Unsupported file type: {target}")
        return [target], target.parent

    if recursive:
        npy_files = list(target.rglob("*.npy"))
        npz_files = list(target.rglob("*.npz"))
    else:
        npy_files = list(target.glob("*.npy"))
        npz_files = list(target.glob("*.npz"))

    all_files = sorted(npy_files + npz_files)
    return all_files, target


def resolve_target_path(target_path):
    target = Path(target_path)
    if target.exists():
        return target

    run_candidate = RUNS_ROOT / str(target_path)
    if run_candidate.exists():
        return run_candidate

    raise FileNotFoundError(
        f"Path not found: {target_path}. "
        f"You can pass an absolute path or a run name under {RUNS_ROOT}."
    )


def default_visualization_dir(target_path):
    target = resolve_target_path(target_path)

    if target.is_file():
        stem = target.stem
    else:
        stem = target.name
    return VISUALS_ROOT / stem


def visualize_and_save_outputs(target_path, save_dir=None, cmap="jet", recursive=True):
    all_files, input_root = collect_target_files(target_path, recursive=recursive)
    if not all_files:
        print(f"No .npy or .npz files found in {target_path}")
        return

    save_root = Path(save_dir) if save_dir is not None else default_visualization_dir(target_path)
    save_root.mkdir(parents=True, exist_ok=True)

    npy_count = sum(1 for path in all_files if path.suffix.lower() == ".npy")
    npz_count = sum(1 for path in all_files if path.suffix.lower() == ".npz")
    print(f"Found {len(all_files)} files ({npy_count} .npy, {npz_count} .npz)")
    print(f"Saving visualizations to: {save_root}")

    for file_path in tqdm(all_files, desc="Visualizing"):
        try:
            save_path = infer_save_path(file_path, input_root, save_root)
            if file_path.suffix.lower() == ".npz":
                visualize_npz_file(file_path, save_path, cmap=cmap)
            else:
                visualize_npy_file(file_path, save_path, cmap=cmap)
        except Exception as exc:
            print(f"Failed on {file_path}: {exc}")
            plt.close("all")

    print(f"Done. Images saved to {save_root}")


if __name__ == "__main__":
    target_path = "May07-204141"
    visualize_and_save_outputs(target_path, recursive=True)
