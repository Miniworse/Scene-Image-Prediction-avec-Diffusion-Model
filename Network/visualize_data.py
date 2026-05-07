import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


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


def visualize_npz_file(file_path, save_dir, cmap="jet"):
    loaded = np.load(file_path)
    keys = [key for key in loaded.keys() if not key.startswith("__")]

    panels = []
    for key in ("scene", "observe", "ifft", "outputs", "predicted", "noise", "xt"):
        if key in loaded:
            panels.append((key, to_2d_image(loaded[key])))

    if "visibility" in loaded:
        vis_real, vis_imag = split_visibility_channels(loaded["visibility"])
        panels.append(("visibility_real", vis_real))
        if vis_imag is not None:
            panels.append(("visibility_imag", vis_imag))

    if "timestep" in loaded:
        timestep = loaded["timestep"]
        timestep_value = int(np.asarray(timestep).reshape(-1)[0])
    else:
        timestep_value = None

    if not panels:
        fallback_keys = []
        for key in keys:
            value = loaded[key]
            if np.asarray(value).ndim >= 2:
                fallback_keys.append((key, to_2d_image(value)))
        panels = fallback_keys

    if not panels:
        print(f"Skipping {file_path.name}: no visualizable arrays found")
        return

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
    if timestep_value is not None:
        header = f"{header} | t={timestep_value}"
    fig.suptitle(header, fontsize=14)
    fig.tight_layout()

    save_path = os.path.join(save_dir, f"{file_path.stem}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_npy_file(file_path, save_dir, cmap="jet"):
    data = np.load(file_path)
    image_2d = to_2d_image(data)

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_scalar(ax, image_2d, file_path.stem, cmap)
    fig.tight_layout()

    save_path = os.path.join(save_dir, f"{file_path.stem}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def visualize_and_save_outputs(folder_path, save_dir="./visualizations", cmap="jet"):
    folder = Path(folder_path)
    npy_files = list(folder.glob("*.npy"))
    npz_files = list(folder.glob("*.npz"))
    all_files = npy_files + npz_files

    if not all_files:
        print(f"No .npy or .npz files found in {folder_path}")
        return

    os.makedirs(save_dir, exist_ok=True)
    print(f"Found {len(all_files)} files ({len(npy_files)} .npy, {len(npz_files)} .npz)")
    print(f"Saving visualizations to: {save_dir}")

    for file_path in tqdm(all_files, desc="Visualizing"):
        try:
            if file_path.suffix.lower() == ".npz":
                visualize_npz_file(file_path, save_dir, cmap=cmap)
            else:
                visualize_npy_file(file_path, save_dir, cmap=cmap)
        except Exception as exc:
            print(f"Failed on {file_path.name}: {exc}")
            plt.close("all")

    print(f"Done. Images saved to {save_dir}")


if __name__ == "__main__":
    folder_path = r"D:\Documents\Self_Files\Projects\SceneGenerating\Network\trainlog\May06-222842"
    save_dir = f"./output_images/{Path(folder_path).name}"
    visualize_and_save_outputs(folder_path, save_dir=save_dir)
