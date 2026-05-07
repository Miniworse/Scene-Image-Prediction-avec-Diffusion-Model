import os
from datetime import datetime


def make_run_name():
    return datetime.now().strftime("%b%d-%H%M%S")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def create_run_layout(runs_root, run_name):
    run_dir = ensure_dir(os.path.join(runs_root, run_name))
    checkpoints_dir = ensure_dir(os.path.join(run_dir, "checkpoints"))
    train_dir = ensure_dir(os.path.join(run_dir, "train"))
    test_dir = ensure_dir(os.path.join(run_dir, "test"))
    summaries_dir = ensure_dir(os.path.join(run_dir, "summaries"))

    return {
        "run_name": run_name,
        "run_dir": run_dir,
        "checkpoints_dir": checkpoints_dir,
        "train_dir": train_dir,
        "test_dir": test_dir,
        "summaries_dir": summaries_dir,
        "latest_checkpoint": os.path.join(checkpoints_dir, "latest.pth"),
        "best_checkpoint": os.path.join(checkpoints_dir, "best.pth"),
    }


def resolve_run_layout_for_model(model_path, runs_root, data_name=None):
    model_path = os.path.normpath(model_path)
    checkpoints_dir = os.path.dirname(model_path)
    run_dir = os.path.dirname(checkpoints_dir)

    if os.path.basename(checkpoints_dir) != "checkpoints":
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        run_name = f"eval_{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        layout = create_run_layout(runs_root, run_name)
    else:
        run_name = os.path.basename(run_dir)
        layout = {
            "run_name": run_name,
            "run_dir": run_dir,
            "checkpoints_dir": checkpoints_dir,
            "train_dir": ensure_dir(os.path.join(run_dir, "train")),
            "test_dir": ensure_dir(os.path.join(run_dir, "test")),
            "summaries_dir": ensure_dir(os.path.join(run_dir, "summaries")),
            "latest_checkpoint": os.path.join(checkpoints_dir, "latest.pth"),
            "best_checkpoint": os.path.join(checkpoints_dir, "best.pth"),
        }

    if data_name:
        inference_dir = ensure_dir(os.path.join(layout["run_dir"], f"inference_{data_name}"))
        layout["inference_dir"] = inference_dir

    return layout


def find_latest_checkpoint(runs_root, checkpoint_name="best.pth"):
    if not os.path.exists(runs_root):
        return None

    matches = []
    for entry in os.scandir(runs_root):
        if not entry.is_dir():
            continue
        candidate = os.path.join(entry.path, "checkpoints", checkpoint_name)
        if os.path.exists(candidate):
            matches.append((os.path.getmtime(candidate), candidate))

    if not matches:
        return None

    matches.sort(reverse=True)
    return matches[0][1]
