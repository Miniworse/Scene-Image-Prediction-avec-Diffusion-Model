import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import scripts.diffusion as diffusion
from scripts.Unet_model import ConditionalUNet, load_pretrained_model, save_model
from scripts.get_EMA import EMA
from scripts.params import params_all
from scripts.run_layout import create_run_layout, find_latest_checkpoint, make_run_name, resolve_run_layout_for_model
from scripts.run_summary import write_summary_files


def prepare_image_tensor(data):
    if data.ndim == 2:
        data = np.expand_dims(data, axis=0)
    elif data.ndim == 3 and data.shape[0] not in (1, 3):
        data = data.transpose(2, 0, 1)
    return torch.from_numpy(data).float()


def prepare_visibility_tensor(data):
    if data.ndim == 2:
        data = np.expand_dims(data, axis=0)
    elif data.ndim == 3 and data.shape[-1] in (1, 2):
        data = data.transpose(2, 0, 1)
    elif data.ndim == 3 and data.shape[0] not in (1, 2):
        data = data.transpose(2, 0, 1)
    return torch.from_numpy(data).float()


def normalize_tensor(tensor, eps=1e-6):
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / (std + eps)


def resolve_array_path(data_dir):
    candidates = [
        os.path.join(data_dir, "array.npy"),
        os.path.join(data_dir, "array_position.npy"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Array geometry file not found. Tried: {candidates}")


class SceneObserveDataset(Dataset):
    def __init__(self, data_dir, indices, normalize=True):
        self.data_dir = data_dir
        self.normalize = normalize
        self.array_path = resolve_array_path(data_dir)
        self.array_tensor = torch.from_numpy(np.load(self.array_path)).float()

        scene_files = sorted(glob.glob(os.path.join(data_dir, "scene_*.npy")))
        self.valid_pairs = []
        for scene_file in scene_files:
            index = os.path.basename(scene_file).replace("scene_", "").replace(".npy", "")
            ifft_file = os.path.join(data_dir, f"observe_{index}.npy")
            visibility_file = os.path.join(data_dir, f"visibility_{index}.npy")
            if os.path.exists(ifft_file) and os.path.exists(visibility_file):
                self.valid_pairs.append((scene_file, ifft_file, visibility_file, index))

        if indices is not None:
            self.valid_pairs = [self.valid_pairs[i] for i in indices if i < len(self.valid_pairs)]

        print(f"Found {len(self.valid_pairs)} valid scene/ifft/visibility samples in {data_dir}")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        scene_path, ifft_path, visibility_path, sample_id = self.valid_pairs[idx]
        scene_tensor = prepare_image_tensor(np.load(scene_path))
        ifft_tensor = prepare_image_tensor(np.load(ifft_path))
        visibility_tensor = prepare_visibility_tensor(np.load(visibility_path))
        array_tensor = self.array_tensor.clone()

        if self.normalize:
            scene_tensor = normalize_tensor(scene_tensor)
            ifft_tensor = normalize_tensor(ifft_tensor)
            visibility_tensor = normalize_tensor(visibility_tensor)

        return {
            "scene": scene_tensor,
            "ifft": ifft_tensor,
            "visibility": visibility_tensor,
            "array": array_tensor,
            "id": sample_id,
        }


def split_dataset(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-8, "ratios must sum to 1"

    scene_files = sorted(glob.glob(os.path.join(data_dir, "scene_*.npy")))
    valid_indices = []
    for i, scene_file in enumerate(scene_files):
        index = os.path.basename(scene_file).replace("scene_", "").replace(".npy", "")
        ifft_file = os.path.join(data_dir, f"observe_{index}.npy")
        visibility_file = os.path.join(data_dir, f"visibility_{index}.npy")
        if os.path.exists(ifft_file) and os.path.exists(visibility_file):
            valid_indices.append(i)

    print(f"Total valid samples: {len(valid_indices)}")
    train_val_indices, test_indices = train_test_split(valid_indices, test_size=test_ratio, random_state=seed)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_ratio / (train_ratio + val_ratio),
        random_state=seed,
    )

    print(f"Train set: {len(train_indices)} samples")
    print(f"Validation set: {len(val_indices)} samples")
    print(f"Test set: {len(test_indices)} samples")
    return train_indices, val_indices, test_indices


def move_batch_to_device(batch, device):
    return {
        "scene": batch["scene"].to(device),
        "ifft": batch["ifft"].to(device),
        "visibility": batch["visibility"].to(device),
        "array": batch["array"].to(device),
        "id": batch["id"],
    }


def build_noise_scheduler(device):
    betas = diffusion.get_beta_schedule(1000, schedule_type="sqrt")
    return diffusion.NoiseScheduler(betas, device=device)


def compute_diffusion_loss(model, noise_scheduler, batch, device):
    batch = move_batch_to_device(batch, device)
    scene_imgs = batch["scene"]
    ifft_imgs = batch["ifft"]
    visibility_imgs = batch["visibility"]
    array_info = batch["array"]

    batch_size = scene_imgs.shape[0]
    t = torch.randint(0, 1000, (batch_size,), device=device)
    xt, noise = noise_scheduler.forward_diffusion(scene_imgs, t)

    outputs = model(
        xt,
        t,
        ifft_cond=ifft_imgs,
        visibility=visibility_imgs,
        antenna_xy=array_info,
    )

    velocity = noise_scheduler.get_velocity(scene_imgs, noise, t)
    snr = noise_scheduler.get_snr(t).view(-1, 1, 1, 1)
    weight = snr / (snr + 1)
    loss = (weight * (outputs - velocity) ** 2).mean()

    context = {
        "scene": scene_imgs,
        "ifft": ifft_imgs,
        "visibility": visibility_imgs,
        "array": array_info,
        "xt": xt,
        "outputs": outputs,
        "timestep": t,
        "sample_id": batch["id"][0],
    }
    return loss, context


def evaluate_validation(model, val_loader, noise_scheduler, device):
    if val_loader is None or len(val_loader) == 0:
        return None

    model.eval()
    losses = []
    with torch.no_grad():
        for batch in val_loader:
            loss, _ = compute_diffusion_loss(model, noise_scheduler, batch, device)
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else None


def save_training_snapshot(context, noise_scheduler, target_dir, epoch):
    timestep = context["timestep"]
    at = noise_scheduler.alphas_cumprod[timestep].view(-1, 1, 1, 1)
    predicted_noise = torch.sqrt(at[0]) * context["xt"][0] - torch.sqrt(1 - at[0]) * context["outputs"][0]
    np.savez(
        os.path.join(target_dir, f"train_{epoch}.npz"),
        sample_id=np.array(context["sample_id"]),
        scene=context["scene"][0].detach().cpu().numpy(),
        ifft=context["ifft"][0].detach().cpu().numpy(),
        visibility=context["visibility"][0].detach().cpu().numpy(),
        outputs=context["outputs"][0].detach().cpu().numpy(),
        predicted=predicted_noise.detach().cpu().numpy(),
        xt=context["xt"][0].detach().cpu().numpy(),
        timestep=timestep[0].item(),
    )


def save_training_curve(train_losses, val_losses, output_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    valid_val_points = [loss for loss in val_losses if loss is not None]
    if valid_val_points:
        x = [idx for idx, loss in enumerate(val_losses) if loss is not None]
        y = [loss for loss in val_losses if loss is not None]
        plt.plot(x, y, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.grid(True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def get_params_snapshot(params):
    return {key: value for key, value in params.items()}


def build_train_summary(
    run_layout,
    checkpoint_paths,
    data_dir,
    array_path,
    dataset_sizes,
    sample_shapes,
    params,
    model,
    train_losses,
    val_losses,
    avg_test_loss,
    best_epoch,
    best_val_loss,
):
    return {
        "title": f"Training Summary - {run_layout['run_name']}",
        "run_name": run_layout["run_name"],
        "run_dir": run_layout["run_dir"],
        "checkpoint_paths": checkpoint_paths,
        "data_dir": data_dir,
        "array_path": array_path,
        "dataset_sizes": dataset_sizes,
        "sample_shapes": sample_shapes,
        "model_config": {
            "architecture": model.__class__.__name__,
            "n_channels": getattr(model, "n_channels", None),
            "n_classes": getattr(model, "n_classes", None),
            "time_emb_dim": getattr(model, "time_emb_dim", None),
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
        },
        "training_objective": {
            "diffusion_timesteps": 1000,
            "beta_schedule": "sqrt",
            "prediction_target": "velocity",
            "loss": "weighted_mse",
            "weight_formula": "snr / (snr + 1)",
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau",
            "ema_decay": 0.999,
        },
        "sampling_for_test_visualization": {
            "sampler": "ddim",
            "steps": 250,
            "eta": 0,
            "criterion": "MSE(predicted_scene, target_scene)",
        },
        "params": get_params_snapshot(params),
        "results": {
            "train_loss_first": float(train_losses[0]) if train_losses else None,
            "train_loss_last": float(train_losses[-1]) if train_losses else None,
            "train_loss_min": float(min(train_losses)) if train_losses else None,
            "train_loss_curve": [float(loss) for loss in train_losses],
            "val_loss_last": float(val_losses[-1]) if val_losses and val_losses[-1] is not None else None,
            "val_loss_min": float(min(loss for loss in val_losses if loss is not None)) if any(loss is not None for loss in val_losses) else None,
            "val_loss_curve": [float(loss) if loss is not None else None for loss in val_losses],
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
            "test_loss": float(avg_test_loss) if avg_test_loss is not None else None,
        },
    }


def train_model(
    model,
    train_loader,
    val_loader,
    run_layout,
    params,
    resume_checkpoint_path=None,
    device="cpu",
):
    writer = SummaryWriter(log_dir=os.path.join(run_layout["run_dir"], "tensorboard"))
    ema = EMA(model, decay=0.999, device=device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    start_epoch = 0
    train_losses = []
    val_losses = []
    best_val_loss = None
    best_epoch = None

    if params.load_pretrained and resume_checkpoint_path:
        model, ema, start_epoch, best_val_loss = load_pretrained_model(
            model,
            ema,
            resume_checkpoint_path,
            device,
            load_optimizer=True,
            optimizer=optimizer,
        )
        if start_epoch is None:
            start_epoch = 0
        else:
            start_epoch += 1
        print(f"Resuming training from epoch {start_epoch + 1}")

    noise_scheduler = build_noise_scheduler(device)
    latest_checkpoint_path = run_layout["latest_checkpoint"]
    best_checkpoint_path = run_layout["best_checkpoint"]

    for epoch in range(start_epoch, params.epochs):
        model.train()
        epoch_loss = 0.0
        last_context = None

        for batch_idx, batch in enumerate(train_loader):
            loss, context = compute_diffusion_loss(model, noise_scheduler, batch, device)
            optimizer.zero_grad()
            loss.backward()
            if params.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), params.max_grad_norm)
            optimizer.step()
            ema.update()

            epoch_loss += loss.item()
            last_context = context
            if batch_idx % 10 == 0:
                print(
                    f"Epoch: {epoch + 1}/{params.epochs} | "
                    f"Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.6f}"
                )

        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        val_loss = evaluate_validation(model, val_loader, noise_scheduler, device)
        scheduler_metric = val_loss if val_loss is not None else avg_train_loss
        scheduler.step(scheduler_metric)

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        if val_loss is not None:
            writer.add_scalar("Loss/validation", val_loss, epoch)
        writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], epoch)

        save_model(
            model,
            ema,
            latest_checkpoint_path,
            epoch=epoch,
            optimizer=optimizer,
            best_val_loss=best_val_loss,
        )

        if val_loss is not None and (best_val_loss is None or val_loss < best_val_loss):
            best_val_loss = val_loss
            best_epoch = epoch + 1
            save_model(
                model,
                ema,
                best_checkpoint_path,
                epoch=epoch,
                optimizer=optimizer,
                best_val_loss=best_val_loss,
            )
            print(f"Best checkpoint updated at epoch {epoch + 1} with val loss {best_val_loss:.6f}")

        if params.checkpoint_every and (epoch + 1) % params.checkpoint_every == 0:
            epoch_checkpoint_path = os.path.join(
                run_layout["checkpoints_dir"],
                f"epoch_{epoch + 1:04d}.pth",
            )
            save_model(
                model,
                ema,
                epoch_checkpoint_path,
                epoch=epoch,
                optimizer=optimizer,
                best_val_loss=best_val_loss,
            )
            if last_context is not None:
                save_training_snapshot(last_context, noise_scheduler, run_layout["train_dir"], epoch + 1)

        val_loss_display = f"{val_loss:.6f}" if val_loss is not None else "N/A"
        print(
            f"Epoch: {epoch + 1}/{params.epochs} | "
            f"Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss_display} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        print("-" * 60)

    writer.close()
    save_training_curve(train_losses, val_losses, os.path.join(run_layout["train_dir"], "training_curve.png"))

    checkpoint_paths = {
        "latest": latest_checkpoint_path,
        "best": best_checkpoint_path if os.path.exists(best_checkpoint_path) else None,
    }
    return model, ema, optimizer, train_losses, val_losses, best_epoch, best_val_loss, checkpoint_paths


def test_and_visualize(model, ema, test_loader, output_dir, device="cuda"):
    model.eval()
    criterion = torch.nn.MSELoss()
    test_loss = 0.0
    noise_scheduler = build_noise_scheduler(device)

    with torch.no_grad():
        for index, batch in enumerate(test_loader):
            batch = move_batch_to_device(batch, device)
            scene_imgs = batch["scene"]
            ifft_imgs = batch["ifft"]
            visibility_imgs = batch["visibility"]
            array_info = batch["array"]
            sample_id = batch["id"][0]

            if ema is not None:
                ema.apply_shadow()

            condition = {
                "ifft_cond": ifft_imgs,
                "visibility": visibility_imgs,
                "antenna_xy": array_info,
            }
            pre_noise, predicted = noise_scheduler.ddim_sampling(model, scene_imgs, condition, steps=1000, eta=0)

            if ema is not None:
                ema.restore()

            loss = criterion(predicted, scene_imgs)
            test_loss += loss.item()

            np.savez(
                os.path.join(output_dir, f"tested_{index}.npz"),
                sample_id=np.array(sample_id),
                outputs=predicted[0].detach().cpu().numpy(),
                observe=ifft_imgs[0].detach().cpu().numpy(),
                scene=scene_imgs[0].detach().cpu().numpy(),
                noise=pre_noise[0].detach().cpu().numpy(),
                visibility=visibility_imgs[0].detach().cpu().numpy(),
            )

    avg_test_loss = test_loss / max(len(test_loader), 1)
    print(f"Test Loss: {avg_test_loss:.6f}")
    return avg_test_loss


def main(args):
    params = params_all
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = params.data_dir
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist")
        return
    print(f"Training data directory: {data_dir}")

    train_indices, val_indices, test_indices = split_dataset(
        data_dir,
        train_ratio=params.train_ratio,
        val_ratio=params.val_ratio,
        test_ratio=params.test_ratio,
        seed=params.seed,
    )

    train_dataset = SceneObserveDataset(data_dir, train_indices, normalize=True)
    val_dataset = SceneObserveDataset(data_dir, val_indices, normalize=True)
    test_dataset = SceneObserveDataset(data_dir, test_indices, normalize=True)

    dataset_sizes = {
        "train": len(train_dataset),
        "validation": len(val_dataset),
        "test": len(test_dataset),
    }
    print("\nDataset sizes:")
    print(f"Training set: {dataset_sizes['train']} samples")
    print(f"Validation set: {dataset_sizes['validation']} samples")
    print(f"Test set: {dataset_sizes['test']} samples")

    sample = train_dataset[0]
    array_path = train_dataset.array_path
    sample_shapes = {
        "scene": list(sample["scene"].shape),
        "ifft": list(sample["ifft"].shape),
        "visibility": list(sample["visibility"].shape),
        "array": list(sample["array"].shape),
    }
    print(
        f"\nSample shapes - Scene: {sample['scene'].shape}, "
        f"IFFT: {sample['ifft'].shape}, Visibility: {sample['visibility'].shape}, Array: {sample['array'].shape}"
    )

    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers)

    batch = next(iter(train_loader))
    in_channels = batch["scene"].shape[1]
    out_channels = batch["scene"].shape[1]
    ifft_channels = batch["ifft"].shape[1]
    visibility_channels = batch["visibility"].shape[1]

    model = ConditionalUNet(
        n_channels=in_channels,
        n_classes=out_channels,
        time_emb_dim=512,
        ifft_channels=ifft_channels,
        visibility_channels=visibility_channels,
    )

    resume_candidate = params.model_dir
    if params.load_pretrained and not os.path.exists(resume_candidate):
        fallback_model = find_latest_checkpoint(params.runs_dir, checkpoint_name="latest.pth")
        if fallback_model is not None:
            print(f"Configured resume checkpoint not found. Falling back to: {fallback_model}")
            resume_candidate = fallback_model

    if params.load_pretrained and os.path.exists(resume_candidate):
        run_layout = resolve_run_layout_for_model(resume_candidate, params.runs_dir)
        resume_checkpoint_path = resume_candidate
    else:
        run_layout = create_run_layout(params.runs_dir, make_run_name())
        resume_checkpoint_path = None

    print("\nModel architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Run directory: {run_layout['run_dir']}")
    print(f"Latest checkpoint path: {run_layout['latest_checkpoint']}")
    print(f"Best checkpoint path: {run_layout['best_checkpoint']}")

    model, ema, optimizer, train_losses, val_losses, best_epoch, best_val_loss, checkpoint_paths = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        run_layout=run_layout,
        params=params,
        resume_checkpoint_path=resume_checkpoint_path,
        device=str(device),
    )

    print("\nTesting model...")
    avg_test_loss = test_and_visualize(model, ema, test_loader, run_layout["test_dir"], device)

    if not os.path.exists(checkpoint_paths["best"] or ""):
        save_model(
            model,
            ema,
            run_layout["best_checkpoint"],
            epoch=params.epochs - 1,
            optimizer=optimizer,
            best_val_loss=best_val_loss,
        )
        checkpoint_paths["best"] = run_layout["best_checkpoint"]

    train_summary = build_train_summary(
        run_layout=run_layout,
        checkpoint_paths=checkpoint_paths,
        data_dir=data_dir,
        array_path=array_path,
        dataset_sizes=dataset_sizes,
        sample_shapes=sample_shapes,
        params=params,
        model=model,
        train_losses=train_losses,
        val_losses=val_losses,
        avg_test_loss=avg_test_loss,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
    )
    summary_json, summary_txt = write_summary_files(
        run_layout["summaries_dir"],
        f"{run_layout['run_name']}_training_summary",
        train_summary,
    )
    print(f"Training summary saved to: {summary_json}")
    print(f"Training summary text saved to: {summary_txt}")


if __name__ == "__main__":
    main(None)
