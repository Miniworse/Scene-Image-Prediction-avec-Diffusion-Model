import glob
import os
from datetime import datetime

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

        print(f"Found {len(self.valid_pairs)} valid scene/ifft/visibility samples")

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


def train_model(model, train_loader, model_dir, log_dir, epochs=50, lr=1e-3, load_pretrained=False, device="cpu"):
    writer = SummaryWriter(log_dir=log_dir)
    ema = EMA(model, decay=0.999, device=device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    start_epoch = 0
    train_losses = []
    best_val_loss = float("inf")

    if load_pretrained:
        model, ema, start_epoch, best_val_loss = load_pretrained_model(
            model,
            ema,
            model_dir,
            device,
            load_optimizer=True,
            optimizer=optimizer,
        )
        if start_epoch is None:
            start_epoch = 0
        print(f"Resuming training from epoch {start_epoch + 1}")

    timesteps = 1000
    betas = diffusion.get_beta_schedule(timesteps, schedule_type="sqrt")
    noise_scheduler = diffusion.NoiseScheduler(betas, device=device)

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            batch = move_batch_to_device(batch, device)
            scene_imgs = batch["scene"]
            ifft_imgs = batch["ifft"]
            visibility_imgs = batch["visibility"]
            array_info = batch["array"]

            batch_size = scene_imgs.shape[0]
            sample_id = batch["id"][0]
            t = torch.randint(0, timesteps, (batch_size,), device=device)
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update()

            epoch_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch: {epoch + 1}/{epochs} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.6f}")

        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        train_losses.append(avg_train_loss)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], epoch)

        if epoch % 10 == 0:
            best_val_loss = avg_train_loss
            save_model(model, ema, model_dir, epoch=epoch, optimizer=optimizer, best_val_loss=best_val_loss)

            at = noise_scheduler.alphas_cumprod[t].view(-1, 1, 1, 1)
            predicted_noise = torch.sqrt(at[0]) * xt[0] - torch.sqrt(1 - at[0]) * outputs[0]
            np.savez(
                os.path.join(log_dir, f"train_{epoch}.npz"),
                sample_id=np.array(sample_id),
                scene=scene_imgs[0].detach().cpu().numpy(),
                ifft=ifft_imgs[0].detach().cpu().numpy(),
                visibility=visibility_imgs[0].detach().cpu().numpy(),
                outputs=outputs[0].detach().cpu().numpy(),
                predicted=predicted_noise.detach().cpu().numpy(),
                xt=xt[0].detach().cpu().numpy(),
                timestep=t[0].item(),
            )

        scheduler.step(avg_train_loss)
        print(f"Epoch: {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 60)

    writer.close()

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss")
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, "training_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    return model, ema, optimizer


def test_and_visualize(model, ema, test_loader, log_dir, device="cuda"):
    model.eval()
    criterion = torch.nn.MSELoss()
    test_loss = 0.0
    betas = diffusion.get_beta_schedule(1000, schedule_type="sqrt")
    noise_scheduler = diffusion.NoiseScheduler(betas, device)

    with torch.no_grad():
        for index, batch in enumerate(test_loader):
            batch = move_batch_to_device(batch, device)
            scene_imgs = batch["scene"]
            ifft_imgs = batch["ifft"]
            visibility_imgs = batch["visibility"]
            array_info = batch["array"]
            sample_id = batch["id"][0]

            condition = {
                "ifft_cond": ifft_imgs,
                "visibility": visibility_imgs,
                "antenna_xy": array_info,
            }
            pre_noise, predicted = noise_scheduler.ddim_sampling(model, scene_imgs, condition, steps=250, eta=0)
            loss = criterion(predicted, scene_imgs)
            test_loss += loss.item()

            np.savez(
                os.path.join(log_dir, f"tested_{index}.npz"),
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

    print("\nDataset sizes:")
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")

    sample = train_dataset[0]
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

    print("\nModel architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    model_name = datetime.now().strftime("%b%d-%H%M%S")
    model_dir = os.path.join("./model", model_name + ".pth")
    if params.load_pretrained:
        model_dir = params.model_dir
    log_dir = os.path.join(params.log_dir, model_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Checkpoint path: {model_dir}")
    print(f"Log directory: {log_dir}")

    model, ema, optimizer = train_model(
        model=model,
        train_loader=train_loader,
        model_dir=model_dir,
        log_dir=log_dir,
        epochs=params.epochs,
        lr=params.learning_rate,
        load_pretrained=params.load_pretrained,
        device=str(device),
    )

    print("\nTesting model...")
    test_and_visualize(model, ema, test_loader, log_dir, device)
    save_model(model, ema, model_dir, epoch=params.epochs, optimizer=optimizer, best_val_loss=None)


if __name__ == "__main__":
    main(None)
