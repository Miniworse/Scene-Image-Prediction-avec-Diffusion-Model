import glob
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from tqdm import tqdm

try:
    from torchvision.models import Inception_V3_Weights, inception_v3
except ImportError:
    Inception_V3_Weights = None
    from torchvision.models import inception_v3

import scripts.diffusion as diffusion
from scripts.Unet_model import load_model
from scripts.params import params_all
from scripts.run_layout import find_latest_checkpoint, resolve_run_layout_for_model
from scripts.run_summary import write_summary_files

warnings.filterwarnings("ignore")


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


def get_params_snapshot(params):
    return {key: value for key, value in params.items()}


class PredictDataset:
    def __init__(self, data_dir, normalize=True):
        self.data_dir = data_dir
        self.normalize = normalize
        self.array_path = resolve_array_path(data_dir)
        self.array_tensor = torch.from_numpy(np.load(self.array_path)).float()

        self.valid_samples = []
        for scene_file in sorted(glob.glob(os.path.join(data_dir, "scene_*.npy"))):
            index = os.path.basename(scene_file).replace("scene_", "").replace(".npy", "")
            ifft_file = os.path.join(data_dir, f"observe_{index}.npy")
            visibility_file = os.path.join(data_dir, f"visibility_{index}.npy")
            if os.path.exists(ifft_file) and os.path.exists(visibility_file):
                self.valid_samples.append((scene_file, ifft_file, visibility_file, index))

        print(f"Found {len(self.valid_samples)} valid scene/ifft/visibility samples")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        scene_path, ifft_path, visibility_path, index = self.valid_samples[idx]
        scene_tensor = prepare_image_tensor(np.load(scene_path))
        ifft_tensor = prepare_image_tensor(np.load(ifft_path))
        visibility_tensor = prepare_visibility_tensor(np.load(visibility_path))
        array_tensor = self.array_tensor.clone()

        if self.normalize:
            scene_tensor = normalize_tensor(scene_tensor)
            ifft_tensor = normalize_tensor(ifft_tensor)
            visibility_tensor = normalize_tensor(visibility_tensor)

        return scene_tensor, ifft_tensor, visibility_tensor, array_tensor, index


def build_noise_scheduler(device):
    betas = diffusion.get_beta_schedule(1000, schedule_type="sqrt")
    return diffusion.NoiseScheduler(betas, device)


def predict_and_save(model, ema, dataset, output_dir, device="cuda"):
    model.eval()
    model.to(device)
    os.makedirs(output_dir, exist_ok=True)

    predictions = []
    ground_truths = []
    inputs = []
    indices = []

    noise_scheduler = build_noise_scheduler(device)

    with torch.no_grad():
        for scene, ifft_img, visibility, array_info, index in tqdm(dataset, desc="Predicting"):
            scene = scene.unsqueeze(0).to(device)
            ifft_img = ifft_img.unsqueeze(0).to(device)
            visibility = visibility.unsqueeze(0).to(device)
            array_info = array_info.unsqueeze(0).to(device)

            if ema is not None:
                ema.apply_shadow()

            condition = {
                "ifft_cond": ifft_img,
                "visibility": visibility,
                "antenna_xy": array_info,
            }
            pre_noise, predicted = noise_scheduler.ddim_sampling(model, scene, condition, steps=1000, eta=0)

            if ema is not None:
                ema.restore()

            predictions.append(predicted.cpu().numpy())
            ground_truths.append(scene.cpu().numpy())
            inputs.append(ifft_img.cpu().numpy())
            indices.append(index)

            np.savez(
                os.path.join(output_dir, f"predicted_{index}.npz"),
                sample_id=np.array(index),
                outputs=predicted.squeeze(0).cpu().numpy(),
                observe=ifft_img.squeeze(0).cpu().numpy(),
                scene=scene.squeeze(0).cpu().numpy(),
                noise=pre_noise.squeeze(0).cpu().numpy(),
                visibility=visibility.squeeze(0).cpu().numpy(),
            )

    print(f"\nPredictions saved to {output_dir}")
    return (
        np.concatenate(predictions, axis=0),
        np.concatenate(ground_truths, axis=0),
        np.concatenate(inputs, axis=0),
        indices,
    )


def calculate_ssim(pred, target, data_range=1.0):
    from skimage.metrics import structural_similarity as ssim

    values = []
    for i in range(pred.shape[0]):
        if pred[i].ndim == 3 and pred[i].shape[0] == 3:
            pred_img = pred[i].transpose(1, 2, 0)
            target_img = target[i].transpose(1, 2, 0)
        else:
            pred_img = pred[i].squeeze(0)
            target_img = target[i].squeeze(0)

        if pred_img.ndim == 3 and pred_img.shape[-1] == 3:
            value = ssim(pred_img, target_img, data_range=data_range, channel_axis=2, win_size=11, gaussian_weights=True)
        else:
            value = ssim(pred_img, target_img, data_range=data_range, win_size=11, gaussian_weights=True)
        values.append(value)

    return np.mean(values), np.std(values), values


class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        if Inception_V3_Weights is not None:
            inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
        else:
            inception = inception_v3(pretrained=True, aux_logits=True)
        inception.eval()
        self.features = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - 0.5) * 2
        x = self.features(x)
        return torch.flatten(x, start_dim=1)


def ensure_feature_matrix(features):
    features = np.asarray(features, dtype=np.float64)
    if features.ndim == 1:
        features = features[None, :]
    return features


def calculate_fid(real_features, fake_features):
    real_features = ensure_feature_matrix(real_features)
    fake_features = ensure_feature_matrix(fake_features)

    if real_features.shape[0] < 2 or fake_features.shape[0] < 2:
        print(
            "Warning: FID requires at least 2 samples per set. "
            f"real_features shape={real_features.shape}, "
            f"fake_features shape={fake_features.shape}. Returning NaN."
        )
        return float("nan")

    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)


def extract_features(images, inception_model, device="cuda", batch_size=32):
    inception_model.eval()
    inception_model.to(device)
    features = []
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Extracting features"):
            batch = images[i:i + batch_size]
            if isinstance(batch, np.ndarray):
                if batch.ndim == 4 and batch.shape[-1] == 3:
                    batch = torch.from_numpy(batch.transpose(0, 3, 1, 2)).float()
                else:
                    batch = torch.from_numpy(batch).float()
            batch = batch.to(device)
            batch_features = inception_model(batch).cpu().numpy()
            features.append(ensure_feature_matrix(batch_features))
    return np.concatenate(features, axis=0)


def save_evaluation_results(mean_ssim, std_ssim, fid, all_ssim, output_dir):
    results = {
        "ssim_mean": float(mean_ssim),
        "ssim_std": float(std_ssim),
        "fid_score": float(fid),
        "ssim_values": all_ssim,
    }
    np.save(os.path.join(output_dir, "evaluation_results.npy"), results)

    with open(os.path.join(output_dir, "evaluation_results.txt"), "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("Model Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"SSIM Mean: {mean_ssim:.6f}\n")
        f.write(f"SSIM Std: {std_ssim:.6f}\n")
        f.write(f"FID Score: {fid:.2f}\n\n")
        f.write("=" * 50 + "\n")
        f.write("Individual SSIM Values:\n")
        f.write("=" * 50 + "\n")
        for i, ssim_val in enumerate(all_ssim):
            f.write(f"Sample {i + 1}: {ssim_val:.6f}\n")


def build_inference_summary(
    run_layout,
    model_path,
    data_dir,
    array_path,
    output_dir,
    dataset,
    params,
    mean_ssim,
    std_ssim,
    fid_score,
    all_ssim,
):
    sample_shapes = None
    if len(dataset) > 0:
        sample_scene, sample_ifft, sample_visibility, sample_array, _ = dataset[0]
        sample_shapes = {
            "scene": list(sample_scene.shape),
            "ifft": list(sample_ifft.shape),
            "visibility": list(sample_visibility.shape),
            "array": list(sample_array.shape),
        }

    return {
        "title": f"Inference Summary - {run_layout['run_name']}",
        "run_name": run_layout["run_name"],
        "run_dir": run_layout["run_dir"],
        "model_path": model_path,
        "data_dir": data_dir,
        "array_path": array_path,
        "output_dir": output_dir,
        "dataset_size": len(dataset),
        "sample_shapes": sample_shapes,
        "inference_setup": {
            "architecture": "ConditionalUNet",
            "sampler": "ddim",
            "sampling_steps": 250,
            "eta": 0,
            "normalize_inputs": True,
        },
        "metrics": {
            "ssim_mean": float(mean_ssim),
            "ssim_std": float(std_ssim),
            "fid_score": float(fid_score),
            "ssim_values": [float(x) for x in all_ssim],
        },
        "params": get_params_snapshot(params),
    }


def evaluate_model(model_path, data_dir, run_layout, params, device="cuda"):
    print("=" * 60)
    print("Starting Model Evaluation")
    print("=" * 60)

    print("\n1. Loading model...")
    model, ema = load_model(
        model_path,
        in_channels=1,
        out_channels=1,
        architecture="ConditionalUNet",
        ifft_channels=1,
        visibility_channels=2,
    )
    if model is None:
        return

    print("\n2. Preparing data...")
    dataset = PredictDataset(data_dir, normalize=True)
    if len(dataset) == 0:
        print("No data found!")
        return

    output_dir = run_layout["inference_dir"]

    print("\n3. Making predictions...")
    predictions, ground_truths, inputs, indices = predict_and_save(model, ema, dataset, output_dir, device)

    print("\n4. Calculating SSIM...")
    mean_ssim, std_ssim, all_ssim = calculate_ssim(predictions, ground_truths)
    print(f"SSIM: {mean_ssim:.4f} +/- {std_ssim:.4f}")

    print("\n5. Calculating FID (this may take a while)...")
    inception_model = InceptionV3()
    real_features = extract_features(ground_truths, inception_model, device, batch_size=16)
    fake_features = extract_features(predictions, inception_model, device, batch_size=16)
    print(f"Real feature matrix shape: {real_features.shape}")
    print(f"Fake feature matrix shape: {fake_features.shape}")
    fid_score = calculate_fid(real_features, fake_features)
    print(f"FID Score: {fid_score:.2f}")

    print("\n6. Saving evaluation results...")
    save_evaluation_results(mean_ssim, std_ssim, fid_score, all_ssim, output_dir)

    inference_summary = build_inference_summary(
        run_layout=run_layout,
        model_path=model_path,
        data_dir=data_dir,
        array_path=dataset.array_path,
        output_dir=output_dir,
        dataset=dataset,
        params=params,
        mean_ssim=mean_ssim,
        std_ssim=std_ssim,
        fid_score=fid_score,
        all_ssim=all_ssim,
    )
    summary_json, summary_txt = write_summary_files(
        output_dir,
        f"{run_layout['run_name']}_inference_summary",
        inference_summary,
    )
    print(f"Inference summary saved to: {summary_json}")
    print(f"Inference summary text saved to: {summary_txt}")

    return {
        "ssim_mean": mean_ssim,
        "ssim_std": std_ssim,
        "fid": fid_score,
        "predictions": predictions,
        "ground_truths": ground_truths,
        "indices": indices,
    }


def main():
    params = params_all
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = params.model_dir
    if not os.path.exists(model_path):
        fallback_model = find_latest_checkpoint(params.runs_dir, checkpoint_name="best.pth")
        if fallback_model is not None:
            print(f"Configured model path not found. Falling back to latest checkpoint: {fallback_model}")
            model_path = fallback_model
    data_dir = params.test_dir
    print(f"Model path: {model_path}")
    print(f"Inference data directory: {data_dir}")

    data_name = os.path.basename(os.path.normpath(data_dir))
    run_layout = resolve_run_layout_for_model(model_path, params.runs_dir, data_name=data_name)
    print(f"Run directory: {run_layout['run_dir']}")
    print(f"Inference output directory: {run_layout['inference_dir']}")

    evaluate_model(model_path, data_dir, run_layout, params, device)


if __name__ == "__main__":
    main()
