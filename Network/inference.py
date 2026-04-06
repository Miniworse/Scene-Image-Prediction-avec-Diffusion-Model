import torch
import torch.nn as nn
import numpy as np
import os
import glob
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from scipy import linalg
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
import warnings
import math

import scripts.diffusion as diffusion
from scripts.params import params_all
from scripts.utils import *
from scripts.Unet_model import load_model

warnings.filterwarnings('ignore')


# 2. 预测数据集
class PredictDataset:
    def __init__(self, data_dir, normalize=True):
        """用于预测的数据集"""
        self.data_dir = data_dir
        self.normalize = normalize

        # 获取所有scene文件
        self.scene_files = sorted(glob.glob(os.path.join(data_dir, "scene_*.npy")))

        # 验证每个scene文件是否有对应的observe文件
        self.valid_pairs = []
        for scene_file in self.scene_files:
            base_name = os.path.basename(scene_file)
            index = base_name.replace("scene_", "").replace(".npy", "")
            observe_file = os.path.join(data_dir, f"observe_{index}.npy")

            if os.path.exists(observe_file):
                self.valid_pairs.append((scene_file, observe_file, index))

        print(f"Found {len(self.valid_pairs)} valid scene-observe pairs")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        scene_path, observe_path, index = self.valid_pairs[idx]

        # 加载.npy文件
        scene_data = np.load(scene_path)
        observe_data = np.load(observe_path)

        # 处理数据形状
        if len(scene_data.shape) == 2:  # 单通道

            # scene_data = np.stack([scene_data] * 3, axis=0)
            # observe_data = np.stack([observe_data] * 3, axis=0)
            scene_data = np.expand_dims(scene_data, axis=0)
            observe_data = np.expand_dims(observe_data, axis=0)

        elif len(scene_data.shape) == 3:
            if scene_data.shape[0] != 3:  # 如果通道在最后一个维度
                scene_data = scene_data.transpose(2, 0, 1)
                observe_data = observe_data.transpose(2, 0, 1)

        # 转换为浮点张量
        scene_tensor = torch.from_numpy(scene_data).float()
        observe_tensor = torch.from_numpy(observe_data).float()

        # 归一化
        if self.normalize:
            # scene_tensor = torch.clamp(scene_tensor / 50.0 - 3.0, -3.0, 3.0)
            # observe_tensor = torch.clamp(observe_tensor / 50.0 - 3.0, -3.0, 3.0)
            # scene_tensor = t_normalize(scene_tensor)
            # observe_tensor = t_normalize(observe_tensor)
            scene_tensor = (scene_tensor - scene_tensor.mean()) / scene_tensor.std()
            observe_tensor = (observe_tensor - observe_tensor.mean()) / observe_tensor.std()

        return scene_tensor, observe_tensor, index


# 3. 预测函数
def predict_and_save(model, ema, dataset, output_dir, device='cuda'):
    """进行预测并保存结果"""
    model.eval()
    model.to(device)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    predictions = []
    ground_truths = []
    inputs = []
    indices = []

    betas = diffusion.get_beta_schedule(1000)
    noise_scheduler = diffusion.NoiseScheduler(betas, device)

    with torch.no_grad():
        for scene, observe, index in tqdm(dataset, desc="Predicting"):
            scene = scene.unsqueeze(0).to(device)  # 添加batch维度
            observe = observe.unsqueeze(0).to(device)
            if index == '0001':
                flag = True
            else:
                flag = False
            # 预测

            if ema is not None:
                ema.apply_shadow()

            # x_t =  torch.randn_like(scene).to(device)
            # pre_noise, predicted  = noise_scheduler.sampling(model, x_t, observe)
            # pre_noise, predicted = noise_scheduler.native_sampling2(model, scene, observe, flag)
            # pre_noise, predicted = noise_scheduler.fast_sampling(model, scene, observe)
            pre_noise, predicted = noise_scheduler.ddim_sampling(model, scene, observe, steps=1000, eta=1)

            if ema is not None:
                ema.restore()

            # 保存到列表
            predictions.append(predicted.cpu().numpy())
            ground_truths.append(scene.cpu().numpy())
            inputs.append(observe.cpu().numpy())
            indices.append(index)

            # 这里也保存为npz包含场景、观测、预测与预测噪声四个子图
            pred_save = predicted.squeeze(0).cpu().numpy()  # 移除batch维度
            scene_save = scene.squeeze(0).cpu().numpy()
            observe_save = observe.squeeze(0).cpu().numpy()
            noise_save = pre_noise.squeeze(0).cpu().numpy()

            np.savez(os.path.join(output_dir, f"predicted_{index}.npz"), outputs=pred_save,
                     observe=observe_save, scene=scene_save, noise=noise_save)

            # # 保存为.npy文件
            # pred_np = predicted.squeeze(0).cpu().numpy()  # 移除batch维度
            #
            # # 保存预测结果
            # pred_filename = f"predicted_{index}.npy"
            # pred_path = os.path.join(output_dir, pred_filename)
            # np.save(pred_path, pred_np)
            #
            # # 如果需要，也可以保存为图像
            # if pred_np.shape[0] == 3:  # 3通道图像
            #     img_np = pred_np.transpose(1, 2, 0)
            #     img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            #     img_filename = f"predicted_{index}.png"
            #     img_path = os.path.join(output_dir, img_filename)
            #     Image.fromarray(img_np).save(img_path)

    print(f"\nPredictions saved to {output_dir}")

    # 将所有预测结果合并
    predictions = np.concatenate(predictions, axis=0)
    ground_truths = np.concatenate(ground_truths, axis=0)
    inputs = np.concatenate(inputs, axis=0)

    return predictions, ground_truths, inputs, indices


# 4. SSIM评估函数
def calculate_ssim(pred, target, data_range=1.0):
    """计算SSIM（结构相似性指数）"""
    from skimage.metrics import structural_similarity as ssim

    batch_size = pred.shape[0]
    ssim_values = []

    for i in range(batch_size):
        # 处理数据形状
        if len(pred[i].shape) == 3 and pred[i].shape[0] == 3:  # (C, H, W)
            pred_img = pred[i].transpose(1, 2, 0)
            target_img = target[i].transpose(1, 2, 0)
        else:
            pred_img = pred[i].squeeze(0)
            target_img = target[i].squeeze(0)

        # 如果有多通道，计算多通道SSIM
        if len(pred_img.shape) == 3 and pred_img.shape[-1] == 3:
            ssim_val = ssim(pred_img, target_img,
                            data_range=data_range,
                            channel_axis=2,  # 通道在最后一个维度
                            win_size=11,  # 窗口大小
                            gaussian_weights=True)
        else:
            ssim_val = ssim(pred_img, target_img,
                            data_range=data_range,
                            win_size=11,
                            gaussian_weights=True)

        ssim_values.append(ssim_val)

    return np.mean(ssim_values), np.std(ssim_values), ssim_values


# 5. FID评估函数
class InceptionV3(nn.Module):
    """用于FID计算的InceptionV3特征提取器"""

    def __init__(self):
        super().__init__()
        inception = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        inception.eval()

        # 获取特定层的特征
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
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

    def forward(self, x):
        # 调整输入大小以适应InceptionV3
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # 灰度转RGB
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = (x - 0.5) * 2  # 归一化到[-1, 1]
        return self.features(x).squeeze()


def calculate_fid(real_features, fake_features):
    """计算FID分数"""
    # 计算均值和协方差
    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)

    # 计算平方和
    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    # 计算协方差矩阵的平方根
    covmean = linalg.sqrtm(sigma1.dot(sigma2))

    # 检查并修复虚部
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # 计算FID
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def extract_features(images, inception_model, device='cuda', batch_size=32):
    """从图像中提取特征"""
    inception_model.eval()
    inception_model.to(device)

    features = []

    # 将图像数据处理成适合InceptionV3的格式
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Extracting features"):
            batch = images[i:i + batch_size]

            # 处理数据形状
            if isinstance(batch, np.ndarray):
                if batch.shape[-1] == 3:  # (N, H, W, C) -> (N, C, H, W)
                    batch = torch.from_numpy(batch.transpose(0, 3, 1, 2)).float()
                else:
                    batch = torch.from_numpy(batch).float()

            batch = batch.to(device)

            # 提取特征
            batch_features = inception_model(batch)
            features.append(batch_features.cpu().numpy())

    return np.concatenate(features, axis=0)


# 6. 主评估函数
def evaluate_model(model_path, data_dir, output_dir, device='cuda'):
    """主评估函数"""
    print("=" * 60)
    print("Starting Model Evaluation")
    print("=" * 60)

    # 1. 加载模型
    print("\n1. Loading model...")
    model, ema = load_model(model_path, in_channels=2, out_channels=1)
    if model is None:
        return

    # 2. 准备数据
    print("\n2. Preparing data...")
    dataset = PredictDataset(data_dir, normalize=True)

    if len(dataset) == 0:
        print("No data found!")
        return

    # 3. 进行预测
    print("\n3. Making predictions...")
    predictions, ground_truths, inputs, indices = predict_and_save(
        model, ema, dataset, output_dir, device
    )

    # 4. 计算SSIM
    print("\n4. Calculating SSIM...")
    mean_ssim, std_ssim, all_ssim = calculate_ssim(predictions, ground_truths)
    print(f"SSIM: {mean_ssim:.4f} ± {std_ssim:.4f}")

    # 5. 计算FID（需要更长时间）
    print("\n5. Calculating FID (this may take a while)...")

    # 准备Inception模型
    inception_model = InceptionV3()

    # 将真实图像和预测图像处理成相同格式
    # 真实图像 (ground_truths)
    real_images = []
    for i in range(len(ground_truths)):
        img = ground_truths[i]
        if img.shape[0] == 3:  # (C, H, W) -> (H, W, C)
            img = img.transpose(1, 2, 0)
        real_images.append(img)
    real_images = np.array(real_images)

    # 预测图像
    fake_images = []
    for i in range(len(predictions)):
        img = predictions[i]
        if img.shape[0] == 3:  # (C, H, W) -> (H, W, C)
            img = img.transpose(1, 2, 0)
        fake_images.append(img)
    fake_images = np.array(fake_images)

    # 提取特征
    real_features = extract_features(real_images, inception_model, device, batch_size=16)
    fake_features = extract_features(fake_images, inception_model, device, batch_size=16)

    # 计算FID
    fid_score = calculate_fid(real_features, fake_features)
    print(f"FID Score: {fid_score:.2f}")

    # 6. 可视化结果
    # print("\n6. Visualizing results...")
    # visualize_results(inputs, ground_truths, predictions, output_dir, indices[:4])

    # 7. 保存评估结果
    print("\n7. Saving evaluation results...")
    save_evaluation_results(mean_ssim, std_ssim, fid_score, all_ssim, output_dir)

    return {
        'ssim_mean': mean_ssim,
        'ssim_std': std_ssim,
        'fid': fid_score,
        'predictions': predictions,
        'ground_truths': ground_truths,
        'indices': indices
    }


# 7. 可视化函数
def visualize_results(inputs, ground_truths, predictions, output_dir, indices, n_samples=4):
    """可视化结果"""
    n_samples = min(n_samples, len(inputs))

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        # 获取图像数据
        input_img = inputs[i]
        gt_img = ground_truths[i]
        pred_img = predictions[i]

        # 处理形状
        if input_img.shape[0] == 3:
            input_img = input_img.transpose(1, 2, 0)
            gt_img = gt_img.transpose(1, 2, 0)
            pred_img = pred_img.transpose(1, 2, 0)

        # 计算差异
        diff_img = np.abs(gt_img - pred_img)
        diff_img = diff_img / diff_img.max() if diff_img.max() > 0 else diff_img

        # 显示图像
        axes[i, 0].imshow(np.clip(input_img, 0, 1))
        axes[i, 0].set_title(f'Input (Scene)\nIndex: {indices[i]}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(np.clip(gt_img, 0, 1))
        axes[i, 1].set_title('Ground Truth (Observe)')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(np.clip(pred_img, 0, 1))
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')

        axes[i, 3].imshow(diff_img, cmap='hot')
        axes[i, 3].set_title('Difference (Heatmap)')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visualization_results.png'), dpi=300, bbox_inches='tight')
    plt.show()


# 8. 保存评估结果
def save_evaluation_results(mean_ssim, std_ssim, fid, all_ssim, output_dir):
    """保存评估结果到文件"""
    results = {
        'ssim_mean': float(mean_ssim),
        'ssim_std': float(std_ssim),
        'fid_score': float(fid),
        'ssim_values': all_ssim
    }

    # 保存为.npy文件
    np.save(os.path.join(output_dir, 'evaluation_results.npy'), results)

    # 保存为文本文件
    with open(os.path.join(output_dir, 'evaluation_results.txt'), 'w') as f:
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

    print(f"Evaluation results saved to {output_dir}")


# 9. 批量处理多个模型
def evaluate_multiple_models(model_paths, data_dir, output_base_dir, device='cuda'):
    """评估多个模型"""
    results = {}

    for model_path in model_paths:
        model_name = os.path.basename(model_path).replace('.pth', '')
        output_dir = os.path.join(output_base_dir, model_name)

        print(f"\n{'=' * 60}")
        print(f"Evaluating model: {model_name}")
        print(f"{'=' * 60}")

        result = evaluate_model(model_path, data_dir, output_dir, device)

        if result is not None:
            results[model_name] = result

    # 比较不同模型的结果
    if results:
        print("\n" + "=" * 60)
        print("Model Comparison")
        print("=" * 60)

        print("\n{:<20} {:<15} {:<15}".format("Model", "SSIM", "FID"))
        print("-" * 50)

        for model_name, result in results.items():
            print("{:<20} {:<15.4f} {:<15.2f}".format(
                model_name,
                result['ssim_mean'],
                result['fid']
            ))

        # 保存比较结果
        comparison_file = os.path.join(output_base_dir, 'model_comparison.txt')
        with open(comparison_file, 'w') as f:
            f.write("Model Comparison Results\n")
            f.write("=" * 50 + "\n\n")
            f.write("{:<20} {:<15} {:<15}\n".format("Model", "SSIM", "FID"))
            f.write("-" * 50 + "\n")
            for model_name, result in results.items():
                f.write("{:<20} {:<15.4f} {:<15.2f}\n".format(
                    model_name,
                    result['ssim_mean'],
                    result['fid']
                ))

        print(f"\nComparison results saved to {comparison_file}")

    return results


# 10. 主函数
def main():
    params = params_all

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 配置路径
    model_path = params.model_dir
    data_dir = params.test_dir
    output_dir = params.output_dir

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    data_name = os.path.basename(data_dir)  # 'TB'
    output_name = f"{model_name}_{data_name}"
    output_dir = os.path.join(output_dir, output_name)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 评估单个模型
    results = evaluate_model(model_path, data_dir, output_dir, device)

    # 如果需要评估多个模型
    # model_paths = [
    #     'model_v1.pth',
    #     'model_v2.pth',
    #     'model_v3.pth'
    # ]
    # results = evaluate_multiple_models(model_paths, data_dir, './model_comparisons', device)


# 11. 单独运行预测（不评估）
def run_prediction_only(model_path, data_dir, output_dir, device='cuda'):
    """只进行预测，不进行评估"""
    print("Running prediction only...")

    # 加载模型
    model, ema = load_model(model_path, in_channels=1)
    if model is None:
        return

    # 准备数据
    dataset = PredictDataset(data_dir, normalize=True)

    # 进行预测
    predictions, ground_truths, inputs, indices = predict_and_save(
        model, ema, dataset, output_dir, device
    )

    print(f"\nPrediction completed. {len(predictions)} samples saved to {output_dir}")
    return predictions, ground_truths, inputs, indices


if __name__ == "__main__":
    main()