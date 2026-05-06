import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import glob
import random
import math
from datetime import datetime

from argparse import ArgumentParser
from scripts.params import params_all
from scripts.utils import *
from scripts.Unet_model import UNet, load_pretrained_model, save_model

import scripts.diffusion as diffusion
from  scripts.cal_loss import compute_loss
from scripts.get_EMA import EMA

# 2. 自定义数据集类 - 适配您的.npy文件格式
class SceneObserveDataset(Dataset):
    def __init__(self, data_dir, indices, transform=None, normalize=True):
        """
        Args:
            data_dir: 数据目录路径
            indices: 要使用的数据索引列表
            transform: 数据增强转换
            normalize: 是否归一化数据
        """
        self.data_dir = data_dir
        self.indices = indices
        self.transform = transform
        self.normalize = normalize

        # 提取所有可用的scene文件
        self.scene_files = sorted(glob.glob(os.path.join(data_dir, "scene_*.npy")))

        # 验证每个scene文件是否有对应的observe文件
        self.valid_pairs = []
        for scene_file in self.scene_files:
            # 提取序号
            base_name = os.path.basename(scene_file)
            index = base_name.replace("scene_", "").replace(".npy", "")
            observe_file = os.path.join(data_dir, f"observe_{index}.npy")

            if os.path.exists(observe_file):
                self.valid_pairs.append((scene_file, observe_file))

        print(f"Found {len(self.valid_pairs)} valid scene-observe pairs")

        # 只使用指定的indices
        if indices is not None:
            self.valid_pairs = [self.valid_pairs[i] for i in indices if i < len(self.valid_pairs)]

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        scene_path, observe_path = self.valid_pairs[idx]

        # 加载.npy文件
        scene_data = np.load(scene_path)  # 干净图像
        observe_data = np.load(observe_path)  # 加噪图像

        # 检查数据形状并确保为3通道, 不必要
        if len(scene_data.shape) == 2:  # 如果是单通道灰度图
            scene_data = np.expand_dims(scene_data, axis=0)
            observe_data = np.expand_dims(observe_data, axis=0)


            # scene_data = np.stack([scene_data] * 3, axis=0)
            # observe_data = np.stack([observe_data] * 3, axis=0)
        elif len(scene_data.shape) == 3:
            # 确保通道在第一个维度 (C, H, W)
            if scene_data.shape[0] != 3:
                scene_data = scene_data.transpose(2, 0, 1)
                observe_data = observe_data.transpose(2, 0, 1)

        # 转换为浮点张量
        scene_tensor = torch.from_numpy(scene_data).float()
        observe_tensor = torch.from_numpy(observe_data).float()

        # 数据归一化（可选）
        if self.normalize:

            # scene_tensor = torch.clamp(scene_tensor / 500.0, -1.0, 1.0)
            # observe_tensor = torch.clamp(observe_tensor / 500.0, -1.0, 1.0)
            # 在辐射计场景下的归一化操作？
            # scene_tensor = t_normalize(scene_tensor)
            # observe_tensor = t_normalize(observe_tensor)

            scene_tensor = (scene_tensor - scene_tensor.mean()) / scene_tensor.std()
            observe_tensor = (observe_tensor - observe_tensor.mean()) / observe_tensor.std()

        # 数据增强（如果指定）
        if self.transform:
            # 这里可以添加随机裁剪、翻转等增强
            scene_tensor = self.transform(scene_tensor)
            observe_tensor = self.transform(observe_tensor)

        return scene_tensor, observe_tensor


# 3. 数据划分函数
def split_dataset(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    划分数据集为训练集、验证集和测试集
    """
    # 确保比例和为1
    assert train_ratio + val_ratio + test_ratio == 1.0, "比例之和必须为1"

    # 获取所有scene文件
    scene_files = sorted(glob.glob(os.path.join(data_dir, "scene_*.npy")))

    # 验证每个scene文件是否有对应的observe文件
    valid_indices = []
    for i, scene_file in enumerate(scene_files):
        base_name = os.path.basename(scene_file)
        index = base_name.replace("scene_", "").replace(".npy", "")
        observe_file = os.path.join(data_dir, f"observe_{index}.npy")

        if os.path.exists(observe_file):
            valid_indices.append(i)

    print(f"Total valid pairs: {len(valid_indices)}")

    # 划分数据集
    train_val_indices, test_indices = train_test_split(
        valid_indices, test_size=test_ratio, random_state=seed
    )

    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_ratio / (train_ratio + val_ratio), random_state=seed
    )

    print(f"Train set: {len(train_indices)} samples")
    print(f"Validation set: {len(val_indices)} samples")
    print(f"Test set: {len(test_indices)} samples")

    return train_indices, val_indices, test_indices


# 4. 训练函数
def train_model(model, train_loader, val_loader, model_dir, log_dir, epochs=50, lr=0.001, load_pretrained=False, device='cpu'):

    writer = SummaryWriter(log_dir=log_dir)

    ema = EMA(model, decay=0.999, device=device)
    model = model.to(device)
    criterion = nn.MSELoss()  # 使用均方误差损失

    # criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    start_epoch = 0
    train_losses = []
    val_losses = []
    pre_std = []
    t_index = []

    best_val_loss = float('inf')
    best_model_state = None

    if load_pretrained:
        model, ema, start_epoch, best_val_loss = load_pretrained_model(
            model, ema, model_dir, device,
            load_optimizer=True, optimizer=optimizer  # 传入已创建的optimizer
        )
        # 注意：如果保存的epoch是50，通常要从51开始训练
        if start_epoch >= 0:
            start_epoch = start_epoch
            print(f"Resuming training from epoch {start_epoch + 1}")

    T = 1000  # Total timesteps
    betas = diffusion.get_beta_schedule(T, schedule_type='sqrt')
    noise_scheduler = diffusion.NoiseScheduler(betas, device=device)

    for epoch in range(start_epoch, epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_idx, (scene_imgs, observe_imgs) in enumerate(train_loader):
            scene_imgs, observe_imgs = scene_imgs.to(device), observe_imgs.to(device)

            # Here add noise to the
            batch_size = len(scene_imgs)
            t = torch.randint(0, T, (batch_size,)).to(device)
            # Bias toward high noise to force the model to work harder on "blurry" images
            # t = torch.pow(torch.rand((batch_size,)), 0.7) * 1000
            # t = t.long().to(device)

            xt, noise = noise_scheduler.forward_diffusion(scene_imgs, t)

            # observe_noisy = noise_scheduler.add_noise(observe_imgs, t, torch.randn_like(observe_imgs))
            observe_noisy = observe_imgs
            xt_observe_cat = torch.cat([xt, observe_noisy], dim=1)

            # t_m = t.float() / noise_scheduler.T
            outputs = model(xt_observe_cat, t)
            velocity = noise_scheduler.get_velocity(scene_imgs, noise, t)

            # snr = noise_scheduler.get_snr(t)
            # snr = snr.view(-1, 1, 1, 1)
            # weight = torch.clamp(snr, max=5.0)
            # weight = weight + 0.1
            snr = noise_scheduler.get_snr(t).view(-1,1,1,1)
            weight = snr / (snr + 1)

            loss = (weight * (outputs - velocity) ** 2).mean()
            # loss = criterion(outputs, velocity)

            # for i in [0, 100, 300, 600, 900]:
            #     t_test = torch.full((batch_size,), i, device=device)
            #     xt, noise = noise_scheduler.forward_diffusion(scene_imgs, t_test)
            #     pred = model(torch.cat([xt, xt], dim=1), t_test)
            #     print(i, pred.std().item())
            # bins = [0, 200, 400, 600, 800, 1000]
            #
            # for i in range(len(bins) - 1):
            #     mask = (t >= bins[i]) & (t < bins[i + 1])
            #     if mask.sum() > 0:
            #         print(f"{bins[i]}-{bins[i + 1]}:", outputs[mask].std().item())

            # print("t:", t[:5])
            # # print("snr:", snr[:5].flatten())
            # print("pred std:", outputs.std().item())

            pre_x_0 = noise_scheduler.deblur(xt, t, noise)


            # pred = model(torch.cat([observe_imgs, observe_imgs], dim=1), torch.zeros_like(t))
            # loss = criterion(pred, scene_imgs)
            # predicted_noise = pred
            # xt = observe_imgs
            # outputs = scene_imgs

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            ema.update()
            train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch + 1}/{epochs} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.6f}')

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        # pre_std.append(outputs.std().item())
        # t_index.append(t)

        # # 验证阶段
        # model.eval()
        # val_loss = 0.0
        # with torch.no_grad():
        #     for scene_imgs, observe_imgs in val_loader:
        #         scene_imgs, observe_imgs = scene_imgs.to(device), observe_imgs.to(device)
        #         outputs = model(scene_imgs)
        #         loss = criterion(outputs, observe_imgs)
        #         val_loss += loss.item()
        #
        # avg_val_loss = val_loss / len(val_loader)
        # val_losses.append(avg_val_loss)

        # 保存最佳模型
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     best_model_state = model.state_dict().copy()
        #     torch.save(model.state_dict(), './model/scene_to_observe_model.pth')
        #     print(f'New best model saved with validation loss: {best_val_loss:.6f}')

        if epoch % 10 == 0:
            best_val_loss = avg_train_loss
            save_model(model, ema, model_dir, epoch=epoch, optimizer=optimizer, best_val_loss=best_val_loss)

            # predicted_noise = noise_scheduler.deblur(xt[0], t[0].item(),outputs[0])
            # v-prediction
            at = noise_scheduler.alphas_cumprod[t].view(-1, 1, 1, 1)
            predicted_noise = torch.sqrt(at[0]) * xt[0] - torch.sqrt(1 - at[0]) * outputs[0]

            scene_save = scene_imgs[0].cpu().detach().numpy()
            x_t_save = xt[0].cpu().detach().numpy()
            outputs_save = outputs[0].cpu().detach().numpy()
            predicted_save = predicted_noise.cpu().detach().numpy()
            t_value = t[0].item()

            os.makedirs(log_dir, exist_ok=True)
            np.savez(os.path.join(log_dir, f"train_{epoch}.npz"), **{f"x_{t_value}": x_t_save}, outputs = outputs_save, predicted = predicted_save, scene = scene_save)


        scheduler.step(avg_train_loss)

        print(
            f'Epoch: {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.6f}  | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 60)

    writer.close()
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'training_curve.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # # 查看std曲线 - 双Y轴
    # fig, ax1 = plt.subplots(figsize=(10, 5))
    #
    # # 左Y轴：t_index
    # color1 = 'tab:blue'
    # ax1.set_xlabel('Epoch')
    # ax1.set_ylabel('t_index', color=color1)
    # ax1.plot(t_index, label='t_index', color=color1)
    # ax1.tick_params(axis='y', labelcolor=color1)
    # ax1.grid(True, alpha=0.3)
    #
    # # 右Y轴：pre_std
    # ax2 = ax1.twinx()
    # color2 = 'tab:orange'
    # ax2.set_ylabel('pre_std', color=color2)
    # ax2.plot(pre_std, label='pre_std', color=color2)
    # ax2.tick_params(axis='y', labelcolor=color2)
    #
    # # 图例合并
    # lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    #
    # plt.title('pre std of t')
    # plt.savefig(os.path.join(log_dir, 'prestd_curve.png'), dpi=300, bbox_inches='tight')
    # plt.show()

    return model, ema, optimizer


# 5. 测试和可视化函数
def test_and_visualize(model, ema, test_loader, log_dir, device='cuda', num_samples=4):
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0.0
    betas = diffusion.get_beta_schedule(1000)
    noise_scheduler = diffusion.NoiseScheduler(betas, device)

    flag = True
    with torch.no_grad():
        for index, (scene_imgs, observe_imgs) in enumerate(test_loader):
            scene_imgs, observe_imgs = scene_imgs.to(device), observe_imgs.to(device)


            # pre_noise, predicted = noise_scheduler.native_sampling2(model, scene_imgs, observe_imgs, flag=flag)
            pre_noise, predicted = noise_scheduler.fast_sampling(model, scene_imgs, observe_imgs)
            # pre_noise, predicted = noise_scheduler.ddim_sample(model, scene_imgs, observe_imgs)


            flag = False
            loss = criterion(predicted, scene_imgs)
            test_loss += loss.item()

            scene_np = scene_imgs[0].cpu().numpy()
            observe_np = observe_imgs[0].cpu().numpy()
            predicted_np = predicted[0].detach().squeeze(0).cpu().numpy()
            noise_np = pre_noise[0].detach().squeeze(0).cpu().numpy()

            np.savez(os.path.join(log_dir, f"tested_{index}.npz"), outputs=predicted_np,
                     observe=observe_np, scene=scene_np, noise=noise_np)

    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss:.6f}')

    # # 可视化一些结果
    # scene_imgs, observe_imgs = next(iter(test_loader))
    # scene_imgs, observe_imgs = scene_imgs.to(device), observe_imgs.to(device)
    # for index, (scene_imgs, observe_imgs) in enumerate(test_loader):
    #     pre_noise, predicted = noise_scheduler.native_sampling2(model, scene_imgs, observe_imgs)
    #     predicted = model(scene_imgs[:num_samples])  # 预测前几个样本
    #
    #
    # # 转换为numpy用于显示
    # scene_np = scene_imgs[:num_samples].cpu().numpy()
    # observe_np = observe_imgs[:num_samples].cpu().numpy()
    # predicted_np = predicted.detach().cpu().numpy()
    #
    # # 如果通道在第一个维度，调整为最后一个维度用于显示
    # if scene_np.shape[1] == 3:  # (B, C, H, W)
    #     scene_np = scene_np.transpose(0, 2, 3, 1)
    #     observe_np = observe_np.transpose(0, 2, 3, 1)
    #     predicted_np = predicted_np.transpose(0, 2, 3, 1)
    #
    # # 显示结果
    # fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    # if num_samples == 1:
    #     axes = axes.reshape(1, -1)
    #
    # for i in range(num_samples):
    #     # 确保值在[0,1]范围内
    #     scene_img = np.clip(scene_np[i], 0, 1)
    #     observe_img = np.clip(observe_np[i], 0, 1)
    #     predicted_img = np.clip(predicted_np[i], 0, 1)
    #
    #     # 如果图像是单通道，转换为3通道显示
    #     if len(scene_img.shape) == 2:
    #         scene_img = np.stack([scene_img] * 3, axis=-1)
    #         observe_img = np.stack([observe_img] * 3, axis=-1)
    #         predicted_img = np.stack([predicted_img] * 3, axis=-1)
    #
    #     axes[i, 0].imshow(scene_img)
    #     axes[i, 0].set_title('Scene (Clean Input)')
    #     axes[i, 0].axis('off')
    #
    #     axes[i, 1].imshow(observe_img)
    #     axes[i, 1].set_title('Observe (Ground Truth Noisy)')
    #     axes[i, 1].axis('off')
    #
    #     axes[i, 2].imshow(predicted_img)
    #     axes[i, 2].set_title('Predicted Noisy')
    #     axes[i, 2].axis('off')
    #
    # plt.tight_layout()
    # plt.savefig('test_results.png', dpi=300, bbox_inches='tight')
    # plt.show()

    return avg_test_loss


# 6. 主函数
def main(args):

    params = params_all


    # # 设置随机种子以确保可重复性
    # torch.manual_seed(params.seed)
    # np.random.seed(params.seed)
    # random.seed(params.seed)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 数据目录路径
    data_dir = params.data_dir

    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"数据目录 {data_dir} 不存在，请检查路径！")
        return

    # 划分数据集
    train_indices, val_indices, test_indices = split_dataset(
        data_dir,
        train_ratio=params.train_ratio,
        val_ratio=params.val_ratio,
        test_ratio=params.test_ratio,
        seed=params.seed
    )

    # 创建数据集
    train_dataset = SceneObserveDataset(data_dir, train_indices, normalize=True)
    val_dataset = SceneObserveDataset(data_dir, val_indices, normalize=True)
    test_dataset = SceneObserveDataset(data_dir, test_indices, normalize=True)

    print(f"\nDataset sizes:")
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")

    # 检查一个样本的形状
    sample_scene, sample_observe = train_dataset[0]
    print(f"\nSample shape - Scene: {sample_scene.shape}, Observe: {sample_observe.shape}")

    # 创建数据加载器
    batch_size = params.batch_size
    num_workers = params.num_workers
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 确定输入通道数
    sample_scene, sample_observe = next(iter(train_loader))
    input_data =  torch.cat([sample_scene, sample_observe], dim=1)
    in_channels = input_data.shape[1]
    out_channels = sample_scene.shape[1]
    print(f"\nInput channels: {in_channels}")

    # 创建模型
    model = UNet(n_channels=in_channels, n_classes=out_channels, time_emb_dim=512)

    # 打印模型结构
    print(f"\nModel architecture:")
    print(model)
    print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')

    # 训练模型
    epochs = params.epochs
    learning_rate = params.learning_rate
    log_dir = params.log_dir
    load_pretrained = params.load_pretrained

    print(f"\nStarting training...")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")

    model_name = datetime.now().strftime("%b%d-%H%M%S")
    model_dir = os.path.join('./model', model_name + '.pth')

    if load_pretrained:
        model_dir = params.model_dir

    log_dir = os.path.join(log_dir, model_name)

    model, ema, optimizer = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_dir=model_dir,
        log_dir=log_dir,
        epochs=epochs,
        lr=learning_rate,
        load_pretrained=load_pretrained,
        device=str(device)
    )

    # 测试模型
    print(f"\nTesting model...")
    test_loss = test_and_visualize(model, ema, test_loader, log_dir, device, num_samples=num_workers)
    # print(f'Test loss: {test_loss: .6f} \n')

    # 保存模型
    save_model(model, ema, model_dir, epoch=epochs, optimizer=optimizer, best_val_loss=None)

    # # 保存完整模型（包括架构）
    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'in_channels': in_channels,
    #     'model_architecture': 'UNet'
    # }, 'scene_to_observe_model_full.pth')
    # print('Full model saved to scene_to_observe_model_full.pth')

    # # 保存训练历史（如果需要）
    # history = {
    #     'train_indices': train_indices,
    #     'val_indices': val_indices,
    #     'test_indices': test_indices,
    #     'input_channels': in_channels,
    #     # 'test_loss': test_loss
    # }
    # np.save('training_history.npy', history)
    # print('Training history saved to training_history.npy')


# 7. 数据检查函数
def check_data_statistics(data_dir):
    """检查数据集的统计信息"""
    print(f"Checking data statistics in {data_dir}...")

    # 获取所有scene文件
    scene_files = sorted(glob.glob(os.path.join(data_dir, "scene_*.npy")))

    if not scene_files:
        print("No scene files found!")
        return

    print(f"Found {len(scene_files)} scene files")

    # 检查前几个文件
    for i, scene_file in enumerate(scene_files[:3]):
        base_name = os.path.basename(scene_file)
        index = base_name.replace("scene_", "").replace(".npy", "")
        observe_file = os.path.join(data_dir, f"observe_{index}.npy")

        if os.path.exists(observe_file):
            scene_data = np.load(scene_file)
            observe_data = np.load(observe_file)

            print(f"\nFile {i + 1}: {base_name}")
            print(f"  Scene shape: {scene_data.shape}, dtype: {scene_data.dtype}")
            print(f"  Scene range: [{scene_data.min():.3f}, {scene_data.max():.3f}]")
            print(f"  Observe shape: {observe_data.shape}, dtype: {observe_data.dtype}")
            print(f"  Observe range: [{observe_data.min():.3f}, {observe_data.max():.3f}]")

            # 可视化第一个样本
            if i == 0:
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))

                if len(scene_data.shape) == 2:
                    axes[0].imshow(scene_data, cmap='gray')
                    axes[1].imshow(observe_data, cmap='gray')
                elif len(scene_data.shape) == 3 and scene_data.shape[2] == 3:
                    axes[0].imshow(scene_data)
                    axes[1].imshow(observe_data)
                elif len(scene_data.shape) == 3 and scene_data.shape[0] == 3:
                    axes[0].imshow(scene_data.transpose(1, 2, 0))
                    axes[1].imshow(observe_data.transpose(1, 2, 0))

                axes[0].set_title('Scene (Clean)')
                axes[1].set_title('Observe (Noisy)')
                plt.tight_layout()
                plt.show()
        else:
            print(f"  Warning: No corresponding observe file for {base_name}")


# 运行主程序
if __name__ == "__main__":
    parser = ArgumentParser(
        description='train (or resume training) a tfdiff model')
    parser.add_argument('--task_id', type=int,
                        help='use case of tfdiff model, 0/1/2/3 for WiFi/FMCW/MIMO/EEG respectively')
    parser.add_argument('--model_dir', default=None,
                        help='directory in which to store model checkpoints and training logs')
    parser.add_argument('--data_dir', default=None, nargs='+',
                        help='space separated list of directories from which to read csi files for training')
    parser.add_argument('--log_dir', default=None)
    parser.add_argument('--max_iter', default=None, type=int,
                        help='maximum number of training iteration')
    parser.add_argument('--batch_size', default=None, type=int)

    # 运行主训练程序
    main(parser.parse_args())