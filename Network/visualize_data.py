import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm import tqdm


def visualize_and_save_npy_files(folder_path, save_dir="./visualizations", cmap='jet'):
    """
    批量可视化 [1, 256, 256] 的 npy 文件或 npz 文件并保存

    对于 npz 文件，假设包含四个变量，绘制在 2x2 子图中

    Args:
        folder_path: 包含 npy/npz 文件的文件夹路径
        save_dir: 保存图像的目录
        cmap: colormap 名称
    """
    # 获取所有 npy 和 npz 文件
    npy_files = list(Path(folder_path).glob("*.npy"))
    npz_files = list(Path(folder_path).glob("*.npz"))
    all_files = npy_files + npz_files

    if not all_files:
        print(f"文件夹 {folder_path} 中没有找到 .npy 或 .npz 文件")
        return

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    print(f"找到 {len(all_files)} 个文件（{len(npy_files)} 个 .npy, {len(npz_files)} 个 .npz）")
    print(f"图像将保存到: {save_dir}")

    # 使用 tqdm 显示进度条
    for file_path in tqdm(all_files, desc="处理中"):
        try:
            file_ext = file_path.suffix.lower()

            if file_ext == '.npy':
                # 处理 .npy 文件
                data = np.load(file_path)

                # 去掉第一个维度，得到 [256, 256]
                data_2d = data[0] if data.shape[0] == 1 else data

                # 创建单个图像
                plt.figure(figsize=(8, 6))
                im = plt.imshow(data_2d, cmap=cmap)
                plt.colorbar(im)
                plt.axis('off')

                # 保存
                save_path = os.path.join(save_dir, f"{file_path.stem}.png")
                plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
                plt.close()

            elif file_ext == '.npz':
                # 处理 .npz 文件
                loaded = np.load(file_path)

                # 获取所有变量名（排除以 '__' 开头的）
                var_names = [key for key in loaded.keys() if not key.startswith('__')]

                if len(var_names) == 0:
                    print(f"  警告: {file_path.name} 中没有变量")
                    continue

                # 创建 2x2 子图
                fig, axes = plt.subplots(2, 2, figsize=(14, 12))
                axes = axes.flatten()

                # 绘制每个变量
                for idx, var_name in enumerate(var_names[:4]):  # 最多显示4个
                    if idx >= 4:
                        break

                    var_data = loaded[var_name]

                    # 如果数据是 [1, 256, 256]，去掉第一个维度
                    if var_data.ndim == 3 and var_data.shape[0] == 1:
                        var_data_2d = var_data[0]
                    elif var_data.ndim == 4 and var_data.shape[0] == 1:
                        var_data_2d = var_data[0][0]
                    else:
                        var_data_2d = var_data

                    # 在子图中显示
                    ax = axes[idx]
                    im = ax.imshow(var_data_2d, cmap=cmap)
                    ax.set_title(f"{var_name}\n{var_data_2d.shape}", fontsize=10)
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # 隐藏多余的子图（如果变量少于4个）
                for idx in range(len(var_names), 4):
                    axes[idx].axis('off')

                plt.suptitle(f"{file_path.stem}", fontsize=14)
                plt.tight_layout()

                # 保存
                save_path = os.path.join(save_dir, f"{file_path.stem}.png")
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()

        except Exception as e:
            print(f"\n处理文件 {file_path.name} 时出错: {e}")
            plt.close()
            continue

    print(f"\n处理完成！所有图像已保存到: {save_dir}")
    print(f"共处理 {len(all_files)} 个文件")


# 使用示例
folder_path = "D:\Documents\Self_Files\Projects\SceneGenerating\\Network\\output\\t_sample_V2"  # 输入文件夹路径
save_dir = f"./output_images/{Path(folder_path).name}"  # 自动生成同名子目录
visualize_and_save_npy_files(folder_path, save_dir=save_dir)