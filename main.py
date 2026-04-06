import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import os
import datetime


def generate_scene(size=256, max_shapes=20, max_attempts=1000, min_size= 10, max_size= 50):
    """
    生成一个包含随机形状的场景

    参数:
    - size: 图像大小 (size x size)
    - max_shapes: 最大形状数量
    - max_attempts: 放置形状的最大尝试次数
    - min_size, max_size: 放置形状随机大小范围
    """
    # 创建背景 (值为0)
    image = np.zeros((size, size), dtype=np.uint8)

    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

    # 定义形状类型和颜色映射
    shape_types = ['square', 'circle', 'rectangle', 'triangle']

    # 用于存储已放置形状的位置和大小，避免重叠
    placed_shapes = []
    shapes_created = 0

    for _ in range(max_shapes):
        if shapes_created >= max_shapes:
            break

        for attempt in range(max_attempts):
            # 随机选择形状类型
            shape_type = random.choice(shape_types)

            # 随机大小 (最小10像素，最大50像素)
            # min_size, max_size = 10, 50
            size1 = random.randint(min_size, max_size)
            size2 = random.randint(min_size, max_size)

            # 随机位置
            x = random.randint(0, size - max_size - 1)
            y = random.randint(0, size - max_size - 1)

            # 检查是否与现有形状重叠
            overlap = False
            for placed in placed_shapes:
                px, py, psize = placed
                # 简单的边界框重叠检查
                if not (x + size1 < px or x > px + psize or
                        y + size2 < py or y > py + psize):
                    overlap = True
                    break

            if not overlap:
                # 将形状添加到图像 (值为1)
                if shape_type == 'square':
                    image[y:y + size1, x:x + size1] = 1
                    rect = patches.Rectangle((x, y), size1, size1,
                                             linewidth=1, edgecolor='white',
                                             facecolor='white', alpha=0.7)
                    ax.add_patch(rect)
                    placed_shapes.append((x, y, size1))

                elif shape_type == 'circle':
                    radius = size1 // 2
                    center_x, center_y = x + radius, y + radius

                    # 创建一个圆形掩码
                    for i in range(size):
                        for j in range(size):
                            if (i - center_y) ** 2 + (j - center_x) ** 2 <= radius ** 2:
                                if 0 <= i < size and 0 <= j < size:
                                    image[i, j] = 1

                    circle = patches.Circle((center_x, center_y), radius,
                                            linewidth=1, edgecolor='white',
                                            facecolor='white', alpha=0.7)
                    ax.add_patch(circle)
                    placed_shapes.append((x, y, size1))

                elif shape_type == 'rectangle':
                    image[y:y + size1, x:x + size2] = 1
                    rect = patches.Rectangle((x, y), size2, size1,
                                             linewidth=1, edgecolor='white',
                                             facecolor='white', alpha=0.7)
                    ax.add_patch(rect)
                    placed_shapes.append((x, y, max(size1, size2)))

                elif shape_type == 'triangle':
                    # 创建一个三角形掩码
                    vertices = [(x, y), (x + size1, y), (x + size1 // 2, y + size1)]
                    triangle = patches.Polygon(vertices,
                                               linewidth=1, edgecolor='white',
                                               facecolor='white', alpha=0.7)
                    ax.add_patch(triangle)

                    # 简单的方法填充三角形区域
                    for i in range(y, min(y + size1 + 1, size)):
                        for j in range(x, min(x + size1 + 1, size)):
                            # 检查点是否在三角形内 (简单方法)
                            if (i - y) <= 2 * (j - x) / size1 * size1 and \
                                    (i - y) <= 2 * (1 - (j - x) / size1) * size1:
                                if i < size and j < size:
                                    image[i, j] = 1

                    placed_shapes.append((x, y, size1))

                shapes_created += 1
                break

    # 设置坐标轴
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # 使y轴方向与图像坐标一致

    # 设置背景为黑色，形状为白色
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # 隐藏坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # plt.title(f'Random Shapes Scene ({shapes_created} shapes)',
    #           color='white', fontsize=12)
    plt.tight_layout()

    return image, fig


if __name__ == '__main__':
    # 生成场景
    print("生成随机形状场景...")

    save_dir = 'data\\data20Mars\\Binary'
    os.makedirs(save_dir, exist_ok=True)

    # 生成数据总数
    total_scenes = 10

    for scene_num in range(1, total_scenes + 1):
        print(f"\r正在生成第 {scene_num}/{total_scenes} 个场景...", end="")

        # 生成场景
        image_data, fig = generate_scene(size=256, max_shapes=1, min_size= 80, max_size=120)
        # image_data, fig = generate_scene(size=256, max_shapes=random.randint(8, 20))

        # 保存图像
        image_path = os.path.join(save_dir, f'scene_{scene_num:04d}.png')
        fig.savefig(image_path, facecolor='black', dpi=100, bbox_inches='tight')

        # 保存numpy数据
        data_path = os.path.join(save_dir, f'scene_{scene_num:04d}.npy')
        np.save(data_path, image_data)

        # 关闭图形以释放内存
        plt.close(fig)

    print(f"\n完成！1000个场景已保存到 '{save_dir}' 目录")
    print(f"每个场景包含：")
    print(f"  - PNG图像文件 (scene_XXXX.png)")
    print(f"  - numpy数据文件 (scene_XXXX.npy)")

    # 可选：保存统计信息
    stats_file = os.path.join(save_dir, 'statistics.txt')
    with open(stats_file, 'w') as f:
        f.write("场景统计信息\n")
        f.write("=" * 50 + "\n")
        f.write(f"总场景数: {total_scenes}\n")
        f.write(f"图像尺寸: 256x256\n")
        f.write(f"背景像素值: 0\n")
        f.write(f"形状像素值: 1\n")
        f.write(f"保存时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"统计信息已保存到: {stats_file}")

