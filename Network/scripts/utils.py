import torch
import numpy as np
import matplotlib.pyplot as plt

def t_normalize(TB, methods=1):
    """
    对输入 tensor 进行背景归一化（简洁版）

    Args:
        TB: 输入 tensor，形状为 [batch, height, width] 或 [height, width]
        methods: 方法选择

    Returns:
        TB_large: 背景矩阵
        TB_corrected: 去除背景后的 tensor
    """
    if methods != 1:
        raise ValueError(f"Unsupported method: {methods}")

    # 保存原始维度信息
    original_shape = TB.shape
    original_dim = TB.dim()

    # 如果是3D [batch, h, w]，保存batch大小
    if original_dim == 3:
        batch_size, h, w = original_shape
        # 展平batch维度进行处理
        TB_flat = TB  # 保持原样，后面用keepdim=True的mean
    else:
        h, w = original_shape
        batch_size = 1
        TB_flat = TB.unsqueeze(0)  # 添加batch维度统一处理

    # 收集所有边界像素
    # 上边界: [batch, w]
    top_edge = TB_flat[:, 0, :]
    # 下边界: [batch, w]
    bottom_edge = TB_flat[:, -1, :]
    # 左边界（不含角点）: [batch, h-2]
    left_edge = TB_flat[:, 1:-1, 0]
    # 右边界（不含角点）: [batch, h-2]
    right_edge = TB_flat[:, 1:-1, -1]

    # 计算每个batch的边界平均值
    # 先计算每个边界的和
    top_sum = top_edge.sum(dim=1)  # [batch]
    bottom_sum = bottom_edge.sum(dim=1)  # [batch]
    left_sum = left_edge.sum(dim=1)  # [batch]
    right_sum = right_edge.sum(dim=1)  # [batch]

    # 边界像素总数 = w + w + (h-2) + (h-2) = 2*(h + w - 2)
    total_pixels = 2 * (h + w - 2)

    # 计算平均值
    TB_avg = (top_sum + bottom_sum + left_sum + right_sum) / total_pixels  # [batch]

    # 创建背景矩阵
    if original_dim == 3:
        # [batch, h, w]
        TB_large = TB_avg.view(batch_size, 1, 1).expand(batch_size, h, w)
        TB_corrected = TB - TB_large
    else:
        # [h, w]
        TB_large = torch.full((h, w), TB_avg.item())
        TB_corrected = TB - TB_large

    eps = 1e-8
    std_val = torch.std(TB_corrected, dim=(1, 2))  # [batch]
    std_matrix = std_val.view(batch_size, 1, 1)
    TB_normalized = TB_corrected / (std_matrix + eps)

    return TB_normalized

def visualize(data_dir):
    data = np.load(data_dir)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x0_show = data.squeeze()
    # x0_show = np.transpose(x0_show,(1, 2, 0))
    # 在第一张子图上显示
    ax1.imshow(x0_show)
    ax1.set_title('Image 1')
    plt.colorbar()
    plt.show()