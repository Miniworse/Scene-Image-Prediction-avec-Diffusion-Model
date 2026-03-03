import numpy as np
import matplotlib.pyplot as plt
import os


def save_comparison_png(original, processed, output_path="comparison.png"):
    """
    简洁的并排对比图，保存为PNG
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # 绘制图像
    im1 = ax1.imshow(original, cmap='viridis')
    ax1.set_title('Scene')
    ax1.axis('off')

    im2 = ax2.imshow(processed, cmap='viridis')
    ax2.set_title('Observe')
    ax2.axis('off')

    # 添加colorbar
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
