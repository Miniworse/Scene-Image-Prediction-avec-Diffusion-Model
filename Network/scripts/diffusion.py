import torch
import numpy as np
import os
from sympy.polys.matrices.dense import ddm_iinv


# Define the beta schedule (linear schedule for simplicity)
def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


class NoiseScheduler:
    def __init__(self, betas, device = torch.device('cpu')):
        self.betas = betas
        self.device = device
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod).to(device)

    def add_noise(self, x0, t, noise):
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

    def deblur(self, xt, t, noise):
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        # Estimate x0 from xt and noise
        x0 = (xt - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t
        return x0

    # Forward diffusion process
    def forward_diffusion(self, x0, t):
        noise = torch.randn_like(x0)
        xt = self.add_noise(x0, t, noise)
        return xt, noise

    def fast_sampling(self, model, x_t, y):
        t = torch.tensor([999]).to(self.device)
        x_t_y = torch.cat((x_t, y), dim=1)
        pre_noise = model(x_t_y, t)
        predicted = self.deblur(x_t, t, pre_noise)
        return pre_noise, predicted

    def sampling(self, model, x_t, y):
        for i in range(999, 0, -1):
            t = torch.tensor([i]).to(self.device)
            x_t_y = torch.cat((x_t, y), dim=1)
            pre_noise = model(x_t_y, t)
            x_0 = self.deblur(x_t, t, pre_noise)
            if i>0:
                x_t = self.add_noise(x_0, t = torch.tensor([i-1]), noise=pre_noise)
            predicted = x_0

        return pre_noise, predicted

    def native_sampling(self, model, x_0, y):
        t = torch.tensor([999]).to(self.device)
        x_t = self.add_noise(x_0, t, noise=torch.randn_like(x_0))
        x_t_y = torch.cat((x_t, y), dim=1)
        pre_noise = model(x_t_y, t)
        predicted = self.deblur(x_t, t, pre_noise)

        return pre_noise, predicted

    def native_sampling2(self, model, x_0, y, flag = False):
        x_t = self.add_noise(x_0, torch.tensor([999]), noise=torch.randn_like(x_0))

        for i in range(999, 0, -1):
            t = torch.tensor([i]).to(self.device)
            x_t_y = torch.cat((x_t, y), dim=1)
            pre_noise = model(x_t_y, t)
            x_0 = self.deblur(x_t, t, pre_noise)
            if i>0:
                x_t = self.add_noise(x_0, t = torch.tensor([i-1]), noise=pre_noise)
            predicted = x_0

            if i%10 == 0 & flag:
                # 写一个小接口，
                output_dir = f'./output/t_sample'
                os.makedirs(output_dir, exist_ok=True)
                pred_path = os.path.join(output_dir, f"x_{t.item()}.npy")
                np.save(pred_path, x_t[0].detach().cpu().numpy())


        return pre_noise, predicted


# # Example usage
# T = 1000  # Total timesteps
# betas = linear_beta_schedule(T)
# noise_scheduler = NoiseScheduler(betas)
#
# # Example image (batch_size=1, channels=3, height=32, width=32)
# x0 = torch.randn(1, 3, 32, 32)
#
# from PIL import Image
# x_0 = Image.open('../data/river18.tif')
# x_0 = np.array(x_0)
# x_0 = torch.from_numpy(x_0)
# x0_float = x_0.float() / 255.0
# t = torch.randint(0, T, (1,))
#
# # Perform forward diffusion
# xt, noise = forward_diffusion(x0_float, t, noise_scheduler)
#
# x_0_pre = noise_scheduler.deblur(xt, t, noise)

# print("Shape of noisy image (xt):", xt.shape)
# print("Shape of noise:", noise.shape)


# Visualize Function
# import matplotlib.pyplot as plt
#
# # 创建1行2列的子图
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
#
# x0_show = x_0_pre.squeeze()
# # x0_show = np.transpose(x0_show,(1, 2, 0))
# # 在第一张子图上显示
# ax1.imshow(x0_show)
# ax1.set_title('Image 1')
# ax1.axis('off')
#
# xt_show = xt.squeeze()
# # xt_show = np.transpose(xt_show,(1, 2, 0))
#
# # 在第二张子图上显示
# ax2.imshow(xt_show)
# ax2.set_title('Image 2')
# ax2.axis('off')
#
# plt.tight_layout()
# plt.show()
