import torch
import torch.nn as nn
import torch.nn.functional as F


def get_gradient_loss(pred, target):
    """Calculates the difference in image gradients (edges)"""
    # Horizontal gradients
    grad_x_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    grad_x_target = target[:, :, :, 1:] - target[:, :, :, :-1]
    # Vertical gradients
    grad_y_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    grad_y_target = target[:, :, 1:, :] - target[:, :, :-1, :]

    return F.mse_loss(grad_x_pred, grad_x_target) + F.mse_loss(grad_y_pred, grad_y_target)


def compute_loss(pred_noise, target_noise, pre_x_0, scene_imgs, noise_schedule, t, snr_cap=5.0):

    # 2. Per-element Huber Loss (do not reduce yet)
    # beta=1.0 is the standard for SmoothL1/Huber
    huber_criterion = nn.MSELoss()
    base_loss = huber_criterion(pred_noise, target_noise)  # Shape: [B, C, H, W]

    # 3. Calculate Min-SNR Weights
    # Formula: min(SNR, snr_cap) / SNR
    snr = noise_schedule.get_snr(t)
    mse_loss_weight = torch.clamp(snr, max=snr_cap) / snr

    # 4. Apply SNR weighting to the base Huber loss
    weighted_huber = (base_loss * mse_loss_weight).mean()

    # 5. Add Gradient (Edge) Loss
    # We apply this to ensure the "gentle clutters" on edges are penalized

    edge_loss = get_gradient_loss(pre_x_0, scene_imgs)

    # Total Loss (lambda=0.1 is usually a good starting balance for edges)
    total_loss = weighted_huber + 0.1 * edge_loss

    return total_loss


# --- Inside your Training Loop ---
# optimizer.zero_grad()
#
# # x_t_y is torch.cat([x_t, y], dim=1)
# loss = compute_loss(model, x_t_y, noise, t, scheduler.alphas_cumprod)
#
# loss.backward()
#
# # CRITICAL: Gradient Clipping to stop the "Enormous Value" spikes
# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#
# optimizer.step()