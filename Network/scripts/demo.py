for t in range(T):  # Including T
    # Create tensor of timestep t (with batch dimension)
    t_tensor = torch.tensor([t], dtype=torch.long).to(device)

    # Forward diffusion to get x_t
    xt, noise = noise_scheduler.forward_diffusion(scene_imgs, t_tensor)

    # Save image every 10 steps (or at specific steps you want)
    if t % 10 == 0 or t == T:  # Save at 0, 10, 20, ... and final step
        # Convert to numpy and denormalize if needed
        img_np = xt.squeeze().cpu().numpy()  # Remove batch and channel dimensions -> [256, 256]

        # If your image values are normalized to [-1, 1] or [0, 1], denormalize
        # Assuming normalized to [-1, 1], convert to [0, 255]
        os.makedirs(f'forward_Process_sqrt2', exist_ok=True)
        np.save(f'forward_Process_sqrt2/x_{t}.npy', img_np)

        print(f"Saved image at timestep {t}")