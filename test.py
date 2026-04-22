from safetensors import safe_open

# Load the safetensors file
checkpoint_path = "wangame_1.3b_lingbot_overfit/checkpoint-5/transformer/diffusion_pytorch_model.safetensors"
with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
    keys = list(f.keys())

    # Check patch_embedding_wancamctrl: newly added parameter (Plucker patch embed)
    print("=== patch_embedding_wancamctrl: newly added parameter ===")
    pew_keys = [k for k in keys if "patch_embedding_wancamctrl" in k]
    for key in sorted(pew_keys):
        tensor = f.get_tensor(key)
        nonzero = (tensor != 0).sum().item()
        total = tensor.numel()
        print(f"{key}: dtype={tensor.dtype}, nonzero={nonzero}/{total}, min={tensor.min():.6f}, max={tensor.max():.6f}, mean={tensor.float().mean():.6f}, std={tensor.float().std():.6f}")

    # Check c2ws_mlp: newly added parameter (camera MLP)
    print("\n=== c2ws_mlp: newly added parameter ===")
    mlp_keys = [k for k in keys if "c2ws_mlp" in k]
    for key in sorted(mlp_keys):
        tensor = f.get_tensor(key)
        nonzero = (tensor != 0).sum().item()
        total = tensor.numel()
        print(f"{key}: dtype={tensor.dtype}, nonzero={nonzero}/{total}, min={tensor.min():.6f}, max={tensor.max():.6f}, mean={tensor.float().mean():.6f}, std={tensor.float().std():.6f}")

    # Check cam_conditioner (block 0): newly added parameter (per-block scale/shift)
    print("\n=== cam_conditioner (block 0): newly added parameter ===")
    cam_keys = [k for k in keys if "blocks.0.cam_conditioner" in k]
    for key in sorted(cam_keys):
        tensor = f.get_tensor(key)
        nonzero = (tensor != 0).sum().item()
        total = tensor.numel()
        print(f"{key}: dtype={tensor.dtype}, nonzero={nonzero}/{total}, min={tensor.min():.6f}, max={tensor.max():.6f}, mean={tensor.float().mean():.6f}, std={tensor.float().std():.6f}")

    # Check time_embedder: original parameter
    print("\n=== time_embedder: original parameter ===")
    time_keys = [k for k in keys if "condition_embedder.time_embedder" in k]
    for key in sorted(time_keys):
        tensor = f.get_tensor(key)
        nonzero = (tensor != 0).sum().item()
        total = tensor.numel()
        print(f"{key}: dtype={tensor.dtype}, nonzero={nonzero}/{total}, min={tensor.min():.6f}, max={tensor.max():.6f}, mean={tensor.float().mean():.6f}, std={tensor.float().std():.6f}")

    # Check to_out (block 0): original attention output projection
    print("\n=== to_out (block 0): original parameter ===")
    to_out_keys = [k for k in keys if "blocks.0" in k and "to_out." in k and "cam_conditioner" not in k]
    for key in sorted(to_out_keys):
        tensor = f.get_tensor(key)
        nonzero = (tensor != 0).sum().item()
        total = tensor.numel()
        print(f"{key}: dtype={tensor.dtype}, nonzero={nonzero}/{total}, min={tensor.min():.6f}, max={tensor.max():.6f}, mean={tensor.float().mean():.6f}, std={tensor.float().std():.6f}")
