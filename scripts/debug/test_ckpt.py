from safetensors import safe_open

# Load the safetensors file
# checkpoint_path = "wangame_1.3b/checkpoint-10/transformer/diffusion_pytorch_model.safetensors"
checkpoint_path = "/mnt/weka/home/hao.zhang/mhuo/FastVideo/wangame_1.3b_zelda_with_mouse_ckpt/checkpoint-best-step-5000/transformer/diffusion_pytorch_model.safetensors"
with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
    keys = list(f.keys())
    
    # Check action_embedder
    print("=== action_embedder: newly added parameter ===")
    action_keys = [k for k in keys if "action_embedder" in k]
    for key in action_keys:
        tensor = f.get_tensor(key)
        nonzero = (tensor != 0).sum().item()
        total = tensor.numel()
        if "fc_in" in key:
            print("* xvaier initialization")
        else:
            print("* zero initialization")
        print(f"{key}: dtype={tensor.dtype}, nonzero={nonzero}/{total}, min={tensor.min():.6f}, max={tensor.max():.6f}, mean={tensor.float().mean():.6f}, std={tensor.float().std():.6f}")
    
    # Check time_embedder
    print("\n=== time_embedder: original parameter ===")
    time_keys = [k for k in keys if "condition_embedder.time_embedder" in k]
    for key in time_keys:
        tensor = f.get_tensor(key)
        nonzero = (tensor != 0).sum().item()
        total = tensor.numel()
        print(f"{key}: dtype={tensor.dtype}, nonzero={nonzero}/{total}, min={tensor.min():.6f}, max={tensor.max():.6f}, mean={tensor.float().mean():.6f}, std={tensor.float().std():.6f}")

    # Check text_embedder
    print("\n=== text_embedder: text embedding parameter ===")
    text_keys = [k for k in keys if "condition_embedder.text_embedder" in k]
    if not text_keys:
        # Some checkpoints may keep old naming (fc_in/fc_out) or be converted.
        text_keys = [
            k for k in keys
            if ("text_embedder.fc_" in k or "text_embedder.linear_" in k)
        ]
    for key in text_keys:
        tensor = f.get_tensor(key)
        nonzero = (tensor != 0).sum().item()
        total = tensor.numel()
        print(f"{key}: dtype={tensor.dtype}, nonzero={nonzero}/{total}, min={tensor.min():.6f}, max={tensor.max():.6f}, mean={tensor.float().mean():.6f}, std={tensor.float().std():.6f}")

    # Check to_out_prope (just block 0)
    print("\n=== to_out_prope (block 0): newly added parameter ===")
    prope_keys = [k for k in keys if "blocks.0" in k and "to_out_prope" in k]
    for key in prope_keys:
        tensor = f.get_tensor(key)
        nonzero = (tensor != 0).sum().item()
        total = tensor.numel()
        print("* zero initialization")
        print(f"{key}: dtype={tensor.dtype}, nonzero={nonzero}/{total}, min={tensor.min():.6f}, max={tensor.max():.6f}, mean={tensor.float().mean():.6f}, std={tensor.float().std():.6f}")

    # Check to_out (just block 0) - original attention output projection
    print("\n=== to_out (block 0): original parameter ===")
    to_out_keys = [k for k in keys if "blocks.0" in k and "to_out." in k and "prope" not in k]
    for key in to_out_keys:
        tensor = f.get_tensor(key)
        nonzero = (tensor != 0).sum().item()
        total = tensor.numel()
        print(f"{key}: dtype={tensor.dtype}, nonzero={nonzero}/{total}, min={tensor.min():.6f}, max={tensor.max():.6f}, mean={tensor.float().mean():.6f}, std={tensor.float().std():.6f}")


""" Result

(mhuo-fv) hao.zhang@fs-mbz-gpu-820:~/mhuo/FastVideo$ python '/mnt/weka/home/hao.zhang/mhuo/FastVideo/test.py'
=== action_embedder: newly added parameter ===
* xvaier initialization
condition_embedder.action_embedder.fc_in.bias: nonzero=1536/1536, min=-0.006336, max=0.006700, mean=-0.000037, std=0.002635
* xvaier initialization
condition_embedder.action_embedder.fc_in.weight: nonzero=393216/393216, min=-0.068075, max=0.069139, mean=-0.000072, std=0.036010
* zero initialization
condition_embedder.action_embedder.fc_out.bias: nonzero=1536/1536, min=-0.000849, max=0.000827, mean=-0.000020, std=0.000233
* zero initialization
condition_embedder.action_embedder.fc_out.weight: nonzero=2359296/2359296, min=-0.002519, max=0.002580, mean=-0.000002, std=0.000215

=== time_embedder: original parameter ===
condition_embedder.time_embedder.linear_1.bias: nonzero=1536/1536, min=-0.023771, max=0.007965, mean=-0.000557, std=0.003792
condition_embedder.time_embedder.linear_1.weight: nonzero=393216/393216, min=-0.096235, max=0.123983, mean=-0.000183, std=0.015347
condition_embedder.time_embedder.linear_2.bias: nonzero=1536/1536, min=-0.063983, max=0.049396, mean=0.003240, std=0.007229
condition_embedder.time_embedder.linear_2.weight: nonzero=2359296/2359296, min=-0.112347, max=0.125004, mean=-0.000010, std=0.006386

=== to_out_prope (block 0): newly added parameter ===
* zero initialization
blocks.0.to_out_prope.bias: nonzero=1536/1536, min=-0.000828, max=0.000675, mean=-0.000004, std=0.000195
* zero initialization
blocks.0.to_out_prope.weight: nonzero=2359296/2359296, min=-0.002262, max=0.002512, mean=-0.000000, std=0.000259

=== to_out (block 0): original parameter ===
blocks.0.attn1.to_out.0.bias: nonzero=1536/1536, min=-0.074909, max=0.017941, mean=-0.000163, std=0.005341
blocks.0.attn1.to_out.0.weight: nonzero=2359296/2359296, min=-0.184305, max=0.172199, mean=-0.000005, std=0.024856
blocks.0.attn2.to_out.0.bias: nonzero=1536/1536, min=-0.034722, max=0.078203, mean=-0.000011, std=0.007098
blocks.0.attn2.to_out.0.weight: nonzero=2359296/2359296, min=-0.426505, max=0.462859, mean=0.000001, std=0.029782
"""
