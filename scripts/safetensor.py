import argparse
import os
from safetensors import safe_open

def main():
    file_path = "./Wan2.1-Game-Fun-1.3B-InP-Diffusers/transformer/diffusion_pytorch_model.safetensors"
    if not os.path.exists(file_path):
        print(f"error: {file_path}")
        return

    print(f"safetensors: {os.path.abspath(file_path)}")
    print("-" * 100)

    count = 0
    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            keys = sorted(f.keys())
            for key in keys:
                shape = f.get_slice(key).get_shape()
                print(f"{key}: {shape}")
                count += 1
    except Exception as e:
        print(f"error: {e}")
        return

    print("-" * 100)
    print(f"total: {count}")

if __name__ == "__main__":
    main()