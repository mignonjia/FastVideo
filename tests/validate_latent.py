"""Decode a VAE latent from the preprocessed parquet and compare with original video."""
import os
import numpy as np
import pandas as pd
import torch
import av

from diffusers import AutoencoderKLWan

MODEL_PATH = "/mnt/data/huggingface/hub/models--mignonjia--mg_bidirectional_zelda/snapshots/3644e37956f3a3a49515b801013fc0c665675bab"
PQ_PATH = "/mnt/data/world_model/MC_third_person/preprocessed/aonly_alpha1/merge_0/combined_parquet_dataset/worker_0"
ORIG_VIDEO = "/mnt/data/world_model/MC_third_person/processed/aonly_alpha1/videos/000000.mp4"
OUT_DIR = "/mnt/home/mhuo/FastVideo-MG/tests/videos/latent_validation"
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda")

# Load VAE
print("Loading VAE...")
vae = AutoencoderKLWan.from_pretrained(MODEL_PATH + "/vae", torch_dtype=torch.float32)
vae = vae.to(device).eval()

# Load latent from parquet
print("Loading latent from parquet...")
df = pd.read_parquet(PQ_PATH)
row = df.iloc[0]
vae_shape = tuple(row['vae_latent_shape'])   # (16, 21, 60, 104)
latent = np.frombuffer(row['vae_latent_bytes'], dtype=np.float32).reshape(vae_shape)
latent_t = torch.from_numpy(latent).unsqueeze(0).to(device)  # (1, 16, 21, 60, 104)
print(f"Latent shape: {latent_t.shape}")

# Decode
print("Decoding latent...")
with torch.no_grad():
    decoded = vae.decode(latent_t).sample  # (1, C, T, H, W)
print(f"Decoded shape: {decoded.shape}")

# Convert to uint8 frames: (1, 3, T, H, W) -> (T, H, W, 3)
decoded = decoded.squeeze(0).clamp(-1, 1)        # (3, T, H, W)
decoded = ((decoded + 1) / 2 * 255).byte()        # 0-255
decoded = decoded.permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, 3)
print(f"Decoded frames: {decoded.shape}")

# Write decoded video
out_decoded = os.path.join(OUT_DIR, "decoded.mp4")
with av.open(out_decoded, mode='w') as container:
    stream = container.add_stream('h264', rate=25)
    stream.width = decoded.shape[2]
    stream.height = decoded.shape[1]
    stream.pix_fmt = 'yuv420p'
    for frame_np in decoded:
        frame = av.VideoFrame.from_ndarray(frame_np, format='rgb24')
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
print(f"Saved decoded video: {out_decoded}")

# Copy first N frames of original video
out_orig = os.path.join(OUT_DIR, "original.mp4")
frames_orig = []
with av.open(ORIG_VIDEO) as container:
    for i, frame in enumerate(container.decode(video=0)):
        if i >= decoded.shape[0]:
            break
        frames_orig.append(frame.to_ndarray(format='rgb24'))

with av.open(out_orig, mode='w') as container:
    stream = container.add_stream('h264', rate=25)
    stream.width = frames_orig[0].shape[1]
    stream.height = frames_orig[0].shape[0]
    stream.pix_fmt = 'yuv420p'
    for frame_np in frames_orig:
        frame = av.VideoFrame.from_ndarray(frame_np, format='rgb24')
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
print(f"Saved original video ({len(frames_orig)} frames): {out_orig}")

# Compute PSNR
n = min(len(frames_orig), decoded.shape[0])
orig_arr = np.stack(frames_orig[:n]).astype(np.float32)
dec_arr = decoded[:n].astype(np.float32)
mse = np.mean((orig_arr - dec_arr) ** 2)
psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
print(f"\nPSNR (original vs decoded, {n} frames): {psnr:.2f} dB")
print("Done. Output:", OUT_DIR)
