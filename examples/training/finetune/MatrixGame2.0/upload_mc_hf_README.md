# MC Third-Person Preprocessed Dataset (mignonjia/mc_third_person)

## Structure on HF
Each action type is compressed into split 3GB tar.gz parts:
```
aonly_alpha1/
  aonly_alpha1.tar.gz.000
  aonly_alpha1.tar.gz.001
  ...
cam_view12hold_alpha1/
  cam_view12hold_alpha1.tar.gz.000
  ...
```

## Reconstruct locally
```bash
# For each action type:
cat aonly_alpha1/aonly_alpha1.tar.gz.* | tar -xz -C /your/output/dir/

# Or restore all at once:
for action in aonly_alpha1 cam_view12hold_alpha1 donly_alpha1 sonly_alpha1 \
              static_alpha1 wasd12hold_alpha1 wasd12holdrandview_alpha1 wonly_alpha1; do
    cat ${action}/${action}.tar.gz.* | tar -xz -C /your/output/dir/
done
```

## Restored structure
```
preprocessed/
  aonly_alpha1/
    merge_0/combined_parquet_dataset/worker_0
    merge_1/combined_parquet_dataset/worker_0
    ...
  cam_view12hold_alpha1/
    ...
```

## Parquet schema
| Column | Shape | Description |
|--------|-------|-------------|
| `vae_latent` | (16, 21, 60, 104) float32 | VAE latent, 81 frames @ 480×832 |
| `clip_feature` | (257, 1280) float32 | CLIP ViT-H/14 first-frame embedding |
| `first_frame_latent` | (16, 21, 60, 104) float32 | First-frame VAE latent |
| `keyboard_cond` | (81, 6) float32 | WASD keyboard actions per frame |
| `mouse_cond` | (81, 2) float32 | Mouse delta (yaw, pitch) per frame |
