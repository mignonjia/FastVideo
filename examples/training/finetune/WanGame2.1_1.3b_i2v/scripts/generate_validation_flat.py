import json
import os
import shutil

import cv2

train = "mc"

if train == "zelda":
    height = 480
    width = 832
    num_frames = 81
    action_dir = "/mnt/weka/home/hao.zhang/mhuo/FastVideo/examples/training/finetune/WanGame2.1_1.3b_i2v/actions_81"
elif train == "mc":
    height = 352
    width = 640
    num_frames = 77
    action_dir = "/mnt/weka/home/hao.zhang/mhuo/FastVideo/examples/training/finetune/WanGame2.1_1.3b_i2v/actions"
else:
    raise ValueError(f"Invalid train type: {train}")

# Output path
output_path = (
    "/mnt/weka/home/hao.zhang/mhuo/FastVideo/examples/training/finetune/"
    f"WanGame2.1_1.3b_i2v/validation_{train}_flat.json"
)

# Fixed fields
fixed_fields = {
    "video_path": None,
    "num_inference_steps": 40,
    "height": height,
    "width": width,
    "num_frames": num_frames,
}

# WASDudlr: single key W.npy, single camera u.npy, key+camera w_u.npy
still = os.path.join(action_dir, "still.npy")
key_W = os.path.join(action_dir, "W.npy")
key_S = os.path.join(action_dir, "S.npy")
key_still5_S_rest = os.path.join(action_dir, "still5_S_rest.npy")
key_A = os.path.join(action_dir, "A.npy")
key_D = os.path.join(action_dir, "D.npy")
key_wa = os.path.join(action_dir, "WA.npy")
key_s_u = os.path.join(action_dir, "S_u.npy")
camera_u = os.path.join(action_dir, "u.npy")
camera_d = os.path.join(action_dir, "d.npy")
camera_l = os.path.join(action_dir, "l.npy")
camera_r = os.path.join(action_dir, "r.npy")
key_A_then_D = os.path.join(action_dir, "still5_A28_still4_D28_still.npy")
key_D_then_A = os.path.join(action_dir, "still5_D28_still4_A28_still.npy")
key_W_then_S = os.path.join(action_dir, "still5_W28_still4_S28_still.npy")
key_S_then_W = os.path.join(action_dir, "still5_S28_still4_W28_still.npy")
camera_u_then_d = os.path.join(action_dir, "still5_u28_still4_d28_still.npy")
camera_d_then_u = os.path.join(action_dir, "still5_d28_still4_u28_still.npy")
camera_l_then_r = os.path.join(action_dir, "still5_l28_still4_r28_still.npy")
camera_r_then_l = os.path.join(action_dir, "still5_r28_still4_l28_still.npy")


val_img_flat_list = [
    "/mnt/weka/home/hao.zhang/mhuo/FastVideo/mc_wasd_10/validate/gen_000003.jpg",
    "/mnt/weka/home/hao.zhang/mhuo/FastVideo/mc_wasd_10/validate/gen_000005.jpg",
    "/mnt/weka/home/hao.zhang/mhuo/FastVideo/mc_wasd_10/validate/000006.jpg",
    "/mnt/weka/home/hao.zhang/mhuo/FastVideo/mc_wasd_10/validate/000005.jpg"
]

# Get doom Val data list
val_img_doom_list = [
    "/mnt/weka/home/hao.zhang/mhuo/FastVideo/examples/training/finetune/WanGame2.1_1.3b_i2v/doom/000000.jpg",
    "/mnt/weka/home/hao.zhang/mhuo/FastVideo/examples/training/finetune/WanGame2.1_1.3b_i2v/doom/000001.jpg",
    "/mnt/weka/home/hao.zhang/mhuo/FastVideo/examples/training/finetune/WanGame2.1_1.3b_i2v/doom/000002.jpg",
    "/mnt/weka/home/hao.zhang/mhuo/FastVideo/examples/training/finetune/WanGame2.1_1.3b_i2v/doom/000003.jpg",
]

val_img_list = val_img_flat_list

holder = 0 # placeholder
# 32 placeholders (idx 0-31). Fill in manually.
a0 = ["00 Val-00: W", val_img_list[0], key_W]
a1 = ["01 Val-01: S", val_img_list[1], key_still5_S_rest]
a2 = ["02 Val-02: A", val_img_list[2], key_A]
a3 = ["03 Val-03: D", val_img_list[3], key_D]
a4 = ["04 Val-04: u", val_img_list[0], camera_u]
a5 = ["05 Val-05: d", val_img_list[1], camera_d]
a6 = ["06 Val-06: l", val_img_list[2], camera_l]
a7 = ["07 Val-07: r", val_img_list[3], camera_r]

a8 = ["08 Val-00: W", val_img_list[3], key_W]
a9 = ["09 Val-01: S", val_img_list[2], key_still5_S_rest]
a10 = ["10 Val-02: A", val_img_list[1], key_A]
a11 = ["11 Val-03: D", val_img_list[0], key_D]
a12 = ["12 Val-00: Still", val_img_list[0], key_still5_S_rest]
a13 = ["13 Val-01: Still", val_img_list[1], key_still5_S_rest]
a14 = ["14 Val-02: Still", val_img_list[2], still]
a15 = ["15 Val-03: Still", val_img_list[3], still]

a16 = ["16 Val-00: A then D", val_img_list[0], key_A_then_D]
a17 = ["17 Val-01: D then A", val_img_list[1], key_D_then_A]
a18 = ["18 Val-02: W then S", val_img_list[2], key_W_then_S]
a19 = ["19 Val-03: S then W", val_img_list[3], key_S_then_W]
a20 = ["20 Val-00: u then d", val_img_list[0], camera_u_then_d]
a21 = ["21 Val-01: d then u", val_img_list[1], camera_d_then_u]
a22 = ["22 Val-02: l then r", val_img_list[2], camera_l_then_r]
a23 = ["23 Val-03: r then l", val_img_list[3], camera_r_then_l]

a24 = ["24 Val-00: A then D", val_img_list[3], key_A_then_D]
a25 = ["25 Val-01: D then A", val_img_list[2], key_D_then_A]
a26 = ["26 Val-02: W then S", val_img_list[1], key_W_then_S]
a27 = ["27 Val-03: S then W", val_img_list[0], key_S_then_W]
a28 = ["28 Val-00: u then d", val_img_list[3], camera_u_then_d]
a29 = ["29 Val-01: d then u", val_img_list[2], camera_d_then_u]
a30 = ["30 Val-02: l then r", val_img_list[1], camera_l_then_r]
a31 = ["31 Val-03: r then l", val_img_list[0], camera_r_then_l]


Val_entries = {
    0: a0,
    1: a1,
    2: a2,
    3: a3,
    4: a4,
    5: a5,
    6: a6,
    7: a7,
    8: a8,
    9: a9,
    10: a10,
    11: a11,
    12: a12,
    13: a13,
    14: a14,
    15: a15,
    16: a16,
    17: a17,
    18: a18,
    19: a19,
    20: a20,
    21: a21,
    22: a22,
    23: a23,
    24: a24,
    25: a25,
    26: a26,
    27: a27,
    28: a28,
    29: a29,
    30: a30,
    31: a31,
}

data = []
for idx in range(32):
    if idx not in Val_entries:
        raise ValueError(f"Missing entry for idx {idx}")
    caption, image_path, action_path = Val_entries[idx]
    data.append(
        {
            "caption": caption,
            "image_path": image_path,
            "action_path": action_path,
            **fixed_fields,
        }
    )

output = {"data": data}
with open(output_path, "w") as f:
    json.dump(output, f, indent=4)

print(f"Generated {len(data)} entries to {output_path}")

# Check file all exists

with open(output_path) as f:
    data = json.load(f)

missing = []
for i, item in enumerate(data['data']):
    for key in ('image_path', 'action_path'):
        path = item.get(key)
        if path:
            import os
            if not os.path.isfile(path):
                missing.append((i, key, path))
if missing:
    print('Missing paths:')
    for idx, key, path in missing:
        print(f'  [{idx}] {key}: {path}')
else:
    print('All paths exist.')


