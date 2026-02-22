python scripts/visualize_lingbot_trajectory.py --action examples/training/finetune/WanGame2.1_1.3b_i2v/actions_81/W.npy --num-frames 81 --fpv pred
python scripts/visualize_lingbot_trajectory.py --action examples/training/finetune/WanGame2.1_1.3b_i2v/actions_81/S.npy --num-frames 81 --fpv pred
python scripts/visualize_lingbot_trajectory.py --action examples/training/finetune/WanGame2.1_1.3b_i2v/actions_81/A.npy --num-frames 81 --fpv pred
python scripts/visualize_lingbot_trajectory.py --action examples/training/finetune/WanGame2.1_1.3b_i2v/actions_81/D.npy --num-frames 81 --fpv pred

python scripts/visualize_lingbot_trajectory.py --action examples/training/finetune/WanGame2.1_1.3b_i2v/actions_81/u.npy --num-frames 81 --fpv pred
python scripts/visualize_lingbot_trajectory.py --action examples/training/finetune/WanGame2.1_1.3b_i2v/actions_81/d.npy --num-frames 81 --fpv pred
python scripts/visualize_lingbot_trajectory.py --action examples/training/finetune/WanGame2.1_1.3b_i2v/actions_81/l.npy --num-frames 81 --fpv pred
python scripts/visualize_lingbot_trajectory.py --action examples/training/finetune/WanGame2.1_1.3b_i2v/actions_81/r.npy --num-frames 81 --fpv pred

python scripts/visualize_lingbot_trajectory.py --poses examples/inference/basic/lingbotworld_examples/00/poses.npy --intrinsics examples/inference/basic/lingbotworld_examples/00/intrinsics.npy  --fpv gt --fpv-fps 24

python scripts/visualize_lingbot_trajectory.py --poses examples/inference/basic/lingbotworld_examples/01/poses.npy --intrinsics examples/inference/basic/lingbotworld_examples/01/intrinsics.npy  --fpv gt --fpv-fps 24

python scripts/visualize_lingbot_trajectory.py --poses examples/inference/basic/lingbotworld_examples/02/poses.npy --intrinsics examples/inference/basic/lingbotworld_examples/02/intrinsics.npy  --fpv gt --fpv-fps 24