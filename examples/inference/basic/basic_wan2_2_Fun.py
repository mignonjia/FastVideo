from fastvideo import VideoGenerator

# from fastvideo.configs.sample import SamplingParam

OUTPUT_PATH = "video_samples_wan2_1_Fun"
OUTPUT_NAME = "wan2.1_test"
def main():
    # FastVideo will automatically use the optimal default arguments for the
    # model.
    # If a local path is provided, FastVideo will make a best effort
    # attempt to identify the optimal arguments.
    generator = VideoGenerator.from_pretrained(
        "weizhou03/Wan2.1-Game-Fun-1.3B-InP-Diffusers",
        # "alibaba-pai/Wan2.2-Fun-A14B-Control",
        # FastVideo will automatically handle distributed setup
        num_gpus=1,
        use_fsdp_inference=False, # set to True if GPU is out of memory
        dit_cpu_offload=True, # DiT need to be offloaded for MoE
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        # Set pin_cpu_memory to false if CPU RAM is limited and there're no frequent CPU-GPU transfer
        pin_cpu_memory=True,
        # image_encoder_cpu_offload=False,
    )

    # prompt = "A first-person Minecraft-style voxel world featuring blocky cubic geometry and pixelated textures."
    prompt = "一只棕色的狗摇着头，坐在舒适房间里的浅色沙发上。在狗的后面，架子上有一幅镶框的画，周围是粉红色的花朵。房间里柔和温暖的灯光营造出舒适的氛围。"
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    # prompt                  = "A young woman with beautiful, clear eyes and blonde hair stands in the forest, wearing a white dress and a crown. Her expression is serene, reminiscent of a movie star, with fair and youthful skin. Her brown long hair flows in the wind. The video quality is very high, with a clear view. High quality, masterpiece, best quality, high resolution, ultra-fine, fantastical."
    # negative_prompt         = "Twisted body, limb deformities, text captions, comic, static, ugly, error, messy code."
    image_path = "assets/images/1.png"
    # image_path = "/mnt/weka/home/hao.zhang/mhuo/FastVideo/mc_wasd_10/validate/000002.jpg"
    # control_video_path = "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/wan_fun/asset_Wan2_2/v1.0/pose.mp4"

    video = generator.generate_video(prompt, negative_prompt=negative_prompt, guidance_scale=6.0, num_inference_steps=40, image_path=image_path, output_path=OUTPUT_PATH, output_video_name=OUTPUT_NAME, save_video=True)

if __name__ == "__main__":
    main()