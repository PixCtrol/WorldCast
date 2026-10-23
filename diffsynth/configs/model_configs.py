wan_series = [
    {
        # Wan-AI/Wan2.1-I2V-14B-720P, transformer/diffusion_pytorch_model*.safetensors (7 shards)
        "model_hash": "6bfcfb3b342cb286ce886889d519a77e",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': True, 'patch_size': [1, 2, 2], 'in_dim': 36, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06},
    },
    {
        # Wan-AI/Wan2.1-I2V-14B-720P, models_t5_umt5-xxl-enc-bf16.pth
        "model_hash": "9c8818c2cbea55eca56c7b447df170da",
        "model_name": "wan_video_text_encoder",
        "model_class": "diffsynth.models.wan_video_text_encoder.WanTextEncoder",
    },
    {
        # Wan-AI/Wan2.1-I2V-14B-720P, Wan2.1_VAE.pth
        "model_hash": "ccc42284ea13e1ad04693284c7a09be6",
        "model_name": "wan_video_vae",
        "model_class": "diffsynth.models.wan_video_vae.WanVideoVAE",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_vae.WanVideoVAEStateDictConverter",
    },
    {
        # Wan-AI/Wan2.1-I2V-14B-720P, models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
        "model_hash": "5941c53e207d62f20f9025686193c40b",
        "model_name": "wan_video_image_encoder",
        "model_class": "diffsynth.models.wan_video_image_encoder.WanImageEncoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_image_encoder.WanImageEncoderStateDictConverter"
    },
]

MODEL_CONFIGS = wan_series
