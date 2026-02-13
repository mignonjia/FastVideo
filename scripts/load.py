import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from fastvideo.models.dits.wangame.model import WanGameActionTransformer3DModel
from fastvideo.models.dits.wangame_lingbot.model import WanLingBotTransformer3DModel
from fastvideo.configs.models.dits.wangamevideo import WanGameVideoConfig, WanLingBotVideoConfig
import re

def main():
    # config = WanGameVideoConfig()
    config = WanLingBotVideoConfig()
    with torch.device("meta"):
        # model = WanGameActionTransformer3DModel(config=config, hf_config={})
        model = WanLingBotTransformer3DModel(config=config, hf_config={})
    
    state_dict = model.state_dict()
    mapping = config.arch_config.param_names_mapping
    expected_format = {}
    for target_name, param in state_dict.items():
        source_name = target_name
        for pattern, replacement in mapping.items():
            pass
        
        print(f"{target_name}: {list(param.shape)}")
        expected_format[target_name] = torch.zeros(param.shape, device="meta")

    print("-" * 60)
    print(f"{len(state_dict)}")

if __name__ == "__main__":
    main()