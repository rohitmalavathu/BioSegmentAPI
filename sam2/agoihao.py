import os
import yaml

config_path = os.path.join(os.getcwd(), "sam2/configs/sam2/sam2_hiera_s.yaml")

if not os.path.exists(config_path):
    print(f"Configuration file not found: {config_path}")
    # You can either exit the program or provide a default configuration
    exit(1)
else:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        print("Configuration file loaded successfully:")
        print(config)