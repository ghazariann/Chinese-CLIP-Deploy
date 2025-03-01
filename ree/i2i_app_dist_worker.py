# i2i_dist_worker.py
# -*- coding: utf-8 -*-

import os
import json
import base64
import csv
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
import sys 

MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(MAIN_DIR)

from cn_clip.clip.model import convert_weights, CLIP
from cn_clip.clip.utils import convert_models_to_fp32, seed_everything

# ---------------------------------------
# Distributed Configuration
# ---------------------------------------
MASTER_ADDR = '18.116.182.172'  # Master server IP
MASTER_PORT = '29500'      # Must match master's MASTER_PORT
WORLD_SIZE = 2             # Total number of processes (master + worker)
RANK = 1                   # Worker rank

# ---------------------------------------
# CLIP Parameters
# ---------------------------------------
PRECISION = "fp32"
VISION_MODEL_NAME = "ViT-B-16"

# ---------------------------------------
# File Paths
# ---------------------------------------
CHECKPOINT_PATH = os.path.join(MAIN_DIR, "data/pretrained_weights/Tencrypted_clip_cn_vit-b-16.pt")

# ---------------------------------------
# Initialize Distributed
# ---------------------------------------
def init_distributed():
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = MASTER_PORT
    dist.init_process_group(
        backend='gloo',  # Use 'nccl' if using GPUs
        init_method=f'tcp://{MASTER_ADDR}:{MASTER_PORT}',
        world_size=WORLD_SIZE,
        rank=RANK
    )
    print("[Worker] Distributed process group initialized.")

# ---------------------------------------
# Load CLIP Model
# ---------------------------------------
def load_model():
    print("[Worker] Initializing the model...")
    # Seed for reproducibility
    seed_everything(42)

    # Initialize distributed process group
    init_distributed()

    # Load model configuration files
    vision_config_file = os.path.join(MAIN_DIR, f"cn_clip/clip/model_configs/{VISION_MODEL_NAME.replace('/', '-')}.json")
    text_config_file = os.path.join(MAIN_DIR, f"cn_clip/clip/model_configs/RoBERTa-wwm-ext-base-chinese.json")  # Assuming text model config)

    assert os.path.exists(vision_config_file), f"Vision config not found: {vision_config_file}"
    assert os.path.exists(text_config_file), f"Text config not found: {text_config_file}"

    with open(vision_config_file, 'r') as fv, open(text_config_file, 'r') as ft:
        model_info = json.load(fv)
        if isinstance(model_info.get('vision_layers'), str):
            model_info['vision_layers'] = eval(model_info['vision_layers'])
        text_config = json.load(ft)
        for k, v in text_config.items():
            model_info[k] = v

    model_info['text_column_permutation'] = True  # Adjust if necessary

    # Initialize CLIP model
    model = CLIP(**model_info)
    convert_weights(model)

    # Set precision
    if PRECISION in ["amp", "fp32"]:
        convert_models_to_fp32(model)
    # model.cuda()  # Uncomment if using GPU
    if PRECISION == "fp16":
        convert_weights(model)

    # Load checkpoint
    assert os.path.exists(CHECKPOINT_PATH), f"Checkpoint not found: {CHECKPOINT_PATH}"
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    state_dict = checkpoint["state_dict"]
    if next(iter(state_dict.items()))[0].startswith('module'):
        # Remove 'module.' prefix if present
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items() if "bert.pooler" not in k}
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Load the model checkpoint.")

    return model

# Initialize model
model = load_model()

# ---------------------------------------
# Distributed Communication
# ---------------------------------------
def process():
    """
    Continuously listen for data from master, process it, and send back results.
    """
    while True:
        try:
            # Prepare to receive embedding
            # The embedding shape should match what the master sends: [1, 197, 768]
            embedding_shape = [1, 197, 768]  # TODO hard code
            embedding = torch.zeros(embedding_shape)
            dist.recv(tensor=embedding, src=0)
            print("[Worker] Receive intermediate embedding.")

            # Process Backbone_forward
            with torch.no_grad():
                sequence_output = model.visual.Backbone_forward(embedding)
                print("[Worker] Encode the intermediate embedding into final embedding.")

            # Send sequence_output back to master
            dist.send(tensor=sequence_output, dst=0)
            print("[Worker]  Send final embedding to TEE.")

        except Exception as e:
            print(f"[Worker] Error during processing: {e}")
            break

# ---------------------------------------
# Main Entry Point
# ---------------------------------------
if __name__ == "__main__":
    print("[Worker] Starting worker process...")
    process()
