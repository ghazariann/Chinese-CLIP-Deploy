# worker.py

# -*- coding: utf-8 -*-
# TODO 
import os
import json
import torch
import torch.distributed as dist
from cn_clip.clip.model import convert_weights, CLIP
from cn_clip.clip.utils import convert_models_to_fp32, seed_everything
import time



def remove_module_prefix(state_dict):
    """Removes the 'module.' prefix from state_dict keys if present."""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value
    return new_state_dict

# -------------------------------
# Torch Distributed Initialization
# -------------------------------

def init_distributed():
    MASTER_ADDR = '18.116.182.172'  # Master server IP
    MASTER_PORT = '29500'           # Must match master's MASTER_PORT
    WORLD_SIZE = 2                  # Total number of processes (master + worker)
    RANK = 1                        # Worker rank

    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = MASTER_PORT
    init_method = f'tcp://{MASTER_ADDR}:{MASTER_PORT}'

    dist.init_process_group(
        backend='gloo',        # Use 'gloo' for CPU
        init_method=init_method,
        world_size=WORLD_SIZE,
        rank=RANK
    )
    print("Distributed process group initialized on worker.")

# -------------------------------
# Model Loading
# -------------------------------

def load_model():
    """Loads the CLIP model up to the backbone part on the worker."""
    # Configuration Parameters (must match master)
    VISION_MODEL_NAME = "ViT-B-16"
    TEXT_MODEL_NAME = "RoBERTa-wwm-ext-base-chinese"
    CHECKPOINT_PATH = "data/pretrained_weights/Tencrypted_clip_cn_vit-b-16.pt"
    PRECISION = "fp32"  # Should match master

    # Initialize GPU (CPU in this case)
    device = 'cpu'

    # Seed for reproducibility
    seed_everything(42)

    # Load Model Configurations
    vision_model_config_file = f"cn_clip/clip/model_configs/{VISION_MODEL_NAME.replace('/', '-')}.json"
    text_model_config_file = f"cn_clip/clip/model_configs/{TEXT_MODEL_NAME.replace('/', '-')}.json"

    assert os.path.exists(vision_model_config_file), f"Vision model config not found at {vision_model_config_file}"
    assert os.path.exists(text_model_config_file), f"Text model config not found at {text_model_config_file}"
    assert os.path.exists(CHECKPOINT_PATH), f"Checkpoint not found at {CHECKPOINT_PATH}"

    with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
        model_info = json.load(fv)
        if isinstance(model_info['vision_layers'], str):
            model_info['vision_layers'] = eval(model_info['vision_layers'])        
        text_config = json.load(ft)
        for k, v in text_config.items():
            model_info[k] = v

    model_info['text_column_permutation'] = True  # Must match master

    # Initialize CLIP Model on CPU
    model = CLIP(**model_info)  # Removed device parameter
    convert_weights(model)    

    # Set Precision
    if PRECISION in ["amp", "fp32"]:
        convert_models_to_fp32(model)
    # model.to('cpu')  # Already on CPU
    if PRECISION == "fp16":
        convert_weights(model)

    # Load Checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    state_dict = checkpoint["state_dict"]
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint '{CHECKPOINT_PATH}' on worker.")

    # Set Model to Evaluation Mode
    model.eval()

    return model

# -------------------------------
# Main Worker Function
# -------------------------------

def main():
    """Main loop for the worker to receive tensors, process them, and send back the results."""
    init_distributed()
    model = load_model()
    print("Worker is ready to receive tensors.")

    try:
        while True:
            # Define fixed shapes
            batch_size = 1
            seq_length = 52
            hidden_dim = 768

            # Initialize tensors with fixed shapes
            embedding_output = torch.zeros(batch_size, seq_length, hidden_dim, dtype=torch.float)
            attn_mask = torch.zeros(batch_size, seq_length, dtype=torch.float)

            # Receive embedding_output and attn_mask from master
            dist.recv(tensor=embedding_output, src=0)
            print(f"Received embedding_output: shape={embedding_output.shape}, dtype={embedding_output.dtype}")

            dist.recv(tensor=attn_mask, src=0)
            print(f"Received attn_mask: shape={attn_mask.shape}, dtype={attn_mask.dtype}")

            # Perform Backbone_forward
            try:
                sequence_output = model.bert.Backbone_forward(embedding_output, attn_mask, (batch_size, seq_length))
                print(f"Performed Backbone_forward: shape={sequence_output.shape}, dtype={sequence_output.dtype}")
            except Exception as e:
                print(f"Error during Backbone_forward on worker: {e}")
                # Optionally, send a tensor filled with zeros or a termination signal
                sequence_output = torch.zeros(batch_size, seq_length, hidden_dim)  # Adjust hidden_dim as necessary

            # Send sequence_output back to master
            dist.send(tensor=sequence_output, dst=0)
            print("Sent sequence_output back to master.")

    except KeyboardInterrupt:
        print("Worker interrupted by user. Shutting down.")

    except Exception as e:
        print(f"Worker encountered an error: {e}")

    finally:
        dist.destroy_process_group()
        print("Worker process group destroyed.")

if __name__ == "__main__":
    main()
