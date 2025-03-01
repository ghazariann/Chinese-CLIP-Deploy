# app_dist.py

# -*- coding: utf-8 -*-

import os
import json
import base64
import csv
import numpy as np
import torch
import torch.distributed as dist
from flask import jsonify, Flask, request, render_template, redirect, url_for
from tqdm import tqdm
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import logging
import time 
import requests
logger = logging.getLogger(__name__)
import subprocess

from cn_clip.clip.model import convert_weights, CLIP
from cn_clip.clip.utils import convert_models_to_fp32, seed_everything

app = Flask(__name__)

# -------------------------------
# Distributed Configuration
# -------------------------------
MASTER_ADDR = '18.116.182.172'  # Master server IP
MASTER_PORT = '29500'           # Must match worker's MASTER_PORT
WORLD_SIZE = 2                  # Total number of processes (master + worker)
RANK = 0                        # Master rank
WORKER_URL = 'http://18.217.90.50:5001/validate_attestation'

# -------------------------------
# Encryption Configuration
# -------------------------------
# AES key must match the client's AES_KEY_HEX
AES_KEY = b'\x01' * 32  # 32-byte key (demo only)
aesgcm = AESGCM(AES_KEY)

def decrypt_data(encrypted_b64: str) -> str:
    """
    Decrypts AES-GCM encrypted data (text).

    Args:
        encrypted_b64: Base64-encoded (nonce + ciphertext).

    Returns:
        Decrypted plaintext (str) or None if decryption fails.
    """
    try:
        encrypted_data = base64.b64decode(encrypted_b64)
        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:]
        plaintext_bytes = aesgcm.decrypt(nonce, ciphertext, None)
        plaintext = plaintext_bytes.decode('utf-8')
        print(f"[Master] Encrypted Text: '{encrypted_b64}'")
        return plaintext
    except Exception as e:
        print(f"[Master] Error decrypting data: {e}")
        return None

def encrypt_image_bytes(image_b64: str) -> str:
    """
    Encrypts image bytes (originally in Base64) using AES-GCM, then returns (nonce + ciphertext) in Base64.

    Args:
        image_b64: The original image data as a Base64-encoded string.

    Returns:
        A Base64-encoded string combining nonce + ciphertext.
    """
    try:
        # Convert base64 image data to raw bytes
        raw_image_bytes = base64.b64decode(image_b64)

        # Generate random 12-byte nonce
        nonce = os.urandom(12)
        ciphertext = aesgcm.encrypt(nonce, raw_image_bytes, None)
        combined = nonce + ciphertext
        encrypted_b64 = base64.b64encode(combined).decode('utf-8')
        print(f"[Master] Encrypted Image Data (Base64): {encrypted_b64[:10]} ...")
        return encrypted_b64
    except Exception as e:
        print(f"[Master] Error encrypting image bytes: {e}")
        return ""

# -------------------------------
# Flask Configuration
# -------------------------------
FLASK_PORT = 8080

# CLIP Parameters
PRECISION = "fp32"
VISION_MODEL_NAME = "ViT-B-16"
TEXT_MODEL_NAME = "RoBERTa-wwm-ext-base-chinese"
TOP_K = 3
CONTEXT_LENGTH = 52

# File Paths
IMAGE_DATA_PATH = "database/valid_imgs.tsv"
IMAGE_FEAT_PATH = "database/valid_imgs.img_feat.jsonl"
CHECKPOINT_PATH = "data/pretrained_weights/Tencrypted_clip_cn_vit-b-16.pt"

# -------------------------------
# Initialize Distributed
# -------------------------------
def init_distributed():
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = MASTER_PORT
    init_method = f'tcp://{MASTER_ADDR}:{MASTER_PORT}'
    dist.init_process_group(
        backend='gloo',
        init_method=init_method,
        world_size=WORLD_SIZE,
        rank=RANK
    )
    print("[Master] Distributed process group initialized.")

# -------------------------------
# Load CLIP Model and Data
# -------------------------------
def load_model_and_data():
    print("[Master] Initializing the application...")
    # Seed
    seed_everything(42)

    # Distributed init
    init_distributed()

    # Load config files
    vision_config_file = f"cn_clip/clip/model_configs/{VISION_MODEL_NAME.replace('/', '-')}.json"
    text_config_file = f"cn_clip/clip/model_configs/{TEXT_MODEL_NAME.replace('/', '-')}.json"
    assert os.path.exists(vision_config_file), f"Vision config not found: {vision_config_file}"
    assert os.path.exists(text_config_file), f"Text config not found: {text_config_file}"

    with open(vision_config_file, 'r') as fv, open(text_config_file, 'r') as ft:
        model_info = json.load(fv)
        if isinstance(model_info.get('vision_layers'), str):
            model_info['vision_layers'] = eval(model_info['vision_layers'])
        text_config = json.load(ft)
        for k, v in text_config.items():
            model_info[k] = v

    model_info['text_column_permutation'] = True

    # Initialize CLIP model
    model = CLIP(**model_info)
    convert_weights(model)

    # Set precision
    if PRECISION in ["amp", "fp32"]:
        convert_models_to_fp32(model)
    if PRECISION == "fp16":
        convert_weights(model)

    # Load checkpoint
    assert os.path.exists(CHECKPOINT_PATH), f"Checkpoint not found: {CHECKPOINT_PATH}"
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    sd = checkpoint["state_dict"]
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}
    model.load_state_dict(sd)
    print(f"[Master] Loaded checkpoint '{CHECKPOINT_PATH}'")

    model.eval()

    # Load image features
    print("[Master] Loading image features...")
    image_ids = []
    image_feats = []
    with open(IMAGE_FEAT_PATH, "r", encoding='utf-8') as fin:
        for line in tqdm(fin, desc="[Master] Reading Image Feats"):
            obj = json.loads(line.strip())
            image_ids.append(obj['image_id'])
            image_feats.append(obj['feature'])
    image_feats_array = np.array(image_feats, dtype=np.float32)
    print(f"[Master] Loaded {len(image_ids)} image features.")

    # Load image data
    print("[Master] Loading image data from TSV...")
    image_id_to_data = {}
    with open(IMAGE_DATA_PATH, 'r', encoding='utf-8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        for row in tqdm(reader, desc="[Master] Reading Image Data"):
            if len(row) < 2:
                continue
            img_id, base64_data = row
            image_id_to_data[int(img_id)] = base64_data
    print(f"[Master] Loaded {len(image_id_to_data)} images in the database.")

    # Convert to tensor
    image_feats_tensor = torch.from_numpy(image_feats_array)

    return model, image_ids, image_feats_tensor, image_id_to_data

# Initialize
model, image_ids, image_feats_tensor, image_id_to_data = load_model_and_data()

# -------------------------------
# Flask Routes
# -------------------------------
@app.route('/', methods=['GET'])
def home():
    """
    Render the home.html template from templates/ directory.
    """
    return render_template('home.html')

@app.route('/result', methods=['POST'])
def get_image():
    """
    1. Decrypt user text from client.
    2. Tokenize -> partial CLIP forward -> send to worker -> receive -> finalize.
    3. Retrieve top-k images.
    4. Encrypt images using AES-GCM before sending to client.
    5. Render template with encrypted images.
    """
    encrypted_user_text = request.form.get('text', '').strip()
    if not encrypted_user_text:
        return redirect(url_for('home'))

    # 1. Decrypt text
    user_text = decrypt_data(encrypted_user_text)
    if user_text is None:
        return render_template('home.html', images=None, error="Error decrypting input text.")
    print(f"[Master] Decrypted user text: {user_text}")

    # 2. Tokenize user_text
    tokens = model.tokenizer.tokenize(user_text)
    max_tokens = CONTEXT_LENGTH - 2
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    token_ids = model.tokenizer.convert_tokens_to_ids(tokens)
    pad_token_id = model.tokenizer.vocab.get('[PAD]', 0)
    padding_length = CONTEXT_LENGTH - len(token_ids)
    if padding_length > 0:
        token_ids += [pad_token_id] * padding_length

    tokens_tensor = torch.tensor([token_ids])
    with torch.no_grad():
        try:
            attn_mask = (tokens_tensor != pad_token_id).type(model.dtype)
            embedding_output = model.bert.F1_forward(tokens_tensor)
            print(f"[Master] embedding_output shape: {embedding_output.shape}")
        except Exception as e:
            print(f"[Master] Error during F1_forward: {e}")
            return render_template('home.html', images=None, error="Text processing (F1) error.")

    # Send to worker
    try:
        dist.send(tensor=embedding_output, dst=1)
        print("[Master] Sent embedding_output to worker.")
        dist.send(tensor=attn_mask, dst=1)
        print("[Master] Sent attn_mask to worker.")

        batch_size, seq_len, hid_dim = embedding_output.shape
        sequence_output = torch.zeros(batch_size, seq_len, hid_dim)
        dist.recv(tensor=sequence_output, src=1)
        print("[Master] Received sequence_output from worker.")
    except Exception as e:
        print(f"[Master] Distributed communication error: {e}")
        return render_template('home.html', images=None, error="Worker communication failed.")

    # F2 part
    with torch.no_grad():
        try:
            text_features = model.bert.F2_forward(sequence_output)[0].type(model.dtype)
            text_features = text_features[:, 0, :] @ model.text_projection
            text_features /= text_features.norm(dim=-1, keepdim=True)
        except Exception as e:
            print(f"[Master] Error during F2_forward: {e}")
            return render_template('home.html', images=None, error="Text processing (F2) error.")

    # Similarity
    similarity = text_features @ image_feats_tensor.t()  # [1, num_images]
    top_k = min(TOP_K, similarity.shape[1])
    _, topk_indices = similarity.topk(top_k, dim=1, largest=True, sorted=True)

    topk_image_ids = [image_ids[idx] for idx in topk_indices[0].tolist()]

    # 3. Retrieve, then encrypt images
    retrieved_images = []
    for img_id in topk_image_ids:
        base64_image = image_id_to_data.get(img_id)  # Original base64
        if base64_image:
            # Encrypt image data using AES-GCM
            encrypted_img_b64 = encrypt_image_bytes(base64_image)
            retrieved_images.append(encrypted_img_b64)
        else:
            print(f"[Master] Image not found for ID: {img_id}")

    if not retrieved_images:
        return render_template('home.html', images=None, error="No images found.")

    # 4. Return template with encrypted images
    return render_template('home.html', images=retrieved_images)

@app.route('/attest', methods=['POST'])
def attest():
    logger.info("Received POST request for attestation.")
    attest_start_time = time.time()

    try:
        # Step 1: Generate attestation report using snpguest utility
        logger.info("Generating attestation report using snpguest utility.")

        # Generate request file
        request_file_path = 'request-file.txt'
        with open(request_file_path, 'w') as f:
            f.write('')  # Empty content as snpguest will generate random data

        # Run snpguest report command
        report_file = 'report.bin'
        cmd_report = [
            '/home/ubuntu/snpguest/target/release/snpguest',
            'report',
            report_file,
            request_file_path,
            '--random'
        ]
        logger.info(f"Running command: {' '.join(cmd_report)}")
        subprocess.run(cmd_report, check=True)
        logger.info(f"Attestation report generated at {report_file}.")

        # Run snpguest certificates command to generate the vlek.pem
        cmd_certificates = [
            '/home/ubuntu/snpguest/target/release/snpguest',
            'certificates',
            'PEM',
            './'
        ]
        logger.info(f"Running command: {' '.join(cmd_certificates)}")
        subprocess.run(cmd_certificates, check=True)
        logger.info("Certificates generated successfully.")

        # Ensure the vlek.pem certificate is available
        vlek_cert_file = './vlek.pem'
        if not os.path.exists(vlek_cert_file):
            logger.error("VLEK certificate not found.")
            return jsonify({
                'status': 'Failure',
                'details': 'VLEK certificate not found.'
            }), 500

        # Step 2: Validate the attestation report signature by sending to worker
        logger.info(f"Sending attestation report and certificate to worker at {WORKER_URL}.")

        with open(report_file, 'rb') as f:
            report_data = f.read()

        with open(vlek_cert_file, 'rb') as f:
            cert_data = f.read()

        # Send report and certificate to worker endpoint for validation
        files = {
            'report': ('report.bin', report_data),
            'certificate': ('vlek.pem', cert_data)
        }

        response = requests.post(WORKER_URL, files=files, timeout=60)

        if response.status_code == 200:
            response_json = response.json()
            validation_result = response_json.get('validation', 'No validation result received.')
            logger.info(f"Validation result received: {validation_result}")
            status = 'Success'
            details = validation_result
        else:
            validation_result = f"Worker returned status code {response.status_code}."
            logger.error(f"Worker returned error: {response.text}")
            status = 'Failure'
            details = validation_result

        attest_time = time.time() - attest_start_time
        logger.info(f"Attestation process completed in {attest_time:.4f} seconds.")

        return jsonify({
            'status': status,
            'details': details,
            'attest_time': f"{attest_time:.4f} seconds"
        }), 200 if status == 'Success' else 500

    except subprocess.CalledProcessError as e:
        logger.error(f"Error running snpguest commands: {e}", exc_info=True)
        return jsonify({
            'status': 'Failure',
            'details': f"Error generating attestation report: {str(e)}"
        }), 500
    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with worker: {e}", exc_info=True)
        return jsonify({
            'status': 'Failure',
            'details': f"Error communicating with worker for attestation validation: {str(e)}"
        }), 500
    except Exception as e:
        logger.error(f"Unexpected error during attestation: {e}", exc_info=True)
        return jsonify({
            'status': 'Failure',
            'details': "An unexpected error occurred during attestation."
        }), 500
        
        
@app.route('/result', methods=['GET'])
def redirect_to_home():
    return redirect(url_for('home'))

# -------------------------------
# Main Entry Point
# -------------------------------
if __name__ == "__main__":
    print(f"[Master] Starting Flask app on port {FLASK_PORT}...")
    # SSL certificate context (uncomment in production)
    context = ('cert.pem', 'key.pem')  # Replace with your certificate and key paths
    app.run(host='0.0.0.0', port=FLASK_PORT, debug=False, ssl_context=context)

    # For demonstration, using plain HTTP:
    # app.run(host='0.0.0.0', port=FLASK_PORT, debug=False)
