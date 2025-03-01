# i2i_app_dist.py
# -*- coding: utf-8 -*-

import sys 
import os

MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(MAIN_DIR)

import json
import base64
import csv
import random
import uuid
import numpy as np
import torch
import torch.distributed as dist

from flask import Flask, request, render_template, redirect, url_for, jsonify, Response
from tqdm import tqdm
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from PIL import Image
from io import BytesIO
import logging
import time
import requests
import subprocess

from cn_clip.clip.model import convert_weights, CLIP
from cn_clip.clip.utils import convert_models_to_fp32, seed_everything

class FlushFileHandler(logging.FileHandler):
    """
    Custom logging handler that flushes the stream after each log entry.
    """
    def emit(self, record):
        super().emit(record)
        self.flush()
        
app = Flask(__name__)

# -------------------------------
# Distributed Configuration
# -------------------------------
WORKER_URL = 'http://18.217.90.50:5001/validate_attestation'
MASTER_ADDR = '18.116.182.172'  # Master server IP (replace with your actual IP)
MASTER_PORT = '29500'           # Must match worker's MASTER_PORT
WORLD_SIZE = 2                  # Total number of processes (master + worker)
RANK = 0                        # Master rank

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
USE_DB = False
USE_DISTRIBUTED = True
USE_ATTEST = True

# File Paths
IMAGE_DATA_PATH = "database/valid_imgs.tsv"
IMAGE_FEAT_PATH = "database/valid_imgs.img_feat.jsonl"
CHECKPOINT_PATH = "data/pretrained_weights/Tencrypted_clip_cn_vit-b-16.pt"
REMOTE_IMAGE_DB_URL = "http://18.217.90.50:5002/fetch_and_encrypt_images"
REMOTE_IMAGE_FEATURES_DB_URL = "http://18.217.90.50:5002/get_image_features"

# -------------------------------
# Encryption Configuration
# -------------------------------
AES_KEY = b'\x89\xc3\xcf\x17\x8fU\x80\xbd\xc0S`#\xf0\xd8\xd7\x8b\x96Q\xcb\xf6C\xdfp\x11P\x0b\x91[`:0\xbc'

  # 32-byte random key
aesgcm = AESGCM(AES_KEY)


# Fetch image features from the DB server
def fetch_image_features():
    try:
        response = requests.get(REMOTE_IMAGE_FEATURES_DB_URL)  # Replace with actual DB server IP
        if response.status_code == 200:
            data = response.json()
            image_ids = data['image_ids']
            image_feats_tensor = torch.tensor(data['image_features'])
            return image_ids, image_feats_tensor
        else:
            logger.error(f"Failed to fetch image features. Status code: {response.status_code}")
            return [], None
    except Exception as e:
        logger.error(f"Error fetching image features: {e}")
        return [], None
    
    
def decrypt_image(encrypted_b64: str) -> bytes:
    """
    Decrypts AES-GCM encrypted image data.

    Args:
        encrypted_b64: Base64-encoded (nonce + ciphertext).

    Returns:
        Decrypted image bytes or None if decryption fails.
    """
    try:
        encrypted_data = base64.b64decode(encrypted_b64)
        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:]
        decrypted = aesgcm.decrypt(nonce, ciphertext, None)
        # logger.info("[Master] 🔓 Decrypted image data successfully.")
        return decrypted
    except Exception as e:
        logger.error(f"Error decrypting image: {e}")
        return None

def encrypt_image_bytes(image_b64: str) -> str:
    """
    Encrypts image bytes using AES-GCM.

    Args:
        image_b64: Base64-encoded image data.

    Returns:
        Base64-encoded (nonce + ciphertext) or empty string if encryption fails.
    """
    try:
        raw_image_bytes = base64.b64decode(image_b64.encode("utf-8"))
        nonce = os.urandom(12)
        ciphertext = aesgcm.encrypt(nonce, raw_image_bytes, None)
        combined = nonce + ciphertext
        encrypted_b64 = base64.b64encode(combined).decode('utf-8')
        # logger.info("[Master] 🔒 Encrypted image data successfully.")
        return encrypted_b64
    except Exception as e:
        logger.error(f"Error encrypting image bytes: {e}")
        return ""


# -------------------------------
# -------------------------------
# Logging Configuration
# log_formatter = logging.Formatter('[TEE] %(message)s')
log_file = 'server.log'

# Ensure the logs directory exists
if not os.path.exists('logs'):
    os.makedirs('logs')

file_handler = FlushFileHandler(log_file)
# file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
# console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

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
    # logger.info("Distributed process group initialized.")

# -------------------------------
# Load CLIP Model and Data
# -------------------------------
def load_model_and_data():
    # logger.info("[Master] 🚀 Initializing the application...")
    # Seed
    seed_everything(42)
    # logger.info("[Master] 🎲 Seed set to 42.")

    # Distributed init
    init_distributed()

    # Load model configuration files
    vision_config_file = f"cn_clip/clip/model_configs/{VISION_MODEL_NAME.replace('/', '-')}.json"
    text_config_file = f"cn_clip/clip/model_configs/{TEXT_MODEL_NAME.replace('/', '-')}.json"

    assert os.path.exists(vision_config_file), f"Vision config not found: {vision_config_file}"
    assert os.path.exists(text_config_file), f"Text config not found: {text_config_file}"
    # logger.info("Configuration files loaded.")

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
        # logger.info(f"Model precision set to {PRECISION}.")
    if PRECISION == "fp16":
        convert_weights(model)

    # Load checkpoint
    assert os.path.exists(CHECKPOINT_PATH), f"Checkpoint not found: {CHECKPOINT_PATH}"
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    sd = checkpoint["state_dict"]
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}
    model.load_state_dict(sd)
    # logger.info("[TEE] Load the model checkpoint.")
    # logger.info("[可信执行环境] 加载模型")
    

    model.eval()

    # logger.info("Getting image features from the Database...")
    image_id_to_data = {}
    image_ids = []
    image_feats_tensor = None
    # logger.info("[TEE] Loading retrieval target embeddings from Database.")
    logger.info("[可信执行环境] 从数据库加载目标图片的特征向量")
    
    # logger.info("[REE] Load the model checkpoint.") # TODO get from REE
    logger.info("[常规执行环境] 加载模型")
    if USE_DB:
        image_ids, image_feats_tensor = fetch_image_features()
    # logger.info(f"Successfully fetched {len(image_ids)} image features from the DB.")
    
    else:
        image_feats = []
        with open(IMAGE_FEAT_PATH, "r", encoding='utf-8') as fin:
            for line in tqdm(fin, desc="Reading Image Feats"):
                obj = json.loads(line.strip())
                image_ids.append(obj['image_id'])
                image_feats.append(obj['feature'])
        image_feats_array = np.array(image_feats, dtype=np.float32)
        image_feats_tensor = torch.from_numpy(image_feats_array)
    # logger.info(f"Got {len(image_ids)} image features.")
    # # Load image data
    # logger.info("Loading image data from TSV...")
    # with open(IMAGE_DATA_PATH, 'r', encoding='utf-8') as tsv_file:
    #     reader = csv.reader(tsv_file, delimiter='\t')
    #     for row in tqdm(reader, desc="Reading Image Data"):
    #         if len(row) < 2:
    #             continue
    #         img_id, base64_data = row
    #         image_id_to_data[int(img_id)] = base64_data
    # logger.info(f"Loaded {len(image_id_to_data)} images in the database.")

    # Convert to tensor
    # image_feats_tensor = image_feats_tensor
    # logger.info("Image features converted to tensor.")

    return model, image_ids, image_feats_tensor, image_id_to_data

# Initialize model and data
model, image_ids, image_feats_tensor, image_id_to_data = load_model_and_data()

# -------------------------------
# Transforms for Input Image
# -------------------------------
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

def _convert_to_rgb(image):
    return image.convert('RGB')

image_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    _convert_to_rgb,
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    )
])

# -------------------------------
# Flask Routes
# -------------------------------

@app.route('/', methods=['GET'])
def home():
    """
    Render the home_i2i.html template from templates/ directory.
    """
    logger.info("Home page accessed.")
    return render_template('home_i2i.html')

@app.route('/get_aes_key', methods=['GET'])
def get_aes_key():
    """
    Provide the AES key to the client in hexadecimal format.
    This endpoint should be secured (e.g., require authentication) in production.
    """
    try:
        aes_key_hex = AES_KEY.hex()
        response = {'aes_key': aes_key_hex}
        logger.info("[可信执行环境] 向客户端发送 AES 密钥")
        # logger.info("[TEE] Send AES key to client.")
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Failed to provide AES key: {e}")
        return jsonify({'error': 'Failed to provide AES key.'}), 500

def fetch_and_encrypt_images(image_ids):
    try:
        response = requests.post(REMOTE_IMAGE_DB_URL, json={'image_ids': image_ids})
        if response.status_code == 200:
            return response.json()['images']  # Assuming the response contains the encrypted images in this field
        else:
            logger.error(f"Failed to fetch images from DB. Status code: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error fetching and encrypting images: {e}")
        return None
    
@app.route('/logs-stream', methods=['GET'])
def logs_stream():
    """
    Stream the log file to the client in real-time using Server-Sent Events (SSE).
    """
    def generate():
        with open(log_file, 'r') as f:
            # Move to the end of the file
            f.seek(0, os.SEEK_END)
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                # SSE requires 'data: ' prefix and double newline
                yield f"data: {line}\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route('/result', methods=['POST'])
def get_similar_images():
    """
    Handle image upload, process it, and return encrypted similar images as JSON.
    """
    encrypted_image_b64 = request.form.get('image', '').strip()
    if not encrypted_image_b64:
        logger.warning("No image provided in the request.")
        return jsonify({'error': 'No image provided.'}), 400

    # logger.info("[Client] Receive AES key, encrypt the image and send it to TEE.")
    logger.info("[客户端] 接收 AES 密钥，加密检索图片，发送至可信执行环境")

    # 1. Decrypt the received image
    decrypted_image_bytes = decrypt_image(encrypted_image_b64)
    if decrypted_image_bytes is None:
        logger.error("Failed to decrypt the uploaded image.")
        return jsonify({'error': 'Error decrypting the uploaded image.'}), 400

    # logger.info("[TEE] Receive the encrypted image and decrypt it.")
    logger.info("[可信执行环境] 接收加密后的图片，解密图片")

    # 2. Convert bytes to PIL Image
    try:
        pil_image = Image.open(BytesIO(decrypted_image_bytes)).convert('RGB')
        # logger.info("Converted bytes to PIL Image.")
    except Exception as e:
        logger.error(f"Error loading decrypted image: {e}")
        return jsonify({'error': 'Error loading the decrypted image.'}), 400

    # 3. Transform the image for the model
    input_tensor = image_transform(pil_image).unsqueeze(0)  # shape = [1, 3, 224, 224]
    # logger.info("[TEE] Encode the image into intermediate embedding.")
    logger.info("[可信执行环境] 将图片编码为中间特征向量")

    # 4. Process image up to F1_forward
    try:
        embedding = model.visual.F1_forward(input_tensor)
        # logger.info(f"Embedding shape: {embedding.shape}")
    except Exception as e:
        logger.error(f"Error during F1_forward: {e}")
        return jsonify({'error': 'Image processing (F1) error.'}), 500

    # 5. Send embedding to worker
    try:
        dist.send(tensor=embedding, dst=1)  # Assuming worker rank is 1
        # logger.info("[TEE] Send intermediate embedding to REE.")
        # logger.info("[REE] Receive intermediate embedding.")
        # logger.info("[REE] Encode the intermediate embedding into final embedding.")
        # logger.info("[REE] Send final embedding to TEE.")

        logger.info("[可信执行环境] 将中间特征向量发送至常规执行环境")
        logger.info("[常规执行环境] 接收中间特征向量")
        logger.info("[常规执行环境] 将中间特征向量编码为最终特征向量")
        logger.info("[常规执行环境] 将最终特征向量发送至可信执行环境")
    except Exception as e:
        logger.error(f"Error sending embedding to worker: {e}")
        return jsonify({'error': 'Failed to communicate with worker.'}), 500

    # 6. Receive sequence_output from worker
    try:
        # sequence_output = model.visual.Backbone_forward(embedding)
        sequence_output = torch.zeros_like(embedding)
        dist.recv(tensor=sequence_output, src=1)
        # logger.info("[TEE] Receive final embedding.")
        logger.info("[可信执行环境] 接收最终特征向量")
    except Exception as e:
        logger.error(f"Error receiving sequence_output from worker: {e}")
        return jsonify({'error': 'Failed to receive data from worker.'}), 500

    # 7. F2 forward to get image features
    try:
        image_features = model.visual.F2_forward(sequence_output)[0].type(model.dtype)
        # logger.info(f"Image features shape: {image_features.shape}")

        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)  # Convert to [1, 512]
            # logger.info(f"Reshaped image_features to {image_features.shape}")

        image_features /= image_features.norm(dim=-1, keepdim=True)
        # logger.info("Normalized image features.")
    except Exception as e:
        logger.error(f"Error during F2_forward: {e}")
        return jsonify({'error': 'Image processing (F2) error.'}), 500

    # Getting image features from the DB
    # logger.info("Fetching image features from database...")
    # try:
    #     image_ids, image_feats_tensor = fetch_image_features()
    #     # time.sleep(3)
    #     logger.info(f"Successfully fetched {len(image_ids)} image features from the DB.")
    # except Exception as e:
    #     logger.error(f"Error fetching image features from DB: {e}")
    #     return jsonify({'error': 'Error fetching image features from the database.'}), 500
    # 8. Compute similarity with all images in the database
    try:
        similarity = image_features @ image_feats_tensor.T  # shape: [1, num_images]
        # logger.info("[TEE] Compute similarity score of retrieval target embeddings and final embedding.")
        logger.info("[可信执行环境] 计算检索图片的最终特征向量和目标图片的特征向量之间的相似度分数")
    except Exception as e:
        logger.error(f"Error computing similarity: {e}")
        return jsonify({'error': 'Error computing similarity.'}), 500

    # 9. Retrieve top-K image IDs
    try:
        top_k = min(TOP_K, similarity.shape[1])
        _, topk_indices = similarity.topk(top_k, dim=1, largest=True, sorted=True)
        topk_image_ids = [image_ids[idx] for idx in topk_indices[0].tolist()]
        # logger.info(f"Retrieved top-{top_k} images")
    except Exception as e:
        logger.error(f"Error retrieving top-K images: {e}")
        return jsonify({'error': 'Error retrieving top-K images.'}), 500

    random_indices = random.sample(image_ids, 2 ** TOP_K)
    combined_image_ids = random_indices + topk_image_ids  # Combine as list of image IDs

    # logger.info("[TEE] Get the top-3 retrieval image indexes.")
    # logger.info("[TEE] Add confusion indexes and send them together to Database.")
    logger.info("[可信执行环境] 获取目标图片中相似度分数前三的图片的索引")
    logger.info("[可信执行环境] 添加混淆索引并将它们一起发送到数据库")
    # Combine random indices and to p-k indices
    # logger.info(f"Requesting random {len(combined_image_ids)} images from DB: {combined_image_ids}")
    # logger.info("[Database] Receive indexes, send images back to TEE.")
    # logger.info("[TEE] Receive images, delete images corresponding to confusion indexes.")
    logger.info("[数据库] 接收索引，将相应图片发送至可信执行环境")
    logger.info("[可信执行环境] 接收图片，删除混淆索引对应的图片")
    # 10. Retrieve and encrypt images
    retrieved_images = fetch_and_encrypt_images(combined_image_ids)

    if not retrieved_images:
        logger.error("No similar images found.")
        return jsonify({'error': 'No similar images found.'}), 404
    filtered_images = [retrieved_images[str(img_id)] for img_id in topk_image_ids if str(img_id) in retrieved_images]

    # logger.info("[TEE] Encrypt images and send it to Client.")
    # logger.info("[Client] Receive encrypted images and decrypt them. Retrieval done!")
    logger.info("[可信执行环境] 加密图片并发送至客户端")
    logger.info("[客户端] 接收并解密图片，至此检索完成")
    # print(filtered_images)
    # 11. Return JSON response with encrypted images
    return jsonify({'images': filtered_images}), 200

@app.route('/attest', methods=['POST'])
def attest():
    attest_start_time = time.time()

    try:
        # Step 1: Generate attestation report using snpguest utility
        # logger.info("Generating attestation report using snpguest utility.")
        # logger.info("[Client] Select one image to do a retrieval.")
        # logger.info("[Client] Send remote attestation request to TEE.")
        # logger.info("[TEE] Send attestation report to client.")
        logger.info("[可信执行环境] 加载模型")
        logger.info("[可信执行环境] 从数据库加载目标图片的特征向量")
        logger.info("[常规执行环境] 加载模型")

        logger.info("[客户端] 选择一张图片进行检索")
        logger.info("[客户端] 向可信执行环境发送远程证明请求")
        logger.info("[可信执行环境] 向客户端发送证明报告")

        # Generate request file
        request_file_path = os.path.join(MAIN_DIR, 'tee/keys/request-file.txt')
        # with open(request_file_path, 'w') as f:
        #     f.write('')  # Empty content as snpguest will generate random data
        # logger.info(f"Generated request file at {request_file_path}.")

        # Run snpguest report command
        report_file = os.path.join(MAIN_DIR, 'tee/keys/report.bin')
        cmd_report = [
            '/home/ubuntu/snpguest/target/release/snpguest',
            'report',
            report_file,
            request_file_path,
            '--random'
        ]
        # logger.info(f"Running command: {' '.join(cmd_report)}")
        # subprocess.run(cmd_report, check=True)
        # logger.info(f"Attestation report generated at {report_file}.")

        # Run snpguest certificates command to generate the vlek.pem
        cmd_certificates = [
            '/home/ubuntu/snpguest/target/release/snpguest',
            'certificates',
            'PEM',
            './'
        ]
        # logger.info(f"Running command: {' '.join(cmd_certificates)}")
        # subprocess.run(cmd_certificates, check=True)
        # logger.info("Certificates generated successfully.")

        # Ensure the vlek.pem certificate is available
        vlek_cert_file = os.path.join(MAIN_DIR, 'tee/keys/vlek.pem')
        if not os.path.exists(vlek_cert_file):
            logger.error("VLEK certificate not found.")
            return jsonify({
                'status': 'Failure',
                'details': 'VLEK certificate not found.'
            }), 500

        # Step 2: Validate the attestation report signature by sending to worker
        # logger.info(f"Sending attestation report and certificate to worker at {WORKER_URL}.")

        with open(report_file, 'rb') as f:
            report_data = f.read()

        with open(vlek_cert_file, 'rb') as f:
            cert_data = f.read()

        # Send report and certificate to worker endpoint for validation
        files = {
            'report': (os.path.join(MAIN_DIR, 'tee/keys/report.bin'), report_data),
            'certificate': (os.path.join(MAIN_DIR, 'tee/keys/vlek.pem'), cert_data)
        }

        response = requests.post(WORKER_URL, files=files, timeout=60)

        if response.status_code == 200:
            response_json = response.json()
            validation_result = response_json.get('validation', 'No validation result received.')
            # logger.info("[Client] Receive attestation report.")
            logger.info("[客户端] 接收证明报告")
            status = '成功'
            details = validation_result
        else:
            validation_result = f"Worker returned status code {response.status_code}."
            logger.error(f"Worker returned error: {response.text}")
            status = 'Failure'
            details = validation_result

        attest_time = time.time() - attest_start_time
        # logger.info(f"Attestation process completed in {attest_time:.4f} seconds.")

        return jsonify({
            'status': status,
            'details': details,
            'attest_time': f"{attest_time:.4f} seconds"
        }), 200 if status == '成功' else 500

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
    logger.info(f"Starting Flask app on port {FLASK_PORT}...")
    # SSL certificate context (uncomment in production)
    # Replace 'cert.pem' and 'key.pem' with your actual certificate and key files
    # Ensure that you have a valid SSL certificate for HTTPS
    context = (os.path.join(MAIN_DIR, 'tee/keys/cert.pem'), os.path.join(MAIN_DIR, 'tee/keys/key.pem')) # Replace with your certificate and key paths
    try:
        app.run(host='0.0.0.0', port=FLASK_PORT, debug=False, ssl_context=context)
    except Exception as e:
        logger.error(f"Failed to start Flask app with SSL: {e}")
        logger.info("Starting Flask app without SSL.")
        app.run(host='0.0.0.0', port=FLASK_PORT, debug=False)
