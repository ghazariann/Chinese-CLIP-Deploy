import base64
import os
import logging
import numpy as np
import requests
import csv
import torch
from flask import Flask, request, jsonify
from tqdm import tqdm
import json
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from urllib3.exceptions import InsecureRequestWarning
import sys

import flask.cli
flask.cli.show_server_banner = lambda *args: None

MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(MAIN_DIR)
# Logger setup
logger = logging.getLogger(__name__)
AES_KEY = b'\x89\xc3\xcf\x17\x8fU\x80\xbd\xc0S`#\xf0\xd8\xd7\x8b\x96Q\xcb\xf6C\xdfp\x11P\x0b\x91[`:0\xbc'

  # 32-byte random key
aesgcm = AESGCM(AES_KEY)
# Constants
IMAGE_FEAT_PATH = os.path.join(MAIN_DIR, "database/valid_imgs.img_feat.jsonl")
IMAGE_DATA_PATH = os.path.join(MAIN_DIR, "database/valid_imgs.tsv")
REMOTE_AES_KEY_URL = "https://18.116.182.172:8080/get_aes_key"  # Replace with the actual remote server URL


import warnings

warnings.simplefilter('ignore', InsecureRequestWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*development server.*")

# Initialize Flask app
app = Flask(__name__)

class Database:
    def __init__(self):
        self.image_id_to_data = {}
        self.image_ids = []
        self.image_feats_tensor = None
        self.load_image_data()
        self.load_image_features()
        
        # Fetch AES key from remote server
        # global aesgcm
        # aes_key = self.fetch_aes_key()
        # aesgcm = AESGCM(aes_key)
        # if not aesgcm:
        #     logger.error("Failed to access AES key")

    def load_image_features(self):
        image_ids = []
        image_feats = []
        with open(IMAGE_FEAT_PATH, "r", encoding='utf-8') as fin:
            for line in tqdm(fin, desc="Reading Image Feats"):
                obj = json.loads(line.strip())
                image_ids.append(obj['image_id'])
                image_feats.append(obj['feature'])
        image_feats_array = np.array(image_feats, dtype=np.float32)
        self.image_feats_tensor = torch.from_numpy(image_feats_array)
        self.image_ids = image_ids
        logger.info(f"Loaded {len(self.image_ids)} image features.")

    def load_image_data(self):
        """
        Load image data (e.g., base64 data) from file.
        """
        try:
            with open(IMAGE_DATA_PATH, 'r', encoding='utf-8') as tsv_file:
                reader = csv.reader(tsv_file, delimiter='\t')
                for row in tqdm(reader, desc="Reading Image Data"):
                    if len(row) < 2:
                        continue
                    img_id, base64_data = row
                    self.image_id_to_data[int(img_id)] = base64_data
            logger.info(f"Loaded {len(self.image_id_to_data)} images in the database.")
        except Exception as e:
            logger.error(f"Error loading image data: {e}")
    
    def fetch_aes_key(self):
        """
        Fetch the AES key from a remote server.
        """
        try:
            response = requests.get(REMOTE_AES_KEY_URL, verify=False)  # Replace with secure flag in production
            if response.status_code == 200:
                aes_key_hex = response.json().get('aes_key')
                return bytes.fromhex(aes_key_hex)
            else:
                logger.error("Failed to fetch AES key.")
        except Exception as e:
            logger.error(f"Error fetching AES key: {e}")
        return None
    
    def fetch_and_encrypt_images(self, image_ids):
        """
        Fetch and encrypt images based on image IDs.

        Args:
            image_ids (list): List of image IDs to fetch and encrypt.

        Returns:
            List of encrypted image data in base64 format.
        """
        encrypted_images = {}
        for img_id in image_ids:
            if img_id in self.image_id_to_data:
                try:
                    image_data_b64 = self.image_id_to_data[img_id]
                    encrypted_data = self.encrypt_image_bytes(image_data_b64)
                    if encrypted_data:
                        encrypted_images[img_id] = encrypted_data
                    else:
                        logger.error(f"Error encrypting image {img_id}.")
                except Exception as e:
                    logger.error(f"Error encrypting image {img_id}: {e}")
            else:
                logger.warning(f"Image ID {img_id} not found in database.")
        logger.info("Receive indexes, send images back to TEE.")
        return encrypted_images
    
    def encrypt_image_bytes(self, image_b64):
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
            return encrypted_b64
        except Exception as e:
            logger.error(f"Error encrypting image bytes: {e}")
            return ""
    
    def get_image_features(self):
        """
        Return image IDs and feature tensors.
        """
        return jsonify({
            'image_ids': self.image_ids,
            'image_features': self.image_feats_tensor.tolist()
        })

# Initialize the Database instance
database = Database()

# Endpoint to get image features (image IDs + feature tensors)
@app.route('/get_image_features', methods=['GET'])
def get_image_features():
    return database.get_image_features()

# Endpoint to fetch and encrypt images based on image IDs
@app.route('/fetch_and_encrypt_images', methods=['POST'])
def fetch_and_encrypt_images():
    image_ids = request.json.get('image_ids')
    if not image_ids:
        return jsonify({'error': 'No image IDs provided'}), 400
    encrypted_images = database.fetch_and_encrypt_images(image_ids)
    if encrypted_images:
        return jsonify({'images': encrypted_images}), 200
    return jsonify({'error': 'Failed to fetch and encrypt images.'}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
