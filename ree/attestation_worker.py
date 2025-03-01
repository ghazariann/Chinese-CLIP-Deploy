# worker/attestation_worker.py

from flask import Flask, request, jsonify
import logging
import subprocess
import os
import warnings 
import urllib3
import sys 
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(MAIN_DIR)


app = Flask(__name__)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration Parameters
CERT_CHAIN_PATH = os.path.join(MAIN_DIR, 'ree/keys/cert_chain.pem')  # Path to cert_chain.pem
# SNPGUEST_PATH = './snpguest'         # Path to snpguest executable

@app.route('/validate_attestation', methods=['POST'])
def validate_attestation():
    logger.info("Received attestation report and certificate for validation.")
    
    # Get the report and certificate from the request
    report_file = request.files.get('report')
    certificate_file = request.files.get('certificate')

    if not report_file or not certificate_file:
        return jsonify({'validation': 'Error: Missing report or certificate.'}), 400

    # Save the received files
    report_path = os.path.join(MAIN_DIR, 'ree/keys/report.bin') 
    cert_path = os.path.join(MAIN_DIR, 'ree/keys/vlek.pem') 
    
    report_file.save(report_path)
    certificate_file.save(cert_path)
    
    # Verify the certificate using OpenSSL
    if not os.path.exists(CERT_CHAIN_PATH):
        return jsonify({'validation': 'Error: Certificate chain file not found.'}), 500

    try:
        logger.info("Verifying VLEK certificate using OpenSSL.")
        # Run OpenSSL verify command
        cmd_verify_cert = ['sudo', 'openssl', 'verify', '--CAfile', CERT_CHAIN_PATH, cert_path]
        result = subprocess.run(cmd_verify_cert, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        cert_verification_result = result.stdout.decode().strip()
        stderr_output = result.stderr.decode().strip()

        # Print both the output and error streams
        # print("Standard Output:", stdout_output)
        # print("Standard Error:", stderr_output)
        # cert_verification_result = "VLEK certificate verified successfully."
        logger.info(cert_verification_result)
    except subprocess.CalledProcessError as e:
        cert_verification_result = f"VLEK certificate verification failed: {e.stderr.decode()}"
        logger.error(cert_verification_result)
        return jsonify({'validation': cert_verification_result}), 500
    
    # Verify the attestation report using snpguest
    try:
        logger.info("Verifying attestation report using snpguest.")
        cmd_verify_attestation = [
            '/home/ubuntu/snpguest/target/release/snpguest', 'verify', 'attestation', "./ree/keys/", report_path
        ]
        result = subprocess.run(cmd_verify_attestation, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        attestation_verification_result = result.stdout.decode().strip()
        stderr_output = result.stderr.decode().strip()

      
        # attestation_verification_result = "Attestation verified successfully."
        logger.info(attestation_verification_result)
    except subprocess.CalledProcessError as e:
        attestation_verification_result = f"Attestation verification failed: {e.stderr.decode()}"
        logger.error(attestation_verification_result)
        return jsonify({'validation': attestation_verification_result}), 500

    # Send the validation results back to the master
    validation_result = f"{cert_verification_result} ./ {attestation_verification_result}"
    return jsonify({'validation': validation_result}), 200

if __name__ == '__main__':
    import flask.cli
    import warnings

    flask.cli.show_server_banner = lambda *args: None
    warnings.filterwarnings("ignore", category=UserWarning, message=".*development server.*")
    warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
    logger.info("Starting Attestation Worker Flask server...")
    app.run(host='0.0.0.0', port=5001, debug=False)  # Set debug=False for production
