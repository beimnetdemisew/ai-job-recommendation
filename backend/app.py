from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import hashlib
import logging
import requests
from extract import process_resume, save_to_database

# Initialize Flask app
app = Flask(__name__)

# Configure CORS with proper settings
CORS(app, resources={r"/process_resume": {"origins": "http://localhost:3000", "methods": ["POST"], "allow_headers": ["Content-Type"]}})

# Configuration constants
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
UPLOAD_FOLDER = 'uploads'

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
)

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Validate file extensions"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_file_hash(file_path):
    """Generate SHA-256 hash of the file"""
    sha_hash = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(4096):
                sha_hash.update(chunk)
        return sha_hash.hexdigest()
    except Exception as e:
        logging.error(f"Hash generation failed: {str(e)}")
        return None
    
@app.route('/process_resume', methods=['POST', 'OPTIONS'])
def handle_resume():
    """Handle resume processing endpoint"""
    file_hash = None  # declare early so we can return it even on failure

    if request.method == 'OPTIONS':
        response = jsonify()
        response.headers.add("Access-Control-Allow-Origin", "http://localhost:3000")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST")
        return response

    if 'resume' not in request.files:
        logging.warning("No file in request")
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['resume']

    if file.filename == '':
        logging.warning("Empty filename")
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        logging.warning(f"Invalid file type: {file.filename}")
        return jsonify({"error": "Invalid file type"}), 400

    try:
        # Save file
        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(temp_path)
        logging.info(f"File saved temporarily: {temp_path}")

        # Generate file hash
        file_hash = generate_file_hash(temp_path)
        logging.info(f"Generated file hash: {file_hash}")
        if not file_hash:
            return jsonify({"error": "File hash generation failed"}), 500

        # Extract entities
        logging.info("Starting resume processing")
        entities = process_resume(temp_path)
        if not entities:
            logging.error("Failed to extract entities")
            return jsonify({"error": "Resume processing failed", "file_hash": file_hash}), 500

        # Save to DB
        logging.info("Attempting database save")
        if not save_to_database(entities, file_hash, filename):
            logging.error("Database save failed")
            return jsonify({"error": "Database operation failed", "file_hash": file_hash}), 500

        # Call job matcher
        job_matching_url = f"http://localhost:5001/match_jobs/{file_hash}"
        matching_response = requests.get(job_matching_url)
        if matching_response.status_code != 200:
            logging.error(f"Job matching failed: {matching_response.text}")
            return jsonify({"error": "Job matching failed", "file_hash": file_hash}), 500

        matches = matching_response.json().get("matches", [])

        # Cleanup
        os.remove(temp_path)
        logging.info("Temporary file removed")

        return jsonify({
            "message": "Resume processed successfully",
            "file_hash": file_hash,
            "entities": entities,
            "job_matches": matches
        }), 200

    except Exception as e:
        logging.error(f"Processing failed: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "file_hash": file_hash,
            "details": str(e)
        }), 500


# --- Run the Flask app ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)
