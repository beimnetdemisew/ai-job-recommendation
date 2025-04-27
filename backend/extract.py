import os
import re
import sys
import time
import logging
import hashlib
import docx
import fitz
import pytesseract
import spacy
import psycopg2
import numpy as np
from PIL import Image, ImageEnhance
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
from typing import Dict, List, Set, Optional
from pydantic_settings import BaseSettings
from spacy.matcher import PhraseMatcher
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from unidecode import unidecode
from psycopg2.pool import SimpleConnectionPool
from psycopg2 import OperationalError, DataError, Error


# --------------------------
# Flask App Setup
# --------------------------
app = Flask(__name__)
CORS(app, resources={r"/process_resume": {"origins": "*"}})

# --------------------------
# Configuration Settings
# --------------------------
class Settings(BaseSettings):
    db_name: str = "aiProject"
    db_user: str = "postgres"
    db_pass: str = "1234"
    db_host: str = "localhost"
    db_port: str = "5432"
    
    class Config:
        env_file = ".env"
        env_prefix = "RESUME_PARSER_"

settings = Settings()

# --------------------------
# Logging Configuration
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("resume_parser.log"), logging.StreamHandler(sys.stdout)]
)

# --------------------------
# Database Connection Pool
# --------------------------
connection_pool = SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    database=settings.db_name,
    user=settings.db_user,
    password=settings.db_pass,
    host=settings.db_host,
    port=settings.db_port
)

# --------------------------
# NLP Models and Skills Data
# --------------------------
nlp = spacy.load("en_core_web_sm")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

SKILLS = {
    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'aws',
    'docker', 'kubernetes', 'tensorflow', 'pytorch', 'react', 'angular',
    'node.js', 'postgresql', 'mongodb', 'git', 'linux', 'sql', 'html',
    'css', 'bash', 'azure', 'google cloud', 'django', 'flask', 'machine learning',
    'deep learning', 'data analysis', 'rest api', 'graphql', 'jenkins', 'ansible'
}

SKILL_SYNONYMS = {
    'js': 'javascript', 'node': 'node.js', 'ml': 'machine learning',
    'dl': 'deep learning', 'gcp': 'google cloud', 'postgres': 'postgresql',
    'ai': 'artificial intelligence', 'ci/cd': 'continuous integration'
}

matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
for skill in SKILLS:
    matcher.add("SKILLS", [nlp.make_doc(skill)])

# --------------------------
# File Handling Utilities
# --------------------------
def validate_file_path(input_path: str) -> str:
    try:
        abs_path = os.path.abspath(input_path)
        logging.info(f"Resolved path: {abs_path}")

        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Path does not exist: {abs_path}")
        
        if not os.path.isfile(abs_path):
            raise IsADirectoryError(f"Path is a directory: {abs_path}")
        
        valid_ext = ('.pdf', '.docx')
        if not abs_path.lower().endswith(valid_ext):
            raise ValueError(f"Invalid file type. Supported: {valid_ext}")
        
        return abs_path
    
    except Exception as e:
        logging.error(f"Path validation failed: {str(e)}")
        return None

def generate_resume_hash(file_path: str) -> str:
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

# --------------------------
# Text Processing
# --------------------------
def extract_text(file_path: str) -> Optional[str]:
    try:
        if file_path.lower().endswith(".docx"):
            doc = docx.Document(file_path)
            return "\n".join(para.text for para in doc.paragraphs)
        
        elif file_path.lower().endswith(".pdf"):
            with fitz.open(file_path) as pdf:
                return "\n".join(page.get_text() for page in pdf)
        
        raise ValueError(f"Unsupported file type: {os.path.splitext(file_path)[1]}") 
    
    except Exception as e:
        logging.error(f"Text extraction failed: {e}")
        return None

# --------------------------
# Entity Extraction
# --------------------------
def normalize_skill(skill: str) -> str:
    """Standardize skill names"""
    skill = unidecode(skill).lower().strip()
    skill = re.sub(r'[^a-z0-9+#. ]', ' ', skill)
    skill = re.sub(r'\s+', ' ', skill)
    return SKILL_SYNONYMS.get(skill, skill)

def extract_skills(text: str) -> Set[str]:
    """Hybrid skill detection"""
    skills = set()
    doc = nlp(text)
    
    # Pattern matching
    for _, start, end in matcher(doc):
        skills.add(normalize_skill(doc[start:end].text))
    
    # Semantic matching
    sentences = [sent.text for sent in doc.sents]
    corpus = list(SKILLS)
    corpus_embeddings = sentence_model.encode(corpus)
    
    for sent in sentences:
        sent_embedding = sentence_model.encode([sent])
        scores = cosine_similarity(sent_embedding, corpus_embeddings)[0]
        skills.update(corpus[i] for i in np.where(scores > 0.4)[0])
    
    return skills

def extract_experience(text: str) -> List[str]:
    """Experience duration extraction"""
    exp_pattern = re.compile(r"\b(\d{1,2}[+]?)\s*(years?|yrs)\b", re.IGNORECASE)
    return list(set(f"{m.group(1)} {m.group(2)}".lower() for m in exp_pattern.finditer(text)))

def extract_education(text: str) -> List[str]:
    """Education information extraction"""
    education = set()
    doc = nlp(text)
    
    # Institutions
    for ent in doc.ents:
        if ent.label_ in ["ORG", "FAC"] and any(
            kw in ent.text.lower() 
            for kw in {"university", "college", "institute"}
        ):
            education.add(ent.text)
    
    # Degrees
    degrees = re.findall(
        r"\b(B\.?Sc|B\.?A|B\.?Tech|M\.?Sc|M\.?A|Ph\.?D)\b", 
        text, 
        re.IGNORECASE
    )
    education.update(degrees)
    
    return list(education)

def extract_location(text: str) -> List[str]:
    """Location extraction for Ethiopian cities"""
    locations = set()
    doc = nlp(text)
    ethiopian_cities = {
        "addis ababa", "bahir dar", "dire dawa", "mekelle", "hawassa",
        "gondar", "jimma", "axum", "lalibela", "adama", "arba minch"
    }
    
    for ent in doc.ents:
        if ent.label_ == "GPE" and ent.text.lower() in ethiopian_cities:
            locations.add(ent.text.title())
    
    return list(locations)

# --------------------------
# Database Connection and Saving Data
# --------------------------
def save_to_database(entities: Dict[str, List[str]], file_hash: str, filename: str) -> bool:
    """Store extracted entities in the database"""
    max_retries = 3
    for attempt in range(max_retries):
        conn = None
        try:
            conn = connection_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute(""" 
                    CREATE TABLE IF NOT EXISTS entities (
                        id SERIAL PRIMARY KEY,
                        file_hash VARCHAR(64) UNIQUE NOT NULL,
                        filename TEXT NOT NULL,
                        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        skills TEXT[],
                        experience TEXT[],
                        education TEXT[],
                        location TEXT[]
                    )
                """)

                # Safely cast all list fields to lists in case None is returned
                skills = entities.get("skills") or []
                experience = entities.get("experience") or []
                education = entities.get("education") or []
                location = entities.get("location") or []

                cursor.execute(""" 
                    INSERT INTO entities
                    (file_hash, filename, skills, experience, education, location)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (file_hash) DO NOTHING
                """, (
                    file_hash,
                    filename,
                    skills,
                    experience,
                    education,
                    location
                ))
                conn.commit()
                
                if cursor.rowcount > 0:
                    logging.info(f"Saved resume: {filename}")
                    return True
                logging.info(f"Duplicate resume: {filename}")
                return False

        except OperationalError as e:
            logging.warning(f"Connection error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                logging.error("Max connection retries reached")
                return False
            time.sleep(2 ** attempt)

        except DataError as e:
            logging.error(f"Data validation error: {e}")
            return False

        except Error as e:
            logging.error(f"Database error: {e}")
            return False

        finally:
            if conn:
                connection_pool.putconn(conn)
    return False

# --------------------------
# Resume Processing
# --------------------------
def process_resume(file_path: str) -> Optional[Dict[str, List[str]]]:
    """Process the resume and extract entities"""
    try:
        text = extract_text(file_path)
        if not text:
            logging.error(f"Text extraction failed for {file_path}")
            return None

        entities = {
    "skills": list(extract_skills(text)) or [],
    "experience": extract_experience(text) or [],
    "education": extract_education(text) or [],
    "location": extract_location(text) or []
}

        logging.info(f"Extracted entities: {entities}")
        return entities

    except Exception as e:
        logging.error(f"Error during resume processing: {e}")
        return None

# --------------------------
# API Endpoint
# --------------------------
@app.route('/process_resume', methods=['POST'])
def process_resume_api():
    file = request.files.get('resume')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    
    try:
        os.makedirs("uploads", exist_ok=True)
        temp_path = os.path.join("uploads", file.filename)
        file.save(temp_path)
        
        entities = process_resume(temp_path)
        if not entities:
            return jsonify({"error": "Failed to extract entities"}), 500
        
        file_hash = generate_resume_hash(temp_path)
        saved = save_to_database(entities, file_hash, file.filename)
        
        if saved:
            return jsonify(entities), 200
        else:
            return jsonify({"error": "Duplicate resume or failed to save"}), 400
    
    except Exception as e:
        logging.error(f"Error during file processing: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
