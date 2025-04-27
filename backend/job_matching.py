import psycopg2
import re
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from psycopg2.pool import SimpleConnectionPool

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# --- DB Connection Pool Setup ---
class DBConnectionPool:
    def __init__(self, dbname, user, password, host, port):
        self.pool = SimpleConnectionPool(1, 10, dbname=dbname, user=user, password=password, host=host, port=port)

    def get_connection(self):
        return self.pool.getconn()

    def release_connection(self, conn):
        self.pool.putconn(conn)

# Initialize the database connection pool
db_pool = DBConnectionPool(dbname="aiProject", user="postgres", password="1234", host="localhost", port="5432")

# --- Initialize spaCy NLP Model ---
nlp = spacy.load("en_core_web_sm")

# --- Fetch Resume Data from DB ---
def fetch_resume_data(file_hash):
    """Fetch resume data (skills, experience, education, location) from the database."""
    try:
        conn = db_pool.get_connection()
        with conn.cursor() as cur:
            cur.execute("""SELECT skills, experience, education, location 
                           FROM entities 
                           WHERE file_hash = %s""", (file_hash,))
            result = cur.fetchone()

        db_pool.release_connection(conn)

        if result:
            skills, experience, education, location = result
            return {
                'skills': skills,
                'experience': experience,
                'education': education,
                'location': location
            }
        return None
    except Exception as e:
        print(f"❌ DB error: {e}")
        return None

# --- Fetch Job Descriptions ---
def fetch_jobs():
    """Fetch all job descriptions from the database."""
    try:
        conn = db_pool.get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT jobid, title, description FROM job")
            jobs = cur.fetchall()

        db_pool.release_connection(conn)
        return jobs
    except Exception as e:
        print(f"❌ Error fetching jobs: {e}")
        return []

# --- Text Preprocessing with spaCy ---
def clean_text(text):
    """Clean and preprocess the input text."""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop])

# --- Match Jobs API ---
@app.route('/match_jobs/<file_hash>', methods=['GET'])
def match_jobs(file_hash):
    """Match resume against available jobs based on text similarity."""
    
    # Fetch the resume data using the file hash
    resume_data = fetch_resume_data(file_hash)
    if not resume_data:
        return jsonify({"error": "Resume not found"}), 404

    # Fetch the job descriptions from the database
    jobs = fetch_jobs()
    if not jobs:
        return jsonify({"error": "No jobs available"}), 404

    # Prepare the resume and job descriptions for similarity computation
    resume_text = f"{resume_data['skills']} {resume_data['experience']} {resume_data['education']} {resume_data['location']}"
    job_descriptions = [f"{title} {desc}" for _, title, desc in jobs]
    documents = [clean_text(resume_text)] + [clean_text(desc) for desc in job_descriptions]

    # Use TF-IDF Vectorizer to convert text to vector format
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Calculate cosine similarity between resume and job descriptions
    cosine_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

    # Rank jobs based on the cosine similarity scores
    ranked_jobs = sorted(
        [(jobs[i][0], jobs[i][1], cosine_scores[i]) for i in range(len(jobs))],
        key=lambda x: x[2], reverse=True
    )

    # Format results with relevance labels
    results = [
        {
            "job_id": jobid,
            "title": title,
            "match_score": round(score * 100, 2),
            "relevance": "High" if score >= 0.5 else "Medium" if score >= 0.3 else "Low"
        }
        for jobid, title, score in ranked_jobs[:5]  # Top 5 matches
    ]

    return jsonify({"matches": results}), 200

# --- Run Server ---
if __name__ == '__main__':
    app.run(debug=True, port=5001)
