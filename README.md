This project is an AI-powered job recommendation platform designed to match candidates with the best-fit job opportunities based on their resume data. It leverages NLP techniques to extract key information (skills, education, experience) from uploaded resumes and uses content-based filtering to recommend relevant jobs. 

Key Features 

📄 Resume Parsing: Supports DOCX and PDF formats, with OCR for scanned files. 

🧠 Entity Extraction: Automatically extracts skills, experience, education, and location using spaCy.  

🔥 Job Matching: Calculates similarity between candidate profiles and job postings. 

⚡ FastAPI Backend: Handles resume upload, parsing, and recommendation generation.  

🎯 React Frontend: Simple and intuitive interface for users to upload resumes and view personalized job matches.  

🛢️ PostgreSQL Database: Stores parsed resume information and job postings for efficient querying.  


Tech Stack Frontend: 

React.js  

Backend: FastAPI (for parsing and matching services)  

Database: PostgreSQL  

NLP: spaCy, scikit-learn  

Storage:AWS S3 (optional for file uploads) 

How It Works

1. User uploads a resume.
2. The system parses the resume and extracts structured data.  
3. The extracted profile is matched against available job listings.  
4. The user receives a ranked list of recommended jobs based on profile-job similarity.

Future Improvements 

🎯 Integrate ML-based learning from user interactions to improve recommendations.

🌍 Add location-based filtering for better regional job matches.  

🔒 Enhance security for resume data handling and storage.
