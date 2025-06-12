# match_students_jobs.py

import json
import pickle
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
from bs4 import BeautifulSoup

# Ensure required NLTK data is downloaded
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")




def load_students(filepath='students.json'):
    """Load students from a JSON file."""
    with open(filepath, 'r') as file:
        return json.load(file)

def load_jsonl_file(filepath):
    with open(filepath, 'r') as file:
        return [json.loads(line.strip()) for line in file]

def preprocess_jobs(jobs):
    """
    Convert job descriptions (HTML) into token lists suitable for BM25.
    Returns:
        job_texts: List of token lists for BM25
        job_index: Original indices of jobs that produced valid tokens
    """
    job_texts = []
    job_index = []

    for idx, job in enumerate(jobs):
        title = job.get('title', '')
        tags = job.get('tagsAndSkills', '').replace(',', ' ')
        raw_description = job.get('jobDescription', '')

        # Convert HTML to plain text
        soup = BeautifulSoup(raw_description, 'html.parser')
        plain_description = soup.get_text(separator=' ', strip=True)

        combined_text = f"{title} {tags} {plain_description}".strip()
        if not combined_text:
            continue

        tokens = word_tokenize(combined_text.lower())
        tokens = [t for t in tokens if t.isalpha()]  # keep alphabetic tokens only

        if not tokens:
            continue

        job_texts.append(tokens)
        job_index.append(idx)

    if not job_texts:
        raise ValueError("❌ No valid job descriptions found.")

    return job_texts, job_index


def build_bm25_model(job_texts):
    """Build and return a BM25Okapi model trained on the given token lists."""
    return BM25Okapi(job_texts)


def match_students_to_jobs(students, jobs, bm25, job_index, top_n=10):
    """
    For each student, compute BM25 scores against all jobs and return structured match data.
    Returns a dictionary with student names as keys and a list of matches as values.
    Each match is a dict with company, title, score, and a brief description snippet.
    """
    all_matches = {}

    for student in students:
        # Construct full name (fallback to "Unnamed" if missing)
        first_name = student.get('first_name', '')
        last_name = student.get('last_name', '')
        student_name = f"{first_name} {last_name}".strip() or "Unnamed"

        # Extract preferences, skills, interests
        job_preferences = student.get('job_preferences', {})
        job_preferences_list = []
        job_roles = []

        if isinstance(job_preferences, dict):
            for key, value in job_preferences.items():
                if isinstance(value, list):
                    if key.lower() in ['job_roles', 'job_titles']:
                        job_roles.extend(value)
                    else:
                        job_preferences_list.extend(value)
                elif isinstance(value, str):
                    if key.lower() in ['job_roles', 'job_titles']:
                        job_roles.append(value)
                    else:
                        job_preferences_list.append(value)

        skills = student.get('skills', [])
        interests = student.get('interests', [])

        # Weight job roles more heavily
        query_terms = job_roles * 5 + job_preferences_list * 2 + skills + interests
        if not query_terms:
            # Skip if no terms to query
            all_matches[student_name] = []
            continue

        # Tokenize and clean query
        query = " ".join(query_terms)
        query_tokens = word_tokenize(query.lower())
        query_tokens = [t for t in query_tokens if t.isalpha()]

        # Compute BM25 scores
        scores = bm25.get_scores(query_tokens)
        ranked = sorted(zip(job_index, scores), key=lambda x: x[1], reverse=True)
        top_matches = ranked[:top_n]

        student_matches = []
        for idx, score in top_matches:
            job = jobs[idx]
            company = job.get('companyName', 'Unknown Company')
            title = job.get('title', 'No Title')
            description_html = job.get('jobDescription', '')
            description_text = BeautifulSoup(description_html, 'html.parser').get_text(separator=' ', strip=True)
            snippet = description_text[:150] + ('...' if len(description_text) > 150 else '')

            student_matches.append({
                'company': company,
                'title': title,
                'score': float(score),
                'snippet': snippet
            })

        all_matches[student_name] = student_matches

    return all_matches


if __name__ == '__main__':
    # 1. Load data files
    students = load_students('students.json')
    jobs = []
    # Load each file and append to the final list
    for part_file in ["part_1.jsonl", "part_2.jsonl", "part_3.jsonl"]:
        data = load_jsonl_file(part_file)
        jobs.extend(data)

    # 2. Preview job count
    print(f"Total jobs loaded: {len(jobs)}")
    print(type(jobs))
    # 3. Preprocess and build BM25 model
    job_texts, job_index = preprocess_jobs(jobs)
    bm25 = build_bm25_model(job_texts)

    # 4. Compute matches
    matches = match_students_to_jobs(students, jobs, bm25, job_index, top_n=10)
    print(matches)

    # 5. Save matches to a pickle file
    with open('student_job_matches.pkl', 'wb') as pkl_file:
        pickle.dump(matches, pkl_file)
    print("✅ Matching results saved to 'student_job_matches.pkl'.")
