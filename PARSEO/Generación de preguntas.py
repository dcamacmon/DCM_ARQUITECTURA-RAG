import os
import re
import pandas as pd
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI
from Cargar_vectorstore import get_vectordb500
import unicodedata

# Load environment variables
dotenv_path = ""
load_dotenv(dotenv_path)

# Configuration
chunk_dir = os.getenv("CHUNK_DIR2")
openai_key = os.getenv("OPENAI_API_KEY2")
persist_directory = 'docs/chroma500_3/'
output_dir = "docs/preguntas_generadas20"

# Initialize OpenAI client
client = OpenAI(api_key=openai_key)

# Load vector store
vectordb = get_vectordb500()
print(f"Total vectors in the vector store: {vectordb._collection.count()}")
data = vectordb._collection.get(include=["documents", "metadatas"])

# Group chunks by 'pathology'
guidelines_by_pathology = defaultdict(list)
for text, metadata in zip(data["documents"], data["metadatas"]):
    pathology_name = metadata.get("pathology", "Unknown")
    guidelines_by_pathology[pathology_name].append(text)

# Function to sanitize file names
def sanitize_filename(name, max_length=100):
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'[^\w\-_\.]', '', name)
    return name[:max_length]

# Function to split text into chunks
def split_text(text, max_length=12000):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

# Function to generate questions
def generate_questions(text, pathology_name):
    prompt = f"""
You are a medical expert. Based on the following clinical guideline for the pathology "{pathology_name}", generate 10 clear and specific questions that could be used to assess medical knowledge or review key concepts.

Content:
\"\"\" 
{text} 
\"\"\"

Write a list of 20 questions:
"""
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1000
    )
    return response.choices[0].message.content

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop over each pathology and generate questions
for pathology_name, chunks in guidelines_by_pathology.items():
    full_text = "\n".join(chunks)
    split_chunks = split_text(full_text)

    all_questions = []

    for i, chunk in enumerate(split_chunks):
        print(f"🧠 Generating questions for {pathology_name} - chunk {i+1}/{len(split_chunks)}")
        try:
            questions = generate_questions(chunk, pathology_name)
            all_questions.append(f"### Questions from chunk {i+1}:\n{questions}")
        except Exception as e:
            print(f"⚠️ Error generating questions for {pathology_name}, chunk {i+1}: {e}")

    combined_questions = "\n\n".join(all_questions)

    safe_filename = sanitize_filename(pathology_name)
    output_path = os.path.join(output_dir, f"{safe_filename}_questions.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(combined_questions)

    print(f"✅ Saved questions for {pathology_name} to {output_path}\n{'='*80}")
