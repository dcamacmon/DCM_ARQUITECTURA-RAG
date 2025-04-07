# Evalación del modelo

import os
import re
import pandas as pd
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI
from Cargar_vectorstore import get_vectordb
import unicodedata

# Load environment variables
dotenv_path = r"C:\Users\Daniel\Desktop\DOCUMENTOS\TFM\PDF STORE\PARSEO\.env"
load_dotenv(dotenv_path)

# Configuration
chunk_dir = os.getenv("CHUNK_DIR2")
openai_key = os.getenv("OPENAI_API_KEY2")
persist_directory = 'docs/chroma/'
output_dir = "docs/preguntas_generadas/"

# Initialize OpenAI client
client = OpenAI(api_key=openai_key)

# Load vector store
vectordb = get_vectordb()
print(f"Total vectors in the vector store: {vectordb._collection.count()}")
data = vectordb._collection.get(include=["documents", "metadatas"])

# Group chunks by 'pathology'
guidelines_by_pathology = defaultdict(list)

for text, metadata in zip(data["documents"], data["metadatas"]):
    pathology_name = metadata.get("pathology", "Unknown")  # Use 'pathology' as the key
    guidelines_by_pathology[pathology_name].append(text)

# Function to sanitize file names (avoid issues on Windows)
def sanitize_filename(name, max_length=100):
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = re.sub(r'\s+', '_', name)  # Replaces spaces with underscores
    name = re.sub(r'[^\w\-_\.]', '', name)  # Only valid characters
    return name[:max_length]  # Cut to 100 characters

# Function to generate questions
def generate_questions(text, pathology_name):
    prompt = f"""
You are a medical expert. Based on the following clinical guideline for the pathology "{pathology_name}", generate 10 clear and specific questions that could be used to assess medical knowledge or review key concepts.

Content:
\"\"\" 
{text} 
\"\"\"

Write a list of 10 questions:
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # You can switch to "gpt-3.5-turbo" if needed
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
    
    # Optional: truncate if text is too long for the model
    if len(full_text) > 12000:
        full_text = full_text[:12000]
    
    questions = generate_questions(full_text, pathology_name)
    
    # Save to file
    safe_filename = sanitize_filename(pathology_name)
    with open(os.path.join(output_dir, f"{safe_filename}_questions.txt"), "w", encoding="utf-8") as f:
        f.write(questions)
    
    # Print to console (optional)
    print(f"📘 {pathology_name}:\n{questions}\n{'='*80}")


