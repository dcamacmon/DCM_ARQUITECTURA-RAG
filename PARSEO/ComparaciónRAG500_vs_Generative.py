import os
import json
import pandas as pd
import re
from google.generativeai import GenerativeModel
import google.generativeai as genai
from Cargar_vectorstore import get_vectordb500
from dotenv import load_dotenv

# Cargar vectorstore
vectorstore = get_vectordb500()

# Cargar variables de entorno
dotenv_path = ""
load_dotenv(dotenv_path)

openai_key = os.getenv("OPENAI_API_KEY2")
gemini_api_key=os.getenv("GEMINI_API_KEY")

# Configurar clave API de Gemini
genai.configure(api_key=gemini_api_key)
model = GenerativeModel("models/gemini-1.5-pro")

def retrieve_context_for_pathology(pathology, vectorstore):
    try:
        documents = vectorstore.get(where={"pathology": pathology})
        chunks = documents["documents"]
        context = "\n\n".join(chunks)
        return context
    except Exception as e:
        print(f"Error retrieving context for {pathology}: {e}")
        return "ERROR"

def extract_value_regex(block, field):
    field = field.lower()
    if not block:
        return "ERROR"
    for line in block:
        clean_line = re.sub(r"[*_`]", "", line).strip().lower()
        if field in clean_line:
            match = re.search(r"\b(10|\d)(?:/10)?\b", line)
            if match:
                return int(match.group(1))
    return "ERROR"

def is_rag_model(name):
    return "rag" in name.lower()

def compare_responses_v2(question, responseA, responseB, modelA, modelB, context_full, context_retrieved):
    both_rag = is_rag_model(modelA) and is_rag_model(modelB)
    a_is_rag = is_rag_model(modelA)
    b_is_rag = is_rag_model(modelB)

    instructions = """
You are a Clinical QA Evaluation Expert AI.

Your task is to compare two answers (A and B) to a clinical question, and evaluate them on specific dimensions.

Evaluation Instructions:
Carefully analyze both answer A and Answer B against the Full Guideline Context and the specific Retrieved Context. Provide scores on a scale of 1 (very poor) to 10 (excellent).
"""

    if not a_is_rag:
        instructions += """
1. Assess Answer A (Baseline LLM):
a. Relevance: How relevant is Answer A to the Question (Score 1-10)
b. Factual Accuracy: How factually accurate is Answer A compared to the Full Guideline Context? (Score 1-10)
c. Completeness: How comprehensively does Answer A address the Question based on the Full Guideline Context? (Score 1-10)
d. Conciseness: Is Answer A concise? (Score 1-10)
"""

    if b_is_rag or a_is_rag:
        instructions += f"""
2. Assess Answer {'B' if b_is_rag else 'A'} (RAG LLM):
a. Faithfulness: CRITICAL – How faithful is Answer {'B' if b_is_rag else 'A'} *only* to the Retrieved Context it was given? (Score 1-10)
b. Relevance: How relevant is it to the Question? (Score 1-10)
c. Factual Accuracy: Compared to Full Guideline Context (Score 1-10)
d. Completeness_Given_Retrieval: Using *only* Retrieved Context (Score 1-10)
e. Completeness_Overall: Based on Full Guideline (Score 1-10)
f. Conciseness: Is it concise? (Score 1-10)
"""

    instructions += """
3. Comparison and Justification:
a. Compare Answer A and Answer B based on all your assessments.
b. Which answer is better overall for answering the specific Question accurately, safely, and reliably based on the full guideline? ('A', 'B', or 'Comparable')
c. Provide a detailed step-by-step reasoning for your choice.
"""

    # Formato
    format_a = (
        f"Answer A ({modelA}):\n"
        f"- Relevance: [1-10]\n"
        f"- Factual Accuracy vs Full Guideline Context: [1-10]\n"
        f"- Completeness vs Full Guideline Context: [1-10]\n"
        f"- Conciseness: [1-10]"
    )

    format_b = (
        f"Answer B ({modelB}):\n"
        f"- Faithfulness to Retrieved Context: [1-10]\n"
        f"- Relevance: [1-10]\n"
        f"- Factual Accuracy vs Full Guideline Context: [1-10]\n"
        f"- Completeness using ONLY Retrieved Context: [1-10]\n"
        f"- Completeness vs Full Guideline Context: [1-10]\n"
        f"- Conciseness: [1-10]"
    )

    prompt = f"""
{instructions}

Use this exact format:

Gemini evaluation output:
{format_a}

{format_b}

Which answer is better overall: [A/B/Comparable]
Justification: [Brief explanation comparing factual accuracy, completeness, and retrieved context use]

QUESTION:
{question}

FULL GUIDELINE CONTEXT:
{context_full}

RETRIEVED CONTEXT:
{context_retrieved}

ANSWER A ({modelA}):
{responseA["respuesta"]}

ANSWER B ({modelB}):
{responseB["respuesta"]}
"""

    # Evaluación
    response = model.generate_content(prompt)
    text = response.text.strip()
    print(f"\n Gemini evaluation output:\n{text}\n")

    # Parsear salida
    try:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        block_a, block_b = [], []
        current_block = None
        comment = ""

        for line in lines:
            if line.lower().startswith("answer a"):
                current_block = block_a
            elif line.lower().startswith("answer b"):
                current_block = block_b
            elif "justification" in line.lower() or "overall" in line.lower():
                current_block = None
                comment += line + "\n"
            elif current_block is not None:
                current_block.append(line)
            else:
                comment += line + "\n"

        return {
            "question": question,
            "model_comparison": f"{modelA} vs {modelB}",
            "modelA": {
                "name": modelA,
                "faithfulness": None,
                "relevance": extract_value_regex(block_a, "Relevance"),
                "accuracy": extract_value_regex(block_a, "Factual Accuracy"),
                "completeness_given": None,
                "completeness_overall": extract_value_regex(block_a, "Completeness"),
                "conciseness": extract_value_regex(block_a, "Conciseness")
            },
            "modelB": {
                "name": modelB,
                "faithfulness": extract_value_regex(block_b, "Faithfulness"),
                "relevance": extract_value_regex(block_b, "Relevance"),
                "accuracy": extract_value_regex(block_b, "Factual Accuracy"),
                "completeness_given": extract_value_regex(block_b, "Completeness using ONLY Retrieved Context"),
                "completeness_overall": extract_value_regex(block_b, "Completeness vs Full Guideline Context"),
                "conciseness": extract_value_regex(block_b, "Conciseness")
            },
            "comment": comment.strip()
        }

    except Exception as e:
        print(f"⚠️ Error parsing evaluation: {e}")
        return {
            "question": question,
            "model_comparison": f"{modelA} vs {modelB}",
            "modelA": {"name": modelA, "faithfulness": None, "relevance": "ERROR", "accuracy": "ERROR", "completeness_given": None, "completeness_overall": "ERROR", "conciseness": "ERROR"},
            "modelB": {"name": modelB, "faithfulness": "ERROR", "relevance": "ERROR", "accuracy": "ERROR", "completeness_given": "ERROR", "completeness_overall": "ERROR", "conciseness": "ERROR"},
            "comment": f"Error: {e}"
        }

# Solo compara Generative vs RAG500
def run_comparisons_generative_vs_rag500(responses_dir, vectorstore_context, max_pathologies=None, skip=0):
    output_dir = "docs/comparisons_gemini2a2"
    os.makedirs(output_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(responses_dir) if f.endswith("_respuestas_generativas.json")])
    pathologies = [f.replace("_respuestas_generativas.json", "") for f in files]
    pathologies = pathologies[skip:max_pathologies + skip] if max_pathologies else pathologies[skip:]

    for pathology in pathologies:
        print(f"\n📋 Evaluating pathology: {pathology}")

        try:
            with open(os.path.join(responses_dir, f"{pathology}_respuestas_generativas.json"), "r", encoding="utf-8") as f:
                generative = json.load(f)
            with open(os.path.join(responses_dir, f"{pathology}_respuestas_rag500.json"), "r", encoding="utf-8") as f:
                rag500 = json.load(f)
        except Exception as e:
            print(f" Could not load files for {pathology}: {e}")
            continue

        context_full = retrieve_context_for_pathology(pathology, vectorstore_context)
        if context_full == "ERROR":
            continue

        results = []
        for i in range(len(generative)):
            question = generative[i]["pregunta"]
            results.append(compare_responses_v2(
                question,
                generative[i],
                rag500[i],
                "Generative",
                "RAG500",
                context_full,
                "\n".join(rag500[i]["contexto"])
            ))

        output_json = os.path.join(output_dir, f"gen_vs_rag500_evaluation_{pathology}.json")
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(f"✅ Results saved to {output_json}")

# Ejecutar
responses_dir = "docs/respuestas_generadas_contexto"
max_pathologies = 5  
skip = 0
run_comparisons_generative_vs_rag500(responses_dir, vectorstore, max_pathologies=max_pathologies, skip=skip)
