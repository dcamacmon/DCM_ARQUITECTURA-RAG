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

# Inicializar modelo Gemini
model = GenerativeModel("models/gemini-1.5-pro")

# Recuperar contexto completo (Full Guideline Context)
def retrieve_context_for_pathology(pathology, vectorstore):
    try:
        documents = vectorstore.get(where={"pathology": pathology})
        chunks = documents["documents"]
        context = "\n\n".join(chunks)
        return context
    except Exception as e:
        print(f"Error retrieving context for {pathology}: {e}")
        return "ERROR"

# Extraer valor numérico
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

# Función para saber si es un modelo RAG (y realizar correctamente la comparación)
def is_rag_model(name):
    return "rag" in name.lower()

# Definición de la función que hará la comparación entre respuestas, determinando si se comparan dos modelos RAG o un modelo RAG con el modelo LLM generativo.
def compare_responses_v2(question, responseA, responseB, modelA, modelB, context_full, context_retrieved):
    both_rag = is_rag_model(modelA) and is_rag_model(modelB)
    a_is_rag = is_rag_model(modelA)
    b_is_rag = is_rag_model(modelB)

    # Instrucciones dinámicas de evaluación
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
a. Faithfulness: CRITICAL – How faithful is Answer {'B' if b_is_rag else 'A'} *only* to the Retrieved Context it was given? Does it hallucinate information *not* in that specific snippet, even if true according to the full guideline? (Score 1-10)
b. Relevance: How relevant is Answer {'B' if b_is_rag else 'A'} to the Question (Score 1-10)
c. Factual Accuracy: How factually accurate is Answer {'B' if b_is_rag else 'A'} compared to the Full Guideline Context? (Score 1-10)
d. Completeness_Given_Retrieval: How comprehensively does Answer {'B' if b_is_rag else 'A'} address the question using *only* information found in its ‘Retrieved Context’? (Score 1-10)
e. Completeness_Overall: How comprehensively does Answer {'B' if b_is_rag else 'A'} address the Question based on the *entire* Full Guideline Context? (Score 1-10)
f. Conciseness: Is Answer {'B' if b_is_rag else 'A'} concise? (Score 1-10)
"""

    instructions += """
3. Comparison and Justification:
a. Compare Answer A and Answer B based on all your assessments.
b. Which answer is better overall for answering the specific Question accurately, safely, and reliably based on the full guideline? ('A', 'B', or 'Comparable')
c. Provide a detailed step-by-step reasoning for your choice. Discuss the impact of RAG. Specifically comment on:
    i. Differences in Factual Accuracy and Safety
    ii. Whether Answer B’s faithfulness to its limited Retrieved Context aligned with the overall guideline truth
    iii. If the Retrieved Context seemed sufficient/good based on comparing Answer B’s Completeness_Given_Retrieval vs Completeness_Overall and its Faithfulness vs Factual Accuracy
"""

    # Reparación del formato para que la respuesta generada sea homogénea
    faithfulness_line_a = "- Faithfulness to Retrieved Context: [1-10]\n" if a_is_rag else ""
    completeness_given_line_a = "- Completeness using ONLY Retrieved Context: [1-10]\n" if a_is_rag else ""

    format_a = (
        f"Answer A ({modelA}):\n"
        f"{faithfulness_line_a}"
        f"- Relevance: [1-10]\n"
        f"- Factual Accuracy vs Full Guideline Context: [1-10]\n"
        f"{completeness_given_line_a}"
        f"- Completeness vs Full Guideline Context: [1-10]\n"
        f"- Conciseness: [1-10]"
    )

    faithfulness_line_b = "- Faithfulness to Retrieved Context: [1-10]\n" if b_is_rag else ""
    completeness_given_line_b = "- Completeness using ONLY Retrieved Context: [1-10]\n" if b_is_rag else ""

    format_b = (
        f"Answer B ({modelB}):\n"
        f"{faithfulness_line_b}"
        f"- Relevance: [1-10]\n"
        f"- Factual Accuracy vs Full Guideline Context: [1-10]\n"
        f"{completeness_given_line_b}"
        f"- Completeness vs Full Guideline Context: [1-10]\n"
        f"- Conciseness: [1-10]"
    )

    # Definición del prompt personalizado que ejecutará el LLM evaluador
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

    # Generar respuesta
    response = model.generate_content(prompt)
    text = response.text.strip()
    print(f"\n Gemini evaluation output:\n{text}\n")

    # Parsing
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
                "faithfulness": extract_value_regex(block_a, "Faithfulness") if a_is_rag else None,
                "relevance": extract_value_regex(block_a, "Relevance"),
                "accuracy": extract_value_regex(block_a, "Factual Accuracy"),
                "completeness_given": extract_value_regex(block_a, "Completeness using ONLY Retrieved Context") if a_is_rag else None,
                "completeness_overall": extract_value_regex(block_a, "Completeness vs Full Guideline Context"),
                "conciseness": extract_value_regex(block_a, "Conciseness")
            },
            "modelB": {
                "name": modelB,
                "faithfulness": extract_value_regex(block_b, "Faithfulness") if b_is_rag else None,
                "relevance": extract_value_regex(block_b, "Relevance"),
                "accuracy": extract_value_regex(block_b, "Factual Accuracy"),
                "completeness_given": extract_value_regex(block_b, "Completeness using ONLY Retrieved Context") if b_is_rag else None,
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
            "modelA": {"name": modelA, "faithfulness": "ERROR", "relevance": "ERROR", "accuracy": "ERROR", "completeness_given": "ERROR", "completeness_overall": "ERROR", "conciseness": "ERROR"},
            "modelB": {"name": modelB, "faithfulness": "ERROR", "relevance": "ERROR", "accuracy": "ERROR", "completeness_given": "ERROR", "completeness_overall": "ERROR", "conciseness": "ERROR"},
            "comment": f"Error: {e}"
        }

# Ejecución de las comparaciones en función de las respuestas generadas por los diferentes modelos
# Se define un límite de patologías y un SKIP para evitar saturar la API de Gemini AI
def run_comparisons_in_batches_v2(responses_dir, vectorstore_context, max_pathologies=None, skip=0):
    output_dir = "docs/comparisons_gemini"
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
            with open(os.path.join(responses_dir, f"{pathology}_respuestas_rag1000.json"), "r", encoding="utf-8") as f:
                rag1000 = json.load(f)
        except Exception as e:
            print(f" Could not load files for {pathology}: {e}")
            continue

        context_full = retrieve_context_for_pathology(pathology, vectorstore_context)
        if context_full == "ERROR":
            continue

        #Se obtienen las respuestas generadas por las 3 posibles comparaciones
        results = []
        for i in range(len(generative)):
            question = generative[i]["pregunta"]
            results.append(compare_responses_v2(question, generative[i], rag500[i], "Generative", "RAG500", context_full, "\n".join(rag500[i]["contexto"])))
            results.append(compare_responses_v2(question, generative[i], rag1000[i], "Generative", "RAG1000", context_full, "\n".join(rag1000[i]["contexto"])))
            results.append(compare_responses_v2(question, rag500[i], rag1000[i], "RAG500", "RAG1000", context_full, "\n".join(rag1000[i]["contexto"])))

        # Se extraen los resultados de las comparaciones en archivos JSON divididos por patologías
        output_json = os.path.join(output_dir, f"model_evaluation_{pathology}.json")
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(f"✅ Results saved to {output_json}")

# Ejecución del script, determinando el directorio de las respuestas y el bloque de patologías a analizar.
responses_dir = "docs/respuestas_generadas_contexto"
max_pathologies = None
skip = 16
run_comparisons_in_batches_v2(responses_dir, vectorstore, max_pathologies=max_pathologies, skip=skip)
