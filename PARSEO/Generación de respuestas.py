# GENERACIÓN DE RESPUESTAS

import os
import json
import time
from RAG_RERANKER_MEMORY_LANGSMITH_1000 import rag_chain1000
from RAG_RERANKER_MEMORY_LANGSMITH_500 import rag_chain500
from RAG_sin_Retrieval import generative_query
import cohere

# Ruta a las preguntas generadas anteriormente
questions_dir = "docs/preguntas_generadas/"
output_dir = "docs/respuestas_generadas_contexto/"

# Función para cargar las preguntas desde los archivos de texto
def cargar_preguntas(questions_dir):
    preguntas = {}
    for filename in os.listdir(questions_dir):
        if filename.endswith("_questions.txt"):
            with open(os.path.join(questions_dir, filename), 'r', encoding="utf-8") as file:
                pathology_name = filename.replace("_questions.txt", "")
                preguntas[pathology_name] = file.read().splitlines()
    return preguntas

# Función para ejecutar el modelo generativo (LLM)
def ejecutar_modelo_generativo(pregunta):
    respuesta_generativa = generative_query(pregunta)
    return respuesta_generativa

# Función para ejecutar el modelo RAG con re-ranking y memoria de chunks=500
def ejecutar_modelo_rag_con_limite500(pregunta, idx):
    try:
        result = rag_chain500.invoke({"question": pregunta})
        respuesta_rag = result["answer"]
        contexto = [doc.page_content for doc in result.get("source_documents", [])]
        return {"respuesta": respuesta_rag, "contexto": contexto}
    except cohere.errors.TooManyRequestsError:
        print("Límite de solicitudes alcanzado, esperando 60 segundos...")
        time.sleep(60)
        return ejecutar_modelo_rag_con_limite500(pregunta, idx)

# Función para ejecutar el modelo RAG con re-ranking y memoria de chunks=1000
def ejecutar_modelo_rag_con_limite1000(pregunta, idx):
    try:
        result = rag_chain1000.invoke({"question": pregunta})
        respuesta_rag = result["answer"]
        contexto = [doc.page_content for doc in result.get("source_documents", [])]
        return {"respuesta": respuesta_rag, "contexto": contexto}
    except cohere.errors.TooManyRequestsError:
        print("Límite de solicitudes alcanzado, esperando 60 segundos...")
        time.sleep(60)
        return ejecutar_modelo_rag_con_limite1000(pregunta, idx)

# Función para procesar las preguntas y generar las respuestas, diferenciando bloques por patología
# Se especifica un Start y un End para procesar un rango de patologías y no saturar 
def procesar_preguntas_y_respuestas(questions_dir, output_dir, start=0, end=None):
    os.makedirs(output_dir, exist_ok=True)

    preguntas = cargar_preguntas(questions_dir)
    nombres_patologias = sorted(preguntas.keys())

    if end is None:
        end = len(nombres_patologias)

    patologias_seleccionadas = nombres_patologias[start:end]

    for pathology_name in patologias_seleccionadas:
        preguntas_list = preguntas[pathology_name]
        respuestas_generativas = []
        respuestas_rag500 = []
        respuestas_rag1000 = []

        for idx, pregunta in enumerate(preguntas_list):
            respuesta_generativa = ejecutar_modelo_generativo(pregunta)
            respuestas_generativas.append({"pregunta": pregunta, "respuesta": respuesta_generativa})

            resultado_rag500 = ejecutar_modelo_rag_con_limite500(pregunta, idx)
            respuestas_rag500.append({
                "pregunta": pregunta,
                "respuesta": resultado_rag500["respuesta"],
                "contexto": resultado_rag500["contexto"]
            })

            resultado_rag1000 = ejecutar_modelo_rag_con_limite1000(pregunta, idx)
            respuestas_rag1000.append({
                "pregunta": pregunta,
                "respuesta": resultado_rag1000["respuesta"],
                "contexto": resultado_rag1000["contexto"]
            })

            if (idx + 1) % 10 == 0:
                time.sleep(60)

        with open(os.path.join(output_dir, f"{pathology_name}_respuestas_generativas.json"), 'w', encoding="utf-8") as f:
            json.dump(respuestas_generativas, f, ensure_ascii=False, indent=4)

        with open(os.path.join(output_dir, f"{pathology_name}_respuestas_rag500.json"), 'w', encoding="utf-8") as f:
            json.dump(respuestas_rag500, f, ensure_ascii=False, indent=4)

        with open(os.path.join(output_dir, f"{pathology_name}_respuestas_rag1000.json"), 'w', encoding="utf-8") as f:
            json.dump(respuestas_rag1000, f, ensure_ascii=False, indent=4)

# Ejecutar
#procesar_preguntas_y_respuestas(questions_dir, output_dir, start=0, end=1)
#procesar_preguntas_y_respuestas(questions_dir, output_dir, start=3, end=6)
#procesar_preguntas_y_respuestas(questions_dir, output_dir, start=6, end=9)
#procesar_preguntas_y_respuestas(questions_dir, output_dir, start=9, end=12)
#procesar_preguntas_y_respuestas(questions_dir, output_dir, start=12, end=15)
procesar_preguntas_y_respuestas(questions_dir, output_dir, start=15, end=None)