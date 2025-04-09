# GENERACIÓN DE RESPUESTAS

# GENERACIÓN DE RESPUESTAS

import os
import json
import time
from RAG_RERANKER_MEMORY import rag_chain  # Importar el modelo RAG con re-ranking y memoria
from RAG_sin_Retrieval import generative_query  # Importar el modelo generativo
import cohere  # Importar la librería de Cohere para manejar el límite de solicitudes

# Ruta a las preguntas generadas (asegúrate de que el directorio contenga los archivos de texto generados)
questions_dir = "docs/preguntas_generadas/"
output_dir = "docs/respuestas_generadas/"

# Función para cargar las preguntas desde los archivos de texto
def cargar_preguntas(questions_dir):
    preguntas = {}
    for filename in os.listdir(questions_dir):
        if filename.endswith("_questions.txt"):
            with open(os.path.join(questions_dir, filename), 'r', encoding="utf-8") as file:
                pathology_name = filename.replace("_questions.txt", "")
                preguntas[pathology_name] = file.read().splitlines()  # Cada línea es una pregunta
    return preguntas

# Función para ejecutar el modelo generativo
def ejecutar_modelo_generativo(pregunta):
    respuesta_generativa = generative_query(pregunta)
    return respuesta_generativa

# Función para ejecutar el modelo RAG con re-ranking y memoria
def ejecutar_modelo_rag(pregunta):
    result = rag_chain.invoke({"question": pregunta})
    respuesta_rag = result["answer"]
    return respuesta_rag

# Función para ejecutar el modelo RAG con manejo del límite de solicitudes de Cohere
def ejecutar_modelo_rag_con_limite(pregunta, idx):
    try:
        result = rag_chain.invoke({"question": pregunta})
        respuesta_rag = result["answer"]
        return respuesta_rag
    except cohere.errors.TooManyRequestsError:  # Captura el error de demasiadas solicitudes
        print("Límite de solicitudes alcanzado, esperando 60 segundos...")
        time.sleep(60)  # Espera 60 segundos antes de intentar nuevamente
        return ejecutar_modelo_rag_con_limite(pregunta, idx)  # Intenta nuevamente después de la espera

# Función para procesar las preguntas y generar las respuestas
def procesar_preguntas_y_respuestas(questions_dir, output_dir, start=0, end=None):
    os.makedirs(output_dir, exist_ok=True)

    preguntas = cargar_preguntas(questions_dir)
    nombres_patologias = sorted(preguntas.keys())  # Asegura orden alfabético

    if end is None:
        end = len(nombres_patologias)

    # Solo selecciona las patologías del rango deseado
    patologias_seleccionadas = nombres_patologias[start:end]

    for pathology_name in patologias_seleccionadas:
        preguntas_list = preguntas[pathology_name]
        respuestas_generativas = []
        respuestas_rag = []

        for idx, pregunta in enumerate(preguntas_list):
            respuesta_generativa = ejecutar_modelo_generativo(pregunta)
            respuestas_generativas.append(respuesta_generativa)

            respuesta_rag = ejecutar_modelo_rag_con_limite(pregunta, idx)
            respuestas_rag.append(respuesta_rag)

            if (idx + 1) % 10 == 0:
                time.sleep(60)

        with open(os.path.join(output_dir, f"{pathology_name}_respuestas_generativas.json"), 'w', encoding="utf-8") as f:
            json.dump(respuestas_generativas, f, ensure_ascii=False, indent=4)

        with open(os.path.join(output_dir, f"{pathology_name}_respuestas_rag.json"), 'w', encoding="utf-8") as f:
            json.dump(respuestas_rag, f, ensure_ascii=False, indent=4)


# Primera tanda (0 a 8): 8 archivos
procesar_preguntas_y_respuestas(questions_dir, output_dir, start=0, end=8)
