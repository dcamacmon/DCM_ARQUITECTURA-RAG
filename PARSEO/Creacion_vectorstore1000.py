# VECTORSTORE 1000

#Preprocesamiento de guías clínicas PDF a Embeddings en VectorStore

#Carga del entorno

# Cargar las dependencias necesarias
import os
import xml.etree.ElementTree as ET
from markdown2 import markdown

import requests
from dotenv import load_dotenv
import time
import re
import random
import tiktoken

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import openai

import pandas as pd
import glob

import chromadb
import chromadb.config

# Cargar las variables desde el archivo .env
dotenv_path = r"C:\Users\Daniel\Desktop\DOCUMENTOS\TFM\PDF STORE\PARSEO\.env"
load_dotenv(dotenv_path)

# Acceder a las variables de entorno

chunk_dir = os.getenv("CHUNK_DIR1000")

# Configuración de OpenAI
openai_key = os.getenv("OPENAI_API_KEY2")


#CHUNKING CON RECURSIVECHARACTERTEXTSPLITTER

# Directorios de guías clínicas en XML y tablas en Markdown
xml_dir = os.getenv("GROBID_MOD")
md_dir = os.getenv("LLAMACLOUD_TAB_DIR")


# Asegurarse de que el directorio de salida exista
os.makedirs(chunk_dir, exist_ok=True)

# Función para parsear XML y extraer texto
def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    text = ""
    for elem in root.iter():
        if elem.text:
            text += elem.text.strip() + " "
    return text

# Función para parsear Markdown
def parse_md(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    html_content = markdown(content)
    return html_content

# Emparejar archivos XML y MD basándose en una convención de nombres común
xml_files = sorted([f for f in os.listdir(xml_dir) if f.endswith(".xml")])
md_files = sorted([f for f in os.listdir(md_dir) if f.endswith(".md")])

# Asegurar que la cantidad de archivos coincide
if len(xml_files) != len(md_files):
    raise ValueError("El número de archivos XML y MD no coincide")

# Iterar sobre los archivos XML y MD, parsearlos y procesarlos
for xml_file, md_file in zip(xml_files, md_files):
    xml_path = os.path.join(xml_dir, xml_file)
    md_path = os.path.join(md_dir, md_file)
    
    # Obtener el contenido de los archivos
    xml_content = parse_xml(xml_path)
    md_content = parse_md(md_path)
    
    # Combinar el contenido XML y MD
    combined_content = f"{xml_content}\n{md_content}"

    # Chunking del contenido combinado
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = text_splitter.split_text(combined_content)
    
    # Guardar los chunks en archivos individuales en el directorio de salida
    for idx, chunk in enumerate(chunks):
        # Definir el nombre del archivo de salida para cada chunk
        output_file = os.path.join(chunk_dir, f"{xml_file}_{md_file}_chunk{idx + 1}.txt")
        
        # Guardar el chunk en un archivo
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(chunk)
    
    print(f"Se generaron {len(chunks)} chunks para los archivos {xml_file} y {md_file}.")


#EMBEDDING Y ALMACENAMIENTO EN VECTORSTORE

# Ruta de persistencia para Chroma
persist_directory = 'docs/chroma1000/'

# Crear el objeto de embeddings de OpenAI
embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

# Crear el objeto vector store de Chroma
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

#cargar metadatos
metadata_df = pd.read_excel("C:/Users/Daniel/Desktop/DOCUMENTOS/TFM/PDF STORE/PARSEO/Metadata.xlsx")
metadata_dict = metadata_df.set_index("ID").to_dict(orient="index")

# Leer los archivos en chunk_dir
texts = []
metadatas = []

# Iterar sobre los archivos en el directorio chunk_dir
for filename in os.listdir(chunk_dir):
    if filename.endswith(".txt"):  # Asegurar solo archivos .txt
        file_path = os.path.join(chunk_dir, filename)
        
        # Leer contenido del archivo
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read().strip()
            if text:  # Solo agregar si el texto no está vacío
                texts.append(text)
                
                # 🔹 Extraer el ID base del archivo (desde CLEAN_ hasta el primer .)
                match = re.search(r"(?:CLEAN_)?([^._]+)", filename)
                if match:
                    base_id = match.group(1)
                else:
                    base_id = filename.replace(".txt", "")  # En caso de error, usar el nombre completo

                # 🔹 Buscar metadatos en metadata_dict con el ID base
                file_metadata = metadata_dict.get(base_id, {})  # Si no encuentra, devuelve {}

                # 🔹 Agregar metadatos correctamente
                metadatas.append({
                    "source": filename,
                    "name": file_metadata.get("Name", "Desconocido"),
                    "original_source": file_metadata.get("Fuente original", "Desconocida"),
                    "url": file_metadata.get("URL", "No disponible"),
                    "year": file_metadata.get("Year", "Desconocido"),
                    "pathology": file_metadata.get("Pathology", "Desconocida"),
                    "doi": file_metadata.get("DOI", "No disponible"),
                    "pubmed": file_metadata.get("PubMed", "No disponible"),
                })
# Agregar los textos y metadatos al vector store
vectordb.add_texts(texts, metadatas=metadatas)

print(f"Se han agregado {len(texts)} chunks al vector store.")

# Ahora los embeddings están almacenados en el vector store de Chroma.