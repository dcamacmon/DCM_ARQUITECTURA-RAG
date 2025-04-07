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
input_dir = os.getenv("INPUT_DIR")
grobid_output_dir = os.getenv("GROBID_DIR")
grobid_cleaned_output_dir = os.getenv("GROBID_TAB")



grobid_tab_dir = os.getenv("GROBID_TAB")
grobid_mod_dir = os.getenv("GROBID_MOD")  # GROBID_MOD será usado como el directorio de salida


# Configuración
api_key = "llx-nkQXJqK4wKhGLbvbHur4FOZxM3Nou7Jm6LNjk5x5E2Q48FKK"  # Reemplaza con tu clave API real

LlamaCloud_dir = os.getenv("LLAMACLOUD_DIR")
llamacloud_output_dir = os.getenv("LLAMACLOUD_OUTPUT_DIR")

chunk_dir = os.getenv("CHUNK_DIR")

# Configuración de OpenAI
openai_key = os.getenv("OPENAI_API_KEY2")

#Parseado de PDF del texto a XML en GROBID

# Endpoint de la API de GROBID
grobid_url = "http://localhost:8070/api/processFulltextDocument"

# Crear directorio de salida si no existe
if not os.path.exists(grobid_output_dir):
    os.makedirs(grobid_output_dir)

# Parsear cada PDF del directorio input_dir
for filename in os.listdir(input_dir):
    if filename.endswith(".pdf"):
        file_path = os.path.join(input_dir, filename)
        
        with open(file_path, 'rb') as pdf_file:
            files = {'input': pdf_file}
            response = requests.post(grobid_url, files=files, data={'consolidate_citations': '1'})
            
            # Manejo de la respuesta
            if response.status_code == 200:
                output_path = os.path.join(grobid_output_dir, filename.replace('.pdf', '.xml'))
                with open(output_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(response.text)
                print(f"Parseo exitoso: {output_path}")
            elif response.status_code == 202:
                print(f"El procesamiento de {filename} ha sido aceptado, pero aún no ha terminado. Esperando...")
                # Espera y vuelve a intentar después de 5 segundos (puedes ajustar el tiempo)
                time.sleep(5)
                # Reintentar la solicitud
                retry_response = requests.post(grobid_url, files=files, data={'consolidate_citations': '1'})
                if retry_response.status_code == 200:
                    output_path = os.path.join(grobid_output_dir, filename.replace('.pdf', '.xml'))
                    with open(output_path, 'w', encoding='utf-8') as output_file:
                        output_file.write(retry_response.text)
                    print(f"Parseo exitoso tras reintento: {output_path}")
                else:
                    print(f"Error al procesar {filename} después de reintentos: {retry_response.status_code}")
            else:
                print(f"Error al parsear {filename}: {response.status_code}")

print("Proceso finalizado.")

##Limpieza de XML a texto en GROBID

# Definir el namespace
ns = {
    "tei": "http://www.tei-c.org/ns/1.0",
    "ns0": "http://www.tei-c.org/ns/1.0"
}

# Expresión regular para eliminar referencias en el texto
ref_pattern = re.compile(r"\(\d+(?:[,-]\d+)*\)")

# Función para limpiar el texto
def clean_text(element):
    if element.text:
        element.text = re.sub(ref_pattern, "", element.text).strip()
    if element.tail:
        element.tail = re.sub(ref_pattern, "", element.tail).strip()
    for child in element:
        clean_text(child)

# Función para limpiar un archivo XML
def clean_xml(xml_file, output_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 1. Eliminar referencias bibliográficas
    for parent in root.iter():
        refs_to_remove = [ref for ref in parent.findall("tei:ref", ns) if ref.get("type") == "bibr"]
        for ref in refs_to_remove:
            parent.remove(ref)

    # 2. Limpiar referencias en el texto
    for body in root.findall(".//tei:body", ns):
        clean_text(body)

    # 3. Eliminar sección de autores en <biblStruct>
    for biblStruct in root.findall(".//tei:biblStruct", ns):
        analytic = biblStruct.find("tei:analytic", ns)
        if analytic is not None:
            biblStruct.remove(analytic)

    # 4. Eliminar identificadores ORCID
    for idno in root.findall(".//tei:idno[@type='ORCID']", ns):
        parent = root.find(f".//tei:*[tei:idno[@type='ORCID']]", ns)
        if parent is not None:
            parent.remove(idno)

    # 5. Eliminar encodingDesc y profileDesc
    for tag in ["encodingDesc", "profileDesc"]:
        for elem in root.findall(f".//tei:{tag}", ns):
            parent = root.find(f".//tei:*[tei:{tag}]", ns)
            if parent is not None:
                parent.remove(elem)

    # 6. Eliminar secciones específicas
    secciones_a_eliminar = ["AUTHOR CONTRIBUTIONS", "ACKNOWLEDGMENTS"]
    for parent in root.findall(".//tei:div", ns):
        for div in parent.findall("tei:div", ns):
            head = div.find("tei:head", ns)
            if head is not None and head.text.strip().upper() in secciones_a_eliminar:
                parent.remove(div)

    # 7. Eliminar lista de referencias
    for list_bibl in root.findall(".//tei:listBibl", ns):
        parent = root.find(f".//tei:*[tei:listBibl]", ns)
        if parent is not None:
            parent.remove(list_bibl)

    # 8. Eliminar tablas
    for figure in root.findall(".//tei:figure[@type='table']", ns):
        for parent in root.iter():
            if figure in parent:
                parent.remove(figure)
                break

    # 9. Eliminar figuras
    for figure in root.findall(".//tei:figure", ns):
        for parent in root.iter():
            if figure in parent:
                parent.remove(figure)
                break
    # Guardar el XML limpio
    tree.write(output_file, encoding="utf-8", xml_declaration=True)

# Crear el directorio de salida si no existe
os.makedirs(grobid_mod_dir, exist_ok=True)

# Procesar todos los archivos XML en el directorio de entrada
for filename in os.listdir(grobid_tab_dir):
    if filename.endswith(".xml"):
        xml_file = os.path.join(grobid_tab_dir, filename)
        cleaned_file = os.path.join(grobid_mod_dir, f"CLEAN_{filename}")
        clean_xml(xml_file, cleaned_file)
        print(f"✅ Archivo limpio guardado como: {cleaned_file}")

#Parseado de PDF de las tablas a MarkDown en LlamaParse

upload_url = "https://api.cloud.llamaindex.ai/api/parsing/upload"
result_url_template = "https://api.cloud.llamaindex.ai/api/v1/parsing/job/{job_id}/result/raw/markdown"

# Crear directorio de salida si no existe
os.makedirs(LlamaCloud_dir, exist_ok=True)

# Encabezados para las solicitudes
headers = {
    "Authorization": f"Bearer {api_key}"
}

# Lista para almacenar los archivos procesados con éxito
processed_files = []

def obtener_resultado(job_id):
    result_url = result_url_template.format(job_id=job_id)
    while True:
        response = requests.get(result_url, headers=headers)
        
        if response.status_code == 200:
            # Si la respuesta es 200, el trabajo está completo y los resultados están listos
            return response.text
        
        elif response.status_code in [202, 404]:
            # Si la respuesta es 202, significa que el trabajo aún está procesando
            print("El archivo está siendo procesado, esperando...")
            time.sleep(10)  # Espera de 10 segundos antes de volver a verificar el estado
            
        else:
            # Si ocurre un error diferente, se muestra el mensaje de error
            print(f"Error al obtener el resultado: {response.status_code} - {response.text}")
            return None

# Procesar cada archivo PDF en el directorio de entrada
for filename in os.listdir(input_dir):
    if filename.endswith(".pdf"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(LlamaCloud_dir, filename.replace(".pdf", ".md"))

        with open(input_path, "rb") as pdf_file:
            files = {"file": pdf_file}
            response = requests.post(upload_url, headers=headers, files=files)

            if response.status_code == 200:
                response_data = response.json()
                job_id = response_data.get("id")
                if job_id:
                    print(f"Archivo '{filename}' enviado correctamente. ID de trabajo: {job_id}")
                    markdown_content = obtener_resultado(job_id)
                    if markdown_content:
                        with open(output_path, "w", encoding="utf-8") as md_file:
                            md_file.write(markdown_content)
                        processed_files.append(filename)
                        print(f"Archivo '{filename}' procesado y guardado en '{output_path}'.")
                else:
                    print(f"No se recibió un ID de trabajo para el archivo '{filename}'.")
            elif response.status_code == 429:
                print("Límite de páginas alcanzado. Proceso detenido.")
                break
            else:
                print(f"Error al enviar '{filename}': {response.status_code} - {response.text}")

# Informar sobre los archivos procesados con éxito
print("Proceso completado.")
print("Archivos procesados con éxito:")
for file in processed_files:
    print(file)


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
        chunk_size=500,
        chunk_overlap=75
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
persist_directory = 'docs/chroma/'

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
