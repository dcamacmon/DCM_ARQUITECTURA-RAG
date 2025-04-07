#Creación vectorstore

import os
import chromadb
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import pandas as pd
import re
from dotenv import load_dotenv

# Cargar variables de entorno
dotenv_path = r"C:\Users\Daniel\Desktop\DOCUMENTOS\TFM\PDF STORE\PARSEO\.env"
load_dotenv(dotenv_path)

# Configuración
chunk_dir = os.getenv("CHUNK_DIR2")
openai_key = os.getenv("OPENAI_API_KEY2")
persist_directory = 'docs/chroma/'

# Crear objeto de embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

# Crear o cargar vectorstore
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# Cargar metadatos
metadata_df = pd.read_excel("C:/Users/Daniel/Desktop/DOCUMENTOS/TFM/PDF STORE/PARSEO/Metadata.xlsx")
metadata_dict = metadata_df.set_index("ID").to_dict(orient="index")

# Leer archivos y agregar solo si no existen
texts = []
metadatas = []
existing_ids = set(vectordb._collection.get()["ids"])

for filename in os.listdir(chunk_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(chunk_dir, filename)
        
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read().strip()
            if text:
                match = re.search(r"^(.*?)(?:_chunk\d+)?\.txt$", filename)
                base_id = match.group(1) if match else filename.replace(".txt", "")
                file_metadata = metadata_dict.get(base_id, {})

                metadata = {
                    "source": filename,
                    "name": file_metadata.get("Name", "Desconocido"),
                    "original_source": file_metadata.get("Fuente original", "Desconocida"),
                    "year": file_metadata.get("Year", "Desconocido"),
                    "pathology": file_metadata.get("Pathology", "Desconocida"),
                    "doi": file_metadata.get("DOI", "No disponible"),
                    "pubmed": file_metadata.get("PubMed", "No disponible"),
                }

                if metadata["source"] not in existing_ids:
                    texts.append(text)
                    metadatas.append(metadata)
                    existing_ids.add(metadata["source"])  # Evitar duplicados

# Agregar textos al vectorstore
if texts:
    vectordb.add_texts(texts, metadatas=metadatas)
    print(f"✅ Se han agregado {len(texts)} chunks al vector store.")

print(f"📌 Total de vectores en el vector store: {vectordb._collection.count()}")
