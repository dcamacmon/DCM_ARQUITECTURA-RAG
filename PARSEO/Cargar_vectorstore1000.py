#Cargar vectorstore

import os
import chromadb
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv

# Cargar variables de entorno
dotenv_path = r"C:\Users\Daniel\Desktop\DOCUMENTOS\TFM\PDF STORE\PARSEO\.env"
load_dotenv(dotenv_path)

# Configuración
openai_key = os.getenv("OPENAI_API_KEY2")
persist_directory = 'docs/chroma1000/'

# Crear el objeto de embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

# Función para recuperar el vectorstore sin modificarlo
def get_vectordb1000():
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
