#Cargar vectorstore

import os
import chromadb
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv

# Cargar variables de entorno
dotenv_path = ""
load_dotenv(dotenv_path)

# Configuración del entorno
openai_key = os.getenv("OPENAI_API_KEY2")
persist_directory500 = 'docs/chroma500/'
persist_directory1000 = 'docs/chroma1000/'

# Crear el objeto de embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_key)

# Función para recuperar ambos vectorstores
def get_vectordb500():
    vectordb = Chroma(persist_directory=persist_directory500, embedding_function=embeddings)
    # Verificar cuántos vectores tiene
    print(f"Total de vectores en el vector store: {vectordb._collection.count()}")
    return vectordb

def get_vectordb1000():
    vectordb = Chroma(persist_directory=persist_directory1000, embedding_function=embeddings)
    # Verificar cuántos vectores tiene
    print(f"Total de vectores en el vector store: {vectordb._collection.count()}")
    return vectordb