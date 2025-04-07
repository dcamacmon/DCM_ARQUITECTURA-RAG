#Creación de RAG

from Cargar_vectorstore import get_vectordb
import os
import openai  # Si necesitas interactuar con OpenAI
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain  # Usamos ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory  # Importamos memoria conversacional

# Configuración de OpenAI
from dotenv import load_dotenv
dotenv_path = r"C:\Users\Daniel\Desktop\DOCUMENTOS\TFM\PDF STORE\PARSEO\.env"
load_dotenv(dotenv_path)
openai_key = os.getenv("OPENAI_API_KEY2")

import os
import openai  # Si necesitas interactuar con OpenAI
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain  # Usamos ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory  # Importamos memoria conversacional


# Cargar el vector store (asumimos que get_vectordb ya lo define en vectorstore.py)
vectordb = get_vectordb()
print(f"Total de vectores en el vector store: {vectordb._collection.count()}")


# Configuración de OpenAI (asegúrate de que tienes la clave de API en tu entorno)
openai_key = os.getenv("OPENAI_API_KEY2")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_key)

# Definir el recuperador de información (retriever)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})  # Recupera los 5 mejores documentos

# Definir un template de prompt personalizado
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="Using the following information: {context}, answer the question: {question}. "
             "If you do not have enough information, say that you don't know. Do not make up information."
)

# Crear la cadena RAG con el prompt personalizado
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Utilizamos el tipo "stuff" para que devuelva la respuesta con los documentos relevantes
    retriever=retriever,
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=True  # Esto devuelve los chunks utilizados
)

# Realizar una consulta
query = "Is urate-lowering therapy (ULT) recommended for patients with asymptomatic hyperuricemia?"

# Buscar los documentos relevantes
docs = retriever.get_relevant_documents(query)

# Ejecutar la cadena RAG con la consulta
result = rag_chain({"query": query})  # Esto devuelve tanto la respuesta como las fuentes

# Obtener la respuesta generada
respuesta = result["result"]  # Respuesta del modelo

# Obtener las fuentes utilizadas
fuentes = result["source_documents"]  # Lista de documentos usados

# Mostrar respuesta generada
print("\n  Respuesta Generada:")
print(respuesta)

# Mostrar las fuentes utilizadas
print("\n  Fuentes utilizadas:")
for doc in fuentes:
    metadata = doc.metadata or {}  # Evitar errores si metadata es None

    # Acceder a los metadatos
    titulo = metadata.get("name", "Título no disponible")
    fuente_original = metadata.get("original_source", "Fuente desconocida")
    año = metadata.get("year", "Año no disponible")
    patologia = metadata.get("pathology", "Patología no especificada")
    doi = metadata.get("doi", "DOI no disponible")
    pubmed = metadata.get("pubmed", "PubMed no disponible")
    
    # Acceder al campo 'source' (filename)
    fuente = metadata.get("source", "Fuente desconocida")  # Aquí obtienes el filename
    
    print(f"- {titulo} ({año}) - {fuente_original} [Patología: {patologia}]")
    print(f"  Fuente: {fuente}")  # Aquí imprimes la fuente (filename)
    print(f"  DOI: {doi}")
    print(f"  PubMed: {pubmed}")

print(f"Total de vectores en el vector store: {vectordb._collection.count()}")
