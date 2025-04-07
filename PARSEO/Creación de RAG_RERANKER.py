# MODELO RAG 2. SIMPLE + RERANKER

import os
from Cargar_vectorstore import get_vectordb
import openai  # Si necesitas interactuar con OpenAI
import cohere  # Para usar el modelo de re-ranking de Cohere
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain  # Usamos ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory  # Importamos memoria conversacional
from langchain.prompts import PromptTemplate

# Cargar la clave de la API desde un archivo .env
from dotenv import load_dotenv
dotenv_path = r"C:\Users\Daniel\Desktop\DOCUMENTOS\TFM\PDF STORE\PARSEO\.env"
load_dotenv(dotenv_path)
openai_key = os.getenv("OPENAI_API_KEY2")
cohere_api_key = os.getenv("COHERE_API_KEY")  # Asegúrate de que tienes la clave de Cohere en tu entorno

# Configuración de Cohere
co = cohere.Client(cohere_api_key)

# Cargar el vector store (asumimos que get_vectordb ya lo define en vectorstore.py)
vectordb = get_vectordb()
print(f"Total de vectores en el vector store: {vectordb._collection.count()}")

# Configuración de OpenAI (asegúrate de que tienes la clave de API en tu entorno)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_key)

# Definir el recuperador de información (retriever)
retriever = vectordb.as_retriever(search_kwargs={"k": 10})  # Recupera los 10 mejores documentos para re-ranking

# Crear la función de re-ranking de Cohere
def cohere_rerank(documents, query):
    # Extraer el texto de los documentos y preparar las solicitudes para Cohere
    document_texts = [doc.page_content for doc in documents]
    
    # Usar Cohere para obtener un puntaje de relevancia de los documentos
    response = co.rerank(
        query=query,
        documents=document_texts
    )
    
    # Ordenar los documentos según el puntaje de relevancia
    scores = [result.relevance_score for result in response.results]
    ranked_documents = [documents[i] for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)]
    
    return ranked_documents

# Definir un template de prompt personalizado
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="Using the following information: {context}, answer the question: {question}. "
             "If you do not have enough information, say that you don't know. Do not make up information."
)

# Crear la cadena RAG con el prompt personalizado y el reranker de Cohere
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Utilizamos el tipo "stuff" para que devuelva la respuesta con los documentos relevantes
    retriever=retriever,
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=True  # Esto devuelve los chunks utilizados
)

# Realizar una consulta
query = "Is urate-lowering therapy (ULT) recommended for patients with asymptomatic hyperuricemia?"

# Buscar los documentos relevantes (Recuperamos más documentos de los que necesitamos)
docs = retriever.get_relevant_documents(query)

# Aplicar el re-ranking con Cohere
ranked_docs = cohere_rerank(docs, query)
top_docs = ranked_docs[:5]  # Tomamos solo los 5 primeros documentos

# Ejecutar la cadena RAG con la consulta usando los documentos reordenados
result = rag_chain({"query": query, "context": top_docs})  # Pasar los documentos reordenados al contexto

# Obtener la respuesta generada
respuesta = result["result"]  # Respuesta del modelo

# Obtener las fuentes utilizadas
fuentes = result["source_documents"][:5] # Lista de documentos usados

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