# MODELO RAG . SIMPLE + RERANKER + MEMORY + LANGSMITH

import os
import openai
import cohere
from dotenv import load_dotenv

# 🔹 IMPORTS ACTUALIZADOS (LangChain ha cambiado)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores.base import VectorStoreRetriever
from Cargar_vectorstore import get_vectordb

#TRAZABILIDAD LANGSMITH
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks import StdOutCallbackHandler

from langchain_core.callbacks import CallbackManager
import langsmith

# Activar trazabilidad si la variable está definida
if os.getenv("LANGCHAIN_API_KEY"):  # Usamos os.getenv para comprobar la variable de entorno
    tracer = LangChainTracer(project_name="TFM-RAG-Reranker")
    callback_manager = CallbackManager([tracer, StdOutCallbackHandler()])
else:
    tracer = None
    callback_manager = CallbackManager([StdOutCallbackHandler()])

#1. Cargar claves de API desde .env
dotenv_path = r"C:\Users\Daniel\Desktop\DOCUMENTOS\TFM\PDF STORE\PARSEO\.env"
load_dotenv(dotenv_path)
openai_key = os.getenv("OPENAI_API_KEY2")
cohere_api_key = os.getenv("COHERE_API_KEY")

# 2. Inicializar clientes de OpenAI y Cohere
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_key)
co = cohere.Client(cohere_api_key)

# 3. Cargar el vectorstore
vectordb = get_vectordb()
retriever = vectordb.as_retriever(search_kwargs={"k": 10})


# 4. Definir memoria conversacional (✅ FIX: output_key="answer")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

# 5. Definir re-ranking con Cohere
def cohere_rerank(documents, query):
    document_texts = [doc.page_content for doc in documents]
    response = co.rerank(query=query, documents=document_texts)
    scores = [result.relevance_score for result in response.results]
    ranked_documents = [documents[i] for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)]
    return ranked_documents

# 6. Crear un nuevo recuperador con re-ranking
def custom_retriever(query):
    docs = retriever.get_relevant_documents(query)
    ranked_docs = cohere_rerank(docs, query)
    return ranked_docs[:5]  # Tomamos los 5 mejores documentos

class CustomRetriever(VectorStoreRetriever):
    def _get_relevant_documents(self, query):  # 🔹 FIX: Ahora usa "_get_relevant_documents"
        return custom_retriever(query)

# 7. Crear instancia del recuperador con re-ranking
reranked_retriever = CustomRetriever(vectorstore=vectordb)

# 8. Definir un prompt personalizado
custom_prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=(
        "You are an AI assistant. Here is the conversation history:\n{chat_history}\n\n"
        "Using the following retrieved documents:\n{context}\n\n"
        "Answer the user's question:\n{question}\n\n"
        "If you do not have enough information, say that you don't know. Do not make up information."
    )
)

# 9. Crear la cadena RAG con memoria, re-ranking y prompt personalizado
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=reranked_retriever,
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": custom_prompt},
    callbacks=callback_manager
)
