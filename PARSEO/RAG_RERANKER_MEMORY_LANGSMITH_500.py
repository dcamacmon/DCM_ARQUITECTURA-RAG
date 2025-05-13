# MODELO RAG . SIMPLE + RERANKER + MEMORY + LANGSMITH

#Carga de los paquetes necesarios
import os
import openai
import cohere
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores.base import VectorStoreRetriever
from Cargar_vectorstore import get_vectordb500

#TRAZABILIDAD LANGSMITH
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks import StdOutCallbackHandler

from langchain_core.callbacks import CallbackManager
import langsmith

# Activación de la trazabilidad si la variable está definida
if os.getenv("LANGCHAIN_API_KEY"): 
    tracer = LangChainTracer(project_name="TFM")
    callback_manager = CallbackManager([tracer, StdOutCallbackHandler()])
else:
    tracer = None
    callback_manager = CallbackManager([StdOutCallbackHandler()])

#1. Cargar claves de API desde .env
dotenv_path = ""
load_dotenv(dotenv_path)
openai_key = os.getenv("OPENAI_API_KEY2")
cohere_api_key = os.getenv("COHERE_API_KEY")

# 2. Inicialización de los clientes de OpenAI y Cohere
llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=openai_key)
co = cohere.Client(cohere_api_key)

# 3. Carga  del vectorstore y se emplea como recuperador inicial
vectordb = get_vectordb500()
retriever = vectordb.as_retriever(search_kwargs={"k": 10})


# 4. Definición de la memoria conversacional, empleando la respuesta como output
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

# 5. Definición de la función de re-ranking con Cohere
def cohere_rerank(documents, query):
    document_texts = [doc.page_content for doc in documents]
    response = co.rerank(query=query, documents=document_texts)
    scores = [result.relevance_score for result in response.results]
    ranked_documents = [documents[i] for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)]
    return ranked_documents

# 6. Creación el nuevo recuperador personalizado con la función de re-ranking, tomando únicamente los 5 mejores documentos entre los recuperados inicialmente
def custom_retriever(query):
    docs = retriever.get_relevant_documents(query)
    ranked_docs = cohere_rerank(docs, query)
    return ranked_docs[:5] 

class CustomRetriever(VectorStoreRetriever):
    def _get_relevant_documents(self, query):
        return custom_retriever(query)

# 7. Creación de la instancia del recuperador con re-ranking
reranked_retriever = CustomRetriever(vectorstore=vectordb)

# 8. Definición del prompt personalizado, emleando la memoria y el contexto recuperado, así como la pregunta del usuario
custom_prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=(
        "You are a medical AI assistant specialized in **rheumatology**, with a strong focus on clinical diagnosis.\n"
        "Act as a board-certified rheumatologist providing evidence-based, concise, and medically accurate answers.\n\n"
        "Here is the previous conversation history:\n{chat_history}\n\n"
        "You have access to the following clinical guidelines and medical literature:\n{context}\n\n"
        "Based on the above information, respond to the user's question:\n{question}\n\n"
        "Only use the provided information to answer. If there is insufficient data to give a reliable response, clearly state that the information is not available. Do not speculate or fabricate any content."
    )
)

# 9. Creación de la cadena RAG con memoria, re-ranking y prompt personalizado
rag_chain500 = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=reranked_retriever,
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": custom_prompt},
    callbacks=callback_manager
)
