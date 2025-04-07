import os
import openai
from langchain_community.chat_models import ChatOpenAI  # Importar desde langchain_community
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Cargar variables de entorno
dotenv_path = r"C:\Users\Daniel\Desktop\DOCUMENTOS\TFM\PDF STORE\PARSEO\.env"
load_dotenv(dotenv_path)
openai_key = os.getenv("OPENAI_API_KEY2")

# Configuración de OpenAI
openai.api_key = openai_key
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_key)

# Definir un template de prompt personalizado
custom_prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question: {question}. "
             "Provide a concise and informative response based solely on your knowledge. "
             "If you do not know the answer, say that you don't know."
)

# Crear la cadena generativa utilizando el modelo sin recuperación
def generative_query(query):
    # Crear el prompt para el modelo
    prompt = custom_prompt.format(question=query)
    
    # Asegurarse de que el prompt esté en el formato adecuado para `invoke`
    messages = [{"role": "user", "content": prompt}]
    
    # Obtener la respuesta del modelo generativo usando `invoke`
    response = llm.invoke(messages)
    
    # Acceder al contenido de la respuesta generada
    content = response.content  # Acceder directamente al atributo 'content' del objeto AIMessage
    
    return content

# Realizar una consulta generativa
query = "Is urate-lowering therapy (ULT) recommended for patients with asymptomatic hyperuricemia?"
respuesta = generative_query(query)

# Mostrar solo el contenido de la respuesta generada
print("\n  Respuesta Generada:")
print(respuesta)
