#  Integración de la Arquitectura de Generación Aumentada con Recuperación para Guías Clínicas: Desarrollo de un Chatbot de Apoyo para Reumatólogos

Este repositorio contiene el código, los datos y los resultados del Trabajo de Fin de Máster (TFM) de **Daniel Camacho Montaño**, centrado en el desarrollo de un sistema RAG (Retrieval-Augmented Generation) aplicado a guías clínicas reumatológicas para asistir en la práctica clínica mediante un chatbot.

🔗 Repositorio oficial: [https://github.com/dcamacmon/DCM_ARQUITECTURA-RAG](https://github.com/dcamacmon/DCM_ARQUITECTURA-RAG)

---

##  Descripción del proyecto

Se diseñó una arquitectura RAG personalizada que combina técnicas de recuperación semántica y generación de lenguaje natural, utilizando guías clínicas reumatológicas como fuente de conocimiento. El objetivo fue evaluar el impacto de distintas variantes del sistema en la generación de respuestas clínicas.

### Modelos evaluados:

- **LLM Baseline**: modelo sin recuperación de contexto.
- **RAG500**: arquitectura RAG con fragmentos (chunks) de 500 caracteres.
- **RAG1000**: arquitectura RAG con fragmentos de 1000 caracteres.

Se realizaron tres experimentos utilizando distintos modelos de OpenAI:

1. **GPT-3.5 Turbo**
2. **GPT-4o mini**
3. **GPT-4.1 mini**

Las respuestas generadas fueron evaluadas automáticamente mediante un modelo externo (**Gemini AI**) utilizando cinco métricas principales.

---

##  Métricas de evaluación

- **Fidelidad** al contexto recuperado  
- **Relevancia** de la información  
- **Precisión** factual  
- **Completitud**, tanto general como contextual  
- **Concisión**  

Los resultados fueron analizados estadísticamente mediante:
- **Test binomial** para preferencias por modelo.
- **Test de Wilcoxon** para diferencias significativas entre pares.

---

##  Tecnologías y herramientas

- **Python 3.11**
- **LangChain**
- **OpenAI API**
- **Gemini AI**
- **ChromaDB / FAISS** para recuperación vectorial
- **Streamlit** (opcional, para interfaz)
- **Pandas, NumPy, SciPy, Matplotlib, Seaborn**

---

##  Estructura del repositorio

DCM_ARQUITECTURA-RAG/
├── guias_clinicas/ # Guías clínicas en formato PDF recopiladas desde la ACR

├── PARSEO/ # Scripts principales y procesamiento

│ ├── GROBID/ # Contiene los archivos XML resultantes del parseo de Grobid

│ ├── GROBID_MOD/ #Contiene los archivos XML tras la limpieza
│ ├── LlamaCloud/ # Contiene los archivos Markdown resultantes del parseo de Grobid
│ ├── LlamaCloud_TAB/ #Contiene los archivos Markdown tras la limpieza
│ ├── CHUNK1000/ # Contiene los chunks de 1000 caracteres y 150 de overlap
│ ├── CHUNK500/ # Contiene los chunks de 500 caracteres y 75 de overlap
│ ├── preprocessing.py #Contiene el workflow del preprocesamiento: parseo con grobid y llamacloud, limpieza, chunking con RecursiveCharacter, embedding con text-embedding-3-large y almacenamiento en el vectorstore de ChromaDB
│ ├── Cargar_vectorstore.py # Creación de las funciones de recuperación del vectorstore
│ ├── RAG_RERANKER_MEMORY_LANGSMITH_1000.py # Creación de la cadena RAG empleando el vectorstore de fragmentos de 1000 caracteres
│ ├── RAG_RERANKER_MEMORY_LANGSMITH_500.py # Creación de la cadena RAG empleando el vectorstore de fragmentos de 500 caracteres
│ ├── RAG_sin_Retrieval.py # Creación del modelo puramente generativo
│ ├── Generación de preguntas.py # Script de generación de las preguntas evaluadoras 
│ ├── Generación de respuestas.py # Script de generación de las respuestas en base a los tres modelos generados, a partir de las preguntas generadas por el LLM 
│ ├── ComparaciónRAG500_vs_Generative.py # Script de comparación entre los modelos RAG500 y el modelo Generative mediante Gemini 1.5 pro
│ ├── ComparaciónRAG500_RAG1000_Generative.py # Script de comparaciones 2 a 2 de los modelos mediante Gemini 1.5 pro
│ ├── GRADIO.py # Script de ejecución de la interfaz visual empleando el modelo RAG1000
│ ├── Metadata.xlsx #Documento excel con los metadatos de los archivos
│ ├── Evaluación de los modelos.ipynb #Notebook de evaluación de los resultados de las comparaciones entre los modelos
├── docs/ # Documentos obtenidos resultantes de los scripts
│ ├── chroma/ #Directorio que contiene ambos vectorstore
│ ├──preguntas_generadas/ #Directorio con el contenido de las preguntas (10 por patologia)
│ ├──preguntas_generadas20/ #Directorio con el contenido de las preguntas (20 por patologia)
│ ├──comparisons_gemini/ #Directorio con el resultado de las comparaciones de los modelos con GPT-3.5 Turbo
│ ├──comparisons_gemini_340/ #Directorio con el resultado de las comparaciones de los modelos con GPT-4o-mini
│ ├──comparisons_gemini_gpt4o/ #Directorio con el resultado de las comparaciones de los modelos con GPT-4.1-mini y con 20 preguntas por patologia
│ ├──respuestas_generadas_contexto/ #Directorio con las respuestas generadas por los modelos con GPT-3.5 Turbo
│ ├──respuestas_generadas_contexto_340/ #Directorio con las respuestas generadas por los modelos con GPT-4o-mini
│ ├──respuestas_generadas_contexto_gpt4o/ #Directorio con las respuestas generadas por los modelos con GPT-4.1-mini y con 20 preguntas por patologia
├── README.md # Este archivo

---

## ▶️ Cómo ejecutar

1. Clona el repositorio:

```bash
git clone https://github.com/dcamacmon/DCM_ARQUITECTURA-RAG.git
cd DCM_ARQUITECTURA-RAG
