import gradio as gr
from RAG_RERANKER_MEMORY_LANGSMITH_1000 import rag_chain1000  # Make sure the path is correct

# Optional: Conversation history storage (not persistent)
chat_history = []

# Función para generar las respuestas a partir de los tres modelos
def answer_question(question):
    if not question.strip():
        return "Please enter a question.", ""

    # Se invoca el modelo RAG con re-ranking y memoria
    result = rag_chain1000.invoke({"question": question})
    answer = result["answer"]
    sources = result["source_documents"][:5]

    # Se determina el formato que mostrarán los metadatos de los documentos recuperados
    sources_info = ""
    for doc in sources:
        metadata = doc.metadata or {}
        title = metadata.get("name", "Title not available")
        original_source = metadata.get("original_source", "Unknown source")
        year = metadata.get("year", "Year not available")
        pathology = metadata.get("pathology", "Unspecified pathology")
        doi = metadata.get("doi", "DOI not available")
        pubmed = metadata.get("pubmed", "PubMed not available")
        source = metadata.get("source", "Unknown file")

        sources_info += (
            f"- {title} ({year}) - {original_source} [Pathology: {pathology}]\n"
            f"  File: {source}\n"
            f"  DOI: {doi}\n"
            f"  PubMed: {pubmed}\n\n"
        )

    return answer, sources_info

# Creación de la interfaz de Gradio
demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="Ask a question about clinical guidelines", placeholder="E.g.: What is the treatment for severe asthma?", lines=2),
    outputs=[
        gr.Textbox(label="Model's Answer"),
        gr.Textbox(label="Sources Used")
    ],
    title="RAG Assistant with Re-Ranking and Memory",
    description="Ask anything about the processed clinical guidelines. The model will retrieve the most relevant documents, re-rank them using Cohere, and generate an informed answer. The conversation context is preserved during the session.",
)

# Ejecución de la apliación de GRADIO
if __name__ == "__main__":
    demo.launch()
