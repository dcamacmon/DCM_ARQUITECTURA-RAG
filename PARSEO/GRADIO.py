#GRADIO. CHATBOT

import gradio as gr
from RAG_RERANKER_MEMORY import rag_chain  # Asegúrate de que el path y nombre del archivo es correcto

# Historial de conversación (reseteado al cerrar sesión)
chat_history = []

# Función para responder preguntas
def responder_pregunta(pregunta):
    if not pregunta.strip():
        return "Por favor, introduce una pregunta."

    # Invocar la cadena RAG
    result = rag_chain.invoke({"question": pregunta})
    respuesta = result["answer"]
    fuentes = result["source_documents"][:5]

    # Formatear las fuentes
    fuentes_info = ""
    for doc in fuentes:
        metadata = doc.metadata or {}
        titulo = metadata.get("name", "Título no disponible")
        fuente_original = metadata.get("original_source", "Fuente desconocida")
        año = metadata.get("year", "Año no disponible")
        patologia = metadata.get("pathology", "Patología no especificada")
        doi = metadata.get("doi", "DOI no disponible")
        pubmed = metadata.get("pubmed", "PubMed no disponible")
        fuente = metadata.get("source", "Fuente desconocida")

        fuentes_info += (
            f"- {titulo} ({año}) - {fuente_original} [Patología: {patologia}]\n"
            f"  Fuente: {fuente}\n"
            f"  DOI: {doi}\n"
            f"  PubMed: {pubmed}\n\n"
        )

    return respuesta, fuentes_info

# Crear la interfaz
demo = gr.Interface(
    fn=responder_pregunta,
    inputs=gr.Textbox(label="Pregunta sobre guías clínicas", placeholder="Ej: What is the treatment for severe asthma?", lines=2),
    outputs=[
        gr.Textbox(label="Respuesta del modelo"),
        gr.Textbox(label="Fuentes utilizadas")
    ],
    title="Asistente RAG con Re-Ranking y Memoria",
    description="Pregunta sobre las guías clínicas procesadas. El modelo buscará información relevante, reordenará las fuentes más útiles y generará una respuesta.",
)

# Ejecutar la app
if __name__ == "__main__":
    demo.launch()
