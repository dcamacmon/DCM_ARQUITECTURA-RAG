import gradio as gr
from RAG_RERANKER_MEMORY_LANGSMITH import rag_chain  # Make sure the path is correct

# Optional: Conversation history storage (not persistent)
chat_history = []

# Function to answer questions
def answer_question(question):
    if not question.strip():
        return "Please enter a question.", ""

    # Invoke the RAG chain
    result = rag_chain.invoke({"question": question})
    answer = result["answer"]
    sources = result["source_documents"][:5]

    # Format source information
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

# Create the Gradio interface
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

# Launch the app
if __name__ == "__main__":
    demo.launch()
