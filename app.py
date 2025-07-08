import gradio as gr
from src.rag_pipeline import answer_question

def rag_interface(question):
    answer, sources = answer_question(question)
    sources_formatted = "\n\n".join([f"- {s}" for s in sources])
    return answer.strip(), sources_formatted

iface = gr.Interface(
    fn=rag_interface,
    inputs="text",
    outputs=["text", "text"],
    title="CrediTrust Complaint Assistant",
    description="Ask about customer complaint trends across financial products."
)

if __name__ == "__main__":
    iface.launch()
