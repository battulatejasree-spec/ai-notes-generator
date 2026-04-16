import os
import gradio as gr
from langchain_google_genai import ChatGoogleGenerativeAI

# API KEY
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# MODEL
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY
)

# MEMORY
history_memory = []

# FUNCTION
def generate_notes(topic):
    global history_memory

    if not topic or not topic.strip():
        return "⚠️ Please enter a topic."

    try:
        context = ""
        for t, n in history_memory:
            context += f"Topic: {t}\nNotes: {n}\n\n"

        prompt = f"""
        {context}

        Create clear, structured study notes on: {topic}

        Include:
        - Definition
        - Key Concepts
        - Important Points
        - Examples
        - Summary

        Keep it simple and student-friendly.
        """

        response = llm.invoke(prompt)

        if isinstance(response.content, list):
            notes = response.content[0].get("text", "")
        else:
            notes = response.content

        if not notes:
            return "⚠️ No response generated"

        history_memory.append((topic, notes))

        return notes

    except Exception as e:
        return f"❌ Error: {str(e)}"

# CLEAR
def clear_all():
    global history_memory
    history_memory = []
    return "", ""

# UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("## 📘 AI Notebook Generator")
    gr.Markdown("### Generate structured notes instantly")

    topic_input = gr.Textbox(
        placeholder="Enter topic (e.g., Machine Learning)",
        label="Topic"
    )

    with gr.Row():
        generate_btn = gr.Button("Generate Notes 🚀", variant="primary")
        clear_btn = gr.Button("Clear 🧹")

    output_box = gr.Markdown()

    generate_btn.click(generate_notes, inputs=topic_input, outputs=output_box)
    topic_input.submit(generate_notes, inputs=topic_input, outputs=output_box)
    clear_btn.click(clear_all, None, [topic_input, output_box])

# LAUNCH
demo.launch()
