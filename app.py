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

    gr.Markdown("## 🤖 AI Notes Chatbot")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask any topic...", label="Your Question")

    def chat(user_input, chat_history):
        if not user_input:
            return "", chat_history

        response = generate_notes(user_input)

        chat_history.append((user_input, response))
        return "", chat_history

    msg.submit(chat, [msg, chatbot], [msg, chatbot])

demo.launch()
