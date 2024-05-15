import gradio as gr

from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatUpstage(streaming=True)

def predict(message, history):
    # Change this to your function
    history_langchain_format = [("system", "You are a helpful assistant.")]
    for human, ai in history:
        history_langchain_format += ("human", human)
        history_langchain_format += ("ai", ai)
    history_langchain_format += ("human", message)

    print(history_langchain_format)
    
    chain = ChatPromptTemplate.from_messages(history_langchain_format) | llm | StrOutputParser()

    generator = chain.stream({})

    assistant = ""
    for gen in generator:
        assistant += gen
        yield assistant

with gr.Blocks() as demo:
    chatbot = gr.ChatInterface(
        predict, 
        examples=["How to eat healthy?", "Best Places in Korea", "How to make a chatbot?"],
        title="Solar Chatbot",
        description="Upstage Solar Chatbot",
    )
    chatbot.chatbot.height = 400

if __name__ == "__main__":
    demo.launch()