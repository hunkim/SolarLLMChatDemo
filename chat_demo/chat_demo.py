import gradio as gr

from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatUpstage()

def predict(message, history):
    # Change this to your function
    history_langchain_format = [("system", "You are a helpful assistant.")]
    for human, ai in history:
        history_langchain_format += ("human", human)
        history_langchain_format += ("ai", ai)
    history_langchain_format += ("human", message)
    
    chain = ChatPromptTemplate.from_messages(history_langchain_format) | llm | StrOutputParser()

    return chain.invoke({})

with gr.Blocks() as demo:
    chatbot = gr.ChatInterface(
        predict, 
        examples=["안녕하세요", "피타고라스에 대해 알려줘", "로마제국의 역사 알려줘"],
        title="Upstage Hackathon 1조",
        description="Upstage Solar demo 입니다."
    )
    chatbot.chatbot.height = 400

if __name__ == "__main__":
    demo.launch()