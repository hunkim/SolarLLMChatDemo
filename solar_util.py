import streamlit as st
from langchain_upstage import ChatUpstage 

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from tokenizers import Tokenizer


solar_tokenizer = Tokenizer.from_pretrained("upstage/solar-pro-preview-tokenizer")


def truncate_to_token_limit(text: str, max_tokens: int = 15000) -> str:
    """
    Truncate text to fit within max token limit using Solar tokenizer.
    """
    tokenizer = Tokenizer.from_pretrained("upstage/solar-pro-tokenizer")
    encoded = tokenizer.encode(text)
    
    if len(encoded.ids) <= max_tokens:
        return text
    
    print(f"Truncating text from {len(encoded.ids)} tokens to {max_tokens} tokens")
    
    # Find the last period within the token limit to avoid cutting mid-sentence
    truncated_ids = encoded.ids[:max_tokens]
    truncated_text = tokenizer.decode(truncated_ids)
    
    # Try to find the last complete sentence
    last_period = truncated_text.rfind('.')
    if last_period > 0:
        truncated_text = truncated_text[:last_period + 1]
    
    return truncated_text


def initialize_solar_llm(MODEL_NAME=None):
    if MODEL_NAME is None:
        MODEL_NAME = st.secrets.get("SOLAR_MODEL_NAME", "solar-pro")

    # Initialize llm with default values
    llm_kwargs = {"model": MODEL_NAME}
    
    # Add base_url if it's set in secrets
    if "SOLAR_BASE_URL" in st.secrets:
        llm_kwargs["base_url"] = st.secrets["SOLAR_BASE_URL"]

    # Add api_key if it's set in secrets
    if "SOLAR_API_KEY" in st.secrets:
        llm_kwargs["api_key"] = st.secrets["SOLAR_API_KEY"]

    return ChatUpstage(**llm_kwargs)


# Define your desired data structure.
# {"original_prompt": "original prompt", "enhanced_prompt": "enhanced prompt", "techniques": "technique"}
# Define your desired data structure.
class PromptEngineering(BaseModel):
    original_prompt: str = Field(description="original prompt")
    enhanced_prompt: str = Field(
        description="enhanced prompt after applying prompt engineering techniques"
    )
    techniques: str = Field(
        description="prompt engineering technique used to enhance the prompt"
    )


parser = JsonOutputParser(pydantic_object=PromptEngineering)

prompt = """Use these prompt engineering technique and enhance user prompt to generate more effective prompt.
Consider the chat history for context if available. 
Please write the promt in Korean.
----
Chat History:
{chat_history}
----
Output should be in json format:
\n{format_instructions}
----
prompt engineering techniques:

Chain of Thought (CoT): This technique encourages the model to think aloud, showing the steps it takes to reach a conclusion. Example: "Imagine you're a detective solving a mystery. Describe your thought process as you deduce who the culprit is in this scenario: [insert scenario]."

Chain of Cause (CoC): This technique focuses on identifying and explaining the causes and effects in a situation. Example: "You're a historian analyzing a historical event. Describe the chain of causes that led to this event: [insert event]."

Program-Aided Language Models (PAL): This technique involves providing a simple program or pseudo-code to guide the model's response. Example: "Write a Python function to calculate the factorial of a number. Then, use this function to find the factorial of 5."

Tree of Thoughts (ToT): This technique visualizes the thought process as a tree, with branches representing different ideas or possibilities. Example: "You're a marketing strategist brainstorming ideas for a new campaign. Present your ideas as a tree of thoughts, with the main idea at the root and branches representing sub-ideas."

Least-to-Most: This technique starts with the simplest or most basic explanation and gradually increases complexity. Example: "Explain the concept of machine learning, starting from the most basic definition and gradually adding more details and complexities."

Self-Consistency: This technique encourages the model to ensure its responses are consistent with previous statements or information. Example: "You're a character in a story. Ensure all your responses are consistent with the character's background and previous statements."
----
originalprompt: {original_prompt}
----

"""

prompt = PromptTemplate(
    template=prompt,
    input_variables=["original_prompt", "chat_history"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)



def prompt_engineering(original_prompt, chat_history=None, llm=None):
    if llm is None:
        llm = initialize_solar_llm()
    chain = prompt | llm | parser

    # Invoke the chain with the joke_query.
    parsed_output = chain.invoke(
        {"original_prompt": original_prompt, "chat_history": chat_history}
    )

    return parsed_output


def result_reference_summary(results):
    results.reverse()
    result_summary = ""
    for i, r in enumerate(results):
        result_summary += f"[{i+1}] {r['title']} - URL: {r['url']}\n{r['content']}\n\n"

    return result_summary


def num_of_tokens(text):
    return len(solar_tokenizer.encode(text).ids)


if __name__ == "__main__":
    print(num_of_tokens("Hello, world!"))
