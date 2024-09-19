import streamlit as st
from langchain_upstage import ChatUpstage as Chat

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from tokenizers import Tokenizer


solar_tokenizer = Tokenizer.from_pretrained("upstage/solar-pro-preview-tokenizer")


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



def prompt_engineering(original_prompt, chat_history=None):
    MODEL_NAME = "solar-pro"
    solar_pro = Chat(model=MODEL_NAME)
    chain = prompt | solar_pro | parser

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
