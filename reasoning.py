import streamlit as st
import time
from pydantic import BaseModel, Field

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.messages import AIMessage, HumanMessage

from langchain_upstage import ChatUpstage as Chat
from langchain_community.tools import DuckDuckGoSearchResults

from solar_util import num_of_tokens

MAX_TOKENS = 2500
MAX_SEARCH_TOKENS = 700
MAX_SEAERCH_RESULTS = 5

llm = Chat(model="solar-pro")
ddg_search = DuckDuckGoSearchResults()

st.set_page_config(page_title="Solar Reasoning", page_icon="🤔")
st.title("Solar Reasoning")

reasoning_examples = """
---
Example 1:

Use Query: If a die is rolled three times, what is the probability of getting a sum of 11? 

Reasoning: 1. Understand the problem: We need to find the probability of getting a sum of 11 when rolling a die three times.
2. Calculate total possible outcomes: A die has 6 faces, so for each roll, there are 6 possibilities. For three rolls, the total possible outcomes are 6^3 = 216.
3. Identify favorable outcomes: List all combinations of rolls that result in a sum of 11. There are 18 such combinations.
4. Calculate probability: Divide the number of favorable outcomes by the total possible outcomes: 18 / 216 = 1/12.
5. Conclusion: The probability of getting a sum of 11 when rolling a die three times is 1/12.

Reasoning Chains: [{'step': 1, 'thought': 'Understand the problem: We need to find the probability of getting a sum of 11 when rolling a die three times.'}, {'step': 2, 'thought': 'Calculate total possible outcomes: A die has 6 faces, so for each roll, there are 6 possibilities. For three rolls, the total possible outcomes are 6^3 = 216.'}, {'step': 3, 'thought': 'Identify favorable outcomes: List all combinations of rolls that result in a sum of 11. There are 18 such combinations.'}, {'step': 4, 'thought': 'Calculate probability: Divide the number of favorable outcomes by the total possible outcomes: 18 / 216 = 1/12.'}, {'step': 5, 'thought': 'Conclusion: The probability of getting a sum of 11 when rolling a die three times is 1/12.'}]
----
Example 2:

User Query: The interactions will be about the science behind culinary techniques. The setting is a cooking class where three friends are discussing various aspects of cooking and sharing their knowledge.
- USER/Jane: A curious learner who wants to understand the science behind cooking
- Mike: An experienced home cook with a passion for experimenting in the kitchen
- Sarah: A food scientist who loves explaining the chemistry behind different cooking processes

Reasoning: 1. Start with the given setting: a cooking class with three friends discussing the science behind culinary techniques.
2. Introduce the topic of resting meat after cooking, with Mike asking Jane if she's ever wondered about it.
3. Have Sarah explain the science behind resting meat, mentioning denatured proteins and juice redistribution.
4. Address the user's question about resting meat, with Sarah confirming that it allows juices to redistribute.
5. Move on to the topic of adding salt to water, with Mike mentioning its effect on boiling point.
6. Have Sarah explain the science behind salt's effect on boiling point, mentioning the higher temperature required for boiling.
7. Address the user's question about cooking speed, with Sarah explaining that it's slightly faster due to the hotter water.
8. Introduce the topic of acids in cooking, with Mike mentioning their use in brightening dishes.
9. Have Sarah explain the science behind acids' effects on flavor and tenderizing meats.
10. Address the user's question about baking, with Mike mentioning the science involved in baking and Sarah explaining the role of gluten and leavening agents.
11. Conclude the conversation with the characters expressing their fascination with the science behind cooking and their excitement to continue learning and experimenting.

Reasoning Chains: [{'step': 1, 'thought': 'Start with the given setting: a cooking class with three friends discussing the science behind culinary techniques.'}, {'step': 2, 'thought': "Introduce the topic of resting meat after cooking, with Mike asking Jane if she's ever wondered about it."}, {'step': 3, 'thought': 'Have Sarah explain the science behind resting meat, mentioning denatured proteins and juice redistribution.'}, {'step': 4, 'thought': "Address the user's question about resting meat, with Sarah confirming that it allows juices to redistribute."}, {'step': 5, 'thought': 'Move on to the topic of adding salt to water, with Mike mentioning its effect on boiling point.'}, {'step': 6, 'thought': "Have Sarah explain the science behind salt's effect on boiling point, mentioning the higher temperature required for boiling."}, {'step': 7, 'thought': "Address the user's question about cooking speed, with Sarah explaining that it's slightly faster due to the hotter water."}, {'step': 8, 'thought': 'Introduce the topic of acids in cooking, with Mike mentioning their use in brightening dishes.'}, {'step': 9, 'thought': "Have Sarah explain the science behind acids' effects on flavor and tenderizing meats."}, {'step': 10, 'thought': "Address the user's question about baking, with Mike mentioning the science involved in baking and Sarah explaining the role of gluten and leavening agents."}, {'step': 11, 'thought': 'Conclude the conversation with the characters expressing their fascination with the science behind cooking and their excitement to continue learning and experimenting.'}]
----
"""

reasoning_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are Solar, a smart reasoning and answer engine by Upstage, loved by many people.. 
            
For the given query, please provide the best answer using a step-by-step explanation. 
Your response should demonstrate a Chain of Thought (CoT) technique, 
where you think aloud and describe the steps you take to reach a conclusion. 

Please best use of the provided reasoning examples and context.
---
{reasoning_examples}
            """,
        ),
        MessagesPlaceholder("chat_history"),
        (
            "human",
            """For the given query, please provide only the "{task}" 
and ensure your response is consistent with the user's request, 
previous chat history, and provided reasoning if any. 
Remember to use the self-consistency technique to maintain a consistent character of a helpful assistant.
Think step by step and provide the best answer for the query.
---
User Query: 
{prompt}
---
{Reasoning}
---
{ReasoningChains}""",
        ),
    ]
)


query_context_expansion_prompt = """Given a query and context(if provided), 
generate up to three related questions to help answer the original query.
Ensure the questions are short, concise, and keyword-based for search engines. 

Write your response in Python LIST format. 

For example: 
["original query", "related question 1", "related question 2", "related question 3"]

---
Context: {context}
----
History: {chat_history}
---
Orignal query: {query}
"""


# Define your desired data structure.
class List(BaseModel):
    list[str]


def query_context_expansion(query, chat_history, context=None):
    # Set up a parser + inject instructions into the prompt template.
    parser = JsonOutputParser(pydantic_object=List)

    prompt = PromptTemplate(
        template=query_context_expansion_prompt,
        input_variables=["query", "context"],
    )

    chain = prompt | llm | parser
    # Invoke the chain with the joke_query.

    for attempt in range(3):
        try:
            parsed_output = chain.invoke(
                {"query": query, "chat_history": chat_history, "context": context}
            )
            return parsed_output
        except Exception as e:
            st.warning(f"Attempt {attempt + 1} failed. Retrying...")

    st.error("All attempts failed. Returning empty list.")
    return []


GlobalTasks = ["Reasoning (No conclusion)", "Reasoning Chains", "Final Answer"]


def perform_task(user_query, task, task_results, chat_history):
    # Limit chat history to 3000 characters
    limited_history = []
    total_length = 0
    for message in reversed(chat_history):
        message_length = num_of_tokens(message.content)
        if total_length + message_length > MAX_TOKENS:
            st.warning("Chat history is too long. Truncating.")
            break
        limited_history.insert(0, message)
        total_length += message_length

    chain = reasoning_prompt | llm | StrOutputParser()

    return chain.stream(
        {
            "chat_history": limited_history,
            "reasoning_examples": reasoning_examples,
            "prompt": user_query,
            "task": task,
            "Reasoning": task_results.get(GlobalTasks[0], ""),
            "ReasoningChains": task_results.get(GlobalTasks[1], ""),
        }
    )


def search(query, chat_history, context=None):
    with st.status("Extending query..."):
        q_list = query_context_expansion(query, chat_history, context)
        st.write(q_list)

    if not q_list:
        st.error("No related questions found. Returning empty list.")
        return []

    # combine all queries with "OR" operator
    results = ""
    for q in q_list:
        with st.spinner(f"Searching for '{q }'..."):
            results += ddg_search.invoke(q)

    return results


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = "AI" if isinstance(message, AIMessage) else "Human"
    with st.chat_message(role):
        st.markdown(message.content)

q = "3.9 vs 3.11. Which one is bigger?"

search_on = st.checkbox("Search on the web", value=False)

if prompt := st.chat_input(q):
    with st.chat_message("user"):
        st.markdown(prompt)

    if search_on:
        search_result = search(prompt, st.session_state.messages)

        with st.status("Search Results:"):
            st.write(search_result)

        if search_result:
            search_result = str(search_result)[:MAX_SEARCH_TOKENS]
            st.session_state.messages.append(
                HumanMessage(
                    content=f"FYI search result conext: {search_result} for the query, {prompt}"
                )
            )
            st.session_state.messages.append(
                AIMessage(
                    content="Thanks for the information! I will keep in mind. Give me the instruction."
                )
            )
    task_results = {}
    current_time = time.time()
    for task in GlobalTasks:
        if task == GlobalTasks[-1]:
            st.info(f"Thinking: {time.time() - current_time:.2f}s")

            with st.chat_message("assistant"):
                response = st.write_stream(
                    perform_task(prompt, task, task_results, st.session_state.messages)
                )
                task_results[task] = response
            break

        with st.status(f"Performing task: {task}"):
            response = st.write_stream(
                perform_task(prompt, task, task_results, st.session_state.messages)
            )
            task_results[task] = response
    # Store the last task result for future reference
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.session_state.messages.append(AIMessage(content=task_results[GlobalTasks[-1]]))