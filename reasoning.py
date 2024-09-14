# from https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps

import streamlit as st

from pydantic import BaseModel, Field

from langchain_upstage import ChatUpstage as Chat

from langchain_community.tools import DuckDuckGoSearchResults


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import AIMessage, HumanMessage

MAX_TOKENS = 4000
MAX_SEAERCH_RESULTS = 5

MODEL_NAME = "solar-pro"

llm = Chat(model=MODEL_NAME)

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
Example 3:

Instruction: Create a C++ program that connects to a Cassandra database and performs basic CRUD operations (Create, Read, Update, Delete) on a table containing employee information (ID, name, department, salary). Use prepared statements for queries and ensure proper error handling.
 None

Reasoning: 1. Understand the instruction: Create a C++ program that connects to a Cassandra database and performs basic CRUD operations on a table containing employee information using prepared statements and proper error handling.
2. Identify required libraries: Include the necessary libraries for connecting to Cassandra and handling errors.
3. Establish a connection to the Cassandra cluster: Create a cluster and session object, set the contact points, and connect to the cluster.
4. Create a keyspace and table for employee information: Write the CQL queries for creating the keyspace and table, execute them, and handle any errors.
5. Prepare CRUD statements: Write the CQL queries for insert, select, update, and delete operations, and prepare them using the Cassandra session.
6. Perform basic CRUD operations using prepared statements:
   a. Insert an employee record: Generate a UUID for the employee ID, bind the prepared insert statement with the employee data, and execute the query.
   b. Read the inserted employee record: Bind the prepared select statement with the employee ID, execute the query, and display the employee information.
   c. Update the employee's salary: Bind the prepared update statement with the new salary and employee ID, and execute the query.
   d. Delete the employee record: Bind the prepared delete statement with the employee ID, and execute the query.
7. Handle errors: Check the error codes for each query execution and print error messages if necessary.
8. Clean up and close the connection: Free the prepared statements, UUID generator, and close the session and cluster objects.
9. Compile and run the program: Provide instructions for installing the DataStax C/C++ driver, compiling the program, and running it with a local Cassandra cluster.

Reasoning Chains: [{'step': 1, 'thought': 'Understand the instruction: Create a C++ program that connects to a Cassandra database and performs basic CRUD operations on a table containing employee information using prepared statements and proper error handling.'}, {'step': 2, 'thought': 'Identify required libraries: Include the necessary libraries for connecting to Cassandra and handling errors.'}, {'step': 3, 'thought': 'Establish a connection to the Cassandra cluster: Create a cluster and session object, set the contact points, and connect to the cluster.'}, {'step': 4, 'thought': 'Create a keyspace and table for employee information: Write the CQL queries for creating the keyspace and table, execute them, and handle any errors.'}, {'step': 5, 'thought': 'Prepare CRUD statements: Write the CQL queries for insert, select, update, and delete operations, and prepare them using the Cassandra session.'}, {'step': 6, 'thought': "Perform basic CRUD operations using prepared statements:\n   a. Insert an employee record: Generate a UUID for the employee ID, bind the prepared insert statement with the employee data, and execute the query.\n   b. Read the inserted employee record: Bind the prepared select statement with the employee ID, execute the query, and display the employee information.\n   c. Update the employee's salary: Bind the prepared update statement with the new salary and employee ID, and execute the query.\n   d. Delete the employee record: Bind the prepared delete statement with the employee ID, and execute the query."}, {'step': 7, 'thought': 'Handle errors: Check the error codes for each query execution and print error messages if necessary.'}, {'step': 8, 'thought': 'Clean up and close the connection: Free the prepared statements, UUID generator, and close the session and cluster objects.'}, {'step': 9, 'thought': 'Compile and run the program: Provide instructions for installing the DataStax C/C++ driver, compiling the program, and running it with a local Cassandra cluster.'}]
---
"""

reasoning_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are Solar, a smart search reasoning and answer engine by Upstage, loved by many people. 
            
            See reasoning examples, context provided for instruction. 
            Follow the instrution in user query and provide best answer for the query using reasoning technique and step by step explanation.
            ---
            {reasoning_examples}
            """,
        ),
        MessagesPlaceholder("chat_history"),
    ]
)

short_answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are Solar, a smart search engine by Upstage, loved by many people. 
            
            Write one word answer if you can say "yes", "no", or direct answer. 
            Otherwise just one or two sentense short answer for the query from the given conetxt.
            Try to understand the user's intention and provide a quick answer.
            If the answer is not in context, please say you don't know and ask to clarify the question.
            """,
        ),
        MessagesPlaceholder("chat_history"),
        (
            "human",
            """Query: {user_query} 
         ----
         Context: {context}""",
        ),
    ]
)

search_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are Solar, a smart search engine by Upstage, loved by many people. 
            
            See the origial query, context, and quick answer, and then provide detailed explanation.

            Try to understand the user's intention and provide the relevant information in detail.
            If the answer is not in context, please say you don't know and ask to clarify the question.
            Do not repeat the short answer.

            When you write the explnation, please cite the source like [1], [2] if possible.
            Thyen, put the cited references including citation number, title, and URL at the end of the answer.
            Each reference should be in a new line in the markdown format like this:

            [1] Title - URL
            [2] Title - URL
            ...
            """,
        ),
        MessagesPlaceholder("chat_history"),
        (
            "human",
            """Query: {user_query} 
         ----
         Short answer: {short_answer}
         ----
         Context: {context}""",
        ),
    ]
)


query_context_expansion_prompt = """
For a given query and context(if provided), expand it with related questions and search the web for answers.
Try to understand the purpose of the query and expand  with upto three related questions 
to privde answer to the original query. 
Note that it's for keyword-based search engines, so it should be short and concise.

Please write in Python LIST format like this:
["number of people in France?", How many people in France?", "France population"]

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


def perform_task(chat_history):
     # Limit chat history to 3000 characters
    limited_history = []
    total_length = 0
    for message in reversed(chat_history):
        message_length = len(message.content)
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
        }
    )


def get_search_desc(user_query, short_answer, context, chat_history):
    chain = search_prompt | llm | StrOutputParser()

    return chain.stream(
        {
            "context": context,
            "chat_history": chat_history,
            "user_query": user_query,
            "short_answer": short_answer,
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
    or_merged_search_query = " OR ".join(q_list)
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

tasks = ["Reasoning", "Reasoning Chains", "Final Answer"]

if prompt := st.chat_input(q):

    search_result = search(prompt, st.session_state.messages)

    with st.status("Search Results:"):
        st.write(search_result)

    if search_result:
        search_result = str(search_result)[:MAX_SEAERCH_RESULTS]
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

    for task in tasks:
        instruction = f"""Please provide {task} for the given query,and context and chat history. 
        Please only provide the {task}.
        ---
        User Query: 
        {prompt}"""
        st.session_state.messages.append(HumanMessage(content=instruction))
        with st.chat_message("user"):
            st.write(instruction)
        with st.chat_message("assistant"):
            response = st.write_stream(perform_task(st.session_state.messages))
        st.session_state.messages.append(AIMessage(content=response))
