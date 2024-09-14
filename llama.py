# from https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps

import streamlit as st

from pydantic import BaseModel, Field

from langchain_groq import ChatGroq as Chat
from langchain_community.tools import DuckDuckGoSearchResults


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import AIMessage, HumanMessage

MAX_TOKENS = 40000
MAX_SEAERCH_RESULTS = 5

MODEL_NAME = "llama-3.1-70b-versatile"

llm = Chat(model=MODEL_NAME)

ddg_search = DuckDuckGoSearchResults()


st.set_page_config(page_title="Llama Reasoning", page_icon="🤔")
st.title("Llama 3.1 70B Reasoning")

reasoning_examples = """
---
Human: Given Instruction, please generate {what}. Please use the following exampels.
    If reasoning and/or reasoning chains are provided, please use them as context to generate the {what}.
    Please only generate the {what} and do not include others.
    
    See the examples below:
    ----
Example 1:

Instruction: If a die is rolled three times, what is the probability of getting a sum of 11? None

Reasoning: 1. Understand the problem: We need to find the probability of getting a sum of 11 when rolling a die three times.
2. Calculate total possible outcomes: A die has 6 faces, so for each roll, there are 6 possibilities. For three rolls, the total possible outcomes are 6^3 = 216.
3. Identify favorable outcomes: List all combinations of rolls that result in a sum of 11. There are 18 such combinations.
4. Calculate probability: Divide the number of favorable outcomes by the total possible outcomes: 18 / 216 = 1/12.
5. Conclusion: The probability of getting a sum of 11 when rolling a die three times is 1/12.

Reasoning Chains: [{'step': 1, 'thought': 'Understand the problem: We need to find the probability of getting a sum of 11 when rolling a die three times.'}, {'step': 2, 'thought': 'Calculate total possible outcomes: A die has 6 faces, so for each roll, there are 6 possibilities. For three rolls, the total possible outcomes are 6^3 = 216.'}, {'step': 3, 'thought': 'Identify favorable outcomes: List all combinations of rolls that result in a sum of 11. There are 18 such combinations.'}, {'step': 4, 'thought': 'Calculate probability: Divide the number of favorable outcomes by the total possible outcomes: 18 / 216 = 1/12.'}, {'step': 5, 'thought': 'Conclusion: The probability of getting a sum of 11 when rolling a die three times is 1/12.'}]
----
Example 2:

Instruction: The interactions will be about the science behind culinary techniques. The setting is a cooking class where three friends are discussing various aspects of cooking and sharing their knowledge.
- USER/Jane: A curious learner who wants to understand the science behind cooking
- Mike: An experienced home cook with a passion for experimenting in the kitchen
- Sarah: A food scientist who loves explaining the chemistry behind different cooking processes

 None

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
----
Example 4:

Instruction: BEGININPUT
BEGINCONTEXT
date: August 15, 2022
author: Sarah Johnson
subject: SharePoint Server 2019 Features and Benefits
to: John Smith
ENDCONTEXT
Hi John,

I hope you're doing well. I wanted to provide you with some information on Microsoft SharePoint Server 2019 and its features and benefits. As you know, our company is considering upgrading our current system, and I believe that SharePoint Server 2019 could be a great fit for us.

SharePoint Server 2019 comes with several new features that can help improve productivity and collaboration within our organization. Some of these features include:

1. Modern Sites: SharePoint Server 2019 introduces modern team sites and communication sites, which offer an improved user experience compared to the classic sites we currently use. These modern sites are responsive by design, making them easily accessible from any device.

2. Improved File Sharing: The new version includes OneDrive for Business integration, allowing users to share files more easily both internally and externally. This feature also supports larger file sizes (up to 15 GB) and provides real-time co-authoring capabilities in Office Online.

3. Hybrid Scenarios: SharePoint Server 2019 allows for better integration between on-premises and cloud environments. This means we can take advantage of cloud-based services like Power BI, Flow, and Planner while still maintaining control over our sensitive data on-premises.

4. Enhanced Search Experience: The search functionality has been significantly improved in this version, providing personalized results based on the user's role and previous searches. This makes it easier for employees to find relevant content quickly.

5. Security and Compliance: SharePoint Server 2019 offers advanced security features such as Data Loss Prevention (DLP), eDiscovery, and Multi-Factor Authentication (MFA). These features help protect our sensitive data and ensure compliance with industry regulations.

6. Accessibility Improvements: The new version includes several accessibility enhancements, such as improved keyboard navigation and support for screen readers. This makes SharePoint more inclusive for all users.

In addition to these features, SharePoint Server 2019 also offers better performance and scalability compared to previous versions. It supports up to 250,000 site collections per content database, which is a significant increase from the 100,000 limit in SharePoint Server 2016.

Overall, I believe that upgrading to SharePoint Server 2019 would greatly benefit our organization by providing us with a modern, user-friendly platform that promotes collaboration and productivity. If you have any questions or concerns about this information, please don't hesitate to reach out.

Best regards,

Sarah Johnson
ENDINPUT

BEGININSTRUCTION
- List three new features of Microsoft SharePoint Server 2019 mentioned in the email.
- What is the maximum file size supported for sharing in SharePoint Server 2019?
- How many site collections per content database does SharePoint Server 2019 support?
Please provide references.
ENDINSTRUCTION None

Reasoning: 1. Read the instruction and identify the required information: three new features, maximum file size supported, and site collections per content database.
2. Scan the email for the mentioned features.
3. Identify the first feature: Modern Sites.
4. Identify the second feature: Improved File Sharing with OneDrive for Business integration.
5. Identify the third feature: Enhanced Search Experience.
6. Locate the information about the maximum file size supported: up to 15 GB.
7. Locate the information about site collections per content database: up to 250,000.
8. Compile the answer with the required information and reference(s).

Reasoning Chains: [{'step': 1, 'thought': 'Read the instruction and identify the required information: three new features, maximum file size supported, and site collections per content database.'}, {'step': 2, 'thought': 'Scan the email for the mentioned features.'}, {'step': 3, 'thought': 'Identify the first feature: Modern Sites.'}, {'step': 4, 'thought': 'Identify the second feature: Improved File Sharing with OneDrive for Business integration.'}, {'step': 5, 'thought': 'Identify the third feature: Enhanced Search Experience.'}, {'step': 6, 'thought': 'Locate the information about the maximum file size supported: up to 15 GB.'}, {'step': 7, 'thought': 'Locate the information about site collections per content database: up to 250,000.'}, {'step': 8, 'thought': 'Compile the answer with the required information and reference(s).'}]
----
Example 5:

Instruction: Rewrite the below text (which is a fragment of a longer script for the video). Fix grammar and other errors and make it sound more professional.

Networks and security, or how to connect the digital world. How can we exchange data quickly and efficiently. There are different types of networks.
A local area network (LAN) is for smaller, more local networks - home, business, school, etc.
A wide area network (WAN) covers larger areas such as cities and even allows computers in different countries to connect.
An intranet is a private enterprise network designed to support an organization's employees in communicating, collaborating, and performing their roles, usually to the exclusion of outside access.
We also have the well-known internet, which is the largest network connecting computers around the world.

But of course someone could try to intercept that communication. And we want to securely exchange data. We want to find vulnerabilities and defend against hackers.
This is why we have something like cryptography. It is a field of study that focuses on the encoding and decoding of info using special algorithms. So only the sender and receiver can know the content of the message.
There are many ways to protect your data, programs, devices and networks. Besides cryptography, we also use antivirus programs that can check if anything looks suspicious. A very popular trend is also the use of artificial intelligence to defend against hackers. But hackers can also use AI to attack. So it's a double-edged sword. None

Reasoning: 1. Read and understand the original text.
2. Identify grammar and other errors.
3. Determine the main points and concepts.
4. Rewrite the text to fix errors and improve professionalism.
5. Ensure all important information is included and clear.
6. Check for factual accuracy and potential confusion.
7. Finalize the revised text and present it as the output.

Reasoning Chains: [{'step': 1, 'thought': 'Read and understand the original text.'}, {'step': 2, 'thought': 'Identify grammar and other errors.'}, {'step': 3, 'thought': 'Determine the main points and concepts.'}, {'step': 4, 'thought': 'Rewrite the text to fix errors and improve professionalism.'}, {'step': 5, 'thought': 'Ensure all important information is included and clear.'}, {'step': 6, 'thought': 'Check for factual accuracy and potential confusion.'}, {'step': 7, 'thought': 'Finalize the revised text and present it as the output.'}]
----
Example 6:

Instruction: How many even perfect square factors does $2^4 \cdot 7^9$ have? None

Reasoning: 1. I need to find the number of factors of $2^4 \cdot 7^9$ that are both even and perfect squares.
2. A factor of $2^4 \cdot 7^9$ must be of the form $2^a \cdot 7^b$, where $0 \leq a \leq 4$ and $0 \leq b \leq 9$.
3. To be even, a factor must have $a > 0$, since $2^0 = 1$ is odd.
4. To be a perfect square, a factor must have both $a$ and $b$ even, since an odd power of a prime is not a perfect square.
5. I need to count how many ways I can choose $a$ and $b$ to be even and positive.
6. For $a$, I have two choices: $2$ or $4$.
7. For $b$, I have five choices: $0, 2, 4, 6, 8$.
8. So the total number of choices is $2 \cdot 5 = 10$.

Reasoning Chains: [{'step': 1, 'thought': 'I need to find the number of factors of $2^4 \\cdot 7^9$ that are both even and perfect squares.'}, {'step': 2, 'thought': 'A factor of $2^4 \\cdot 7^9$ must be of the form $2^a \\cdot 7^b$, where $0 \\leq a \\leq 4$ and $0 \\leq b \\leq 9$.'}, {'step': 3, 'thought': 'To be even, a factor must have $a > 0$, since $2^0 = 1$ is odd.'}, {'step': 4, 'thought': 'To be a perfect square, a factor must have both $a$ and $b$ even, since an odd power of a prime is not a perfect square.'}, {'step': 5, 'thought': 'I need to count how many ways I can choose $a$ and $b$ to be even and positive.'}, {'step': 6, 'thought': 'For $a$, I have two choices: $2$ or $4$.'}, {'step': 7, 'thought': 'For $b$, I have five choices: $0, 2, 4, 6, 8$.'}, {'step': 8, 'thought': 'So the total number of choices is $2 \\cdot 5 = 10$.'}]
----
Example 7:

Instruction: If the city council maintains spending at the same level as this year' s, it can be expected to levy a sales tax of 2 percent next year. Thus, if the council levies a higher tax, it will be because the council is increasing its expenditures. Which one of the following exhibits a pattern of reasoning most closely similar to that of the argument above?
A: If the companies in the state do not increase their workers'wages this year, the prices they charge for their goods can be expected to be much the same as they were last year. Thus, if the companies do increase prices, it will be because they have increased wages.
B: If newspaper publishers wish to publish good papers, they should employ good journalists. Thus, if they employ poor journalists, it will not be surprising if their circulation falls as a result.
C: If shops wish to reduce shoplifting, they should employ more store detectives. Thus, if shops do not, they will suffer reduced profits because of their losses from stolen goods.
D: If house-building costs are not now rising, builders cannot be expected to increase the prices of houses. Thus, if they decrease the prices of houses, it will be because that action will enable them to sell a greater number of houses. Choose A, B, C or D as your solution.

Reasoning: 1. Identify the pattern of reasoning in the given argument: If X remains constant, Y can be expected to be Z. If Y is not Z, it's because X has changed.
2. Analyze each option to find a similar pattern of reasoning:
   A: If X (workers' wages) remains constant, Y (prices) can be expected to be Z (the same as last year). If Y is not Z, it's because X has changed (increased wages).
   B: This option discusses a cause and effect relationship (employing good journalists leads to good papers) but does not follow the same pattern of reasoning.
   C: This option also discusses a cause and effect relationship (employing more store detectives reduces shoplifting) but does not follow the same pattern of reasoning.
   D: This option has a different pattern of reasoning: If X (house-building costs) remains constant, Y (house prices) cannot be expected to increase. If Y decreases, it's because of a different reason (selling more houses).
3. Option A follows the same pattern of reasoning as the given argument.
4. Choose A as the solution.

Reasoning Chains: [{'step': 1, 'thought': "Identify the pattern of reasoning in the given argument: If X remains constant, Y can be expected to be Z. If Y is not Z, it's because X has changed."}, {'step': 2, 'thought': "Analyze each option to find a similar pattern of reasoning:\n   A: If X (workers' wages) remains constant, Y (prices) can be expected to be Z (the same as last year). If Y is not Z, it's because X has changed (increased wages).\n   B: This option discusses a cause and effect relationship (employing good journalists leads to good papers) but does not follow the same pattern of reasoning.\n   C: This option also discusses a cause and effect relationship (employing more store detectives reduces shoplifting) but does not follow the same pattern of reasoning.\n   D: This option has a different pattern of reasoning: If X (house-building costs) remains constant, Y (house prices) cannot be expected to increase. If Y decreases, it's because of a different reason (selling more houses)."}, {'step': 3, 'thought': 'Option A follows the same pattern of reasoning as the given argument.'}, {'step': 4, 'thought': 'Choose A as the solution.'}]
----
Example 9:

Instruction: If z = arctan(e^{1 + (1 + x)^2}), what's the derivative of $\frac{\partial z}{\partial x}$ at x = 0.
Relevant Theorem: The Derivative Chain Rule is a fundamental rule in calculus used to find the derivative of a composite function. A composite function is a function that is formed by combining two or more functions, where the output of one function becomes the input of another function.

The Chain Rule states that if you have a composite function, say h(x) = f(g(x)), then the derivative of h(x) with respect to x, denoted as h'(x) or dh/dx, can be found by taking the derivative of the outer function f with respect to the inner function g(x), and then multiplying it by the derivative of the inner function g(x) with respect to x.

Mathematically, the Chain Rule can be expressed as:

h'(x) = f'(g(x)) * g'(x)

or

dh/dx = (df/dg) * (dg/dx)

The Chain Rule is particularly useful when dealing with complex functions that involve multiple layers of functions, as it allows us to break down the problem into simpler parts and find the derivative step by step. None

Reasoning: 1. Identify the given function: z = arctan(e^{1 + (1 + x)^2})
2. Recognize that this is a composite function, with an outer function (arctan) and an inner function (e^{1 + (1 + x)^2}).
3. Apply the Chain Rule to find the derivative of z with respect to x: dz/dx = (d(arctan)/d(e^{1 + (1 + x)^2})) * (d(e^{1 + (1 + x)^2})/dx)
4. Find the derivative of the outer function (arctan) with respect to the inner function (e^{1 + (1 + x)^2}): d(arctan)/d(e^{1 + (1 + x)^2}) = 1/(1 + (e^{1 + (1 + x)^2})^2)
5. Find the derivative of the inner function (e^{1 + (1 + x)^2}) with respect to x: d(e^{1 + (1 + x)^2})/dx = e^{1 + (1 + x)^2} * 2(1 + x)
6. Combine the derivatives from steps 4 and 5 using the Chain Rule: dz/dx = (1/(1 + (e^{1 + (1 + x)^2})^2)) * (e^{1 + (1 + x)^2} * 2(1 + x))
7. Simplify the expression: dz/dx = (2(1 + x) * e^{1 + (1 + x)^2}) / (1 + e^{2(1 + (1 + x)^2)})
8. Evaluate the derivative at x = 0: dz/dx = (2(1 + 0) * e^{1 + (1 + 0)^2}) / (1 + e^{2(1 + (1 + 0)^2)})
9. Simplify and calculate the numerical value: dz/dx = (2 * e^2) / (1 + e^4) ≈ 0.3017
10. The derivative of z with respect to x at x = 0 is approximately 0.3017.

Reasoning Chains: [{'step': 1, 'thought': 'Identify the given function: z = arctan(e^{1 + (1 + x)^2})'}, {'step': 2, 'thought': 'Recognize that this is a composite function, with an outer function (arctan) and an inner function (e^{1 + (1 + x)^2}).'}, {'step': 3, 'thought': 'Apply the Chain Rule to find the derivative of z with respect to x: dz/dx = (d(arctan)/d(e^{1 + (1 + x)^2})) * (d(e^{1 + (1 + x)^2})/dx)'}, {'step': 4, 'thought': 'Find the derivative of the outer function (arctan) with respect to the inner function (e^{1 + (1 + x)^2}): d(arctan)/d(e^{1 + (1 + x)^2}) = 1/(1 + (e^{1 + (1 + x)^2})^2)'}, {'step': 5, 'thought': 'Find the derivative of the inner function (e^{1 + (1 + x)^2}) with respect to x: d(e^{1 + (1 + x)^2})/dx = e^{1 + (1 + x)^2} * 2(1 + x)'}, {'step': 6, 'thought': 'Combine the derivatives from steps 4 and 5 using the Chain Rule: dz/dx = (1/(1 + (e^{1 + (1 + x)^2})^2)) * (e^{1 + (1 + x)^2} * 2(1 + x))'}, {'step': 7, 'thought': 'Simplify the expression: dz/dx = (2(1 + x) * e^{1 + (1 + x)^2}) / (1 + e^{2(1 + (1 + x)^2)})'}, {'step': 8, 'thought': 'Evaluate the derivative at x = 0: dz/dx = (2(1 + 0) * e^{1 + (1 + 0)^2}) / (1 + e^{2(1 + (1 + 0)^2)})'}, {'step': 9, 'thought': 'Simplify and calculate the numerical value: dz/dx = (2 * e^2) / (1 + e^4) ≈ 0.3017'}, {'step': 10, 'thought': 'The derivative of z with respect to x at x = 0 is approximately 0.3017.'}]
----
Example 10:

Instruction: What is the largest number $c$ such that $2x^2+5x+c=0$ has at least one real solution? Express your answer as a common fraction. None

Reasoning: 1. I need to find the largest value of $c$ for which the quadratic equation $2x^2+5x+c=0$ has at least one real solution.
2. To do this, I'll consider the discriminant of the quadratic equation, which is $b^2-4ac$.
3. The equation has at least one real solution if and only if the discriminant is non-negative, so I want to maximize $c$ subject to the constraint that $b^2-4ac\geq 0$.
4. In this case, $a=2$, $b=5$, and $c$ is the unknown, so I have $5^2-4(2)c\geq 0$.
5. Simplifying, I get $25-8c\geq 0$.
6. Adding $8c$ to both sides, I get $25\geq 8c$.
7. Dividing both sides by $8$, I get $\frac{25}{8}\geq c$.
8. This means that $c$ can be any number less than or equal to $\frac{25}{8}$, but the largest possible value of $c$ is $\frac{25}{8}$ itself.

Reasoning Chains: [{'step': 1, 'thought': 'I need to find the largest value of $c$ for which the quadratic equation $2x^2+5x+c=0$ has at least one real solution.'}, {'step': 2, 'thought': "To do this, I'll consider the discriminant of the quadratic equation, which is $b^2-4ac$."}, {'step': 3, 'thought': 'The equation has at least one real solution if and only if the discriminant is non-negative, so I want to maximize $c$ subject to the constraint that $b^2-4ac\\geq 0$.'}, {'step': 4, 'thought': 'In this case, $a=2$, $b=5$, and $c$ is the unknown, so I have $5^2-4(2)c\\geq 0$.'}, {'step': 5, 'thought': 'Simplifying, I get $25-8c\\geq 0$.'}, {'step': 6, 'thought': 'Adding $8c$ to both sides, I get $25\\geq 8c$.'}, {'step': 7, 'thought': 'Dividing both sides by $8$, I get $\\frac{25}{8}\\geq c$.'}, {'step': 8, 'thought': 'This means that $c$ can be any number less than or equal to $\\frac{25}{8}$, but the largest possible value of $c$ is $\\frac{25}{8}$ itself.'}]    
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
            st.warning("Chat history is too long. Truncating to fit model input.")
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

tasks = ["Reasoning (No conclusion)", "Reasoning Chains", "Final Answer"]

if prompt := st.chat_input(q):

    search_result = search(prompt, st.session_state.messages)

    with st.status("Search Results:"):
        st.write(search_result)

    if search_result:
        search_result = str(search_result)
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
