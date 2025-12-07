from dotenv import load_dotenv
import os

#Import tools for the AI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from todoist_api_python.api import TodoistAPI


load_dotenv()

# load the API keys
todoist_api_key = os.getenv("TODOIST_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

todoist = TodoistAPI(todoist_api_key)

# tool is a python function in the langchain jargon that directs the llm to run
# an action, in this case add_task(). A docstring explaining the function is
# compulsory as it gives the llm context
# Desc(Description) is added by default by the llm id you do not give it a default value
@tool
def add_task(task, desc=None):
    """Add a new task to the user's task list. Use this when the user wants to
    add or create a task
    """
    todoist.add_task(content=task,
                     description=desc)

@tool
def show_tasks():
    """Show all tasks from Todoist. Use this tool when the user wants to see their
    tasks"""
    results_paginator = todoist.get_tasks()
    tasks = []
    for task_list in results_paginator:
        for task in task_list:
            tasks.append(task.content)

    return tasks

# The tools have to be added to a list
tools = [add_task, show_tasks]

llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    google_api_key=gemini_api_key,
    temperature=0.3
)
# System prompt to give the llm perspective as to its purpose
system_prompt = """You are a helpful assistant. You will help the user manage tasks.
if the user asks to show all tasks: for example: "show all tasks", print out all the 
tasks to the user. Print them in a  bullet list format.
"""


prompt = ChatPromptTemplate([
    ("system", system_prompt),
    MessagesPlaceholder("history"),
    ("user", "{input}"),
    MessagesPlaceholder("agent_scratchpad")
])


agent =  create_openai_tools_agent(llm, tools, prompt)
agent_execute = AgentExecutor(agent=agent, tools=tools, verbose=False)

# Chain is a variable to carry the code for langchain, it is where it got its name
# as commands pass to it look like a chain, separated with |, following the chain
# The prompt will be passed to llm which will then be parsed by StrOutParser(), all
# stored in chain
# chain = prompt | llm | StrOutputParser()
# To run the llm, .invoke takes a dictionary as an argument with input as key
# response = chain.invoke({"input": user_prompt})

# To keep track of previous conversation, the history of the conversation is
# appended to the list
history = []
while True:
    user_prompt = input("You: ")
    response = agent_execute.invoke({"input": user_prompt, "history":history})
    print(response["output"])
    history.append(HumanMessage(content=user_prompt))
    history.append(AIMessage(content=response["output"]))

