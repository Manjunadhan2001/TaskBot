
from dotenv import load_dotenv
load_dotenv()

import os
import uuid
import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent


# ---------------------------------
# 1. DATABASE CONNECTION
# ---------------------------------

DB_PATH = os.getenv("DB_PATH", "my_dailytasks.db")

# Create the parent directory if it doesn't exist
os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)

db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")


# ---------------------------------
# 2. CREATE TASKS TABLE
# ---------------------------------

db.run("""
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT CHECK (
        status IN ('Pending', 'InProgress', 'Completed')
    ) DEFAULT 'Pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")


# ---------------------------------
# 3. LLM CONFIGURATION
# ---------------------------------

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# ---------------------------------
# 4. SQL TOOLKIT
# ---------------------------------

toolkit = SQLDatabaseToolkit(
    db=db,
    llm=model
)

tools = toolkit.get_tools()


# ---------------------------------
# 5. SYSTEM PROMPT
# ---------------------------------

system_prompt = """
You are TaskBot, an AI task management assistant.

You interact with a SQLite database containing
a table named 'tasks'.

TABLE SCHEMA:
- id: Integer primary key
- title: Task title
- description: Task description
- status: Pending, InProgress, Completed
- created_at: Task creation timestamp

TASK RULES:

1. Only work with the tasks table.
2. For SELECT queries, return a maximum of 10 records.
3. Always order task listings by created_at DESC.
4. After INSERT, UPDATE or DELETE, execute a
   SELECT query to verify the operation.
5. Never claim an operation succeeded without
   verifying it in the database.
6. Display task lists in a clear Markdown table.
7. Use only these status values:
   Pending, InProgress, Completed.
8. Ask the user for clarification if the task
   title or ID is ambiguous.
9. Before deleting a task, ask the user for
   explicit confirmation.
10. Do not invent task records.

CRUD OPERATIONS:

CREATE:
INSERT INTO tasks(title, description, status)

READ:
SELECT * FROM tasks
ORDER BY created_at DESC
LIMIT 10

UPDATE:
UPDATE tasks
SET status = ?
WHERE id = ?

DELETE:
DELETE FROM tasks
WHERE id = ?

Be friendly and provide clear responses.
"""


# ---------------------------------
# 6. CREATE LANGCHAIN AGENT
# ---------------------------------

@st.cache_resource
def get_agent():

    agent = create_agent(
        model=model,
        tools=tools,
        checkpointer=InMemorySaver(),
        system_prompt=system_prompt
    )

    return agent


agent = get_agent()


# ---------------------------------
# 7. STREAMLIT USER INTERFACE
# ---------------------------------

st.set_page_config(
    page_title="TaskBot",
    page_icon="📋",
    layout="centered"
)

st.title("TaskBot - AI Task Manager 📋")

st.caption(
    "Manage your daily tasks using natural language."
)


# ---------------------------------
# 8. SESSION STATE
# ---------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Unique conversation ID for each user session
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# ---------------------------------
# 9. DISPLAY CHAT HISTORY
# ---------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------
# 10. USER INPUT
# ---------------------------------

prompt = st.chat_input(
    "Enter your task or ask about your tasks..."
)


# ---------------------------------
# 11. PROCESS USER MESSAGE
# ---------------------------------

if prompt:

    # Display user message
    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Generate AI response
    with st.chat_message("assistant"):

        with st.spinner("TaskBot is processing..."):

            try:

                res = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    },
                    {
                        "configurable": {
                            "thread_id": st.session_state.thread_id
                        }
                    }
                )

                result = res["messages"][-1].content

                if not isinstance(result, str):
                    result = str(result)

                st.markdown(result)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result
                })

            except Exception as e:

                st.error(f"Something went wrong: {e}")
