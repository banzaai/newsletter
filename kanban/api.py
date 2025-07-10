from typing import Annotated, Dict, List, Optional
from fastapi import APIRouter, Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from langchain_chroma import Chroma
import msal
import requests
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import db
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts.chat import MessagesPlaceholder, ChatPromptTemplate
from config import model, embeddings
from typing import TypedDict, List
from langchain_core.messages import BaseMessage
import markdown
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
import os
import tempfile
import uuid


load_dotenv()

router = APIRouter()

CLIENT_ID = os.getenv("CLIENT_ID")
TENANT_ID = os.getenv("TENANT_ID")
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPES = ["Group.Read.All", "Tasks.Read"]
CACHE_FILE = os.getenv("CACHE_FILE")
TOKEN_CACHE = os.getenv("TOKEN_CACHE")

filename_real = None
@tool
def generate_plot(data_json: str, plot_type: str = "line") -> dict:
    """
    Generates a plot from JSON data and saves it to a temporary file.
    Returns a dictionary with the filename and a short description.
    """
    import json
    import matplotlib.pyplot as plt
    global filename_real
    data = json.loads(data_json)
    plt.figure()
    if plot_type == "line":
        plt.plot(data["x"], data["y"])
    elif plot_type == "bar":
        plt.bar(data["x"], data["y"])

    # Generate a unique filename
    filename = f"{uuid.uuid4().hex}.png"
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)
    plt.savefig(file_path)
    print(f'filename is {filename}')
    filename_real = filename 
    return {
        "text": "Plot saved to temporary file."
    }


class MyState(MessagesState):
    context: str = Field(default="")

prompt_template = ChatPromptTemplate.from_messages(
    [
        (

        "system",
        
            f"""

                You are a highly intelligent, context-aware chatbot designed to assist with task management queries.
                Always respond in full sentences, providing complete and clear information based on the context provided.
                You are aware of the full message history and the current query.

                ### 🎯 Your Objective

                Your job is to:
                1. **Understand the current query:** in the context of the full message history.
                2. **Analyze the task context: {{context}}** to extract relevant and accurate information.
                3. **Respond** Always respond in full sentences, providing complete and clear information based on the context provided.
                4. Responses should be in markdown format with appropriate headings and bullet points where necessary.
                5. **Use the tools available** to generate plots or visualizations if especifically asked for.
                ---

                IMPORTANT: 
                -All people who appear in the context are on the bench regardless of their tasks completion status.
                
                
                When calling the `generate_plot` tool:
                - Always pass a JSON string with **two keys**: `"x"` and `"y"`.
                - `"x"` should be a list of labels (e.g., names of people).
                - `"y"` should be a list of corresponding numeric values (e.g., task counts).
                - Example: `"x": ["Alice", "Bob"], "y": [3, 5]`

            
                                                   

                

                ---

                ### ⚠️ Edge Cases

                - If the query is **ambiguous**, ask a clarifying question.
                - If the **context is empty**, respond with:
                > "Please specify the query in a different way or provide more context."

                """
        ), MessagesPlaceholder(variable_name="messages")
    ]
)

agent_executor = initialize_agent(
    tools=[generate_plot],
    llm=model,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
)

# Define the graph globally (outside the endpoint)
workflow = StateGraph(MyState)

def call_model(state: MyState):
    prompt = prompt_template.invoke(state)
    response = agent_executor.run(prompt.to_string())
    print(response)
    if 'filename' in response[-1]:
        print('IN HEREEEEEEEEEEE')
        return {
            "messages": [
                {
                    "text": response["text"],
                    "filename": response["filename"]
                }
            ]
        }
    else:
        return {"messages": [response]}


workflow.add_node("model", call_model)
workflow.set_entry_point("model")  # or use add_edge(START, "model") if START is defined
app = workflow.compile(checkpointer=MemorySaver())


class ChecklistItem(BaseModel):
    title: str

class TaskDetails(BaseModel):
    description: Optional[str] = None
    checklist: Optional[Dict[str, ChecklistItem]] = {}

class Task(BaseModel):
    id: str
    title: str
    bucketName: str
    percentComplete: Optional[int] = 0
    priority: Optional[int] = None
    details: TaskDetails

class TaskList(BaseModel):
    tasks: List[Task]


def load_cache():
    try:
        cache = msal.SerializableTokenCache()
        if TOKEN_CACHE:
            # Deserialize directly from the string
            cache.deserialize(TOKEN_CACHE)
        elif os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                cache.deserialize(f.read())
        return cache
    except Exception as e:
        raise RuntimeError(f"Error loading token cache: {e}")


def save_cache(cache):
    try:
        if cache.has_state_changed:
            # Save to file
            with open(CACHE_FILE, "w") as f:
                f.write(cache.serialize())

            # Also print to console for manual copy-paste into Render env var
            print("🔐 Updated TOKEN_CACHE (copy this to Render env var):")
            print(cache.serialize())
    except Exception as e:
        raise RuntimeError(f"Error saving token cache: {e}")


def get_access_token():
    try:
        cache = load_cache()
        app_msal = msal.PublicClientApplication(CLIENT_ID, authority=AUTHORITY, token_cache=cache)
        accounts = app_msal.get_accounts()
        if accounts:
            result = app_msal.acquire_token_silent(SCOPES, account=accounts[0])
        else:
            flow = app_msal.initiate_device_flow(scopes=SCOPES)
            if "user_code" not in flow:
                raise ValueError("Failed to create device flow")
            print(flow["message"])
            result = app_msal.acquire_token_by_device_flow(flow)
        save_cache(cache)
        if "access_token" in result:
            return result["access_token"]
        else:
            return None
    except Exception as e:
        raise RuntimeError(f"Authentication error: {e}")

def get_teams(token):
    try:
        url = "https://graph.microsoft.com/v1.0/groups?$filter=resourceProvisioningOptions/Any(x:x eq 'Team')"
        headers = {'Authorization': f'Bearer {token}'}
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        return r.json().get('value', [])
    except Exception as e:
        raise RuntimeError(f"Failed to fetch teams: {e}")

def get_plans(token, group_id):
    try:
        url = f"https://graph.microsoft.com/v1.0/groups/{group_id}/planner/plans"
        headers = {'Authorization': f'Bearer {token}'}
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        return r.json().get('value', [])
    except Exception as e:
        raise RuntimeError(f"Failed to fetch plans: {e}")

def get_tasks(token, plan_id):
    try:
        url = f"https://graph.microsoft.com/v1.0/planner/plans/{plan_id}/tasks"
        headers = {'Authorization': f'Bearer {token}'}
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        return r.json().get('value', [])
    except Exception as e:
        raise RuntimeError(f"Failed to fetch tasks: {e}")

def get_task_details(token, task_id):
    try:
        url = f"https://graph.microsoft.com/v1.0/planner/tasks/{task_id}/details"
        headers = {'Authorization': f'Bearer {token}'}
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch task details: {e}")

def get_bucket_name(token, bucket_id):
    try:
        url = f"https://graph.microsoft.com/v1.0/planner/buckets/{bucket_id}"
        headers = {'Authorization': f'Bearer {token}'}
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        return r.json().get("name")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch bucket name: {e}")

@router.get("/teams/")
def api_get_teams():
    try:
        token = get_access_token()
        if not token:
            raise HTTPException(status_code=401, detail="Authentication failed")
        teams = get_teams(token)
        print("Teams fetched:")
        for team in teams:
            print(f"- {team['displayName']} (ID: {team['id']})")
        return teams
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/plans/{team_id}/")
def api_get_plans(team_id: str):
    try:
        token = get_access_token()
        return get_plans(token, team_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tasks/{plan_id}/")
def api_get_tasks(plan_id: str):
    try:
        token = get_access_token()
        tasks = get_tasks(token, plan_id)
        enriched = []
        for task in tasks:
            bucket = get_bucket_name(token, task['bucketId'])
            details = get_task_details(token, task['id'])
            task['bucketName'] = bucket
            task['details'] = details
            enriched.append(task)
        return enriched
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save_kanban_info/", response_class=HTMLResponse)
async def save_kanban_info(task_list: Annotated[TaskList, Body(...)]):
    """
    Endpoint to save kanban information.
    This endpoint is used to save the user's kanban information.
    """
    db = Chroma(persist_directory="vector_kanban_db", embedding_function=embeddings)

    try:
        texts = []
        for task in task_list.tasks:
            checklist_items = "\n".join(
                f"- {item.title}" for item in task.details.checklist.values()
            ) if task.details and task.details.checklist else "No checklist items."

            task_text = (
                f"Task ID: {task.id}\n"
                f"Title: {task.title}\n"
                f"Bucket: {task.bucketName}\n"
                f"Completion: {task.percentComplete}%\n"
                f"Priority: {task.priority if task.priority is not None else 'Not set'}\n"
                f"Description: {task.details.description if task.details and task.details.description else 'No description.'}\n"
                f"Checklist:\n{checklist_items}"
            )
            texts.append(task_text)
        
        db.add_texts(texts)

    except Exception as e:
        return HTMLResponse(content=f"Error saving kanban information: {e}")
    

@router.get("/kanban_query/")
async def kanban_query(
    query: Annotated[str, Query(description="Query to search in the kanban")],

):
    print('I am here in the kanban query function')
    print(f"Received query: {query}")
    global filename_real
    # Load context from vector DB
    db = Chroma(persist_directory="vector_kanban_db", embedding_function=embeddings)
    raw_docs = db.get()['documents']
    if not raw_docs:
        return JSONResponse(content={"message": "No kanban data available."})
    context = "\n".join(raw_docs)

    result = app.invoke({"messages": query, "context":context}, config={"configurable": {"thread_id": "1"}})
    response = result["messages"][-1].content if result else "No results found."
    response_markdown = markdown.markdown(response)
    return JSONResponse(content={'html': response_markdown,
                                 'filename': filename_real})

@router.get("/plot")
def get_plot(filename: str = Query(...)):
    global filename_real
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    filename_real=None
    return FileResponse(file_path, media_type="image/png")


