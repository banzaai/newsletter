from typing import Annotated, Dict, List, Optional
from fastapi import APIRouter, Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from langchain_chroma import Chroma
import msal
import requests
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import db
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config import model, embeddings
from typing import TypedDict, List
from langchain_core.messages import BaseMessage


load_dotenv()

router = APIRouter()

CLIENT_ID = os.getenv("CLIENT_ID")
TENANT_ID = os.getenv("TENANT_ID")
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPES = ["Group.Read.All", "Tasks.Read"]
CACHE_FILE = os.getenv("CACHE_FILE")


class MessagesState(TypedDict):
    messages: List[BaseMessage]
    query: str
    context: str


def call_model(state: MessagesState):
    system_prompt = (
        f"""context is:
        {state["context"]}
        The query is:
        {state["query"]}
        message history is:
        {state["messages"]}

        You are a highly intelligent, context-aware chatbot designed to assist with task management queries. Always respond in full sentences, providing complete and clear information based on the context provided.
        You have to be able to compare the differences between the bucket names and the task titles, and you have to be able to understand the subtasks in the checklist.
        You will be provided with:
        - A **context**: a list of tasks, each with the following fields:
        - `title`: the name of the task.
        - `Bucket`: Name of person assigned to the task, this means this person is on the bench regardless of their current task status.
        - `percentComplete`: a number from 0 to 100 indicating task completion.
        - `details`: a description of the task.
        - `checklist`: a dictionary of subtasks related to the main task, always indicate this is a subtask with reference to the main task.

        - A **query**: a natural language question asking for specific information from the context.

        - A **message history**: a list of previous messages exchanged between the user and the assistant.

        ---

        ### 🎯 Your Objective

        Your job is to:
        1. **Understand the current query** in the context of the full message history.
        2. **Analyze the task context** to extract only the most relevant and accurate information.
        3. **Respond** Always respond in full sentences, providing complete and clear information based on the context provided.
        ---

        ### 📌 Important Definitions
        - A person is considered **"on the bench"** if their name appears as a `Bucket` in the context.
        - The `title` is the name of the task the person is working on or has completed.
        - The `checklist` contains subtasks related to the main task.
        - The `details` field may contain additional descriptive information about the task.

        ---

        ### 🧾 Response Guidelines

        - **All information**: Respond with all relevant information from the context that answers the query, try to infer the user's intent.
        - **Be honest**: If the answer is not found in the context, respond with:
        > "The information you're looking for is not available in the provided context."

        ---

        ### 🧪 Examples

        **Example 1**  
        Query: *"What tasks are assigned to people on the bench?"*  
        Response:
        - John Doe: "Update Documentation" (50% complete)
        - Jane Smith: "Code Review" (100% complete)

        **Example 2**  
        Query: *"Which tasks are incomplete?"*  
        Response:
        - "Write Unit Tests" (assigned to Alice, 20% complete)
        - "Design Mockups" (assigned to Bob, 0% complete)

        **Example 3**  
        Query: *"Who is on the bench?"*  
        Response:
        - John Doe
        - Jane Smith

        ---

        ### ⚠️ Edge Cases

        - If the query is **ambiguous**, ask a clarifying question.
        - If the **context is empty**, respond with:
        > "No task data is available to answer your query."

        """
    )

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = model.invoke(messages)
    return {"messages": state["messages"] + [response]}


# Define the graph globally (outside the endpoint)
workflow = StateGraph(MessagesState)
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
        if os.path.exists(CACHE_FILE):
            cache.deserialize(open(CACHE_FILE, "r").read())
        return cache
    except Exception as e:
        raise RuntimeError(f"Error loading token cache: {e}")

def save_cache(cache):
    try:
        if cache.has_state_changed:
            with open(CACHE_FILE, "w") as f:
                f.write(cache.serialize())
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
    

# Global in-memory store
session_memory = {}

@router.get("/kanban_query/")
async def kanban_query(
    query: Annotated[str, Query(description="Query to search in the kanban")],
    thread_id: Annotated[str, Query(description="Session ID for memory")] = "default"
):
    print('I am here in the kanban query function')
    print(f"Received query: {query}")

    # Load context from vector DB
    db = Chroma(persist_directory="vector_kanban_db", embedding_function=embeddings)
    raw_docs = db.get()['documents']
    if not raw_docs:
        return JSONResponse(content={"message": "No kanban data available."})
    context = "\n".join(raw_docs)

    # Retrieve or initialize message history
    if thread_id not in session_memory:
        session_memory[thread_id] = []
    session_memory[thread_id].append(HumanMessage(content=query))

    # Build state
    state: MessagesState = {
        "messages": session_memory[thread_id],
        "query": query,
        "context": context
    }

    result = app.invoke(state, config={"configurable": {"thread_id": thread_id}})
    response = result["messages"][-1].content if result else "No results found."

    # Append assistant response to memory
    session_memory[thread_id].append(result["messages"][-1])

    return response

