import json
import re
from typing import Any, Dict, List, Optional
import msal
import requests
import os
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts.chat import MessagesPlaceholder, ChatPromptTemplate
from config import model
from typing import List
from langchain.agents import initialize_agent, AgentType
import os
import traceback
from kanban.plot import generate_plot



class MyState(MessagesState):
    context: str = Field(default="")

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
f"""
    You are a highly intelligent, context-aware assistant designed to help with task management queries.
    Always respond in full sentences, using markdown formatting with headings and bullet points where appropriate.
    
    make a cohesive and complete answer out of the context as if you answered the question with the complete information just once.
    ---

    ## 🎯 Your Objective

    Your job is to:
    1. **Understand the user query** in the context of the full message history.
    2. **Analyze the task context** provided below to extract relevant and accurate information.
    3. **Respond clearly and completely**, using markdown formatting.
    4. **Use tools** (e.g., for plots) only when explicitly requested.

    ---

    ## 📌 Critical Rules (Must Follow)

    ### ✅ Task Completion Logic
    - A task is considered **completed** **only** if the `Completed` field is **exactly 100%** and not less than that.
    - Any task with `Completed` at 50% or 0% is considered **incomplete**, **ongoing**, or **not finished**.
    - Do **not** treat tasks with 99% or less as completed under any circumstance.

    ### ✅ Bench Status Logic
    - A person is considered **on the bench** if their `bench status` field is `"On the bench"`.
    - This status is **independent of task completion** or any other field.
    - When asked "Who is on the bench?", list **all** individuals with `bench status = "On the bench"`.

    ### ❓ Ambiguity Handling
    - If the query is **ambiguous**, ask a clarifying question before proceeding.

    ### 🚫 Empty Context Handling
    - If the context is empty, respond with:
    > "Please specify the query in a different way or provide more context."

    ---

    ## 🎨 Output Style Options

    **If the user specifies a style, use it. Otherwise, choose the emoji with structured text**

    Choose between these output styles and decide which one is more appropriate:
    - `emoji`: Emoji-enhanced bullet points
    - `table`: Markdown table format
    - `collapsible`: HTML collapsible sections
    - `quote`: Summary in quote blocks

    - Be concise but complete.
    - Ensure clarity and structure in your response and divide the different tasks with the use of a horizontal line to separate content.

    ---

    ## 📄 Task Context
    Each task is summarized using the following format:

    ```
    ### Task: 
    - person: 
    - bench_status:
    - Start: | Due:
    - Priority: | Completed: %
    - Labels: 
    - Description:
    - Checklist:

    ```

    ### Field Descriptions:
    - **Task**: The title or name of the task.
    - **Person**: The individual assigned to or who completed the task.
    - **Bench Status**: Indicates whether the person is "On the bench" or not.
    - **Start / Due**: Start and end dates of the task. If missing, shown as "N/A".
    - **Priority**: A number from 1 (low) to 5 (high) indicating task urgency.
    - **Completed**: Task completion percentage. Only 100% means the task is finished.
    - **Labels**: Categories or tags describing the task (e.g., certification, interview).
    - **Description**: A detailed explanation of the task’s content or purpose.
    - **Checklist**: A list of subtasks or steps related to the main task.

    
    The context is : {{context}}
    ---

    ## 📊 Plotting Instructions

    When calling the `generate_plot` tool:
    - Pass a JSON string with at least two keys: `"x"` and `"y"`.
    - `"x"` should be a list of labels (e.g., names of people).
    - `"y"` should be a list of corresponding numeric values (e.g., task counts).
    - Example: `"x": ["Alice", "Bob"], "y": [3, 5]`

    You may optionally include a `"config"` key to customize the plot:
    - `"plot_type"`: `"line"` (default), `"bar"`, `"scatter"`, `"hist"`, or `"pie"`
    - `"title"`: Title of the plot
    - `"xlabel"` / `"ylabel"`: Axis labels
    - `"color"`: Color of the plot elements
    - `"grid"`: `true` or `false`
    - `"figsize"`: `[width, height]` in inches

    ### 🧠 Plot Type Selection Logic

    If the user has not specified a plot type:
    - Ask: “I can generate a plot to visualize this data. Which type would you prefer? Options: **line**, **bar**, **scatter**, **histogram**, or **pie**.”
    - Wait for the user's response before calling the `generate_plot` tool.
    - Once the user responds, include their choice in the `config.plot_type` field.
    - If the user says “you decide,” infer the best plot type based on the data structure.
"""

        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)


class ChecklistItem(BaseModel):
    title: str

class TaskDetails(BaseModel):
    description: Optional[str] = None
    checklist: Optional[Dict[str, ChecklistItem]] = {}

class Task(BaseModel):
    id: str
    bucketId: Optional[str] = None
    title: str
    bucketName: Optional[str] = None
    percentComplete: Optional[int] = 0
    priority: Optional[int] = None
    startDateTime: Optional[str] = None
    dueDateTime: Optional[str] = None
    details: Optional[TaskDetails] = TaskDetails()
    appliedCategories: Optional[Dict[str, Any]] = {}
    assignments: Optional[Dict[str, Any]] = {}

class TaskList(BaseModel):
    tasks: List[Task]

CLIENT_ID = os.getenv("CLIENT_ID")
TENANT_ID = os.getenv("TENANT_ID")
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPES = ["https://graph.microsoft.com/.default"]
# SCOPES = ["Group.Read.All"]  # or "Tasks.Read.All" if needed 
CACHE_FILE = os.getenv("CACHE_FILE")
TOKEN_CACHE = os.getenv("TOKEN_CACHE")
CLIENT_CREDENTIALS= os.getenv("CLIENT_CREDENTIALS")

filename_real = None


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
    if 'filename' in response[-1]:
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
        app_msal = msal.ConfidentialClientApplication(
            CLIENT_ID,
            authority=AUTHORITY,
            client_credential=CLIENT_CREDENTIALS,
            token_cache=cache
        )

        result = app_msal.acquire_token_for_client(SCOPES)
        if "access_token" in result:
            save_cache(cache)
            return result["access_token"]
        else:
            error_msg = result.get("error_description", "Unknown error")
            print("Token acquisition failed:", error_msg)
            raise RuntimeError(f"Token acquisition failed: {error_msg}")

    except Exception as e:
        print("Authentication error:", traceback.format_exc())
        raise RuntimeError(f"Authentication error: {e}")



def get_teams(token):
    try:
        # Extracted from the URL you provided
        team_id = "4a81625f-fb9f-4d14-a311-80024666fa36"
        url = f"https://graph.microsoft.com/v1.0/groups/{team_id}"
        headers = {'Authorization': f'Bearer {token}'}
        
        r = requests.get(url, headers=headers)
        print("Graph response status:", r.status_code)
        print("Graph response body:", r.text)
        r.raise_for_status()
        
        # Return a list with a single team object to match original return type
        return [r.json()]
    except Exception as e:
        print("Failed to fetch team:", e)
        raise RuntimeError(f"Failed to fetch team: {e}")


def get_plans(token, group_id):
    try:
        # Hardcoded known plan ID
        target_plan_id = "symbPdcQ_kCiuzKvIftaBZYAC2EV"
        url = f"https://graph.microsoft.com/v1.0/planner/plans/{target_plan_id}"
        headers = {'Authorization': f'Bearer {token}'}
        
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        
        plan = r.json()
        return [plan]  # Return as a list to match original structure
    except Exception as e:
        raise RuntimeError(f"Failed to fetch hardcoded plan: {e}")

def get_tasks(token, plan_id):
    try:
        target_plan_id = "symbPdcQ_kCiuzKvIftaBZYAC2EV"
        url = f"https://graph.microsoft.com/v1.0/planner/plans/{target_plan_id}/tasks"
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

def get_buckets(token, plan_id):
    try:
        url = f"https://graph.microsoft.com/v1.0/planner/plans/{plan_id}/buckets"
        headers = {'Authorization': f'Bearer {token}'}
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        return r.json().get("value", [])
    except Exception as e:
        raise RuntimeError(f"Failed to fetch buckets: {e}")



def preprocess_query(query: str) -> Dict[str, str]:
    """
    Enhance the query by normalizing, removing punctuation,
    mapping synonyms to canonical terms, and detecting intent and key attributes.
    Returns a dictionary with cleaned query and detected metadata.
    """
    query = query.lower()
    query = re.sub(r"[^\w\s]", "", query)
    query = query.strip()

    # Synonym mapping
    synonyms = {
        "not done": "incomplete",
        "unfinished": "incomplete",
        "not completed": "incomplete",
        "done": "complete",
        "finished": "complete",
        "idle": "on the bench",
        "available": "on the bench",
        "working": "not on the bench",
        "busy": "not on the bench",
        "assigned to": "for",
        "responsible for": "for",
        "certified in": "has certificate",
        "has certification in": "has certificate",
        "obtained certificate": "has certificate",
        "earned certificate": "has certificate",
        "got certificate": "has certificate",
        "passed": "has certificate",
        "when did": "date of",
        "when was": "date of",
        "should get": "due certificate",
        "needs to get": "due certificate",
        "must obtain": "due certificate",
        "planned certificate": "due certificate",
        "upcoming certificate": "due certificate"
    }

    for phrase, replacement in synonyms.items():
        query = query.replace(phrase, replacement)

    # Intent detection
    intent = "general"
    if "has certificate" in query or "due certificate" in query:
        intent = "certificate"
    elif "date of" in query or "last" in query or "month" in query or "week" in query:
        intent = "date"
    elif "on the bench" in query or "not on the bench" in query:
        intent = "bench_status"


    # Certificate name extraction (simple heuristic)
    certificate_name = None
    match = re.search(r"(green|azure|aws|scrum|python|data)\s+certificate", query)
    if match:
        certificate_name = match.group(0)

    return_str = json.dumps({
        "cleaned_query": query,
        "intent": intent,
        "certificate_name": certificate_name or "",
    })

    return return_str
