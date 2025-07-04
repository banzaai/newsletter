from typing import Annotated, List
from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from langchain_chroma import Chroma
import msal
import requests
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from config import embeddings

load_dotenv()

router = APIRouter()

CLIENT_ID = os.getenv("CLIENT_ID")
TENANT_ID = os.getenv("TENANT_ID")
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPES = ["Group.Read.All", "Tasks.Read"]
CACHE_FILE = os.getenv("CACHE_FILE")

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
    
from pydantic import BaseModel
from typing import Optional, Dict, Any

class TaskDetails(BaseModel):
    description: Optional[str]
    checklist: Optional[Dict[str, Any]]

class KanbanItem(BaseModel):
    title: str
    bucketName: str
    details: TaskDetails
    # Add other fields as needed


@router.post("/save_kanban_info/", response_class=HTMLResponse)
async def save_kanban_info(response: List[KanbanItem]):
    """
    Endpoint to save kanban information.
    This endpoint is used to save the user's kanban information.
    """
    db = Chroma(persist_directory="vector_kanban_db", embedding_function=embeddings)

    try:
        text = response[0].tasks.replace("'", '"')
        db.add_texts([text])
        return HTMLResponse(content="Kanban information saved successfully!")
    except Exception as e:
        return HTMLResponse(content=f"Error saving kanban information: {e}")
    

@router.get("/kanban_query/")
async def kanban_query(event: str = Query(..., description="The user's query to search the Kanban")):
    """
    Endpoint to query kanban information.
    This endpoint retrieves Kanban information based on the user's query.
    """
    db = Chroma(persist_directory="vector_kanban_db", embedding_function=embeddings)

    result = db.similarity_search(event)

    response = [
        {
            "content": doc.page_content
        } for doc in result
    ]
    print(f"Query result: {response}")
    return {"events": response[0]['content']} if response else {"message": "No events found."}