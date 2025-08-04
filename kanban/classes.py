from typing import Any, Dict, List, Optional
import msal
import requests
import os
from pydantic import BaseModel
from typing import List
import os
import traceback
from typing import List
from langchain.schema import Document
from .kanban_config import AUTHORITY, CACHE_FILE, CLIENT_CREDENTIALS, CLIENT_ID, SCOPES, TOKEN_CACHE
from labels import Category



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
    createdDateTime: Optional[str] = None
    percentComplete: Optional[int] = 0
    priority: Optional[int] = None
    startDateTime: Optional[str] = None
    dueDateTime: Optional[str] = None
    details: Optional[TaskDetails] = TaskDetails()
    appliedCategories: Optional[Dict[str, Any]] = {}
    assignments: Optional[Dict[str, Any]] = {}

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



def build_document(task: Task) -> Document:
    bucket_name = task.bucketName
    on_bench = not ("[" in bucket_name and "]" in bucket_name)
    bench_status = "On the bench" if on_bench else "Not on the bench"

    description = task.details.description if task.details and task.details.description else "No description."

    #Detect certificate-related tasks
    cert_keywords = ["cert", "certificate", "exam", "diploma", "certificates", "certification", "certifications"]

    # Combine title and description
    combined_text = f"{task.title} {description}".lower()

    # Combine checklist items into a single string
    checklist_items = "\n".join(
        f"- {item.title}" for item in task.details.checklist.values()
    ) if task.details and task.details.checklist else ""

    # Combine all text to search
    full_text = f"{combined_text} {checklist_items}".lower()

    # Check for any certificate-related keywords
    has_certificate = any(kw in full_text for kw in cert_keywords)
        
    # ✅ Rich and semantic `page_content`
    summary = (
        f"### Task: {task.title}\n"
        f"**Summary**: The summary for this task is: {full_text}\n"
        f"**Assignee**: {bucket_name}\n"
        f"**Bench status**: {bench_status}\n"
        f"**Start date**: {task.startDateTime or 'N/A'}\n"
        f"**Due date**: {task.dueDateTime or 'N/A'}\n"
        f"**Priority**: {task.priority} | **Completed**: {task.percentComplete}%\n"
        f"**Labels**: {",".join([Category[key].value for key, value in task.appliedCategories.items() if value]) or 'None'}\n"
        f"**Certificate-related**: {'Yes it is certificate related ' if has_certificate else 'Not certificate related'}\n"
        f"\n**Description**:\n{description}\n"
        f"\n**Checklist**:\n{checklist_items}"
    )

    return Document(
        page_content=summary,
        metadata={
            "task_id": task.id,
            "title": task.title,
            "person": bucket_name,
            "bench_status": bench_status,
            "priority": task.priority,
            "percent_complete": task.percentComplete,
            "start": task.startDateTime or "N/A",
            "due": task.dueDateTime or "N/A",
            "has_description": bool(description.strip()),
            "description" : description or "",
            "has_checklist": bool(task.details and task.details.checklist),
            "checklist": checklist_items,
            "labels": ", ".join(Category[key].value for key, value in task.appliedCategories.items() if value),
            "has_certificate": has_certificate,
            "createdDateTime": task.createdDateTime or "N/A"
        }
    )

