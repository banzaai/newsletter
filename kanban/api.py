import json
import os
import tempfile
import uuid
from langchain_chroma import Chroma
from dotenv import load_dotenv
from auth import verify_token
from config import embeddings
import markdown
from typing import Annotated
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .classes import TaskList, build_document, get_access_token, get_buckets, get_plans, get_task_details, get_tasks, get_teams
from fastapi import APIRouter, Depends, Query, Body, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from pathlib import Path
from .kanban_config import agent
from db import connection


load_dotenv(override=True)

router = APIRouter()


@router.get("/chart/{chart_id}")
def get_chart(chart_id: str):
    # Determine environment
    environment = os.getenv("ENVIRONMENT")

    # Set chart path based on environment
    if environment == "local":
        chart_dir = os.getenv("DIR_CHART", tempfile.gettempdir())
    else:
        chart_dir = tempfile.gettempdir()  # Render allows writing to /tmp

    chart_path = os.path.join(chart_dir, f"{chart_id}.html")
    print(f"Fetching chart from: {chart_path}")

    # Check if file exists
    if not os.path.exists(chart_path):
        return {"error": "Chart not found"}

    return FileResponse(chart_path, media_type="text/html")


@router.get("/me/")
def get_user(email: str = Depends(verify_token)):
    return {"email": email}

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
        buckets = get_buckets(token, plan_id)
        bucket_map = {b['id']: b['name'] for b in buckets}
        enriched = []
        for task in tasks:
            bucket_name = bucket_map.get(task['bucketId'], "Unknown")
            details = get_task_details(token, task['id'])
            task['bucketName'] = bucket_name
            task['details'] = details 
            enriched.append(task)
        return enriched
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/save_kanban_info/", response_class=HTMLResponse)
async def save_kanban_info(task_list: Annotated[TaskList, Body(...)], email: str = Depends(verify_token)):
    print('📥 Received request to save kanban info')

    try:
        docs = [build_document(task) for task in task_list.tasks]

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunked_docs = text_splitter.split_documents(docs)

        db = connection.supabase  # SupabaseVectorStore or Chroma

        # ✅ Add documents to vector store (Chroma or Supabase)
        db.add_documents(chunked_docs)

        # 📝 Save to local file for inspection/debugging
        output_path = Path("kanban_docs.jsonl")
        with output_path.open("w", encoding="utf-8") as f:
            for doc in chunked_docs:
                json.dump({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }, f)
                f.write("\n")

        print(f"✅ Added {len(chunked_docs)} documents to vector store and saved to {output_path}")
        return HTMLResponse(content="Kanban information saved successfully.")

    except Exception as e:
        print(f"❌ Error during kanban save: {e}")
        return HTMLResponse(content=f"Error saving kanban information: {e}")


@router.get("/kanban_query/")
async def kanban_query(query: Annotated[str, Query(description="Your question")]):
    try:
        answer = await agent.arun(query)

        html = markdown.markdown(
            answer,
            extensions=[
                "extra",        # Enables tables, definition lists, footnotes
                "nl2br",        # Converts newlines to <br>
                "sane_lists",   # Prevents unexpected list breaks
            ],
            output_format="html5"
        )
        print("--- RAW ANSWER ---")
        print(answer)
        print("--- HTML CONVERTED ---")
        print(html)

        return JSONResponse(content={"html": html})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})


