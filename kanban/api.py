import json
import re
from typing import Annotated, Optional
from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
from config import embeddings
import markdown
import os
import tempfile
import uuid
from fastapi import APIRouter, Body
from fastapi.responses import HTMLResponse
from typing import Annotated
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from kanban.classes import TaskList, build_context, build_document, get_access_token, get_buckets, get_plans, get_task_details, get_tasks, get_teams, handle_date_query, preprocess_query, summarize_batches, synthesize_summary
from labels import Category
from kanban.classes import app 
from kanban.plot import filename
from langgraph.graph import StateGraph, MessagesState
from fastapi.responses import JSONResponse
import uuid

load_dotenv(override=True)

router = APIRouter()


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
async def save_kanban_info(task_list: Annotated[TaskList, Body(...)]):
    print('📥 Received request to save kanban info')

    try:
        db = Chroma(persist_directory="vector_kanban_db", embedding_function=embeddings)
        db.delete_collection()  # Consider specifying collection name if needed
        db = Chroma(persist_directory="vector_kanban_db", embedding_function=embeddings)

        docs = [build_document(task) for task in task_list.tasks]

        # Optional: Chunk long documents
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=20)
        chunked_docs = text_splitter.split_documents(docs)

        db.add_documents(chunked_docs)

        print(f"✅ Added {len(chunked_docs)} documents to vector store.")
        return HTMLResponse(content="Kanban information saved successfully.")


    except Exception as e:
        print(f"❌ Error during kanban save: {e}")
        return HTMLResponse(content=f"Error saving kanban information: {e}")



@router.get("/kanban_query/")
async def kanban_query(query: Annotated[str, Query(description="Query to search in the kanban")]):
    print('🔍 Entered /kanban_query/ endpoint')
    print(f"Received query: {query}")

    try:
        query_info = preprocess_query(query)
        query_str = query_info['cleaned_query']
        intent = query_info.get('intent', 'general')

        db = Chroma(persist_directory="vector_kanban_db", embedding_function=embeddings)
        docs_with_scores = db.similarity_search_with_score(query_str, k = 80)

        # # Retrieve documents with scores
        # docs_with_scores = retriever.invoke(query_str, return_score=True)

        # Filter by similarity threshold
        threshold = 0.2
        filtered_docs = [(doc, score) for doc, score in docs_with_scores if score >= threshold]

        # Sort and select top N
        top_n = 50
        top_docs = [doc for doc, _ in sorted(filtered_docs, key=lambda x: x[1], reverse=True)[:top_n]]
        print(top_docs)
        if not top_docs:
            return JSONResponse(content={"html": "No relevant documents found."})

        batch_summaries = summarize_batches(top_docs, query_str, intent)
        final_summary = synthesize_summary(batch_summaries, query_str)

        response_html = markdown.markdown(final_summary)
        return JSONResponse(content={'html': response_html})

    except Exception as e:
        print(f"❌ Error during kanban query: {e}")
        return JSONResponse(content={"message": f"Error during kanban query: {e}"})


@router.get("/plot")
def get_plot(filename_: str = Query(...)):
    global filename
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename_)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    filename=None
    return FileResponse(file_path, media_type="image/png")


##############################################################


@router.get("/kanban_query_filtered/")
async def kanban_query_filtered(
    bencher: Annotated[Optional[str], Query(description="Filter by assignee name")] = None,
    incomplete_only: Annotated[bool, Query(description="Only include tasks with percentComplete < 100")] = False,
):
    print("🔍 Entered /kanban_query_filtered/ endpoint")

    # Load vector DB
    db = Chroma(persist_directory="vector_kanban_db", embedding_function=embeddings)

    # Build metadata filter
    filter_conditions = []
    filter_descriptions = []

    if bencher:
        filter_conditions.append({"person": {"$eq": bencher}})
        filter_descriptions.append(f"assigned to **{bencher}**")

    if incomplete_only:
        filter_conditions.append({"percent_complete": {"$lt": 100}})
        filter_descriptions.append("with less than 100% completion")

    metadata_filter = {"$and": filter_conditions} if filter_conditions else {}
    filter_summary = (
        "Showing tasks " + " and ".join(filter_descriptions) + "."
        if filter_descriptions else "Showing all tasks."
    )

    try:
        print("📦 Metadata filter:", metadata_filter)
        filtered_docs = db.get(where=metadata_filter) if metadata_filter else db.get()
    except Exception as e:
        print("❌ Error during metadata filtering:", e)
        return JSONResponse(content={"message": f"Metadata filtering failed: {e}"}, status_code=500)

    if not filtered_docs or not filtered_docs.get("documents"):
        return JSONResponse(content={"message": "No matching tasks found."}, status_code=404)

    print("✅ Returning filtered docs:", filtered_docs)

    final_result = app.invoke(
        {"messages": filter_summary, "context":filtered_docs},
        config={"configurable": {"thread_id": str(uuid.uuid4())}}
    )
    response = final_result["messages"][-1].content if final_result else "No results found."
    response_markdown = markdown.markdown(response)

    return JSONResponse(content={'html': response_markdown, 'filename': filename})
