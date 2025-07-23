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
from kanban.classes import TaskList, get_access_token, get_buckets, get_plans, get_task_details, get_tasks, get_teams, preprocess_query
from labels import Category
from kanban.classes import app 
from kanban.plot import filename

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
        # Clear and reinitialize the vector store
        db = Chroma(persist_directory="vector_kanban_db", embedding_function=embeddings)
        db.delete_collection()
        db = Chroma(persist_directory="vector_kanban_db", embedding_function=embeddings)

        docs = []

        for task in task_list.tasks:
            bucket_name = task.bucketName
            on_bench = not (bucket_name.startswith("[") and bucket_name.endswith("]"))
            bench_status = "On the bench" if on_bench else "Not on the bench"

            description = task.details.description if task.details and task.details.description else "No description."
            checklist_items = "\n".join(
                f"- {item.title}" for item in task.details.checklist.values()
            ) if task.details and task.details.checklist else "No checklist items."

            labels = ", ".join(Category[key].value for key in task.appliedCategories.keys()) if task.appliedCategories else "No labels"

            summary = (
                f"### Task: {task.title}\n"
                f"- person: {bucket_name}\n"
                f"- bench_status: {bench_status}\n"
                f"- Start: {task.startDateTime or 'N/A'} | Due: {task.dueDateTime or 'N/A'}\n"
                f"- Priority: {task.priority} | Completed: {task.percentComplete}%\n"
                f"- Labels: {labels}\n"
                f"- Description: {description}\n"
                f"- Checklist:\n{checklist_items}\n"
            )

            doc = Document(
                page_content=summary,
                metadata={
                    "person": bucket_name,
                    "bench_status": bench_status,
                    "priority": task.priority,
                    "percent_complete": task.percentComplete,
                    "start": task.startDateTime,
                    "due": task.dueDateTime,
                    "has_description": bool(description.strip()),
                    "has_checklist": bool(task.details and task.details.checklist),
                    "labels": str(labels.split(", ")) if labels != "No labels" else ''
                }
            )

            docs.append(doc)

        # Optional: Chunk long documents if needed
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=100, chunk_overlap=50)
        chunked_docs = text_splitter.split_documents(docs)

        db.add_documents(chunked_docs)

        print(f"✅ Added {len(chunked_docs)} documents to vector store.")
        return HTMLResponse(content="Kanban information saved successfully.")

    except Exception as e:
        print(f"❌ Error during kanban save: {e}")
        return HTMLResponse(content=f"Error saving kanban information: {e}")

from fastapi.responses import JSONResponse
import uuid

@router.get("/kanban_query/")
async def kanban_query(
    query: Annotated[str, Query(description="Query to search in the kanban")],
):
    print('🔍 Entered /kanban_query/ endpoint')
    print(f"Received query: {query}")
    global filename
    query = preprocess_query(query)

    try:
        # Load vector store and retrieve relevant documents
        db = Chroma(persist_directory="vector_kanban_db", embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 100})
        docs = retriever.invoke(query)

        def build_context(doc_batch):
            blocks = []
            for doc in doc_batch:
                person = doc.metadata.get("person", "Unknown")
                bench_status = doc.metadata.get("bench_status", "Unknown")
                priority = doc.metadata.get("priority", "Unknown")
                percent_complete = doc.metadata.get("percent_complete", "Unknown")
                labels = ", ".join(doc.metadata.get("labels", []))
                summary = doc.page_content.strip()

                block = (
                    f"### Name person bench: {person}\n"
                    f"- Bench Status: {bench_status}\n"
                    f"- Priority: {priority}\n"
                    f"- Completion: {percent_complete}%\n"
                    f"- Labels: {labels}\n\n"
                    f"{summary}\n"
                    f"{'-'*40}"
                )
                blocks.append(block)
            return "\n\n".join(blocks)

        # Batch processing
        batch_size = 20
        batch_summaries = []

        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            context = build_context(batch)

            result = app.invoke(
                {"messages": query, "context": context},
                config={"configurable": {"thread_id": str(uuid.uuid4())}}  
            )
            summary = result["messages"][-1].content if result else "No summary"
            batch_summaries.append(summary)

        # Final synthesis with memory (optional)
        final_context = "\n\n".join(batch_summaries)
        final_result = app.invoke(
            {"messages": query, "context": final_context},
            config={"configurable": {"thread_id": "kanban-final"}}
        )
        response = final_result["messages"][-1].content if final_result else "No results found."
        response_markdown = markdown.markdown(response)

        return JSONResponse(content={'html': response_markdown, 'filename': filename})

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
