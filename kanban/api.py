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
from kanban.classes import TaskList, build_context, build_document, get_access_token, get_buckets, get_plans, get_task_details, get_tasks, get_teams, handle_date_query, preprocess_query, summarize_batches, synthesize_summary
from labels import Category
from kanban.classes import app 
from kanban.plot import filename
from langgraph.graph import StateGraph, MessagesState
from fastapi.responses import JSONResponse
import uuid
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
import markdown
from langchain.memory import ConversationSummaryMemory
from langchain.chains import RetrievalQA
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from config import model
from langchain.tools import tool
from datetime import datetime, timedelta
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationSummaryMemory
from langchain.llms import OpenAI  # or your LLM
from langchain.tools import tool
from datetime import datetime, timedelta
from fastapi import APIRouter, Body
from fastapi.responses import HTMLResponse
from typing import Annotated
import json
from pathlib import Path

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

        docs = [build_document(task) for task in task_list.tasks]

        # Optional: Chunk long documents
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000,
            chunk_overlap=20
        )
        chunked_docs = text_splitter.split_documents(docs)

        db.add_documents(chunked_docs)

        # 🔽 Write original or chunked docs to file
        output_path = Path("kanban_docs.jsonl")
        with output_path.open("w", encoding="utf-8") as f:
            for doc in chunked_docs:
                # Assumes `doc.page_content` and `doc.metadata` exist
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


vectordb = Chroma(persist_directory="vector_kanban_db", embedding_function=embeddings)
llm = model

retriever = vectordb.as_retriever(search_kwargs={"k": 200})  # increase k for broader fetch
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever)

@tool
def who_on_bench() -> str:
    """Returns a markdown list of people currently on the bench."""
    docs = retriever.get_relevant_documents("")
    people = sorted({doc.metadata.get("person") for doc in docs if doc.metadata.get("bench_status") == "On the bench"})
    if not people:
        return "No one is currently on the bench."
    lines = ["### People Currently On The Bench:\n"]
    for p in people:
        lines.append(f"- **{p}**")
    return "\n".join(lines)


@tool
def uncompleted_tasks_for_person(person_name: str) -> str:
    """Returns uncompleted tasks (<100%) for a person as a markdown table."""
    docs = retriever.get_relevant_documents("")
    tasks = [
        doc for doc in docs
        if doc.metadata.get("person") == person_name and doc.metadata.get("percent_complete", 100) < 100
    ]
    if not tasks:
        return f"No uncompleted tasks for **{person_name}**."

    lines = [
        f"### Uncompleted Tasks for **{person_name}**\n",
        "| Task | Percent Complete | Due Date | Priority |",
        "|-------|-----------------|----------|----------|"
    ]
    for doc in tasks:
        content = doc.page_content.split("\n")[0].strip()
        percent = doc.metadata.get("percent_complete", 0)
        due = doc.metadata.get("due", "N/A")
        priority = doc.metadata.get("priority", "N/A")
        lines.append(f"| {content} | {percent}% | {due} | {priority} |")

    return "\n".join(lines)


@tool
def completed_tasks_for_person(person_name: str) -> str:
    """Returns completed tasks (100%) for a person as a markdown table."""
    docs = retriever.get_relevant_documents("")
    tasks = [
        doc for doc in docs
        if doc.metadata.get("person") == person_name and doc.metadata.get("percent_complete", 0) == 100
    ]
    if not tasks:
        return f"No completed tasks for **{person_name}**."

    lines = [
        f"### Completed Tasks for **{person_name}**\n",
        "| Task | Due Date | Priority |",
        "|-------|----------|----------|"
    ]
    for doc in tasks:
        content = doc.page_content.split("\n")[0].strip()
        due = doc.metadata.get("due", "N/A")
        priority = doc.metadata.get("priority", "N/A")
        lines.append(f"| {content} | {due} | {priority} |")

    return "\n".join(lines)


@tool
def overdue_tasks_for_person(person_name: str) -> str:
    """Returns overdue, uncompleted tasks for a person in markdown list."""
    now_iso = datetime.now().isoformat()
    docs = retriever.get_relevant_documents("")
    overdue = [
        doc for doc in docs
        if doc.metadata.get("person") == person_name
           and doc.metadata.get("due")
           and doc.metadata["due"] < now_iso
           and doc.metadata.get("percent_complete", 100) < 100
    ]
    if not overdue:
        return f"No overdue tasks for **{person_name}**."
    
    lines = [f"### Overdue Tasks for **{person_name}**:\n"]
    for doc in overdue:
        title = doc.page_content.split("\n")[0].strip()
        due = doc.metadata.get("due")
        percent = doc.metadata.get("percent_complete", 0)
        lines.append(f"- **{title}** (Due: {due}, {percent}% complete)")
    return "\n".join(lines)


@tool
def high_priority_uncompleted_tasks() -> str:
    """Returns all high priority tasks that are not completed, as markdown table."""
    docs = retriever.get_relevant_documents("")
    tasks = [
        doc for doc in docs
        if doc.metadata.get("priority") in ["High", 5, "5"] and doc.metadata.get("percent_complete", 100) < 100
    ]
    if not tasks:
        return "No high priority uncompleted tasks."

    lines = [
        "### High Priority Uncompleted Tasks\n",
        "| Task | Person | Percent Complete | Due Date |",
        "|-------|--------|-----------------|----------|"
    ]
    for doc in tasks:
        task = doc.page_content.split("\n")[0].strip()
        person = doc.metadata.get("person", "N/A")
        percent = doc.metadata.get("percent_complete", 0)
        due = doc.metadata.get("due", "N/A")
        lines.append(f"| {task} | {person} | {percent}% | {due} |")

    return "\n".join(lines)

@tool
def kanban_stats_summary() -> str:
    """
    Returns statistics on people, bench duration, uncompleted tasks, certifications,
    and additional breakdowns by categories.
    """
    from collections import defaultdict, Counter
    from dateutil.parser import parse

    docs = retriever.get_relevant_documents("")
    now = datetime.now()
    
    bench_people = set()
    total_bench_days = 0
    bench_start_dates = defaultdict(list)
    uncompleted_tasks = defaultdict(int)
    completed_tasks = defaultdict(int)
    certification_counts = defaultdict(int)
    priority_counts = defaultdict(lambda: defaultdict(int))
    label_counts = Counter()

    for doc in docs:
        m = doc.metadata
        person = m.get("person", "Unknown")
        if not person:
            continue

        # Bench duration tracking
        if m.get("bench_status") == "On the bench":
            bench_people.add(person)
            start = m.get("start")
            if start:
                try:
                    dt = parse(start)
                    bench_start_dates[person].append(dt)
                except:
                    continue

        # Count uncompleted and completed tasks
        if m.get("percent_complete", 100) < 100:
            uncompleted_tasks[person] += 1
        elif m.get("percent_complete") == 100:
            completed_tasks[person] += 1

        # Count certificates
        if m.get("has_certificate"):
            certification_counts[person] += 1

        # Track priorities
        priority = str(m.get("priority", "Unknown"))
        priority_counts[priority]["total"] += 1
        if m.get("percent_complete", 100) < 100:
            priority_counts[priority]["uncompleted"] += 1

        # Track labels
        for label in m.get("labels", []):
            label_counts[label.lower()] += 1

    avg_bench_days = 0
    for person in bench_people:
        if bench_start_dates[person]:
            earliest = min(bench_start_dates[person])
            try:
                days_on_bench = (now - earliest).days
                total_bench_days += days_on_bench
            except:
                continue

    avg_bench_days = round(total_bench_days / len(bench_people), 1) if bench_people else 0
    avg_uncompleted = round(sum(uncompleted_tasks.values()) / len(uncompleted_tasks), 1) if uncompleted_tasks else 0
    avg_completed = round(sum(completed_tasks.values()) / len(completed_tasks), 1) if completed_tasks else 0
    avg_certifications = round(sum(certification_counts.values()) / len(certification_counts), 1) if certification_counts else 0

    # Markdown output
    lines = [
        f"### Kanban Statistical Summary\n",
        f"- 👥 **People on the bench**: {len(bench_people)}",
        f"- 📊 **Average days on bench**: {avg_bench_days}",
        f"- 📄 **Average uncompleted tasks per person**: {avg_uncompleted}",
        f"- ✅ **Average completed tasks per person**: {avg_completed}",
        f"- 🏁 **Average certifications per person**: {avg_certifications}",
        "\n---\n",
        "### Top People by Uncompleted Tasks",
    ]
    for person, count in sorted(uncompleted_tasks.items(), key=lambda x: -x[1])[:5]:
        lines.append(f"- **{person}**: {count} uncompleted tasks")

    lines.append("\n### Top People by Completed Tasks")
    for person, count in sorted(completed_tasks.items(), key=lambda x: -x[1])[:5]:
        lines.append(f"- **{person}**: {count} completed tasks")

    lines.append("\n### People with Most Certifications")
    for person, count in sorted(certification_counts.items(), key=lambda x: -x[1])[:5]:
        lines.append(f"- **{person}**: {count} certifications")

    lines.append("\n### Task Priority Distribution")
    for priority, counts in sorted(priority_counts.items(), key=lambda x: x[0]):
        lines.append(f"- **Priority {priority}**: {counts['total']} total, {counts['uncompleted']} uncompleted")

    lines.append("\n### Most Common Labels")
    for label, count in label_counts.most_common(5):
        lines.append(f"- **{label}**: {count} tasks")

    return "\n".join(lines)


@tool
def tasks_with_label(label: str) -> str:
    """Returns tasks that have the specified label as a markdown list."""
    docs = retriever.get_relevant_documents("")
    matching_tasks = [
        doc for doc in docs
        if label.lower() in [l.lower() for l in doc.metadata.get("labels", [])]
    ]
    if not matching_tasks:
        return f"No tasks found with label '**{label}**'."

    lines = [f"### Tasks with Label '**{label}**':\n"]
    for doc in matching_tasks:
        title = doc.page_content.split("\n")[0].strip()
        person = doc.metadata.get("person", "N/A")
        percent = doc.metadata.get("percent_complete", 0)
        due = doc.metadata.get("due", "N/A")
        lines.append(f"- **{title}** (Person: {person}, {percent}% complete, Due: {due})")

    return "\n".join(lines)


@tool
def tasks_with_checklist(include_completed: bool = False) -> str:
    """Returns tasks with checklist items as a markdown list, optionally including completed."""
    docs = retriever.get_relevant_documents("")
    if include_completed:
        filtered = [doc for doc in docs if doc.metadata.get("has_checklist")]
    else:
        filtered = [doc for doc in docs if doc.metadata.get("has_checklist") and doc.metadata.get("percent_complete", 100) < 100]

    if not filtered:
        return "No matching tasks with checklist."

    lines = ["### Tasks with Checklists:\n"]
    for doc in filtered:
        title = doc.page_content.split("\n")[0].strip()
        person = doc.metadata.get("person", "N/A")
        percent = doc.metadata.get("percent_complete", 0)
        due = doc.metadata.get("due", "N/A")
        lines.append(f"- **{title}** (Person: {person}, {percent}% complete, Due: {due})")

    return "\n".join(lines)


@tool
def tasks_started_after(date_iso: str) -> str:
    """Returns tasks started after the specified ISO date string as a markdown list."""
    docs = retriever.get_relevant_documents("")
    filtered = [
        doc for doc in docs
        if doc.metadata.get("start") and doc.metadata["start"] >= date_iso
    ]
    if not filtered:
        return f"No tasks started after **{date_iso}**."

    lines = [f"### Tasks Started After **{date_iso}**:\n"]
    for doc in filtered:
        title = doc.page_content.split("\n")[0].strip()
        start = doc.metadata.get("start", "N/A")
        person = doc.metadata.get("person", "N/A")
        lines.append(f"- **{title}** (Started: {start}, Person: {person})")

    return "\n".join(lines)



@tool
def last_completed_task_for_person(person_name: str) -> str:
    """Returns the most recent completed task (100%) for a given person."""
    docs = retriever.get_relevant_documents("")
    completed = [
        doc for doc in docs
        if doc.metadata.get("person") == person_name and doc.metadata.get("percent_complete", 0) == 100 and doc.metadata.get("due")
    ]

    if not completed:
        return f"No completed tasks found for **{person_name}**."

    # Sort by due date (most recent first)
    completed.sort(key=lambda doc: doc.metadata["due"], reverse=True)
    doc = completed[0]
    title = doc.page_content.split("\n")[0].strip()
    due = doc.metadata.get("due", "N/A")
    return f"**{person_name}**'s last completed task was:\n- **{title}** (Due: {due})"

@tool
def certificates_completed_since(days_ago: int = 30) -> str:
    """
    Returns a grouped list of people who completed certificate-related tasks in the past X days.
    """
    from collections import defaultdict
    from dateutil.parser import parse

    cutoff_dt = datetime.now() - timedelta(days=days_ago)
    tasks_by_person = defaultdict(list)

    try:
        results = retriever.vectorstore.get(where={"has_certificate": True})
        docs = zip(results["documents"], results["metadatas"])

        for content, metadata in docs:
            if metadata.get("percent_complete", 0) != 100:
                continue
            due_str = metadata.get("due")
            if not due_str:
                continue
            try:
                due_date = parse(due_str)

                person = metadata.get("person", "Unknown")
                title = metadata.get("title", "Untitled Task")
                tasks_by_person[person].append((title, due_date.date().isoformat()))
            except:
                continue
    except Exception as e:
        return f"❌ Error retrieving certificate tasks: {e}"

    if not tasks_by_person:
        return f"No certificate-related tasks completed in the past {days_ago} days."

    # Format with collapsible sections
    lines = [f"<h3>Certificate-Related Tasks Completed in the Past {days_ago} Days</h3>"]
    for person, tasks in sorted(tasks_by_person.items()):
        lines.append(f"<details><summary><strong>{person}</strong></summary><ul>")
        for title, due in tasks:
            lines.append(f"<li>🏁 <strong>{title}</strong> (Due: {due})</li>")
        lines.append("</ul></details>\n")

    return "\n".join(lines)

from datetime import datetime, timezone
from dateutil.parser import parse

@tool
def bench_duration_for_person(person_name: str) -> str:
    """
    Calculates how long a person has been on the bench, using the earliest task's start date.
    """
    person_name_lc = person_name.lower().strip()
    docs = retriever.get_relevant_documents("")

    valid_dates = []
    for doc in docs:
        doc_person = doc.metadata.get("person", "").lower().strip()
        bench_status = doc.metadata.get("bench_status", "")
        start_str = doc.metadata.get("start", "")

        if doc_person != person_name_lc or bench_status != "On the bench":
            continue

        if not start_str or start_str in ["N/A", "-", None]:
            continue

        try:
            parsed = parse(start_str)
            valid_dates.append(parsed)
        except Exception:
            continue  # Ignore bad dates

    if not valid_dates:
        return f"I couldn't find any valid task start dates for **{person_name}** while on the bench."

    earliest = min(valid_dates)

    # Ensure both datetimes are timezone-aware
    now_utc = datetime.now(timezone.utc)
    if earliest.tzinfo is None:
        earliest = earliest.replace(tzinfo=timezone.utc)

    delta_days = (now_utc - earliest).days
    return f"**{person_name}** has been on the bench for **{delta_days} days**, since **{earliest.date().isoformat()}**."

@tool
def certificates_for_person_since(person_name: str) -> str:
    """Returns certificate-related tasks completed or not by a person in the past X days."""
    from dateutil.parser import parse
    docs = retriever.get_relevant_documents(person_name)
    person_name = person_name.strip().lower()

    certs = []
    for doc in docs:
        m = doc.metadata
        doc_person = m.get("person", "").lower()
        if person_name not in doc_person:
            continue
        if not m.get("has_certificate"):
            continue
        try:
            due = parse(m.get("due"))
            certs.append((m.get("title", "Untitled Task"), due.date().isoformat(), m.get("person", "Unknown")))
        except:
            continue

    if not certs:
        return f"No completed certificate tasks found for '{person_name}"

    full_name = certs[0][2]
    lines = [f"### Certificates for **{full_name}**:\n"]
    lines += [f"- 🏁 **{title}** (Due: {date})" for title, date, _ in certs]
    return "\n".join(lines)

@tool
def who_completed_certificate(keyword: str, completed_only: bool = True) -> str:
    """
    Returns a list of people who obtained a certification/training matching the keyword.
    Includes extra metadata for context. Optionally filters by completed tasks only.
    """
    from dateutil.parser import parse

    docs = retriever.get_relevant_documents("")
    matches = []

    keyword_lc = keyword.lower()

    for doc in docs:
        title = doc.metadata.get("title", "") or doc.page_content
        if keyword_lc not in title.lower():
            continue

        if completed_only and doc.metadata.get("percent_complete", 0) < 100:
            continue

        person = doc.metadata.get("person", "Unknown")
        percent = doc.metadata.get("percent_complete", 0)
        due_raw = doc.metadata.get("due", "")
        labels = doc.metadata.get("labels", [])
        priority = doc.metadata.get("priority", "N/A")

        try:
            due = parse(due_raw).date().isoformat() if due_raw else "N/A"
        except Exception:
            due = "N/A"

        matches.append({
            "person": person,
            "title": title.strip(),
            "due": due,
            "percent": percent,
            "labels": labels,
            "priority": priority,
        })

    if not matches:
        return f"No one found with certification containing '**{keyword}**'."

    lines = [f"### Certifications Matching '**{keyword}**'\n"]

    for m in sorted(matches, key=lambda x: (x["person"], x["due"])):
        lines.append(f"- **{m['person']}**:")
        lines.append(f"  - 🏁 **{m['title']}**")
        lines.append(f"    - Due: `{m['due']}`")
        lines.append(f"    - Progress: `{m['percent']}%`")
        lines.append(f"    - Priority: `{m['priority']}`")
        lines.append(f"    - Labels: `{', '.join(m['labels']) if isinstance(m['labels'], list) else m['labels']}`")
        lines.append("")

    return "\n".join(lines)


@tool
def general_qa(question: str) -> str:
    """
    Use this for any free-text Kanban question or when no other tool applies.
    Answers based on semantic search + LLM. Includes fallback guidance if no good match is found.
    """
    if not question.strip():
        return "❗ Please ask a specific question about your Kanban data, like:\n\n- Who has overdue tasks?\n- What are the high-priority issues?\n- Who completed certifications?"

    try:
        # Run the LLM-based QA chain
        answer = qa_chain.run(question)

        # Optional: check for null/default responses
        if not answer or answer.strip().lower() in ["i don't know", "not sure", "no relevant information found"]:
            return "🤔 I couldn't find a confident answer. Try rephrasing your question or be more specific (e.g., include a person's name or keyword)."

        return answer.strip()

    except Exception as e:
        return f"❌ An error occurred while processing your question: `{e}`"


# Tools list remains the same
tools = [
    who_on_bench,
    uncompleted_tasks_for_person,
    completed_tasks_for_person,
    overdue_tasks_for_person,
    high_priority_uncompleted_tasks,
    tasks_with_label,
    tasks_with_checklist,
    tasks_started_after,
    certificates_for_person_since,
    certificates_completed_since,
    last_completed_task_for_person,
    general_qa,
    bench_duration_for_person,
    kanban_stats_summary
]



# ─── 3) Conversation Memory ───────────────────────────────────────────────────

memory = ConversationSummaryMemory(
    llm=llm, 
    max_token_limit=1024,
    return_messages=False,
    memory_key="chat_history",
)

# ─── 4) Agent Initialization ──────────────────────────────────────────────────

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    memory=memory,
    verbose=True,
)


# ─── 5) FastAPI endpoint ─────────────────────────────────────────────────────
@router.get("/kanban_query/")
async def kanban_query(query: Annotated[str, Query(description="Your question")]):
    try:
        answer = await agent.arun(query)

        import markdown
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
