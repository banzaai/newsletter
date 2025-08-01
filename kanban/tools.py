import re
from typing import Annotated, Optional
from urllib import parse
from langchain.tools import tool
from datetime import datetime, timezone
from collections import defaultdict, Counter
from config import model
from langchain.chains import RetrievalQA
from dateutil.parser import parse
from datetime import datetime
from dateutil.parser import parse
from db import connection

vectordb = connection.supabase
llm = model

retriever = vectordb.as_retriever(search_kwargs={"k": 700})  
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


@tool
def filter_tasks_by_label(label: str, status_filter: Optional[str] = None) -> str:
    """
    Returns tasks matching the given label, grouped by completion status (Completed or In Progress).
    Includes full task descriptions and metadata.
    Filters by status if 'completed' or 'not completed' is passed. **Can be chained with other tools**
    """
    docs = retriever.get_relevant_documents("")
    now = datetime.now(timezone.utc)

    matching = [
        doc for doc in docs
        if label.lower() in [
            tag.strip().lower() for tag in doc.metadata.get("labels", "").split(",")
        ] or label.lower() in doc.metadata.get("title", "").lower()
    ]

    if not matching:
        return f"No tasks found with label '**{label}**'. You might want to check for overdue, high-priority, or general tasks."

    completed_tasks = []
    in_progress_tasks = []

    for doc in matching:
        title = doc.page_content.split("\n")[0].strip()
        person = doc.metadata.get("person", "N/A")
        percent = doc.metadata.get("percent_complete", 0)
        due = doc.metadata.get("due", "Not specified")

        task_line = f"- **{title}** (Assigned to: {person}, {percent}% complete) → Due: {due}"

        # Apply status filter
        if status_filter == "completed" and percent == 100:
            completed_tasks.append(task_line)
        elif status_filter == "not completed" and percent < 100:
            in_progress_tasks.append(task_line)
        elif status_filter is None:
            # No filter: include all
            if percent == 100:
                completed_tasks.append(task_line)
            else:
                in_progress_tasks.append(task_line)

    output = [f"🏷️ Tasks with Label '**{label}**':\n"]

    if in_progress_tasks:
        output.append("🔄 In Progress:\n" + "\n".join(in_progress_tasks))
    if completed_tasks:
        output.append("✅ Completed:\n" + "\n".join(completed_tasks))

    return "\n\n".join(output)

@tool
def get_opportunities(status_filter: Optional[str] = None) -> str:
    """
    Returns a markdown list of opportunities grouped by person.
    You can filter by status: 'completed', 'not completed', or leave empty for all.
    Opportunities are considered completed if 'percent_complete' is 100. **Can be chained with other tools.**
    """
    docs = retriever.get_relevant_documents("opportunity")
    now = datetime.now(timezone.utc)

    grouped_opportunities = {}

    for doc in docs:
        if doc.metadata.get("bench_status") != "On the bench":
            continue

        m = doc.metadata
        doc_person = m.get("person", "").lower()
        title = m.get("title", "Untitled")
        description = m.get("description", "")
        due_str = m.get("due", "")
        pc_val = m.get("percent_complete", 0)

        combined_text = f"{description} {title}".lower()
        if any(keyword in combined_text for keyword in [
            "opp", "opportunity", "interviews", "interview", 'opps',
            "opportunities", "joined", "engaged in"
        ]):
            try:
                due = parse(due_str) if due_str else None
                due_date_str = str(due.date()) if due else "Not set"
                status = "✅ Completed" if pc_val == 100 else "⏳ Not yet obtained"
            except Exception:
                due_date_str = "Invalid date"
                status = "⚠️ Invalid due date"

            # Apply status filter
            if status_filter == "completed" and pc_val < 100:
                continue
            if status_filter == "not completed" and pc_val == 100:
                continue

            if doc_person not in grouped_opportunities:
                grouped_opportunities[doc_person] = []

            grouped_opportunities[doc_person].append(
                f"- **{title}**: Due: {due_date_str}, Status: {status}, Completion: {pc_val}% with description {description}"
            )

    if not grouped_opportunities:
        return "No opportunities found."

    markdown_output = []
    for person, opps in grouped_opportunities.items():
        markdown_output.append(f"### {person.title()} has these opportunities:\n" + "\n".join(opps))

    return "\n".join(markdown_output)


@tool
def contains_rfp_request_for_proposal(keywords=None):
    """
    Checks if any of the specified keywords (default: RFP-related) are present
    in the title, description, or checklist of relevant documents.
    """
    if keywords is None:
        keywords = ["rfp", "request for proposal", "rfps"]

    pattern = re.compile(r"|".join(re.escape(k) for k in keywords), re.IGNORECASE)
    docs = retriever.get_relevant_documents("")

    for doc in docs:
        title = doc.metadata.get("title", "")
        description = doc.metadata.get("description", "")
        checklist = doc.metadata.get("checklist", [])

        if pattern.search(title) or pattern.search(description):
            return True

        if any(pattern.search(entry) for entry in checklist):
            return True

    return False



@tool
def who_on_bench() -> str:
    """Returns a markdown list of people who are currently on the bench. 
    Use this to find available individuals. """

    docs = retriever.get_relevant_documents("who is on the bench")
    
    bench_people = [doc.metadata.get("person") for doc in docs if doc.metadata.get("bench_status") == "On the bench"]
    
    if not bench_people:
        return "No one is currently on the bench."
    
    return "\n".join(f"- {person}" for person in bench_people)


@tool
def task_due(is_due_by: str = datetime.now(), due_filter: str = None, person_name: str = None):
    """
    Returns the tasks that are due by a certain date or person, which could be in the future or past which depends on the due filter
    depending on the query. This tool can be chained with other tools.
    """
    docs = retriever.get_relevant_documents("")  # Consider passing a query here for better filtering
    tasks = {}

    for doc in docs:
        m = doc.metadata
        due = m.get("due")
        task = m.get("title")
        description = m.get("description")
        checklist = m.get("checklist", "No checklist")
        person = m.get("person")

        # Skip if person_name is specified and doesn't match
        if person_name and person_name.lower() not in person.lower():
            continue

        # Apply due_filter logic
        if due_filter in ['before', 'finished']:
            if due > is_due_by:
                continue
        elif due_filter in ['after', 'upcoming']:
            if due < is_due_by:
                continue
        # If no due_filter, include all

        key = person_name.lower() if person_name else person.lower()
        tasks.setdefault(key, []).append(
            f"- **{task}** for {person}: the due date is: {due} with description {description} and checklist: {checklist}"
        )

    # Format output
    markdown_output = []
    for person, task_list in tasks.items():
        markdown_output.append(f"### {person} has these tasks:\n" + "\n".join(task_list) + "\nThat are still due.")

    return "\n".join(markdown_output) if markdown_output else "No matching tasks found."


@tool
def tasks_for_person(person_name: str, filter_status: str = None) -> str:
    """
    Returns tasks for a person filtered by completion status ('completed', 'uncompleted', or None) as a markdown table.
    If filter_status is None, returns both completed and uncompleted tasks.
    """
    docs = retriever.get_relevant_documents("")
    person_name_lower = person_name.lower()

    def is_task_for_person(doc):
        return person_name_lower in (doc.metadata.get("person") or "").lower()

    def is_task_matching_status(doc):
        percent = doc.metadata.get("percent_complete", 0)
        if filter_status is None:
            return True
        elif filter_status.lower() == "completed":
            return percent == 100
        elif filter_status.lower() == "uncompleted":
            return percent < 100
        return False

    tasks = [doc for doc in docs if is_task_for_person(doc) and is_task_matching_status(doc)]

    if not tasks:
        status_text = filter_status or "any"
        return f"No {status_text} tasks for **{person_name}**."

    status_text = filter_status or "All"
    lines = [
        f"### {status_text.capitalize()} Tasks for **{person_name}**\n",
        "| Task | Percent Complete | Due Date | Priority | Description |",
        "|------|------------------|----------|----------|-------------|"
    ]

    for doc in tasks:
        content = doc.page_content.split("\n")[0].strip()
        percent = doc.metadata.get("percent_complete", 0)
        due_date = doc.metadata.get("due")

        if due_date:
            if isinstance(due_date, str):
                try:
                    due_date = datetime.fromisoformat(due_date)
                    due = due_date.date().isoformat()
                except ValueError:
                    due = due_date
            elif isinstance(due_date, datetime):
                due = due_date.date().isoformat()
            else:
                due = str(due_date)
        else:
            due = "N/A"

        priority = doc.metadata.get("priority", "N/A")
        description = doc.metadata.get("description", "N/A")
        lines.append(f"| {content} | {percent}% | {due} | {priority} | {description} |")

    return "\n".join(lines)

@tool
def overdue_tasks_for_person(person_name: str) -> str:
    """Returns overdue, uncompleted tasks for a person in markdown list."""
    now_iso = datetime.now().isoformat()
    docs = retriever.get_relevant_documents("")
    overdue = [
        doc for doc in docs
        if person_name.lower() in (doc.metadata.get("person") or "").lower()
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

def parse_naive_datetime(date_str):
    try:
        dt = parse(date_str)
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    except Exception:
        return None

@tool
def kanban_stats_summary(person_name: str = None) -> str:
    """
    Returns statistics on people or, if specified, a specific person: bench duration, uncompleted tasks,
    certifications, and additional breakdowns by categories. Includes task overview if person_name is provided.
    """
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

        if person_name and person_name.lower() not in person.lower():
            continue

        if m.get("bench_status") == "On the bench":
            bench_people.add(person)
            start = m.get("start")
            if start:
                dt = parse_naive_datetime(start)
                if dt:
                    bench_start_dates[person].append(dt)

        percent = m.get("percent_complete", 0)
        if percent < 100:
            uncompleted_tasks[person] += 1
        elif percent == 100:
            completed_tasks[person] += 1

        if m.get("has_certificate"):
            certification_counts[person] += 1

        priority = str(m.get("priority", "Unknown"))
        priority_counts[priority]["total"] += 1
        if percent < 100:
            priority_counts[priority]["uncompleted"] += 1

        raw_labels = m.get("labels")
        if isinstance(raw_labels, str) and raw_labels.strip():
            for label in raw_labels.split(','):
                label_counts[label.strip().lower()] += 1

    for person in bench_people:
        if bench_start_dates[person]:
            earliest = min(bench_start_dates[person])
            try:
                total_bench_days += (now - earliest).days
            except Exception:
                continue

    avg_bench_days = round(total_bench_days / len(bench_people), 1) if bench_people else 0
    avg_uncompleted = round(sum(uncompleted_tasks.values()) / len(uncompleted_tasks), 1) if uncompleted_tasks else 0
    avg_completed = round(sum(completed_tasks.values()) / len(completed_tasks), 1) if completed_tasks else 0
    avg_certifications = round(sum(certification_counts.values()) / len(certification_counts), 1) if certification_counts else 0

    lines = []

    if person_name:
        lines.append(f"### 📌 Detailed Stats for **{person_name}**\n")
        lines.append(f"- 🪑 On the bench: {'Yes' if person_name in bench_people else 'No'}")
        if person_name in bench_start_dates and bench_start_dates[person_name]:
            days = (now - min(bench_start_dates[person_name])).days
            lines.append(f"- ⏳ Days on bench: {days}")
        lines.append(f"- 📄 Uncompleted tasks: {uncompleted_tasks.get(person_name, 0)}")
        lines.append(f"- ✅ Completed tasks: {completed_tasks.get(person_name, 0)}")
        lines.append(f"- 🏁 Certifications: {certification_counts.get(person_name, 0)}")

        # 🔄 Include task overview
        lines.append("\n---\n")
        lines.append(tasks_for_person(person_name))  # Show all tasks

    else:
        lines.extend([
            f"### 📊 Kanban Statistical Summary\n",
            f"- 👥 People on the bench: {len(bench_people)}",
            f"- ⏳ Average days on bench: {avg_bench_days}",
            f"- 📄 Average uncompleted tasks per person: {avg_uncompleted}",
            f"- ✅ Average completed tasks per person: {avg_completed}",
            f"- 🏁 Average certifications per person: {avg_certifications}",
            "\n---\n",
            "### 🔝 Top People by Uncompleted Tasks",
        ])
        for person, count in sorted(uncompleted_tasks.items(), key=lambda x: -x[1])[:5]:
            lines.append(f"- **{person}**: {count} uncompleted tasks")

        lines.append("\n### 🔝 Top People by Completed Tasks")
        for person, count in sorted(completed_tasks.items(), key=lambda x: -x[1])[:5]:
            lines.append(f"- **{person}**: {count} completed tasks")

        lines.append("\n### 🏅 People with Most Certifications")
        for person, count in sorted(certification_counts.items(), key=lambda x: -x[1])[:5]:
            lines.append(f"- **{person}**: {count} certifications")

        lines.append("\n### ⚠️ Task Priority Distribution")
        for priority, counts in sorted(priority_counts.items(), key=lambda x: x[0]):
            lines.append(f"- **Priority {priority}**: {counts['total']} total, {counts['uncompleted']} uncompleted")

        lines.append("\n### 🏷️ Most Common Labels")
        for label, count in label_counts.most_common(5):
            lines.append(f"- **{label}**: {count} tasks")

    return "\n".join(lines)

@tool
def tasks_with_checklist() -> str:
    """Returns tasks with checklist items as a markdown list, optionally including completed, can be chained with other tools"""
    docs = retriever.get_relevant_documents("")

    filtered = [doc for doc in docs if doc.metadata.get("has_checklist")]

    if not filtered:
        return "No matching tasks with checklist."

    lines = ["### Tasks with Checklists:\n"]
    for doc in filtered:
        title = doc.page_content.split("\n")[0].strip()
        person = doc.metadata.get("person", "N/A")
        percent = doc.metadata.get("percent_complete", 0)
        due = doc.metadata.get("due", "N/A")
        checklist = doc.metadata.get("checklist")
        description = doc.metadata.get("description")
        lines.append(f"- **{title}** (Person: {person}, {percent}% complete, Due: {due})\
                     with checlist: {checklist} and  with the description of the task: {description}")

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
    """Returns the most recent completed task (100%) for a given person and its description """
    docs = retriever.get_relevant_documents("")
    completed = [
        doc for doc in docs
        if person_name.lower() in (doc.metadata.get("person") or "").lower() and doc.metadata.get("percent_complete", 0) == 100 and doc.metadata.get("due")
    ]

    if not completed:
        return f"No completed tasks found for **{person_name}**."

    # Sort by due date (most recent first)
    completed.sort(key=lambda doc: doc.metadata["due"], reverse=True)
    doc = completed[0]
    title = doc.page_content.split("\n")[0].strip()
    due = doc.metadata.get("due", "N/A")
    description = doc.metadata.get("description")
    return f"**{person_name}**'s last completed task was:\n- **{title}** (Due: {due} with description: {description})"


@tool
def certificates_completed_since(days_ago: int = 30) -> str:
    """
    Returns people or the person from the query who completed certificate-related tasks in the past X days with the available info about that certificate.
    """
    tasks_by_person = defaultdict(list)

    try:
        docs = retriever.get_relevant_documents("")
        for doc in docs:
            metadata = doc.metadata
            if not metadata.get("has_certificate"):
                continue
            if metadata.get("percent_complete", 0) != 100:
                continue
            due_str = metadata.get("due")
            if not due_str:
                continue
            try:
                checklist = metadata.get("checklist") 
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
    lines = [f"Certificate-Related Tasks Completed in the Past {days_ago} Days"]
    for person, tasks in sorted(tasks_by_person.items()):
        lines.append(f"{person}")
        for title, due in tasks:
            lines.append(f"{title} (Due: {due})")
            if checklist:
                lines.append(f"{checklist}")

    return lines



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

from datetime import datetime
from collections import defaultdict, Counter
from dateutil.parser import parse
from langchain.tools import tool
from pydantic import BaseModel, Field

class KanbanInput(BaseModel):
    person_name: str = Field(None, description="Name of the person to summarize")
    filter_status: str = Field(None, description="Task filter: 'completed', 'uncompleted', or None for all")

def parse_naive_datetime(date_str):
    try:
        dt = parse(date_str)
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    except Exception:
        return None

@tool(args_schema=KanbanInput)
def kanban_stats_summary(person_name: str = None, filter_status: str = None) -> str:
    """
    Returns kanban statistics and optionally a task overview for a specific person.
    If person_name is provided, includes task table filtered by completion status.
    """
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

    person_name_lower = person_name.lower() if person_name else None

    def is_task_for_person(doc):
        return person_name_lower in (doc.metadata.get("person") or "").lower()

    def is_task_matching_status(doc):
        percent = doc.metadata.get("percent_complete", 0)
        if filter_status is None:
            return True
        elif filter_status.lower() == "completed":
            return percent == 100
        elif filter_status.lower() == "uncompleted":
            return percent < 100
        return False

    task_lines = []

    for doc in docs:
        m = doc.metadata
        person = m.get("person", "Unknown")
        if not person:
            continue

        if person_name and person_name_lower not in person.lower():
            continue

        # Bench tracking
        if m.get("bench_status") == "On the bench":
            bench_people.add(person)
            start = m.get("start")
            if start:
                dt = parse_naive_datetime(start)
                if dt:
                    bench_start_dates[person].append(dt)

        # Task completion
        percent = m.get("percent_complete", 0)
        if percent < 100:
            uncompleted_tasks[person] += 1
        elif percent == 100:
            completed_tasks[person] += 1

        # Certifications
        if m.get("has_certificate"):
            certification_counts[person] += 1

        # Priorities
        priority = str(m.get("priority", "Unknown"))
        priority_counts[priority]["total"] += 1
        if percent < 100:
            priority_counts[priority]["uncompleted"] += 1

        # Labels
        raw_labels = m.get("labels")
        if isinstance(raw_labels, str) and raw_labels.strip():
            for label in raw_labels.split(','):
                label_counts[label.strip().lower()] += 1

        # Task table (only if person_name is provided)
        if person_name and is_task_for_person(doc) and is_task_matching_status(doc):
            content = doc.page_content.split("\n")[0].strip()
            due_date = m.get("due")
            if due_date:
                if isinstance(due_date, str):
                    try:
                        due_date = datetime.fromisoformat(due_date)
                        due = due_date.date().isoformat()
                    except ValueError:
                        due = due_date
                elif isinstance(due_date, datetime):
                    due = due_date.date().isoformat()
                else:
                    due = str(due_date)
            else:
                due = "N/A"

            task_lines.append(f"| {content} | {percent}% | {due} | {m.get('priority', 'N/A')} | {m.get('description', 'N/A')} |")

    for person in bench_people:
        if bench_start_dates[person]:
            earliest = min(bench_start_dates[person])
            try:
                total_bench_days += (now - earliest).days
            except Exception:
                continue

    avg_bench_days = round(total_bench_days / len(bench_people), 1) if bench_people else 0
    avg_uncompleted = round(sum(uncompleted_tasks.values()) / len(uncompleted_tasks), 1) if uncompleted_tasks else 0
    avg_completed = round(sum(completed_tasks.values()) / len(completed_tasks), 1) if completed_tasks else 0
    avg_certifications = round(sum(certification_counts.values()) / len(certification_counts), 1) if certification_counts else 0

    lines = []

    if person_name:
        lines.append(f"### 📌 Detailed Stats for **{person_name}**\n")
        lines.append(f"- 🪑 On the bench: {'Yes' if person_name in bench_people else 'No'}")
        if person_name in bench_start_dates and bench_start_dates[person_name]:
            days = (now - min(bench_start_dates[person_name])).days
            lines.append(f"- ⏳ Days on bench: {days}")
        lines.append(f"- 📄 Uncompleted tasks: {uncompleted_tasks.get(person_name, 0)}")
        lines.append(f"- ✅ Completed tasks: {completed_tasks.get(person_name, 0)}")
        lines.append(f"- 🏁 Certifications: {certification_counts.get(person_name, 0)}")

        if task_lines:
            lines.append("\n---\n")
            status_text = filter_status or "All"
            lines.append(f"### {status_text.capitalize()} Tasks for **{person_name}**\n")
            lines.append("| Task | Percent Complete | Due Date | Priority | Description |")
            lines.append("|------|------------------|----------|----------|-------------|")
            lines.extend(task_lines)
        else:
            lines.append(f"\nNo {filter_status or 'any'} tasks found for **{person_name}**.")

    else:
        lines.extend([
            f"### 📊 Kanban Statistical Summary\n",
            f"- 👥 People on the bench: {len(bench_people)}",
            f"- ⏳ Average days on bench: {avg_bench_days}",
            f"- 📄 Average uncompleted tasks per person: {avg_uncompleted}",
            f"- ✅ Average completed tasks per person: {avg_completed}",
            f"- 🏁 Average certifications per person: {avg_certifications}",
            "\n---\n",
            "### 🔝 Top People by Uncompleted Tasks",
        ])
        for person, count in sorted(uncompleted_tasks.items(), key=lambda x: -x[1])[:5]:
            lines.append(f"- **{person}**: {count} uncompleted tasks")

        lines.append("\n### 🔝 Top People by Completed Tasks")
        for person, count in sorted(completed_tasks.items(), key=lambda x: -x[1])[:5]:
            lines.append(f"- **{person}**: {count} completed tasks")

        lines.append("\n### 🏅 People with Most Certifications")
        for person, count in sorted(certification_counts.items(), key=lambda x: -x[1])[:5]:
            lines.append(f"- **{person}**: {count} certifications")

        lines.append("\n### ⚠️ Task Priority Distribution")
        for priority, counts in sorted(priority_counts.items(), key=lambda x: x[0]):
            lines.append(f"- **Priority {priority}**: {counts['total']} total, {counts['uncompleted']} uncompleted")

        lines.append("\n### 🏷️ Most Common Labels")
        for label, count in label_counts.most_common(5):
            lines.append(f"- **{label}**: {count} tasks")

    return "\n".join(lines)



@tool
def general_qa(question: str) -> str:
    """
    Use this for any free-text Kanban question or when no other tool applies.
    Answers based on semantic search + LLM. Includes fallback guidance if no good match is found.
    Also summarizes retrieved content and contrasts it with the original question.
    """
    if not question.strip():
        return (
            "❗ Please ask a specific question about your Kanban data, like:\n\n"
            "- Who has overdue tasks?\n"
            "- What are the high-priority issues?\n"
            "- Who completed certifications?"
        )

    try:
        # Retrieve relevant documents
        docs = retriever.get_relevant_documents(question)

        if not docs:
            return (
                "🤔 I couldn't find any relevant documents. Try rephrasing your question "
                "or be more specific (e.g., include a person's name or keyword)."
            )

        # Step 1: Summarize the retrieved documents
        summary_prompt = f"Summarize the following Kanban-related documents:\n\n{docs}"
        summary = llm.invoke(summary_prompt)

        # Step 2: Contrast the summary with the original question
        contrast_prompt = (
            f"Given the user's question:\n\n\"{question}\"\n\n"
            f"And the summary of relevant documents:\n\n\"{summary}\"\n\n"
            "Provide a direct answer to the question, and explain briefly how the summary supports it."
        )
        final_answer = llm.invoke(contrast_prompt)

        return final_answer.strip()

    except Exception as e:
        return f"❌ An error occurred while processing your question: `{e}`"
