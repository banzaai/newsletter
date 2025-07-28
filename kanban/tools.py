from typing import List
from urllib import parse
from langchain.tools import tool
from datetime import datetime, timedelta, timezone
from collections import defaultdict, Counter
from config import model, embeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
import ast  # Add this at the top if not already imported
from dateutil.parser import parse
from langchain.docstore.document import Document
from datetime import datetime
from dateutil.parser import parse

vectordb = Chroma(persist_directory="vector_kanban_db", embedding_function=embeddings)
llm = model

retriever = vectordb.as_retriever(search_kwargs={"k": 700})  # increase k for broader fetch
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


@tool
def filter_tasks_by_label(label: str) -> str:
    """Returns a markdown list of tasks that match the given label. If none found, lets the agent try other tools."""
    docs = retriever.get_relevant_documents("")
    matching = [
        doc for doc in docs
        if label.lower() in (doc.metadata.get("labels").lower() or "")
    ]

    if not matching:
        return f"No tasks found with label '**{label}**'. You might want to check for overdue, high-priority, or general tasks."

    lines = [f"### Tasks with Label '**{label}**':\n"]
    for doc in matching:
        title = doc.page_content.split("\n")[0].strip()
        bench_status = doc.metadata.get("bench_status")
        person = doc.metadata.get("person", "N/A")
        percent = doc.metadata.get("percent_complete", 0)
        due = doc.metadata.get("due", "N/A")

        priority = doc.metadata.get("priority")
        pc =doc.metadata.get("percent_complete")
        start = doc.metadata.get("start")
        due = doc.metadata.get("due")
        has_des = doc.metadata.get("has_description")
        has_check = doc.metadata.get("has_checklist")
        labels = doc.metadata.get("labels")
        has_cer =doc.metadata.get("has_certificate")
        checklist = doc.metadata.get("checklist")
        lines.append(f"- **{title}** (Person: {person} is {bench_status}, {percent}% complete, Due: {due}) --- start {start} and ends {due}, it has the labels {labels} and is {pc} completed. the checklist for this task is {checklist}")

    return "\n".join(lines)


@tool
def get_opportunities(docs: List[Document], person_name: str) -> str:
    """Returns a markdown list of opportunities a person has or had, based on document descriptions."""
    if not docs:
        docs = retriever.get_relevant_documents("")
    person_name = person_name.strip().lower()

    opportunities = []

    for doc in docs:
        m = doc.metadata
        doc_person = m.get("person", "").lower()
        description = m.get("description", "")

        if person_name not in doc_person:
            continue

        # Look for keywords that suggest an opportunity
        if any(keyword in description.lower() for keyword in ["opportunity", "opportunities" "assigned to", "joined", "started", "engaged in"]):
            title = m.get("title", "Untitled")
            opportunities.append(f"- **{title}**: {description}")

    if not opportunities:
        return f"No opportunities found for **{person_name}**."

    return f"### Opportunities for **{person_name.title()}**:\n" + "\n".join(opportunities)



@tool
def who_on_bench(docs: str) -> str:
    """Returns a markdown list of people currently on the bench."""
    print('IN HEREEEEEEE')
    docs = retriever.get_relevant_documents(str)
    people = sorted({doc.metadata.get("person") for doc in docs if doc.metadata.get("bench_status") == "On the bench"})
    if not people:
        return "No one is currently on the bench."
    lines = ["### People Currently On The Bench:\n"]
    for p in people:
        lines.append(f"- **{p}**")
    return "\n".join(lines)


@tool
def uncompleted_tasks_for_person(person_name: str, docs: List[Document]) -> str:
    """Returns uncompleted tasks (<100%) for a person as a markdown table."""
    if not docs:
        docs = retriever.get_relevant_documents("")
    tasks = [
        doc for doc in docs
        if person_name.lower() in (doc.metadata.get("person") or "").lower()
        and doc.metadata.get("percent_complete", 100) < 100
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
def completed_tasks_for_person(person_name: str, docs: List[Document]) -> str:
    """Returns completed tasks (100%) for a person as a markdown table."""
    if not docs:
        docs = retriever.get_relevant_documents("")
    tasks = [
        doc for doc in docs
        if person_name.lower() in (doc.metadata.get("person") or "").lower()
        and doc.metadata.get("percent_complete", 0) == 100
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
def overdue_tasks_for_person(person_name: str, docs: List[Document]) -> str:
    """Returns overdue, uncompleted tasks for a person in markdown list."""
    now_iso = datetime.now().isoformat()
    if not docs:
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
def high_priority_uncompleted_tasks(docs: List[Document]) -> str:
    """Returns all high priority tasks that are not completed, as markdown table."""
    if not docs:
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
def kanban_stats_summary(docs: List[Document]) -> str:
    """
    Returns statistics on people, bench duration, uncompleted tasks, certifications,
    and additional breakdowns by categories.
    """
    if not docs:
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


        # Inside your loop:
        raw_labels = m.get("labels", [])
        if isinstance(raw_labels, str):
            try:
                raw_labels = ast.literal_eval(raw_labels)
            except:
                raw_labels = []

        for label in raw_labels:
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
def tasks_with_checklist(docs: List[Document], include_completed: bool = False) -> str:
    """Returns tasks with checklist items as a markdown list, optionally including completed."""
    if not docs:
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
def tasks_started_after(date_iso: str, docs: List[Document]) -> str:
    """Returns tasks started after the specified ISO date string as a markdown list."""
    if not docs:
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
def last_completed_task_for_person(person_name: str, docs: List[Document]) -> str:
    """Returns the most recent completed task (100%) for a given person."""
    if not docs:
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
    return f"**{person_name}**'s last completed task was:\n- **{title}** (Due: {due})"


@tool
def certificates_completed_since(days_ago: int = 30) -> str:
    """
    Returns a grouped list of people who completed certificate-related tasks in the past X days.
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
                keywords = ['certificate', 'certification', 'certificates', 'certifications']
                checklist_items = [item.lower() for item in metadata.get("checklist", [])]

                if any(keyword in checklist_items for keyword in keywords):
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
    lines = [f"<h3>Certificate-Related Tasks Completed in the Past {days_ago} Days</h3>"]
    for person, tasks in sorted(tasks_by_person.items()):
        lines.append(f"<details><summary><strong>{person}</strong></summary><ul>")
        for title, due in tasks:
            lines.append(f"<li>🏁 <strong>{title}</strong> (Due: {due})</li>")
            if checklist:
                lines.append(f"<li> <strong>{checklist}</strong> )</li>")
        lines.append("</ul></details>\n")

    return "\n".join(lines)



@tool
def bench_duration_for_person(person_name: str, docs: List[Document]) -> str:
    """
    Calculates how long a person has been on the bench, using the earliest task's start date.
    """
    person_name_lc = person_name.lower().strip()
    if not docs:
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
def certificates_for_person_since(person_name: str, docs: List[Document]) -> str:
    """Returns certificate-related tasks completed or not by a person, including due date status."""
    if not docs: 
        docs = retriever.get_relevant_documents(person_name)
    person_name = person_name.strip().lower()

    certs = []
    now = datetime.now(timezone.utc)  

    for doc in docs:
        m = doc.metadata
        doc_person = m.get("person", "").lower()

        if person_name not in doc_person:
            continue

        if not m.get("has_certificate"):
            continue

        title = m.get("title", "Untitled Task")
        person = m.get("person", "Unknown")
        due_str = m.get("due")

        if due_str:
            try:
                due = parse(due_str)
                due_date_str = due.date()
                if due > now:
                    status = "⏳ Not yet obtained"
                else:
                    status = "✅ Completed"
            except Exception:
                due_date_str = "Invalid date"
                status = "⚠️ Invalid due date"
        else:
            due_date_str = "Not set"
            status = "❓ Due date missing"

        certs.append((title, due_date_str, person, status))

    if not certs:
        return f"No certificate-related tasks found for '{person_name}'."

    full_name = certs[0][2]
    lines = [f"### Certificates for **{full_name}**:\n"]
    for title, due, _, status in certs:
        lines.append(f"- 🏁 **{title}** (Due: {due}) — {status}")

    return "\n".join(lines)

@tool
def who_completed_certificate(docs: List[Document], keyword: str, completed_only: bool = True) -> str:
    """
    Returns a list of people who obtained a certification/training matching the keyword.
    Includes extra metadata for context. Optionally filters by completed tasks only.
    """

    if not docs:
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
def general_qa(question: str, docs: List[Document] = None) -> str:
    """
    Use this for any free-text Kanban question or when no other tool applies.
    Answers based on semantic search + LLM. Includes fallback guidance if no good match is found.
    """
    if not question.strip():
        return (
            "❗ Please ask a specific question about your Kanban data, like:\n\n"
            "- Who has overdue tasks?\n"
            "- What are the high-priority issues?\n"
            "- Who completed certifications?"
        )

    try:
        # Fallback to semantic search if no docs provided
        if not docs:
            docs = retriever.get_relevant_documents(question)

        # Run the LLM-based QA chain
        answer = qa_chain.run(input_documents=docs, query=question)

        if not answer or answer.strip().lower() in [
            "i don't know", "not sure", "no relevant information found"
        ]:
            return (
                "🤔 I couldn't find a confident answer. Try rephrasing your question "
                "or be more specific (e.g., include a person's name or keyword)."
            )

        return answer.strip()

    except Exception as e:
        return f"❌ An error occurred while processing your question: `{e}`"
