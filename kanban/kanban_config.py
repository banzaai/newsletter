import os
from .tools import bench_duration_for_person, contains_rfp_request_for_proposal, filter_tasks_by_label, certificates_completed_since, generate_pie_chart_from_tool_output, get_opportunities,  general_qa, high_priority_uncompleted_tasks, kanban_stats_summary, last_completed_task_for_person, overdue_tasks_for_person, task_due, tasks_for_person, tasks_started_after, tasks_with_checklist, who_on_bench
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from config import model

tools = [
    generate_pie_chart_from_tool_output,
    who_on_bench,
    tasks_for_person,
    contains_rfp_request_for_proposal,
    overdue_tasks_for_person,
    high_priority_uncompleted_tasks,
    tasks_with_checklist,
    tasks_started_after,
    certificates_completed_since,
    last_completed_task_for_person,
    general_qa,
    bench_duration_for_person,
    kanban_stats_summary,
    get_opportunities,
    filter_tasks_by_label,
    task_due
]


memory = ConversationBufferMemory(
    llm=model, 
    max_token_limit=1024,
    return_messages=True,
    memory_key="chat_history",
)


agent = initialize_agent(
    tools=tools,
    llm=model,
    agent=AgentType.OPENAI_FUNCTIONS,
    memory=memory,
    verbose=True,
)


CLIENT_ID = os.getenv("CLIENT_ID")
TENANT_ID = os.getenv("TENANT_ID")
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPES = ["https://graph.microsoft.com/.default"]
CACHE_FILE = os.getenv("CACHE_FILE")
TOKEN_CACHE = os.getenv("TOKEN_CACHE")
CLIENT_CREDENTIALS= os.getenv("CLIENT_CREDENTIALS")
