import json
import os
from fastapi import Body, HTTPException, APIRouter, Query
import numpy as np
from pydantic import BaseModel, Field
from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain.agents import initialize_agent, AgentType
from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from supafunc import create_client
from config import model, embeddings
from langchain_chroma import Chroma
from db import save_event

router = APIRouter()

class UserInputEvent(BaseModel):
    name: Annotated[str, Field(description="Name of the user")]
    event: Annotated[str, Field(description="Event to be created")]


# Define prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", ''' You are a helpful assistant designed to guide users in creating events based on their stories.

When a user starts a conversation, always begin with a warm and welcoming message.

Listen carefully to the user's description. If they describe an event:

1. Acknowledge their input positively.
2. Ask follow-up questions to gather more details about the event, such as:
   - Purpose
   - Date and time
   - Location
   - Number of attendees
   - Any special requirements
3. Continue asking questions until you clearly understand the event and can categorize it (e.g., birthday, meeting, wedding, etc.).

Once you have enough information to define the event:
- Confirm the details with the user to ensure accuracy.
- Then ask: **"Would you like to be the organizer of this event, or are you just proposing the idea?"**

When the user responds:
- If they say anything like “yes”, “I want to organize”, “I’ll organize it”, or similar → set `organize_event` to `true`.
- If they say anything like “just proposing”, “no”, “I don’t want to organize”, “someone else should organize”, or similar → set `organize_event` to `false`.

⚠️ As soon as the user gives a clear answer, you must immediately call the `save_event` function with:
- `name`: the user's name
- `content`: the full event description you generated
- `organize_event`: the boolean value based on their response
- `category`: the event type based on the user's description

Do not repeat the question or say the event is saved unless you have called the function.


'''
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Initialize agent
agent_executor = initialize_agent(
    tools=[save_event],
    llm=model,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
)

# LangGraph node that wraps the agent
def agent_node(state: MessagesState):
    messages = state["messages"]

    # Apply the prompt template
    prompt = prompt_template.invoke({"messages": messages})

    # Run the agent with the formatted prompt
    result = agent_executor.run(prompt.to_string())

    return {"messages": messages + [HumanMessage(content=result)]}

# LangGraph workflow
workflow = StateGraph(state_schema=MessagesState)
workflow.add_node("agent", agent_node)
workflow.set_entry_point("agent")
workflow.set_finish_point("agent")

memory = MemorySaver()
appl = workflow.compile(checkpointer=memory)

@router.post('/event/')
async def make_event(user_input: Annotated[UserInputEvent, Body(...)]):
    """Endpoint to create an event based on user input."""
    try:
        input_messages = [HumanMessage(content=f"{user_input.name} says: {user_input.event}")]
        response = appl.invoke({"messages": input_messages}, config={"configurable": {"thread_id": "abc345"}})
        return {"response": response["messages"][-1].content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/event/")
async def get_event(query: Annotated[str, Query(description="Query to search for an event")]):
    """Endpoint to retrieve a user's event by name."""
    supabase_url = os.getenv("VECTOR_DB_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if os.getenv("ENVIRONMENT", "production") == "production":

        supabase = create_client(supabase_url, supabase_key, is_async=False)
        data = supabase.table("vector").select("*").execute().data

        query_vector = embeddings.embed_query(query)

        def cosine_similarity(a, b):
            a = np.array(a)
            b = np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        scored = []
        for item in data:
            embedding = json.loads(item["embedding"]) if isinstance(item["embedding"], str) else item["embedding"]
            score = cosine_similarity(query_vector, embedding)
            scored.append({**item, "score": score})

        response = [
            {
                "event_id": item.get("event_id"),
                "name": item.get("name"),
                "content": item.get("text")
            }
            for item in sorted(scored, key=lambda x: x["score"], reverse=True)[:10]
        ]

    else:
        db = Chroma(persist_directory="vector_db", embedding_function=embeddings)
        results = db.similarity_search(query, k=10)
        response = [
            {
                "event_id": doc.metadata.get("event_id"),
                "name": doc.metadata.get("name"),
                "content": doc.page_content
            } for doc in results
        ]

    bot_answer_response = bot_answer(response, query)
    return {"events": bot_answer_response} if response else {"events": "No events found."}


def bot_answer(response: list[dict], query: str) -> str:
    if not response:
        return "No events found."

    # Format the response into a string
    content_summary = "\n\n".join(
        f"Event ID: {item['event_id']}\nName: {item['name']}\nContent: {item['content']}"
        for item in response
    )

    prompt_text = (
        f"You are a helpful assistant that provides comprehensive answers based on the user's query.\n"
        f"Query: {query}\n\n"
        f"Content:\n{content_summary}\n\n"
        "Please provide a detailed and informative answer."
    )

    return model.invoke(prompt_text).content.strip()