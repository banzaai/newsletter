import os
from fastapi import Body, HTTPException, Query, APIRouter
import numpy as np
from pydantic import BaseModel, Field
from typing import Annotated
import numpy as np
import json
from langchain_core.messages import HumanMessage
from langchain.agents import initialize_agent, AgentType
from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from supafunc import create_client

from config import model, embeddings
from langchain_chroma import Chroma
from db import save_story

router = APIRouter()

# Input schema
class UserInputStory(BaseModel):
    name: Annotated[str, Field(description="Name of the user")]
    event: Annotated[str, Field(description="Story to be analyzed")]

prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system", '''You are a helpful and friendly assistant designed to support users in improving their written stories, especially user stories or short narratives.

                When a user starts a conversation, always begin with a warm and welcoming message.

                If the user provides a story:

                Acknowledge and appreciate their effort.

                Improve the story by:

                Correcting grammar and spelling.
                Enhancing clarity and flow.
                Adding stylistic improvements while preserving the original meaning.
                Return the improved version to the user.

                After sharing the improved story:
                Ask the user if they are satisfied with the result.
                If the user expresses satisfaction with the improved story (e.g., says "yes", "looks good", "save", "I like it", etc.), 
                you must immediately call the save_story function with the user's name and the improved story. Do not say it is saved unless you have called the function.
                If the user is not satisfied, ask them what they would like to change or improve, and continue refining the story.

                If the user asks a general question, answer it clearly and helpfully.
                Whenever possible, relate your answer back to helping the user improve or refine their story.

                If the user hasn’t provided a story yet, gently encourage them to share one.
                Your goal is to make the user feel supported and confident in expressing their ideas clearly and creatively.''',
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

# Initialize agent
agent_executor = initialize_agent(
    tools=[save_story],
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


@router.post("/story/")
async def analyze_story(user_input: Annotated[UserInputStory, Body(...)]):
    try:
        input_messages = [HumanMessage(content=f"{user_input.name} says: {user_input.event}")]
        response = appl.invoke({"messages": input_messages}, config={"configurable": {"thread_id": "abc345"}})
        return {"response": response["messages"][-1].content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/story/")
async def get_story(query: Annotated[str, Query(description="Query to search for a story")]):
    """Endpoint to retrieve a user's story by name."""
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
                "story_id": item.get("story_id"),
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
                "story_id": doc.metadata.get("story_id"),
                "name": doc.metadata.get("name"),
                "content": doc.page_content
            } for doc in results
        ]

    bot_answer_response = bot_answer(response, query)
    return {"stories": bot_answer_response} if response else {"stories": "No stories found."}


def bot_answer(response: list[dict], query: str) -> str:
    if not response:
        return "No stories found."

    # Format the response into a string
    content_summary = "\n\n".join(
        f"Story ID: {item['story_id']}\nName: {item['name']}\nContent: {item['content']}"
        for item in response
    )

    prompt_text = (
        f"You are a helpful assistant that provides comprehensive answers based on the user's query.\n"
        f"Query: {query}\n\n"
        f"Content:\n{content_summary}\n\n"
        "Please provide a detailed and informative answer."
    )

    return model.invoke(prompt_text).content.strip()
