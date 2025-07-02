from fastapi import Body, HTTPException, Query, APIRouter
from pydantic import BaseModel, Field
from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain.agents import initialize_agent, AgentType
from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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

                If the user confirms they are satisfied (e.g., says "yes", "looks good", "I like it", etc.), call the save_story function to store the story in the database.

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

# FastAPI endpoint
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

    # Load the persisted vector store
    db = Chroma(persist_directory="vector_db", embedding_function=embeddings)

    # Perform a similarity search
    results = db.similarity_search(query, k=1)  # k = number of results

    response = [
        {
            "story_id": doc.metadata.get("story_id"),
            "name": doc.metadata.get("name"),
            "content": doc.page_content
        } for doc in results
    ]

    return {"stories": response[0]['content']} if response else {"stories": "No stories found."}
