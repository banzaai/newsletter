from fastapi import Body, HTTPException, APIRouter, Query
from pydantic import BaseModel, Field
from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain.agents import initialize_agent, AgentType
from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
        - Purpose.
        - Date and time
        - Location
        - Number of attendees
        - Any special requirements
        3. Continue asking questions until you clearly understand the event and can categorize it (e.g., birthday, meeting, wedding, etc.).

        Once you have enough information to define the event:
        - Confirm the details with the user to ensure accuracy.
        - Ask if they would like to be the organizer of the event. Save this as a boolean variable called `organize_event`.

        Then, call the `save_event` function with the following parameters:
        - `name`: the user's name → `{{user_input.name}}`
        - `content`: the event description you generated
        - `organize_event`: the boolean value from the user's response
        - `category`: the event type based on the user's description

        Finally, confirm to the user that the event has been successfully created.

        If the user asks a general question (not related to creating an event), respond clearly and helpfully.

        At any point, if the user provides an event description, include that description in your response.
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
    
@router.get('/event/')
async def get_event(query: Annotated[str, Query(description="Query to search for an event")]):
    """Endpoint to retrieve the event creation workflow."""
    try:
        db = Chroma(persist_directory="vector_db", embedding_function=embeddings)

        result = db.similarity_search(query)

        response = [
            {
                "event_id": doc.metadata.get("event_id"),
                "name": doc.metadata.get("name"),
                "content": doc.page_content
            } for doc in result
        ]

        return {"events": response[0]['content']} if response else {"message": "No events found."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))