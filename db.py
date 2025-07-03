from distro import name
from fastapi import Depends
from sqlmodel import SQLModel, create_engine, Session, Field
from sqlalchemy.ext.asyncio import create_async_engine
from langchain_core.tools import tool
from langchain.docstore.document import Document
from config import embeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
import os



ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
DATABASE_URL = os.getenv("DATABASE_URL")

if ENVIRONMENT == "local":
    engine = create_engine(DATABASE_URL, echo=True)
else:
    # Use async engine for PostgreSQL on Render
    engine = create_async_engine(DATABASE_URL, echo=True)


class Story(SQLModel, table=True):
    """Model for storing user stories."""
    id: int =Field(default=None, primary_key=True)
    name: str = Field(default=None)
    story: str = Field(default=None)

class Event(SQLModel, table=True):
    """Model for storing user events."""
    id: int = Field(default=None, primary_key=True)
    name: str = Field(default=None)
    event: str = Field(default=None)
    organize_event: bool = Field(default=False)
    category: str = Field(default=None)


@tool
def save_story(name: str, content: str):
    """Immediately call this function to save the user's 
    story to the database when they confirm they are satisfied. Do not say the story is saved unless this function is called."""
    with Session(engine) as session:
        story = Story(name=name, story=content)
        session.add(story)
        session.commit()
        store_vector(story.id, story.name, story.story) 

        return "Story saved successfully."

@tool
def save_event(name: str, content: str, organize_event: bool, category: str):
    """Immediately call this function to save the user's 
    event to the database when they confirm they are satisfied. Do not say the event is saved unless this function is called."""
    with Session(engine) as session:
        event = Event(name=name, event=content, organize_event=organize_event, category=category)
        session.add(event)
        session.commit()
        store_event_vector(event.id, event.name, event.event, event.category, event.organize_event)

    return "Event saved successfully."


def store_vector(story_id: int, name: str, content: str):
    '''
        Stores the vector representation of a user's story in the vector database.
        '''
    text_to_embed = f"Title: {name} Content: {content} Story ID: {story_id}"
    document = Document(
    page_content=text_to_embed,
    metadata={"story_id": story_id, "name": name}
)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents([document])

    # Load existing vector store or create if not exists
    db = Chroma(persist_directory="vector_db", embedding_function=embeddings)

    # Add new documents
    db.add_documents(documents)

    print(f"Storing vector for story ID {story_id}, Name: {name}, Content: {content}")

def store_event_vector(event_id: int, name: str, event: str, category: str, organize_event: bool):
    '''
        Stores the vector representation of a user's event in the vector database.
        '''
    text_to_embed = f"Name: {name} Event: {event} Event ID: {event_id} Category: {category} Organize Event: {organize_event}"
    document = Document(
    page_content=text_to_embed,
    metadata={"event_id": event_id, "name": name, "organize_event": organize_event, "category": category}
)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents([document])

    # Load existing vector store or create if not exists
    db = Chroma(persist_directory="vector_db", embedding_function=embeddings)

    # Add new documents
    db.add_documents(documents)

    print(f"Storing vector for event ID {event_id}, Name: {name}, Event: {event} Category: {category}")


if __name__ == "__main__":
    SQLModel.metadata.create_all(engine)
    print("Database tables created.")