import traceback
from distro import name
from fastapi import Depends
from sqlmodel import SQLModel, create_engine, Session, Field
from langchain_core.tools import tool
from langchain.docstore.document import Document
from config import embeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
import os
from supabase import create_client
from langchain.vectorstores import SupabaseVectorStore

ENVIRONMENT = os.getenv("ENVIRONMENT")
DATABASE_URL = os.getenv("DATABASE_URL")


class Connect:
    def __init__(self):
        self.engine = None
        self.supabase = None
        self.get_db()

    def get_db(self):
        if ENVIRONMENT == "production":
            supabase_url = os.getenv("VECTOR_DB_URL")
            supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            client = create_client(supabase_url, supabase_key)

            self.supabase = SupabaseVectorStore(
                client=client,
                embedding=embeddings,
                table_name="documents"
            )


        elif ENVIRONMENT == "local":
            self.engine = create_engine(DATABASE_URL)
            self.supabase = Chroma(persist_directory="vector_kanban_db", embedding_function=embeddings)


connection = Connect()

@tool
def save_story(name: str, content: str):
    """Immediately call this function to save the user's
    story to the database when they confirm they are satisfied. Do not say the story is saved unless this function is called."""
    try:
        store_vector(id, name, content)
    except Exception as e:
        print("Vector storage failed:", e)
        traceback.print_exc()

    return "Story saved successfully."

@tool
def save_event(name: str, content: str, organize_event: bool, category: str):
    """Immediately call this function to save the user's 
    event to the database when they confirm they are satisfied. Do not say the event is saved unless this function is called."""
    try:
        store_event_vector(id, name, content, category, organize_event)
    except Exception as e:
        print("Vector storage failed:", e)
        traceback.print_exc()
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

    environment = os.getenv("ENVIRONMENT", "production")

    if environment == "production":
        # Generate embedding using your embedding function
        embedding = embeddings.embed_query(text_to_embed)

        response =  connection.supabase.table("stories").insert({
            "id": str(story_id),
            "embedding": embedding,
            "name": name,
            "content": content
        }).execute()

        print(f"Uploaded vector to Supabase for story ID {story_id}: {response}")
    else:
        # Store locally in Chroma
        db = Chroma(persist_directory="vector_db", embedding_function=embeddings)
        db.add_texts(
            texts=[doc.page_content for doc in documents],
            metadatas=[doc.metadata for doc in documents],
            ids=[str(story_id)]
        )

        print(f"Stored vector locally for story ID {story_id}")

def store_event_vector(event_id: int, name: str, event: str, category: str, organize_event: bool):
    """
    Stores the vector representation of a user's event:
    - In Chroma if in development
    - In Supabase if in production
    """

    try:
        # Prepare text and metadata
        text_to_embed = f"Name: {name} Event: {event} Event ID: {event_id} Category: {category} Organize Event: {organize_event}"
        document = Document(
            page_content=text_to_embed,
            metadata={"event_id": event_id, "name": name, "organize_event": organize_event, "category": category}
        )
    except Exception as e:
        print("Error preparing event vector:", e)

    try:
        # Split text if needed
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents([document])

    except Exception as e:
        print("Error splitting event vector:", e)

    environment = os.getenv("ENVIRONMENT", "production")

    try:
        if environment == "production":
            # Generate embedding using your embedding function
            embedding = embeddings.embed_query(text_to_embed)

            response = connection.supabase.table("events").insert({
                "id": str(event_id),
                "embedding": embedding,
                "name": name,
                "content": event,
                "category": category,
                "organize_event": organize_event
            }).execute()

            print(f"Uploaded vector to Supabase for event ID {event_id}: {response}")
        else:
            # Store locally in Chroma
            db = Chroma(persist_directory="vector_db", embedding_function=embeddings)

            db.add_texts(
                texts=[doc.page_content for doc in documents],
                metadatas=[doc.metadata for doc in documents],
                ids=[str(event_id)]
            )

            print(f"Stored vector locally for event ID {event_id}")
    except Exception as e:
        print("Error storing event vector:", e)
        traceback.print_exc()

if __name__ == "__main__":
    SQLModel.metadata.create_all(engine)
    print("Database tables created.")