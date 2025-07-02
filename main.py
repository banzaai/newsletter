from fastapi import FastAPI
from story import api as story
from event import api as event

# FastAPI app
app = FastAPI()

app.include_router(story.router, prefix="/api", tags=["story"])
app.include_router(event.router, prefix="/api", tags=["event"])