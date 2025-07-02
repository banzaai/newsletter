import os
from fastapi import FastAPI
from story import api as story
from event import api as event
import uvicorn

# FastAPI app
app = FastAPI()

app.include_router(story.router, prefix="/api", tags=["story"])
app.include_router(event.router, prefix="/api", tags=["event"])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
