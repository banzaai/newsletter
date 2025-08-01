from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from story import api as story
from event import api as event
from kanban import api as kanban
from fastapi import Request
from fastapi.responses import HTMLResponse

# FastAPI app
app = FastAPI()


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


app.include_router(story.router, prefix="/api", tags=["story"])
app.include_router(event.router, prefix="/api", tags=["event"])
app.include_router(kanban.router, prefix="/api", tags=["kanban"])

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/admin", response_class=HTMLResponse)
async def get_admin_page(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})


@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/kanban", response_class=HTMLResponse)
async def kanban(request: Request):
    return templates.TemplateResponse("kanban.html", {"request": request})

@app.get("/query_kanban", response_class=HTMLResponse)
async def query_kanban(request: Request):
    return templates.TemplateResponse("query_kanban.html", {"request": request})


# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8000))
#     uvicorn.run("main:app", host="0.0.0.0", port=port)
