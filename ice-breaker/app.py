from typing import Annotated

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from ice_breaker import linkedin


class Body(BaseModel):
    name: str


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("person.html", {"request": request})


@app.post("/api/linkedin")
async def fetch_linkedin_data(name: Annotated[str, Form()]):
    print(f'name = {name}')
    profile_picture, person_info = linkedin(name)
    return {
        "summary": person_info.summary,
        "interests": person_info.topics_of_interest,
        "facts": person_info.facts,
        "ice_breakers": person_info.ice_breakers,
        "picture_url": profile_picture
    }
