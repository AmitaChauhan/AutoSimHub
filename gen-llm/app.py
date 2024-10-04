import fastapi
from dotenv import load_dotenv
import os

load_dotenv()

app = fastapi.FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/generate")
def generate_text(prompt: str):
    return {"message": "Hello, World!"}