from dotenv import load_dotenv
import os
import fastapi
import interpreter

load_dotenv()

app = fastapi.FastAPI()

@app.get("/")
def root():
    return {"message": "Hello, World!"}

@app.get("/generate")
def generate_text(prompt: str):
    return {"message": "Hello, World!"}