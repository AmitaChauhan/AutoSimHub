import fastapi
from dotenv import load_dotenv
import os
import shelve
import sim_agent as SA
import openai

DB_PATH = "db.shelf"
load_dotenv()

db = shelve.open(DB_PATH)

app = fastapi.FastAPI()

client = openai.OpenAI()

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}

@app.get("/conversation")
def generate_text(conversation_id: str, prompt: str):
    conversation = db.get(conversation_id, [])
    conversation.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=conversation,
        tools=SA.TOOL_DESCRIPTIONS,
        tool_choice="auto",
    )

    if response.choices[0].message.tool_calls:
        for tool_call in response.choices[0].message.tool_calls:
            if tool_call.function.name == "get_simulation":
                tool_response = SA.get_simulation()
            elif tool_call.function.name == "run_simulation":
                tool_response = SA.run_simulation(
                    tool_call.function.arguments
                )
            elif tool_call.function.name == "update_simulation":
                tool_response = SA.update_simulation(
                    tool_call.function.arguments
                )
            else:
                raise ValueError(f"Unknown tool: {tool_call.function.name}")

            conversation.append({
                "role": "tool",
                "content": tool_response
            })

    conversation.append({"role": "assistant", "content": response.choices[0].message})
    db[conversation_id] = conversation
    return {"message": response}
