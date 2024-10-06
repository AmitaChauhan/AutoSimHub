from interpreter import interpreter
import os
import dotenv

dotenv.load_dotenv()

SIM_FILE = "sim.py"

# interpreter.offline = True # Disables online features like Open Procedures
# interpreter.llm.api_base = "http://localhost:1234/v1" # Point this at any OpenAI compatible server

interpreter.llm.model = "openai/gpt-4o"  # Tells OI to send messages in OpenAI's format
interpreter.llm.api_key = os.getenv(
    "OPENAI_API_KEY"
)  # LiteLLM, which we use to talk to LM Studio, requires this
interpreter.auto_run = True

TOOL_DESCRIPTIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_simulation",
            "description": "Get a description of the current simulation",
            "parameters": {},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_simulation",
            "description": "Run the simulation with the given parameters",
            "parameters": {
                "type": "object",
                "properties": {
                    "sim_params": {
                        "type": "object",
                        "description": "The parameters for the simulation",
                    },
                    "stats": {
                        "type": "string",
                        "description": "The statistics to return from the simulation",
                    },
                },
                "required": ["sim_params", "stats"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_simulation",
            "description": "Update the simulation for a new scenario",
            "parameters": {
                "type": "object",
                "properties": {
                    "new_description": {
                        "type": "string",
                        "description": "The new description for the simulation",
                    }
                },
                "required": ["new_description"],
            },
        },
    },
]


def LLM_GET_SIMULATION_PROMPT():
    return f"Give a detailed description of the simulation scenario described in {SIM_FILE}. Include all the parameters which can be manipulated while running the simulation. Abstract away the implementation details. Do not include details about how to run the simulation."


def LLM_RUN_SIMULATION_PROMPT(sim_params: str, stats: str):
    return f"Run the simulation with the following parameters: {sim_params} and return the following statistics: {stats}."


def LLM_UPDATE_SIMULATION_PROMPT(new_description: str):
    return f"Update the simulation scenario in {SIM_FILE} with the following description: {new_description}. Update `{SIM_FILE}` in place and return a JSON description of the new parameters."


def get_simulation():
    return interpreter.chat(LLM_GET_SIMULATION_PROMPT())


def run_simulation(
    sim_params: str = "default parameters", stats: str = "overall throughput"
):
    return interpreter.chat(LLM_RUN_SIMULATION_PROMPT(sim_params, stats))


def update_simulation(new_description: str):
    return interpreter.chat(LLM_UPDATE_SIMULATION_PROMPT(new_description))
