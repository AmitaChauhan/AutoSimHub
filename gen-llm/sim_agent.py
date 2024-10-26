from interpreter import interpreter
import os
import dotenv

dotenv.load_dotenv()

SIM_FILE = "sim.py"
DISPLAY = True

# interpreter.offline = True # Disables online features like Open Procedures
# interpreter.llm.api_base = "http://localhost:1234/v1" # Point this at any OpenAI compatible server

interpreter.llm.model = "openai/gpt-4o"  # Tells OI to send messages in OpenAI's format
interpreter.llm.api_key = os.getenv("OPENAI_API_KEY")
interpreter.auto_run = True

TOOL_DESCRIPTIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_simulation",
            "description": "Get a description of the current simulation",
            "parameters": {
                "type": "object",
                "properties": {
                    "special_request": {
                        "type": "string",
                        "description": "Any aspect of the description of the simulation to describe in more detail.",
                    }
                },
                "required": [],
            },
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
                        "description": "All statistics to return from the simulation",
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


def LLM_GET_SIMULATION_PROMPT(special_request: str = ""):
    if special_request:
        special_request = f"Special request: {special_request}"
    else:
        special_request = ""

    return f"Give a detailed description of the simulation scenario described in {SIM_FILE}. Include all the parameters which can be manipulated while running the simulation. Abstract away the implementation details. Do not include details about how to run the simulation. {special_request}"


def LLM_RUN_SIMULATION_PROMPT(sim_params: str, stats: str):
    return f"Run the simulation by importing {SIM_FILE}, instantiating the `Simulation` object, and calling the `run` method with the given parameters: {sim_params} and return the following statistics: {stats}. If the simulation cannot be run for the given parameters, report it as an error. "


def LLM_UPDATE_SIMULATION_PROMPT(new_description: str):
    return f"Update the simulation scenario in {SIM_FILE} with the following description: {new_description}. Update `{SIM_FILE}` in place and return a JSON description of the new parameters. It should be possible to execute the simulation just like before."


def get_simulation(special_request: str = ""):
    return interpreter.chat(LLM_GET_SIMULATION_PROMPT(special_request), display=DISPLAY)


def run_simulation(
    sim_params: str = "default parameters", stats: str = "overall throughput"
):
    return interpreter.chat(
        LLM_RUN_SIMULATION_PROMPT(sim_params, stats), display=DISPLAY
    )


def update_simulation(new_description: str):
    return interpreter.chat(
        LLM_UPDATE_SIMULATION_PROMPT(new_description), display=DISPLAY
    )

