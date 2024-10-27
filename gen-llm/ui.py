import gradio as gr
import json
from openai import OpenAI

import dotenv
import base64
import os
import logging
import shutil
from interpreter import interpreter

dotenv.load_dotenv()

# Initialize OpenAI client

MAX_ITERATIONS = 5

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


SIM_FILE = "sim.py"
SIM_FILE_BACKUP = "sim_backup.py"
DISPLAY = True

interpreter.llm.model = "openai/gpt-4o"  # Tells OI to send messages in OpenAI's format
interpreter.llm.api_key = os.getenv("OPENAI_API_KEY")
interpreter.auto_run = True

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TOOL_DESCRIPTIONS = [
    {
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
    {
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
    {
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
]


def LLM_GET_SIMULATION_PROMPT(special_request: str = ""):
    if special_request:
        special_request = f"Special request: {special_request}"
    else:
        special_request = ""

    return f"Give a detailed description of the simulation scenario described in {SIM_FILE}. Include all the parameters which can be manipulated while running the simulation. Abstract away the implementation details. Do not include details about how to run the simulation. {special_request}"


def LLM_RUN_SIMULATION_PROMPT(sim_params: str, stats: str):
    return f"Run the simulation by importing {SIM_FILE}, instantiating the `Simulation` object, and calling the `run` method with the given parameters: {sim_params} and return the following statistics: {stats}. If the simulation cannot be run for the given parameters, report it as an error. This is a non-interactive session; DO NOT ask the user for any input."


def LLM_UPDATE_SIMULATION_PROMPT(new_description: str):
    return f"Update the simulation scenario in {SIM_FILE} with the following description: {new_description}. Update `{SIM_FILE}` in place, but keep a backup, and return a JSON description of the new parameters. It should be possible to execute the simulation. Verify that the updated simulation can be imported successfully and executed. This is a non-interactive session; DO NOT ask the user for any input."


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


def reset_simulation():
    shutil.copy(SIM_FILE_BACKUP, SIM_FILE)
    return get_simulation()


def interact_with_gen_llm(messages, iterations=0):
    logging.info(
        "Iterations: %s, # messages: %s",
        iterations,
        len(messages),
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        functions=TOOL_DESCRIPTIONS,
        function_call="auto",
    )
    logging.info("Response received: %s", response)

    message = response.choices[0].message

    if message.function_call:
        # Append the assistant's message indicating the function call
        messages.append(
            {
                "role": message.role,
                "content": None,
                "function_call": {
                    "name": message.function_call.name,
                    "arguments": message.function_call.arguments,
                },
            }
        )
        tool_name = message.function_call.name
        tool_args = json.loads(message.function_call.arguments)
        logging.info("Function call detected: %s", tool_name)
        try:
            if tool_name == "get_simulation":
                tool_response = get_simulation(**tool_args)
            elif tool_name == "run_simulation":
                tool_response = run_simulation(**tool_args)
            elif tool_name == "update_simulation":
                tool_response = update_simulation(**tool_args)
            else:
                logging.error("Unknown tool: %s", tool_name)
                raise ValueError(f"Unknown tool: {tool_name}")
        except Exception as e:
            logging.error("Error in tool call: %s", e)
            tool_response_content = f"Error in tool call: {e}"
        else:
            # Process tool response
            tool_response_content = "\n".join([r["content"] for r in tool_response])

        logging.info("Tool response: %s", tool_response_content)

        # Append tool response to messages
        messages.append(
            {"role": "function", "name": tool_name, "content": tool_response_content}
        )

        logging.info("Calling LLM with updated messages")
        if iterations < MAX_ITERATIONS:
            return interact_with_gen_llm(messages, iterations=iterations + 1)
        else:
            logging.warning("Max iterations reached")
            return None
    else:
        llm_content = message.content.strip()
        if llm_content == "":
            logging.info("Assistant chose not to respond.")
            return None
        else:
            messages.append({"role": message.role, "content": message.content})
            logging.info("Assistant response: %s", llm_content)
            return llm_content


def chat():
    logging.info("Chat interface initialized")
    autosim_logo = base64.b64encode(open("./AutoSimHub.png", "rb").read()).decode()

    with gr.Blocks(fill_height=True, theme=gr.themes.Soft()) as demo:

        with gr.Row(equal_height=True):
            gr.Markdown(
                f"# <img src='data:image/png;base64,{autosim_logo}' width='96px' style='vertical-align: middle; display: inline-block;' /> AutoSimHub"
            )

        gr.Markdown(
            "This is a chat interface for AutoSimHub, a simulation tool for production line modelling. You can ask the agent to perform various tasks, such as running simulations, updating simulations, or getting simulations. The agent will use the tools provided to perform these tasks. You can also ask the agent to explain the code and the simulations."
        )

        chatbot = gr.Chatbot(
            type="messages",
            show_copy_button=True,
            layout="panel",
            scale=1,
            show_copy_all_button=True,
        )
        messages_state = gr.State([])  # To store the messages in OpenAI format
        chat_history_state = gr.State(
            []
        )  # To store the chat history displayed to the user

        with gr.Row(equal_height=True):
            role_dropdown = gr.Dropdown(
                choices=[
                    "New Product Introduction Department",
                    "Design Department",
                    "Planning Department",
                    "Production Department",
                    "Quality Control Department",
                    "Materials Department" "Other",
                ],
                value="Other",
                show_label=False,
                container=False,
                scale=1,
            )

            message_input = gr.Textbox(
                placeholder="Type your message here...",
                lines=1,
                show_label=False,
                container=False,
                scale=6,
            )

            submit_button = gr.Button("Send", variant="primary", scale=1)

        def submit_message(role, message, chat_history, messages):
            # Format and add the user's message to the chat history and messages
            formatted_message = f"**[{role}]:** {message}"
            chat_history.append({"role": "user", "content": formatted_message})
            messages.append({"role": "user", "content": formatted_message})

            # Call LLM with all messages, but only respond if necessary
            llm_response = interact_with_gen_llm(messages)

            if llm_response and llm_response.translate(
                str.maketrans("", "", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ \t")
            ):
                # Format the LLM response and update chat history
                formatted_llm_response = f"{llm_response}"
                chat_history.append(
                    {"role": "assistant", "content": formatted_llm_response}
                )
                messages.append({"role": "assistant", "content": llm_response})

            # Clear the input box for the next message
            return chat_history, messages, gr.update(value="", interactive=True)

        # Button click to submit the message
        submit_button.click(
            submit_message,
            inputs=[role_dropdown, message_input, chat_history_state, messages_state],
            outputs=[chatbot, messages_state, message_input],
        )

        # Initialize chat with the system prompt and an initial message from AutoSim
        def initialize_chat(chat_history, messages):
            system_prompt = (
                "You are AutoSim, a helpful assistant that assists users in running simulations and answering questions. "
                "You should only respond if one of the following is true:\n"
                "1. The user 'tags' you with '@AutoSim' (case insensitive).\n"
                "2. There is some ambiguity which you can resolve between the users with different roles.\n"
                "3. There is discussion about a question which can be answered by running a simulation.\n"
                "In all other cases, you should not respond, and instead wait for the users to continue their discussions.\n"
                "When you respond, you should always format your messages as '**[AutoSim]:** <message>'.\n"
                "If no response is needed, return an empty string.\n\n"
                "In the course of answering any query or question, always describe all the simulations which were run, why they were run, and the results obtained from them before summarizing the results and answering the user."
            )
            messages.append({"role": "system", "content": system_prompt})

            initial_message = """You may begin your discussion. Tag me with ‘@AutoSim’ to ask questions directly, request simulations, or adjust the simulation scenario. I'll also step in to resolve ambiguities or respond to questions that can be answered by a simulation. You may also request a reset to the original simulation state.
            """
            chat_history.append(
                {"role": "assistant", "content": f"**[AutoSim]:** {initial_message}"}
            )
            messages.append({"role": "assistant", "content": initial_message})

            return chat_history, messages

        # Load initial chat history and messages on start
        demo.load(
            initialize_chat,
            inputs=[chat_history_state, messages_state],
            outputs=[chatbot, messages_state],
        )

    return demo


if __name__ == "__main__":
    chat().launch(auth=("admin", "autosimhub"))
