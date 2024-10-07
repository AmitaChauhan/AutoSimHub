import gradio as gr
import json
import sim_agent as SA
import openai
import dotenv
import os
import logging

dotenv.load_dotenv()

client = openai.OpenAI()

messages = []
MAX_ITERATIONS = 5


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def interact_with_gen_llm(iterations: int = 0):
    logging.info(
        "Iterations: %s, # messages: %s",
        iterations,
        len(messages),
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=SA.TOOL_DESCRIPTIONS,
        tool_choice="auto",
    )
    logging.info("Response received: %s", response)

    if response.choices[0].message.tool_calls:
        messages.append(
            {
                "role": "assistant",
                "tool_calls": response.choices[0].message.tool_calls,
            }
        )
        for tool_call in response.choices[0].message.tool_calls:
            logging.info("Tool call detected: %s", tool_call.function.name)
            tool_call_id = tool_call.id
            try:
                if tool_call.function.name == "get_simulation":
                    tool_response = SA.get_simulation(
                        **json.loads(tool_call.function.arguments)
                    )
                elif tool_call.function.name == "run_simulation":
                    tool_response = SA.run_simulation(
                        **json.loads(tool_call.function.arguments)
                    )
                elif tool_call.function.name == "update_simulation":
                    tool_response = SA.update_simulation(
                        **json.loads(tool_call.function.arguments)
                    )
                else:
                    logging.error("Unknown tool: %s", tool_call.function.name)
                    raise ValueError(f"Unknown tool: {tool_call.function.name}")
            except Exception as e:
                logging.error("Error in tool call: %s", e)
                tool_response = [
                    {
                        "content": f"Error in tool call: {e}",
                        "role": "tool",
                    }
                ]

            logging.info("Tool response: %s", len(tool_response))

            tool_response_messages = []
            for idx, resp in enumerate(tool_response):
                if idx < len(tool_response) - 1:
                    tool_response_messages.append(
                        "\n".join(["> " + x for x in resp["content"].split("\n")])
                        + "\n\n"
                    )
                else:
                    tool_response_messages.append(resp["content"])

            tool_response_combined = "\n".join(tool_response_messages)
            logging.info("Tool response: %s", tool_response_combined)

            messages.append(
                {
                    "role": "tool",
                    "content": tool_response_combined,
                    "tool_call_id": tool_call_id,
                }
            )

        logging.info("Calling LLM with updated messages")
        if iterations < MAX_ITERATIONS:
            return interact_with_gen_llm(iterations=iterations + 1)
        else:
            logging.warning("Max iterations reached")
    else:
        messages.append(
            {"role": "assistant", "content": response.choices[0].message.content}
        )
        logging.info("Assistant response: %s", response.choices[0].message.content)

    return response.choices[0].message.content


def chat_interface(prompt, history):
    messages.append({"role": "user", "content": prompt})
    return interact_with_gen_llm()


def chat():
    logging.info("Chat interface initialized")
    return gr.ChatInterface(
        fn=chat_interface,
        title="Gen LLM for Sim Agent",
        description="This is a chat interface for the Gen LLM for Sim Agent. You can ask the agent to perform various tasks, such as running simulations, updating simulations, or getting simulations. The agent will use the tools provided to perform these tasks. You can also ask the agent to explain the code and the simulations.",
    )


if __name__ == "__main__":
    chat().launch()
