import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
google_api_key = os.getenv('GOOGLE_API_KEY')

print(f"Google API Key exists and begins {google_api_key[:2]}")

client = OpenAI(
    api_key=google_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Define a simple tool
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_greeting",
            "description": "Returns a greeting for the given name",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The name to greet"}
                },
                "required": ["name"]
            }
        }
    }
]

def get_greeting(name: str) -> str:
    return f"Hello, {name}!"

# First call — model decides to use the tool
response = client.chat.completions.create(
    model="models/gemini-2.0-flash",
    messages=[{"role": "user", "content": "Say hello to Alice using the greeting tool."}],
    tools=tools
)


# Handle the tool call
tool_call = response.choices[0].message.tool_calls[0]
args = json.loads(tool_call.function.arguments)
result = get_greeting(**args)

# Second call — send tool result back to model
final_response = client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[
        {"role": "user", "content": "Say hello to Alice using the greeting tool."},
        response.choices[0].message,
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        }
    ],
    tools=tools
)

print(final_response.choices[0].message.content)