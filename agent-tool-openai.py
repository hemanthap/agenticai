import os
import json
from dotenv import load_dotenv
from openai import OpenAI

#pip install python-dotenv openai --break-system-packages

load_dotenv(override=True)
groq_api_key = os.getenv('GROQ_API_KEY')

if groq_api_key is None:
    print("Error: GROQ_API_KEY environment variable is not set.")
    exit(1)

print(f"Groq API Key exists and begins {groq_api_key[:3]}")

client = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")

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

messages = [
    {"role": "user", "content": "Say Goodmorning to Alice using the greeting tool."}
]

# First call — model decides to use the tool
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=messages,
    tools=tools
)

print("Model response:", response.choices[0].message)

# Handle the tool call
# This is plain Python — no AI involved here. You're just calling your own function.
if response.choices[0].message.tool_calls:
    
    tool_call = response.choices[0].message.tool_calls[0]
   
    print("Tool call received:", tool_call)
   
    args = json.loads(tool_call.function.arguments)
   
    print("Parsed tool call arguments:", args)
   
    result = get_greeting(**args)

    print(f"Tool result: {result}")
   
    messages.append(response.choices[0].message)

    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

print("Updated messages with tool result:", messages)

# Second call — send tool result back to model
final_response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=messages,
)

print(f' final response: {final_response.choices[0].message.content}')

# Output
# OpenAI API Key exists and begins sk-
# Model response: ChatCompletionMessage(content=None, refusal=None, role='assistant', annotations=[], audio=None, function_call=None, tool_calls=[ChatCompletionMessageFunctionToolCall(id='call_AFeFmlyd7DB5qFaRFCLLDOz8', function=Function(arguments='{"name":"Alice"}', name='get_greeting'), type='function')])
# Tool call received: ChatCompletionMessageFunctionToolCall(id='call_AFeFmlyd7DB5qFaRFCLLDOz8', function=Function(arguments='{"name":"Alice"}', name='get_greeting'), type='function')
# Parsed tool call arguments: {'name': 'Alice'}
# Updated messages with tool result: [{'role': 'user', 'content': 'Say hello to Alice using the greeting tool.'}, ChatCompletionMessage(content=None, refusal=None, role='assistant', annotations=[], audio=None, function_call=None, tool_calls=[ChatCompletionMessageFunctionToolCall(id='call_AFeFmlyd7DB5qFaRFCLLDOz8', function=Function(arguments='{"name":"Alice"}', name='get_greeting'), type='function')]), {'role': 'tool', 'tool_call_id': 'call_AFeFmlyd7DB5qFaRFCLLDOz8', 'content': 'Hello, Alice!'}]
#  final response: Good morning, Alice.