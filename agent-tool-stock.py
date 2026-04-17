import os
import json
import time
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY')  # add this to your .env file

# ─── TOOL DEFINITION (unchanged) ─────────────────────────────────────────────
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Returns the latest stock price and basic info for a given ticker symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol, e.g. AAPL, TSLA, MSFT"
                    }
                },
                "required": ["ticker"]
            }
        }
    }
]

# ─── REAL FUNCTION (now uses Alpha Vantage) ───────────────────────────────────
def get_stock_price(ticker: str) -> str:
    url = "https://www.alphavantage.co/query"

    # 1. GLOBAL QUOTE — current price
    quote_params = {
        "function": "GLOBAL_QUOTE",
        "symbol": ticker,
        "apikey": ALPHA_VANTAGE_KEY
    }
    quote_resp = requests.get(url, params=quote_params).json()
    print("GLOBAL QUOTE RESPONSE:", quote_resp)

    quote = quote_resp.get("Global Quote", {})
    if not quote:
        return json.dumps({"error": f"No price data found for '{ticker}'"})

    # Alpha Vantage free tier requires spacing requests
    time.sleep(1.2)

    # 2. OVERVIEW — 52-week high/low
    overview_params = {
        "function": "OVERVIEW",
        "symbol": ticker,
        "apikey": ALPHA_VANTAGE_KEY
    }
    overview_resp = requests.get(url, params=overview_params).json()
    print("OVERVIEW RESPONSE:", overview_resp)

    week_high = overview_resp.get("52WeekHigh")
    week_low = overview_resp.get("52WeekLow")

    result = {
        "ticker": ticker.upper(),
        "current_price": quote.get("05. price"),
        "day_high": quote.get("03. high"),
        "day_low": quote.get("04. low"),
        "prev_close": quote.get("08. previous close"),
        "change_pct": quote.get("10. change percent"),
        "volume": quote.get("06. volume"),
        "52_week_high": week_high,
        "52_week_low": week_low,
    }

    return json.dumps(result)



# ─── ROUND 1 (unchanged) ──────────────────────────────────────────────────────
messages = [
    {
        "role": "user",
        "content": "What's the current stock price of Apple and Tesla? Is it close to its 52 weeks low?"
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools
)

#1 Response from the First LLM Call
assistant_message = response.choices[0].message

print("Raw response from LLM", assistant_message)
print("Model wants to call:", assistant_message.tool_calls[0].function.name)
print("With args:", assistant_message.tool_calls[0].function.arguments)

# ─── BRIDGE (unchanged) ───────────────────────────────────────────────────────
if assistant_message.tool_calls:
   
    # add the response from the First Call
    messages.append(assistant_message)

    for tool_call in assistant_message.tool_calls:
        
        args = json.loads(tool_call.function.arguments)
        print(f"\nFetching live data for: {args['ticker']}...")

        result = get_stock_price(**args)
        print("Live data returned:", result)

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,   # ← each result paired to its call
            "content": result
        })

        time.sleep(1) 

# ─── ROUND 2 (unchanged) ──────────────────────────────────────────────────────
final_response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools
)

print("\n─── Final Answer ───")
print(final_response.choices[0].message.content)