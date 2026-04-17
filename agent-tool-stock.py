import os
import json
import time
import requests
from dotenv import load_dotenv
from openai import OpenAI
import yfinance as yf

load_dotenv(override=True)
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY')  # add this to your .env file

# ─── TOOL DEFINITION (unchanged) ─────────────────────────────────────────────
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Returns the latest stock price and basic info for a list of ticker symbols",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of stock ticker symbols, e.g. ['AAPL', 'TSLA', 'MSFT']"
                    }
                },
                "required": ["tickers"]
            }
        }
    }
]

def get_sp500_list():
    return yf.tickers_sp500()


# ─── REAL FUNCTION (now uses Alpha Vantage) ───────────────────────────────────
def get_stock_price(tickers: list[str]) -> str:
    results = []

    for ticker in tickers:
        stock = yf.Ticker(ticker)

        hist = stock.history(period="1y")
        if hist.empty:
            results.append({"ticker": ticker.upper(), "error": "No data found"})
            continue

        current_price = hist["Close"].iloc[-1]
        day_high = hist["High"].iloc[-1]
        day_low = hist["Low"].iloc[-1]
        prev_close = hist["Close"].iloc[-2] if len(hist) > 1 else None
        volume = hist["Volume"].iloc[-1]
        week_high = hist["High"].max()
        week_low = hist["Low"].min()

        results.append({
            "ticker": ticker.upper(),
            "current_price": float(current_price),
            "day_high": float(day_high),
            "day_low": float(day_low),
            "prev_close": float(prev_close) if prev_close else None,
            "change_pct": None,
            "volume": int(volume),
            "52_week_high": float(week_high),
            "52_week_low": float(week_low),
        })

    return json.dumps(results)



# ─── ROUND 1 (unchanged) ──────────────────────────────────────────────────────
messages = [
    {
        "role": "user",
        #"content": "Consider the S&P 500 stocks whose 52 weeks low is close to current price."
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

if assistant_message.tool_calls:
    print("Model wants to call:", assistant_message.tool_calls[0].function.name)
    print("With args:", assistant_message.tool_calls[0].function.arguments)
else:
    print("Model did not request a tool call.")

# ─── BRIDGE (unchanged) ───────────────────────────────────────────────────────
if assistant_message.tool_calls:
    messages.append(assistant_message)

    for tool_call in assistant_message.tool_calls:
        args = json.loads(tool_call.function.arguments)

        # Updated for multi‑ticker support
        print(f"\nFetching live data for tickers: {args['tickers']}")

        result = get_stock_price(args['tickers'])
        print("Live data returned:", result)

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
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