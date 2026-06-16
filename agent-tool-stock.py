import os
import json
import time
import yfinance as yf
from dotenv import load_dotenv
from openai import OpenAI

# pip install yfinance --break-system-packages


load_dotenv(override=True)
groq_api_key = os.getenv('GROQ_API_KEY')

if groq_api_key is None:
    print("Error: GROQ_API_KEY environment variable is not set.")
    exit(1)

client = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")

os.makedirs("cache", exist_ok=True)

# ─── TOOL DEFINITIONS ─────────────────────────────────────────────────────────
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Returns stock price and info for a list of ticker symbols. Always pass ALL tickers in a single call, never call this function multiple times.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of ALL stock ticker symbols to fetch at once"
                    }
                },
                "required": ["tickers"]
            }
        }
    }
]

# ─── REAL FUNCTIONS ───────────────────────────────────────────────────────────

def get_stock_price(tickers: list[str]) -> str:
    results = []

    for ticker in tickers:
        clean_ticker = ticker.lstrip("$")
        cache_file = f"cache/{clean_ticker}.json"  # ← was ticker

        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                cached = json.load(f)
            if "prev_day_high" not in cached or cached.get("prev_day_high") is None:
                print(f"DEBUG: cache stale for {clean_ticker}, re-fetching")
                os.remove(cache_file)
            else:
                results.append(cached)
                print(f"DEBUG: loaded {clean_ticker} from cache")
                continue

        try:
            stock = yf.Ticker(clean_ticker)
            fast  = stock.fast_info
            hist  = stock.history(period="1y")

            if hist.empty:
                print(f"DEBUG: {clean_ticker} → no history data")
                results.append({"ticker": clean_ticker, "error": "No data returned"})  # ← was ticker
                continue

            current_price  = fast.last_price
            week_high      = fast.year_high
            week_low       = fast.year_low

            prev_day_high  = float(hist["High"].iloc[-2])  if len(hist) > 1 else None
            prev_day_low   = float(hist["Low"].iloc[-2])   if len(hist) > 1 else None
            prev_close     = float(hist["Close"].iloc[-2]) if len(hist) > 1 else None

            volume         = int(hist["Volume"].iloc[-1])
            day_high       = float(hist["High"].iloc[-1])
            day_low        = float(hist["Low"].iloc[-1])

            pct_above_52w_low  = round((current_price - week_low)  / week_low  * 100, 2)
            pct_below_52w_high = round((week_high - current_price)  / week_high * 100, 2)
            change_pct         = round((current_price - prev_close) / prev_close * 100, 2) if prev_close else None

            result = {
                "ticker":             clean_ticker,  # ← was ticker
                "current_price":      round(current_price, 2),
                "day_high":           round(day_high, 2),
                "day_low":            round(day_low, 2),
                "prev_day_high":      round(prev_day_high, 2) if prev_day_high else None,
                "prev_day_low":       round(prev_day_low,  2) if prev_day_low  else None,
                "prev_close":         round(prev_close,    2) if prev_close    else None,
                "change_pct":         change_pct,
                "volume":             volume,
                "52_week_high":       round(week_high, 2),
                "52_week_low":        round(week_low,  2),
                "pct_above_52w_low":  pct_above_52w_low,
                "pct_below_52w_high": pct_below_52w_high,
            }

            with open(cache_file, "w") as f:
                json.dump(result, f)

            results.append(result)

        except Exception as e:
            print(f"DEBUG: {clean_ticker} → EXCEPTION: {e}")
            results.append({"ticker": clean_ticker, "error": str(e)})  # ← was ticker

    return json.dumps(results)


# ─── BATCHED WRAPPER WITH PYTHON-SIDE FILTERING ───────────────────────────────

def get_stock_price_batched(tickers: list[str], batch_size: int = 25, pct_threshold: float = 5.0) -> str:
    all_results = []
    near_lows   = []
    seen        = set()

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"\nDEBUG: fetching batch {i//batch_size + 1} — tickers {i+1} to {i+len(batch)}")
        batch_results = json.loads(get_stock_price(batch))

        for r in batch_results:
            ticker = r.get("ticker")
            if ticker in seen:
                print(f"DEBUG: skipping duplicate {ticker}")
                continue
            seen.add(ticker)
            all_results.append(r)

            if "error" not in r and r.get("pct_above_52w_low", 999) <= pct_threshold:
                near_lows.append(r)
                print(f"✅ NEAR 52W LOW: {ticker} → {r['pct_above_52w_low']}% above low")

    near_lows.sort(key=lambda x: x["pct_above_52w_low"])

    print(f"\n─── Python Filter: {len(near_lows)} stocks within {pct_threshold}% of 52-week low ───")
    print(f"{'Ticker':<8} {'Current':>10} {'52W Low':>10} {'52W High':>10} {'%AboveLow':>10} {'PrevHigh':>10} {'PrevLow':>10}")
    print("-" * 75)
    
    for s in near_lows:
        print(
            f"{s['ticker']:<8}"
            f"  ${s['current_price']:>8.2f}"
            f"  ${s['52_week_low']:>8.2f}"
            f"  ${s['52_week_high']:>8.2f}"
            f"  {s['pct_above_52w_low']:>8.2f}%"
            f"  ${s.get('prev_day_high') or 0:>8.2f}"
            f"  ${s.get('prev_day_low')  or 0:>8.2f}"
        )

    return json.dumps({
        "filtered_stocks": near_lows,
        "total_analyzed":  len(all_results),
        "threshold_pct":   pct_threshold,
        "count_near_low":  len(near_lows)
    })


# ─── LOAD S&P 500 TICKERS ─────────────────────────────────────────────────────

with open("sp500.json") as f:
    sp500_tickers = json.load(f)

print(f"Loaded {len(sp500_tickers)} S&P 500 tickers from sp500.json")

# ─── INITIAL MESSAGES ────────────────────────────────────────────────────────

messages = [
    {
        "role": "system",
        "content": (
            "You are a financial analysis assistant. "
            "You will receive pre-filtered and pre-sorted stock data from the tool. "
            "Present the filtered_stocks as a clean markdown table with these exact columns:\n"
            "Ticker | Current Price | 52-Week Low | 52-Week High | % Above Low | Prev Day High | Prev Day Low\n"
            "The data is already sorted by % Above Low ascending (closest to 52w low first). "
            "After the table, add a short analysis of why these stocks near 52-week lows "
            "may be interesting to investors. "
            "Do NOT call any tools again after receiving results."
        )
    },
    {
        "role": "user",
        "content": (
            f"Here are the S&P 500 tickers to analyze: {json.dumps(sp500_tickers)}\n\n"
            "Call get_stock_price() with ALL the tickers above. "
            "The tool will return only stocks within 5% of their 52-week low, already sorted. "
            "Present them as a markdown table followed by analysis."
        )
    }
]

# ─── AGENT LOOP ───────────────────────────────────────────────────────────────

first_turn = True

while True:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "get_stock_price"}} if first_turn else "auto"
    )

    assistant_message = response.choices[0].message
    first_turn = False

    if assistant_message.tool_calls:
        print(f"\nDEBUG: LLM requesting tool → {assistant_message.tool_calls[0].function.name}")
    else:
        print("\nDEBUG: LLM returning final answer (no tool calls)")

    if not assistant_message.tool_calls:
        print("\n─── Final Answer ───")
        print(assistant_message.content)
        break

    messages.append(assistant_message)

    for tool_call in assistant_message.tool_calls:
        tool_name = tool_call.function.name

        if tool_name == "get_stock_price":
            print(f"\n[TOOL] get_stock_price called — fetching {len(sp500_tickers)} tickers from sp500.json")
            result = get_stock_price_batched(sp500_tickers, batch_size=25, pct_threshold=5.0)
        else:
            print(f"\n[TOOL] Unknown tool: {tool_name}")
            result = json.dumps({"error": f"Unknown tool {tool_name}"})

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        })

        time.sleep(1)