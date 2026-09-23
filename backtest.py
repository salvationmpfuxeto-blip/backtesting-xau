import yfinance as yf
import pandas as pd
import numpy as np

# ==============================
# SETTINGS
# ==============================

SYMBOLS = {
    "XAUUSD": "GC=F",
    "NAS100": "NQ=F"
}

START_DATE = "2020-01-01"
END_DATE = "2026-09-01"

INITIAL_BALANCE = 10000.0
RISK_PER_TRADE = 0.01

EMA_FAST = 100
EMA_SLOW = 200

# More selective entries to improve win rate
LONG_ENTRY_RSI = 60
SHORT_ENTRY_RSI = 40
LONG_EXIT_RSI = 50
SHORT_EXIT_RSI = 50


# ==============================
# DOWNLOAD DATA
# ==============================

def get_data(symbol):
    print(f"\nDownloading {symbol}...")

    data = yf.download(
        symbol,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        print("No data received.")
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.dropna()
    return data


# ==============================
# CREATE INDICATORS
# ==============================

def indicators(data):
    data["EMA50"] = data["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    data["EMA200"] = data["Close"].ewm(span=EMA_SLOW, adjust=False).mean()

    change = data["Close"].diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = np.where(avg_loss == 0, np.nan, avg_gain / avg_loss)
    data["RSI"] = 100 - (100 / (1 + rs))

    return data.dropna()


# ==============================
# BACKTEST
# ==============================

def backtest(data):
    balance = INITIAL_BALANCE
    trades = []
    position = None
    entry_price = 0.0

    for i in range(1, len(data)):
        price = float(data["Close"].iloc[i])
        ema50 = float(data["EMA50"].iloc[i])
        ema200 = float(data["EMA200"].iloc[i])
        rsi = float(data["RSI"].iloc[i])

        if position is None:
            # LONG: trend up + momentum strong
            if ema50 > ema200 and rsi > LONG_ENTRY_RSI:
                position = "LONG"
                entry_price = price

            # SHORT: trend down + momentum strong
            elif ema50 < ema200 and rsi < SHORT_ENTRY_RSI:
                position = "SHORT"
                entry_price = price

        elif position == "LONG":
            # Exit if trend weakens or RSI loses strength
            if ema50 < ema200 or rsi < LONG_EXIT_RSI:
                profit_percent = (price - entry_price) / entry_price
                profit = balance * RISK_PER_TRADE * (profit_percent * 100)
                balance += profit
                trades.append(profit)
                position = None

        elif position == "SHORT":
            # Exit if trend weakens or RSI loses strength
            if ema50 > ema200 or rsi > SHORT_EXIT_RSI:
                profit_percent = (entry_price - price) / entry_price
                profit = balance * RISK_PER_TRADE * (profit_percent * 100)
                balance += profit
                trades.append(profit)
                position = None

    return balance, trades


# ==============================
# RESULTS
# ==============================

def results(start_balance, final_balance, trades):
    if not trades:
        print("No trades found.")
        return

    wins = [x for x in trades if x > 0]
    losses = [x for x in trades if x < 0]

    win_rate = len(wins) / len(trades) * 100
    total_profit = final_balance - start_balance

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf")

    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0

    print("\n==============================")
    print("BACKTEST RESULTS")
    print("==============================")

    print(f"Starting balance: ${start_balance:,.2f}")
    print(f"Final balance:    ${final_balance:,.2f}")
    print(f"Total profit:     ${total_profit:,.2f}")
    print(f"Total trades:     {len(trades)}")
    print(f"Winning trades:   {len(wins)}")
    print(f"Losing trades:    {len(losses)}")
    print(f"Win rate:         {win_rate:.2f}%")
    print(f"Profit factor:    {profit_factor:.2f}")
    print(f"Average win:      ${avg_win:,.2f}")
    print(f"Average loss:     ${avg_loss:,.2f}")
    print("==============================\n")


# ==============================
# RUN
# ==============================

def main():
    for name, ticker in SYMBOLS.items():
        print("\n")
        print("################################")
        print(f"TESTING {name}")
        print("################################")

        data = get_data(ticker)
        if data is None:
            continue

        data = indicators(data)
        final_balance, trades = backtest(data)
        results(INITIAL_BALANCE, final_balance, trades)


if __name__ == "__main__":
    main()
