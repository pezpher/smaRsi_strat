import os

# Template code for generating different scripts
template_code = '''
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Function to calculate RSI
def calculate_rsi(df, window={window_size}):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to combine RSI and Moving Average Strategy
def rsi_ma_hyper_tune(ticker):
    start = '1980-01-01'
    end = dt.datetime.today()

    # Download data
    data = yf.download(ticker, start=start, end=end)
    data = data.droplevel(1, axis=1)  
    data = data.reset_index()  

    best_wealth = -float('inf')
    second_best_wealth = -float('inf')
    best_params = None
    second_best_params = None

    buy_threshold_range = range(10, 41, 1)  # Buy threshold  
    sell_threshold_range = range(60, 91, 1)  # Sell threshold  
    ma1_range = range(10, 51, 1)  # Short-term MA
    ma2_range = range(50, 201, 1)  # Long-term MA

    for buy_threshold in buy_threshold_range:
        for sell_threshold in sell_threshold_range:
            for ma1 in ma1_range:
                for ma2 in ma2_range:
                    if ma1 >= ma2:
                        continue

                    df = data.copy()

                    df['RSI'] = calculate_rsi(df, window={window_size})
                    df['ma1'] = df['Close'].rolling(ma1).mean()
                    df['ma2'] = df['Close'].rolling(ma2).mean()

                    df.dropna(inplace=True)

                    df['Shares'] = 0

                    df.loc[(df['RSI'] < buy_threshold) & (df['ma1'] > df['ma2']), 'Shares'] = 1

                    df.loc[(df['RSI'] > sell_threshold) & (df['ma1'] < df['ma2']), 'Shares'] = -1

                    df['Close1'] = df['Close'].shift(-1)

                    df.dropna(inplace=True)

                    df['Profit'] = df['Shares'] * (df['Close1'] - df['Close'])

                    df['Wealth'] = df['Profit'].cumsum().round(2)

                    if not df.empty:
                        final_wealth = df['Wealth'].iloc[-1]

                        if final_wealth > best_wealth:
                            second_best_wealth = best_wealth
                            second_best_params = best_params
                            best_wealth = final_wealth
                            best_params = ({window_size}, buy_threshold, sell_threshold, ma1, ma2)
                        elif final_wealth > second_best_wealth:
                            second_best_wealth = final_wealth
                            second_best_params = ({window_size}, buy_threshold, sell_threshold, ma1, ma2)
                    else:
                        pass

    if best_params is not None:
        print(f"Best Params: RSI Window: {best_params[0]}, Buy threshold: {best_params[1]}, Sell threshold: {best_params[2]}, MA1: {best_params[3]}, MA2: {best_params[4]}, Best Wealth: {best_wealth}")
        print(f"Second Best Params: RSI Window: {second_best_params[0]}, Buy threshold: {second_best_params[1]}, Sell threshold: {second_best_params[2]}, MA1: {second_best_params[3]}, MA2: {second_best_params[4]}, Second Best Wealth: {second_best_wealth}")
        with open('results.txt', 'a') as f:
            f.write(f"{ticker} Script main - Best Params: RSI Window: {best_params[0]}, Buy threshold: {best_params[1]}, Sell threshold: {best_params[2]}, MA1: {best_params[3]}, MA2: {best_params[4]}, Best Wealth: {best_wealth}\\n")

        df = data.copy()
        df['RSI'] = calculate_rsi(df, window={window_size})
        df['ma1'] = df['Close'].rolling(best_params[3]).mean()
        df['ma2'] = df['Close'].rolling(best_params[4]).mean()
        df.dropna(inplace=True)
        df['Shares'] = 0
        df.loc[(df['RSI'] < best_params[1]) & (df['ma1'] > df['ma2']), 'Shares'] = 1
        df.loc[(df['RSI'] > best_params[2]) & (df['ma1'] < df['ma2']), 'Shares'] = -1
        df['Close1'] = df['Close'].shift(-1)
        df['Profit'] = df['Shares'] * (df['Close1'] - df['Close'])
        df['Wealth'] = df['Profit'].cumsum()

        sns.set(style="darkgrid")

        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 1, height_ratios=[3, 1])

        ax1 = fig.add_subplot(gs[0])
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price/Wealth')
        sns.lineplot(data=df['Wealth'] + df['Close'].iloc[0], label='Cumulative Wealth', ax=ax1)
        sns.lineplot(data=df['Close'], label='Close Price', ax=ax1)
        sns.lineplot(data=df['ma1'], label=f'{best_params[3]}-Day MA', ax=ax1)
        sns.lineplot(data=df['ma2'], label=f'{best_params[4]}-Day MA', ax=ax1)

        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.set_ylabel('RSI')
        sns.lineplot(data=df['RSI'], label='RSI', ax=ax2, color='purple')
        ax2.axhline(best_params[2], linestyle='--', color='red', alpha=0.7)
        ax2.axhline(best_params[1], linestyle='--', color='green', alpha=0.7)

        plt.setp(ax1.get_xticklabels(), visible=False)

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.title(f'{ticker} Stock Analysis with Optimized RSI and MA (RSI Window: {best_params[0]}, Buy: {best_params[1]}, Sell: {best_params[2]}, MA1: {best_params[3]}, MA2: {best_params[4]})')
        plt.tight_layout()
        plt.show()
    else:
        print("No valid combination of parameters found.")

    return best_params
tickerr = 'spy'
best_params = rsi_ma_hyper_tune(tickerr)
'''
tickerr = 'spy'

# Directory to save the scripts
save_dir = f"{tickerr}_generated_scripts"
os.makedirs(save_dir, exist_ok=True)

# Generate scripts with RSI window sizes from 10 to 30
for window_size in range(10, 31):
    # Replace the placeholder {window_size} with the actual window size
    new_code = template_code.replace("{window_size}", str(window_size))

    # Filename for each script
    filename = os.path.join(save_dir, f"{tickerr}script_rsi_{window_size}.py")

    # Save the new script
    with open(filename, 'w') as f:
        f.write(new_code)

    print(f"Generated {filename}")



