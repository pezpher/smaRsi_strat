import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Function to calculate RSI
def calculate_rsi(df, window=14):
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
    #Parameters
    start = '1980-01-01'  
    end = dt.datetime.today()
    period = '5d'
    interval = '5m'
    
    
    data = yf.download(ticker, start=start, end=end)
    data = data.droplevel(1, axis=1)  
    data = data.reset_index()  

    best_wealth = -float('inf')
    second_best_wealth = -float('inf')
    best_params = None
    second_best_params = None

    # Define the ranges for RSI and MA parameters, change the increments, smaller will result in more compute time.
    window_range = range(10, 31, 10)  # Range of RSI window sizes
    buy_threshold_range = range(10, 41, 10)  # Buy threshold 
    sell_threshold_range = range(60, 91, 10)  # Sell threshold 
    ma1_range = range(10, 51, 10)  # Short-term MA
    ma2_range = range(50, 201, 10)  # Long-term MA

    # Iterate over all combinations of RSI and MA parameters
    for window in window_range:
        for buy_threshold in buy_threshold_range:
            for sell_threshold in sell_threshold_range:
                for ma1 in ma1_range:
                    for ma2 in ma2_range:
                        if ma1 >= ma2:  
                            continue

                        # Create a fresh copy of the original data for each iteration
                        df = data.copy()

                        # Calculate RSI and Moving Averages
                        df['RSI'] = calculate_rsi(df, window=window)
                        df['ma1'] = df['Close'].rolling(ma1).mean()
                        df['ma2'] = df['Close'].rolling(ma2).mean()

                        # Drop NaN values caused by rolling calculations
                        df.dropna(inplace=True)

                        # Generate combined signals
                        df['Shares'] = 0  # Default to no position
                        
                        # Long Signal
                        df.loc[(df['RSI'] < buy_threshold) & (df['ma1'] > df['ma2']), 'Shares'] = 1  
                        
                        # Short Signal
                        df.loc[(df['RSI'] > sell_threshold) & (df['ma1'] < df['ma2']), 'Shares'] = -1

                        # Shift the closing price by 1 day
                        df['Close1'] = df['Close'].shift(-1)

                        # Drop NaN values caused by the shift
                        df.dropna(inplace=True)
                    
                        # Calculate profit based on the signals
                        df['Profit'] = df['Shares']*(df['Close1'] - df['Close'])

                        # Calculate cumulative wealth
                        df['Wealth'] = df['Profit'].cumsum().round(2)

                        # Ensure there are enough rows before trying to access the last row
                        if not df.empty:
                            final_wealth = df['Wealth'].iloc[-1]

                            #Print the current parameters and final wealth
                            print(f"Window: {window}, Buy threshold: {buy_threshold}, Sell threshold: {sell_threshold}, MA1: {ma1}, MA2: {ma2}, final_wealth: {final_wealth}")

                            if final_wealth > best_wealth:
                                second_best_wealth = best_wealth
                                second_best_params = best_params
                                best_wealth = final_wealth
                                best_params = (window, buy_threshold, sell_threshold, ma1, ma2)
                            elif final_wealth > second_best_wealth:
                                second_best_wealth = final_wealth
                                second_best_params = (window, buy_threshold, sell_threshold, ma1, ma2)
                        else:
                            print(f"Skipping combination due to insufficient data.")

   
    if best_params is not None:
        print(f"Best Params: RSI Window: {best_params[0]}, Buy threshold: {best_params[1]}, Sell threshold: {best_params[2]}, MA1: {best_params[3]}, MA2: {best_params[4]}, Best Wealth: {best_wealth}")
        print(f"Second Best Params: RSI Window: {second_best_params[0]}, Buy threshold: {second_best_params[1]}, Sell threshold: {second_best_params[2]}, MA1: {second_best_params[3]}, MA2: {second_best_params[4]}, Second Best Wealth: {second_best_wealth}")
        with open('results.txt', 'a') as f:
            f.write(f"{ticker} Script main - Best Params: RSI Window: {best_params[0]}, Buy threshold: {best_params[1]}, Sell threshold: {best_params[2]}, MA1: {best_params[3]}, MA2: {best_params[4]}, Best Wealth: {best_wealth}\n")

        # Plot the results using the best parameters
        df = data.copy()  # Reset to the original data for the final plot
        df['RSI'] = calculate_rsi(df, window=best_params[0])
        df['ma1'] = df['Close'].rolling(best_params[3]).mean()
        df['ma2'] = df['Close'].rolling(best_params[4]).mean()
        df.dropna(inplace=True)
        df['Shares'] = 0
        df.loc[(df['RSI'] < best_params[1]) & (df['ma1'] > df['ma2']), 'Shares'] = 1  # Long signal
        df.loc[(df['RSI'] > best_params[2]) & (df['ma1'] < df['ma2']), 'Shares'] = -1  # Short signal
        df['Close1'] = df['Close'].shift(-1)
        df['Profit'] = df['Shares'] * (df['Close1'] - df['Close'])
        df['Wealth'] = df['Profit'].cumsum()

        sns.set(style="darkgrid")

        # Create a figure with 2 subplots (one for price/wealth and one for RSI)
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 1, height_ratios=[3, 1])

        # First subplot for stock price, moving averages, and cumulative wealth
        ax1 = fig.add_subplot(gs[0])
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price/Wealth')
        sns.lineplot(data=df['Wealth'] + df['Close'].iloc[0], label='Cumulative Wealth', ax=ax1)
        sns.lineplot(data=df['Close'], label='Close Price', ax=ax1)
        sns.lineplot(data=df['ma1'], label=f'{best_params[3]}-Day MA', ax=ax1)
        sns.lineplot(data=df['ma2'], label=f'{best_params[4]}-Day MA', ax=ax1)

        # Second subplot for RSI
        ax2 = fig.add_subplot(gs[1], sharex=ax1)  # Share x-axis with the top chart
        ax2.set_ylabel('RSI')
        sns.lineplot(data=df['RSI'], label='RSI', ax=ax2, color='purple')
        ax2.axhline(best_params[2], linestyle='--', color='red', alpha=0.7)  # Overbought line
        ax2.axhline(best_params[1], linestyle='--', color='green', alpha=0.7)  # Oversold line

        # Remove x-axis labels from the first plot to avoid clutter
        plt.setp(ax1.get_xticklabels(), visible=False)

      
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.title(f'{ticker} Stock Analysis with Optimized RSI and MA (RSI Window: {best_params[0]}, Buy: {best_params[1]}, Sell: {best_params[2]}, MA1: {best_params[3]}, MA2: {best_params[4]})')
        plt.tight_layout()
        plt.show()
    else:
        print("No valid combination of parameters found.")

    return best_params

# Example
best_params = rsi_ma_hyper_tune(input('Enter ticker symbol: '))
print(f"Best combined RSI and MA parameters: {best_params}")