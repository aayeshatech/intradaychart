import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import seaborn as sns

# Set compatible style with error handling
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('default')
        sns.set_style("darkgrid")

# --- INTRADAY CHART FOR AUGUST 5, 2025 ---
def generate_intraday_chart():
    # Create time range from 9:15 AM to 3:30 PM with 15-minute intervals
    start_time = datetime(2025, 8, 5, 9, 15)
    end_time = datetime(2025, 8, 5, 15, 30)
    times = pd.date_range(start=start_time, end=end_time, freq='15T')
    
    # Initialize price array
    prices = np.zeros(len(times))
    base_price = 24620  # Opening price
    
    # Astrological events with timestamps and price impact
    events = [
        {"time": datetime(2025, 8, 5, 9, 15), "aspect": "Mercury square Jupiter + Void Moon", "price": 24620, "impact": -0.5},
        {"time": datetime(2025, 8, 5, 10, 0), "aspect": "Moon trine Jupiter", "price": 24780, "impact": 1.0},
        {"time": datetime(2025, 8, 5, 11, 30), "aspect": "Mars sextile Jupiter building", "price": 24750, "impact": 0.3},
        {"time": datetime(2025, 8, 5, 12, 30), "aspect": "Sun in Leo (no aspects)", "price": 24740, "impact": 0.0},
        {"time": datetime(2025, 8, 5, 14, 0), "aspect": "Moon square Saturn", "price": 24650, "impact": -0.8},
        {"time": datetime(2025, 8, 5, 15, 0), "aspect": "Venus-Saturn opposition building", "price": 24700, "impact": 0.2},
        {"time": datetime(2025, 8, 5, 15, 30), "aspect": "Close", "price": 24722, "impact": 0.1}
    ]
    
    # Generate price movements
    for i, time in enumerate(times):
        # Find the closest event
        closest_event = min(events, key=lambda x: abs((x["time"] - time).total_seconds()))
        
        # Calculate distance from event
        distance = abs((closest_event["time"] - time).total_seconds()) / 3600  # in hours
        
        # Base movement with random volatility
        volatility = 0.15 if distance < 0.5 else 0.05
        random_change = np.random.normal(0, volatility)
        
        # Event influence (stronger when closer to event)
        event_influence = closest_event["impact"] * np.exp(-distance)
        
        # Calculate price change
        if i == 0:
            prices[i] = base_price
        else:
            change = (event_influence + random_change) * 10
            prices[i] = prices[i-1] + change
    
    # Create DataFrame
    df_intraday = pd.DataFrame({
        'Time': times,
        'Price': prices,
        'Aspect': [min(events, key=lambda x: abs((x["time"] - t).total_seconds()))["aspect"] for t in times]
    })
    
    # Plot intraday chart with adjusted layout
    fig, ax = plt.subplots(figsize=(14, 10))  # Increased height
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.25)  # Adjust margins
    
    # Plot price line
    ax.plot(df_intraday['Time'], df_intraday['Price'], 
            color='royalblue', linewidth=2.5, label='Nifty Price')
    
    # Mark key events with adjusted annotation positions
    for event in events:
        ax.axvline(x=event['time'], color='crimson', linestyle='--', alpha=0.7)
        # Adjust text position to avoid overlap
        y_pos = event['price'] + 50 if event['price'] < 24700 else event['price'] - 50
        ax.text(event['time'], y_pos, event['aspect'], 
                rotation=90, verticalalignment='bottom', fontsize=8, color='crimson',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # Formatting
    ax.set_title('Nifty Intraday Chart - August 5, 2025\n(Astrological Transits & Aspects)', fontsize=16, pad=20)
    ax.set_xlabel('Time (IST)', fontsize=12)
    ax.set_ylabel('Nifty Price', fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.xticks(rotation=45)
    ax.legend(loc='upper left')
    
    # Add closing price annotation
    ax.annotate(f'Close: {df_intraday["Price"].iloc[-1]:.0f}', 
                xy=(df_intraday['Time'].iloc[-1], df_intraday['Price'].iloc[-1]),
                xytext=(df_intraday['Time'].iloc[-1] - timedelta(hours=1), df_intraday['Price'].iloc[-1] + 50),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                fontsize=12, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
    
    return fig

# --- MONTHLY CHART FOR AUGUST 2025 ---
def generate_monthly_chart():
    # Create date range for August 2025
    dates = pd.date_range(start='2025-08-01', end='2025-08-30', freq='D')
    
    # Astrological events with dates and price impact
    events = [
        {"date": datetime(2025, 8, 1), "aspect": "Mercury Retrograde starts", "price": 24520, "impact": 0.5},
        {"date": datetime(2025, 8, 4), "aspect": "Venus Opposition Saturn", "price": 24420, "impact": -1.0},
        {"date": datetime(2025, 8, 5), "aspect": "Moon-Jupiter trine → Moon-Saturn square", "price": 24722, "impact": 1.2},
        {"date": datetime(2025, 8, 7), "aspect": "Full Moon in Aquarius", "price": 24750, "impact": 0.8},
        {"date": datetime(2025, 8, 11), "aspect": "Jupiter Square Saturn", "price": 24750, "impact": -1.5},
        {"date": datetime(2025, 8, 15), "aspect": "Sun enters Virgo", "price": 24720, "impact": 0.3},
        {"date": datetime(2025, 8, 19), "aspect": "Mercury Direct", "price": 24800, "impact": 1.0},
        {"date": datetime(2025, 8, 23), "aspect": "Venus enters Libra", "price": 24880, "impact": 0.8},
        {"date": datetime(2025, 8, 27), "aspect": "Mars Trine Saturn", "price": 24960, "impact": 0.5},
        {"date": datetime(2025, 8, 30), "aspect": "New Moon in Virgo", "price": 25100, "impact": 1.3}
    ]
    
    # Initialize price array
    prices = np.zeros(len(dates))
    base_price = 24500  # July 31 closing price
    
    # Generate price movements
    for i, date in enumerate(dates):
        # Find the closest event
        closest_event = min(events, key=lambda x: abs((x["date"].date() - date.date()).days))
        
        # Calculate distance from event
        distance = abs((closest_event["date"].date() - date.date()).days)
        
        # Base movement with random volatility
        volatility = 0.3 if distance < 2 else 0.1
        random_change = np.random.normal(0, volatility)
        
        # Event influence (stronger when closer to event)
        event_influence = closest_event["impact"] * np.exp(-distance/2)
        
        # Calculate price change
        if i == 0:
            prices[i] = base_price
        else:
            change = (event_influence + random_change) * 20
            prices[i] = prices[i-1] + change
    
    # Create DataFrame
    df_monthly = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Aspect': [min(events, key=lambda x: abs((x["date"].date() - d.date()).days))["aspect"] for d in dates]
    })
    
    # Plot monthly chart with adjusted layout
    fig, ax = plt.subplots(figsize=(16, 10))  # Increased height
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.25)  # Adjust margins
    
    # Plot price line
    ax.plot(df_monthly['Date'], df_monthly['Price'], 
            color='forestgreen', linewidth=2.5, label='Nifty Closing Price')
    
    # Mark key events with adjusted annotation positions
    for event in events:
        ax.axvline(x=event['date'], color='crimson', linestyle='--', alpha=0.7)
        # Adjust text position to avoid overlap
        y_pos = event['price'] + 150 if event['price'] < 24800 else event['price'] - 150
        ax.text(event['date'], y_pos, event['aspect'], 
                rotation=90, verticalalignment='bottom', fontsize=8, color='crimson',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # Formatting
    ax.set_title('Nifty Monthly Chart - August 2025\n(Astrological Transits & Aspects)', fontsize=16, pad=20)
    ax.set_xlabel('Date (August 2025)', fontsize=12)
    ax.set_ylabel('Nifty Closing Price', fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.xticks(rotation=45)
    ax.legend(loc='upper left')
    
    # Add month-end close annotation
    ax.annotate(f'Month Close: {df_monthly["Price"].iloc[-1]:.0f}', 
                xy=(df_monthly['Date'].iloc[-1], df_monthly['Price'].iloc[-1]),
                xytext=(df_monthly['Date'].iloc[-1] - timedelta(days=5), df_monthly['Price'].iloc[-1] + 200),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                fontsize=12, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
    
    return fig

# --- ASTROLOGICAL ASPECT ANALYSIS ---
def analyze_aspects():
    # Create a summary of key aspects
    aspects_data = {
        'Aspect': [
            'Mercury Retrograde (Aug 1-19)', 
            'Venus Opposition Saturn (Aug 4)', 
            'Moon-Jupiter Trine (Aug 5)', 
            'Full Moon in Aquarius (Aug 7)', 
            'Jupiter Square Saturn (Aug 11)', 
            'Mercury Direct (Aug 19)', 
            'Venus enters Libra (Aug 23)', 
            'New Moon in Virgo (Aug 30)'
        ],
        'Date': [
            'Aug 1-19', 
            'Aug 4', 
            'Aug 5', 
            'Aug 7', 
            'Aug 11', 
            'Aug 19', 
            'Aug 23', 
            'Aug 30'
        ],
        'Market Impact': [
            'High Volatility', 
            'Bearish Pressure', 
            'Bullish Surge', 
            'Trend Reversal', 
            'Major Tension', 
            'Clarity Returns', 
            'Financial Rally', 
            'Strong Bullish'
        ],
        'Price Change': [
            '+20 to -100 pts', 
            '-100 pts', 
            '+302 pts', 
            '+50 pts', 
            '-100 pts', 
            '+20 pts', 
            '+20 pts', 
            '+130 pts'
        ],
        'Sector Focus': [
            'All Sectors', 
            'Banking/Realty', 
            'Broad Market', 
            'Technology', 
            'Financials', 
            'Technology', 
            'Banking', 
            'Broad Market'
        ]
    }
    
    df_aspects = pd.DataFrame(aspects_data)
    
    # Create a bar chart of price changes with adjusted layout
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)  # Adjust margins
    
    # Extract numerical values from price change strings
    price_changes = []
    for change in df_aspects['Price Change']:
        num = ''.join(filter(str.isdigit, change))
        if num:
            price_changes.append(int(num))
        else:
            price_changes.append(0)
    
    # Create color map based on impact
    colors = ['red' if 'Bearish' in impact or 'Tension' in impact or 'Volatility' in impact 
              else 'green' for impact in df_aspects['Market Impact']]
    
    bars = ax.bar(df_aspects['Aspect'], price_changes, color=colors, alpha=0.7)
    
    # Add labels
    ax.set_title('Astrological Aspect Impact on Nifty Price Changes', fontsize=14, pad=20)
    ax.set_ylabel('Price Change (Points)', fontsize=12)
    
    # Rotate x-axis labels and adjust alignment
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    return fig, df_aspects

# --- STREAMLIT APP ---
def main():
    st.title('Nifty Astrological Analysis')
    st.write('This app generates Nifty charts based on astrological transits and aspects.')
    
    # Sidebar for chart selection
    st.sidebar.title('Select Chart Type')
    chart_type = st.sidebar.selectbox(
        'Choose a chart to generate:',
        ['Intraday Chart (Aug 5, 2025)', 'Monthly Chart (Aug 2025)', 'Aspect Analysis']
    )
    
    if chart_type == 'Intraday Chart (Aug 5, 2025)':
        st.header('Nifty Intraday Chart - August 5, 2025')
        st.write('Simulated intraday price movements based on astrological transits and aspects.')
        fig = generate_intraday_chart()
        st.pyplot(fig)
        
    elif chart_type == 'Monthly Chart (Aug 2025)':
        st.header('Nifty Monthly Chart - August 2025')
        st.write('Simulated daily closing prices based on astrological transits and aspects.')
        fig = generate_monthly_chart()
        st.pyplot(fig)
        
    elif chart_type == 'Aspect Analysis':
        st.header('Astrological Aspect Analysis')
        st.write('Analysis of key astrological aspects and their impact on Nifty prices.')
        fig, df_aspects = analyze_aspects()
        st.pyplot(fig)
        
        st.subheader('Aspect Data')
        st.dataframe(df_aspects)

if __name__ == "__main__":
    main()
