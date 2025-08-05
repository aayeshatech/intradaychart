import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import seaborn as sns
from matplotlib.patches import Circle, Wedge
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

# Set compatible style with error handling
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('default')
        sns.set_style("darkgrid")

# --- PLANETARY POSITION VISUALIZATION ---
def draw_planetary_wheel(ax, date, size=0.3):
    """Draw a simplified astrological wheel showing planetary positions"""
    # Planet positions for August 2025 (simplified)
    planets = {
        'Sun': {'angle': 135, 'color': 'gold', 'size': 8},
        'Moon': {'angle': 225, 'color': 'silver', 'size': 6},
        'Mercury': {'angle': 120, 'color': 'gray', 'size': 5},
        'Venus': {'angle': 170, 'color': 'lightgreen', 'size': 7},
        'Mars': {'angle': 85, 'color': 'red', 'size': 6},
        'Jupiter': {'angle': 45, 'color': 'orange', 'size': 10},
        'Saturn': {'angle': 315, 'color': 'darkgoldenrod', 'size': 9}
    }
    
    # Draw zodiac wheel
    zodiac = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 
              'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
    
    for i, sign in enumerate(zodiac):
        angle = i * 30
        ax.add_patch(Wedge((0, 0), size, angle, angle+30, width=size*0.8, 
                          facecolor='lightgray', edgecolor='black', alpha=0.3))
        ax.text(0.85*size * np.cos(np.radians(angle+15)), 
                0.85*size * np.sin(np.radians(angle+15)), 
                sign[:3], ha='center', va='center', fontsize=6)
    
    # Draw planets
    for planet, data in planets.items():
        angle_rad = np.radians(data['angle'])
        x = size * 0.6 * np.cos(angle_rad)
        y = size * 0.6 * np.sin(angle_rad)
        ax.add_patch(Circle((x, y), data['size']/200, color=data['color']))
        ax.text(x, y, planet[:1], ha='center', va='center', fontsize=6, fontweight='bold')
    
    ax.set_xlim(-size, size)
    ax.set_ylim(-size, size)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Planetary Positions\n{date.strftime("%b %d, %Y")}', fontsize=8)

# --- ENHANCED INTRADAY CHART ---
def generate_intraday_chart(date_str='2025-08-05'):
    # Parse date
    selected_date = datetime.strptime(date_str, '%Y-%m-%d')
    
    # Create time range from 9:15 AM to 3:30 PM with 15-minute intervals
    start_time = selected_date.replace(hour=9, minute=15)
    end_time = selected_date.replace(hour=15, minute=30)
    times = pd.date_range(start=start_time, end=end_time, freq='15T')
    
    # Initialize price array
    prices = np.zeros(len(times))
    base_price = 24620  # Opening price
    
    # Astrological events with timestamps and price impact
    events = [
        {"time": selected_date.replace(hour=9, minute=15), "aspect": "Mercury square Jupiter + Void Moon", "price": 24620, "impact": -0.5, "type": "bearish"},
        {"time": selected_date.replace(hour=10, minute=0), "aspect": "Moon trine Jupiter", "price": 24780, "impact": 1.0, "type": "bullish"},
        {"time": selected_date.replace(hour=11, minute=30), "aspect": "Mars sextile Jupiter building", "price": 24750, "impact": 0.3, "type": "neutral"},
        {"time": selected_date.replace(hour=12, minute=30), "aspect": "Sun in Leo (no aspects)", "price": 24740, "impact": 0.0, "type": "neutral"},
        {"time": selected_date.replace(hour=14, minute=0), "aspect": "Moon square Saturn", "price": 24650, "impact": -0.8, "type": "bearish"},
        {"time": selected_date.replace(hour=15, minute=0), "aspect": "Venus-Saturn opposition building", "price": 24700, "impact": 0.2, "type": "neutral"},
        {"time": selected_date.replace(hour=15, minute=30), "aspect": "Close", "price": 24722, "impact": 0.1, "type": "neutral"}
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
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 1], width_ratios=[4, 1])
    
    # Main price chart
    ax_main = fig.add_subplot(gs[0, 0])
    
    # Plot price line with gradient color based on trend
    for i in range(1, len(df_intraday)):
        color = 'green' if df_intraday['Price'].iloc[i] > df_intraday['Price'].iloc[i-1] else 'red'
        ax_main.plot(df_intraday['Time'].iloc[i-1:i+1], 
                    df_intraday['Price'].iloc[i-1:i+1], 
                    color=color, linewidth=2.5)
    
    # Mark key events with enhanced visuals
    for event in events:
        # Vertical line for event
        color_map = {'bullish': 'green', 'bearish': 'red', 'neutral': 'blue'}
        ax_main.axvline(x=event['time'], color=color_map[event['type']], 
                       linestyle='--', alpha=0.7, linewidth=2)
        
        # Event marker
        marker_color = color_map[event['type']]
        ax_main.scatter(event['time'], event['price'], color=marker_color, 
                       s=100, zorder=5, edgecolor='black', linewidth=1)
        
        # Annotation with background
        y_pos = event['price'] + 50 if event['price'] < 24700 else event['price'] - 50
        ax_main.annotate(event['aspect'], 
                        xy=(event['time'], event['price']),
                        xytext=(event['time'], y_pos),
                        arrowprops=dict(arrowstyle='->', color=marker_color, lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                        fontsize=9, ha='center')
    
    # Formatting
    ax_main.set_title(f'Nifty Intraday Chart - {selected_date.strftime("%B %d, %Y")}\n(Astrological Transits & Aspects)', 
                     fontsize=16, pad=20)
    ax_main.set_xlabel('Time (IST)', fontsize=12)
    ax_main.set_ylabel('Nifty Price', fontsize=12)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_main.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.setp(ax_main.get_xticklabels(), rotation=45, ha='right')
    
    # Add closing price annotation
    ax_main.annotate(f'Close: {df_intraday["Price"].iloc[-1]:.0f}', 
                    xy=(df_intraday['Time'].iloc[-1], df_intraday['Price'].iloc[-1]),
                    xytext=(df_intraday['Time'].iloc[-1] - timedelta(hours=1), 
                           df_intraday['Price'].iloc[-1] + 50),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                    fontsize=12, fontweight='bold',
                    bbox=dict(facecolor='yellow', alpha=0.7, edgecolor='none', pad=2))
    
    # Planetary wheel
    ax_wheel = fig.add_subplot(gs[0, 1])
    draw_planetary_wheel(ax_wheel, selected_date)
    
    # Volume chart (simulated)
    ax_volume = fig.add_subplot(gs[1, 0])
    volume = np.random.randint(1000, 5000, size=len(times))
    colors_volume = ['green' if df_intraday['Price'].iloc[i] > df_intraday['Price'].iloc[i-1] 
                    else 'red' for i in range(1, len(df_intraday))]
    colors_volume.insert(0, 'green')  # First bar
    
    ax_volume.bar(df_intraday['Time'], volume, color=colors_volume, alpha=0.7)
    ax_volume.set_title('Volume', fontsize=12)
    ax_volume.set_ylabel('Volume', fontsize=10)
    ax_volume.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax_volume.get_xticklabels(), rotation=45, ha='right')
    
    # Aspect strength indicator
    ax_aspect = fig.add_subplot(gs[2, :])
    aspect_times = [event['time'] for event in events]
    aspect_strengths = [abs(event['impact']) for event in events]
    aspect_colors = [color_map[event['type']] for event in events]
    
    ax_aspect.scatter(aspect_times, aspect_strengths, color=aspect_colors, s=100, zorder=3)
    ax_aspect.plot(aspect_times, aspect_strengths, color='gray', alpha=0.5, linestyle='--')
    ax_aspect.set_title('Astrological Aspect Strength Throughout the Day', fontsize=12)
    ax_aspect.set_ylabel('Strength', fontsize=10)
    ax_aspect.set_ylim(0, 1.5)
    ax_aspect.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax_aspect.get_xticklabels(), rotation=45, ha='right')
    
    # Add aspect labels
    for i, (time, strength, event) in enumerate(zip(aspect_times, aspect_strengths, events)):
        ax_aspect.annotate(event['aspect'], 
                          xy=(time, strength),
                          xytext=(time, strength + 0.1),
                          fontsize=8, ha='center',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    return fig

# --- ENHANCED MONTHLY CHART ---
def generate_monthly_chart():
    # Create date range for August 2025
    dates = pd.date_range(start='2025-08-01', end='2025-08-30', freq='D')
    
    # Astrological events with dates and price impact
    events = [
        {"date": datetime(2025, 8, 1), "aspect": "Mercury Retrograde starts", "price": 24520, "impact": 0.5, "type": "neutral"},
        {"date": datetime(2025, 8, 4), "aspect": "Venus Opposition Saturn", "price": 24420, "impact": -1.0, "type": "bearish"},
        {"date": datetime(2025, 8, 5), "aspect": "Moon-Jupiter trine → Moon-Saturn square", "price": 24722, "impact": 1.2, "type": "bullish"},
        {"date": datetime(2025, 8, 7), "aspect": "Full Moon in Aquarius", "price": 24750, "impact": 0.8, "type": "bullish"},
        {"date": datetime(2025, 8, 11), "aspect": "Jupiter Square Saturn", "price": 24750, "impact": -1.5, "type": "bearish"},
        {"date": datetime(2025, 8, 15), "aspect": "Sun enters Virgo", "price": 24720, "impact": 0.3, "type": "neutral"},
        {"date": datetime(2025, 8, 19), "aspect": "Mercury Direct", "price": 24800, "impact": 1.0, "type": "bullish"},
        {"date": datetime(2025, 8, 23), "aspect": "Venus enters Libra", "price": 24880, "impact": 0.8, "type": "bullish"},
        {"date": datetime(2025, 8, 27), "aspect": "Mars Trine Saturn", "price": 24960, "impact": 0.5, "type": "neutral"},
        {"date": datetime(2025, 8, 30), "aspect": "New Moon in Virgo", "price": 25100, "impact": 1.3, "type": "bullish"}
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
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 1], width_ratios=[4, 1])
    
    # Main price chart
    ax_main = fig.add_subplot(gs[0, 0])
    
    # Plot price line with gradient color based on trend
    for i in range(1, len(df_monthly)):
        color = 'green' if df_monthly['Price'].iloc[i] > df_monthly['Price'].iloc[i-1] else 'red'
        ax_main.plot(df_monthly['Date'].iloc[i-1:i+1], 
                    df_monthly['Price'].iloc[i-1:i+1], 
                    color=color, linewidth=2.5)
    
    # Mark key events with enhanced visuals
    for event in events:
        # Vertical line for event
        color_map = {'bullish': 'green', 'bearish': 'red', 'neutral': 'blue'}
        ax_main.axvline(x=event['date'], color=color_map[event['type']], 
                       linestyle='--', alpha=0.7, linewidth=2)
        
        # Event marker
        marker_color = color_map[event['type']]
        ax_main.scatter(event['date'], event['price'], color=marker_color, 
                       s=150, zorder=5, edgecolor='black', linewidth=1.5)
        
        # Annotation with background
        y_pos = event['price'] + 100 if event['price'] < 24800 else event['price'] - 100
        ax_main.annotate(event['aspect'], 
                        xy=(event['date'], event['price']),
                        xytext=(event['date'], y_pos),
                        arrowprops=dict(arrowstyle='->', color=marker_color, lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                        fontsize=9, ha='center')
    
    # Formatting
    ax_main.set_title('Nifty Monthly Chart - August 2025\n(Astrological Transits & Aspects)', 
                     fontsize=16, pad=20)
    ax_main.set_xlabel('Date (August 2025)', fontsize=12)
    ax_main.set_ylabel('Nifty Closing Price', fontsize=12)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax_main.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax_main.get_xticklabels(), rotation=45, ha='right')
    
    # Add month-end close annotation
    ax_main.annotate(f'Month Close: {df_monthly["Price"].iloc[-1]:.0f}', 
                    xy=(df_monthly['Date'].iloc[-1], df_monthly['Price'].iloc[-1]),
                    xytext=(df_monthly['Date'].iloc[-1] - timedelta(days=5), 
                           df_monthly['Price'].iloc[-1] + 200),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                    fontsize=12, fontweight='bold',
                    bbox=dict(facecolor='yellow', alpha=0.7, edgecolor='none', pad=2))
    
    # Planetary positions for key dates
    ax_planets = fig.add_subplot(gs[0, 1])
    key_dates = [datetime(2025, 8, 5), datetime(2025, 8, 11), datetime(2025, 8, 30)]
    for i, date in enumerate(key_dates):
        ax_sub = fig.add_axes([0.75, 0.7-i*0.2, 0.15, 0.15])
        draw_planetary_wheel(ax_sub, date, size=0.5)
    
    # Volume chart (simulated)
    ax_volume = fig.add_subplot(gs[1, 0])
    volume = np.random.randint(10000, 50000, size=len(dates))
    colors_volume = ['green' if df_monthly['Price'].iloc[i] > df_monthly['Price'].iloc[i-1] 
                    else 'red' for i in range(1, len(df_monthly))]
    colors_volume.insert(0, 'green')  # First bar
    
    ax_volume.bar(df_monthly['Date'], volume, color=colors_volume, alpha=0.7)
    ax_volume.set_title('Volume', fontsize=12)
    ax_volume.set_ylabel('Volume', fontsize=10)
    ax_volume.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.setp(ax_volume.get_xticklabels(), rotation=45, ha='right')
    
    # Aspect calendar
    ax_calendar = fig.add_subplot(gs[2, :])
    aspect_dates = [event['date'] for event in events]
    aspect_strengths = [abs(event['impact']) for event in events]
    aspect_colors = [color_map[event['type']] for event in events]
    
    ax_calendar.scatter(aspect_dates, aspect_strengths, color=aspect_colors, s=200, zorder=3)
    ax_calendar.plot(aspect_dates, aspect_strengths, color='gray', alpha=0.5, linestyle='--')
    ax_calendar.set_title('Astrological Aspect Strength Throughout August 2025', fontsize=12)
    ax_calendar.set_ylabel('Strength', fontsize=10)
    ax_calendar.set_ylim(0, 2)
    ax_calendar.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.setp(ax_calendar.get_xticklabels(), rotation=45, ha='right')
    
    # Add aspect labels
    for i, (date, strength, event) in enumerate(zip(aspect_dates, aspect_strengths, events)):
        ax_calendar.annotate(event['aspect'], 
                           xy=(date, strength),
                           xytext=(date, strength + 0.1),
                           fontsize=8, ha='center',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
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
    st.title('🌟 Nifty Astrological Analysis Dashboard')
    st.write('This interactive dashboard generates Nifty charts based on astrological transits and aspects.')
    
    # Sidebar for chart selection
    st.sidebar.title('📊 Chart Selection')
    chart_type = st.sidebar.selectbox(
        'Choose a chart to generate:',
        ['Intraday Chart (Aug 2025)', 'Monthly Chart (Aug 2025)', 'Aspect Analysis']
    )
    
    if chart_type == 'Intraday Chart (Aug 2025)':
        st.header('📈 Nifty Intraday Chart')
        st.write('Simulated intraday price movements based on astrological transits and aspects.')
        
        # Date selector
        selected_date = st.date_input(
            'Select a date in August 2025:',
            value=datetime(2025, 8, 5).date(),
            min_value=datetime(2025, 8, 1).date(),
            max_value=datetime(2025, 8, 31).date()
        )
        
        # Generate chart
        date_str = selected_date.strftime('%Y-%m-%d')
        fig = generate_intraday_chart(date_str)
        st.pyplot(fig)
        
        # Additional information
        st.subheader('🔮 Astrological Insights for ' + selected_date.strftime('%B %d, %Y'))
        st.info("""
        **Key Astrological Events:**
        - **Mercury Retrograde**: Causes market volatility and unpredictable movements
        - **Moon-Jupiter Trine**: Creates bullish sentiment and optimism
        - **Moon-Saturn Square**: Triggers fear and selling pressure
        - **Venus-Saturn Opposition**: Creates financial stress and risk aversion
        
        **Trading Strategy:**
        - Best time to buy: Around Moon-Jupiter trine (10:00 AM)
        - Best time to sell: Before Moon-Saturn square (2:00 PM)
        - Avoid trading during Mercury retrograde periods
        """)
        
    elif chart_type == 'Monthly Chart (Aug 2025)':
        st.header('📊 Nifty Monthly Chart')
        st.write('Simulated daily closing prices based on astrological transits and aspects.')
        fig = generate_monthly_chart()
        st.pyplot(fig)
        
        # Additional information
        st.subheader('🌙 Monthly Astrological Summary')
        st.info("""
        **August 2025 Key Transits:**
        - **Mercury Retrograde (Aug 1-19)**: High volatility, avoid major decisions
        - **Jupiter Square Saturn (Aug 11)**: Major tension between optimism and restriction
        - **Mercury Direct (Aug 19)**: Clarity returns, good for new positions
        - **New Moon in Virgo (Aug 30)**: Strong bullish close to the month
        
        **Market Phases:**
        1. **Early August (1-11)**: Bearish pressure with high volatility
        2. **Mid August (12-18)**: Stabilization and consolidation
        3. **Late August (19-30)**: Bullish surge with financial sector rally
        """)
        
    elif chart_type == 'Aspect Analysis':
        st.header('📋 Astrological Aspect Analysis')
        st.write('Analysis of key astrological aspects and their impact on Nifty prices.')
        fig, df_aspects = analyze_aspects()
        st.pyplot(fig)
        
        st.subheader('📊 Aspect Data Table')
        st.dataframe(df_aspects)
        
        # Additional information
        st.subheader('🔭 Understanding Astrological Aspects')
        st.info("""
        **Aspect Types:**
        - **Conjunction (0°)**: Powerful, combining energies
        - **Square (90°)**: Challenging, creates tension
        - **Trine (120°)**: Harmonious, positive flow
        - **Opposition (180°)**: Polarizing, requires balance
        
        **Planetary Influences:**
        - **Jupiter**: Expansion, optimism, growth
        - **Saturn**: Restriction, discipline, fear
        - **Mercury**: Communication, logic, volatility
        - **Venus**: Finances, relationships, value
        - **Mars**: Action, energy, conflict
        - **Moon**: Emotions, sentiment, cycles
        """)

if __name__ == "__main__":
    main()
