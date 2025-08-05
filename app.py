import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, date
import seaborn as sns
from matplotlib.patches import Circle, Wedge
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
import calendar

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
    # Planet positions for any date (simplified calculation)
    # In a real app, you would use an ephemeris library for accurate positions
    base_date = datetime(2025, 8, 1)
    days_diff = (date - base_date.date()).days if isinstance(date, datetime) else (date - base_date.date()).days
    
    # Base positions for August 2025 (in degrees)
    base_positions = {
        'Sun': 135,
        'Moon': 225,
        'Mercury': 120,
        'Venus': 170,
        'Mars': 85,
        'Jupiter': 45,
        'Saturn': 315
    }
    
    # Adjust positions based on days difference (simplified)
    # Sun moves ~1° per day, Moon ~13° per day, others vary
    daily_movement = {
        'Sun': 1.0,
        'Moon': 13.2,
        'Mercury': 1.5,
        'Venus': 1.2,
        'Mars': 0.5,
        'Jupiter': 0.08,
        'Saturn': 0.03
    }
    
    # Calculate positions for the given date
    planets = {}
    for planet, base_pos in base_positions.items():
        angle = (base_pos + daily_movement[planet] * days_diff) % 360
        planets[planet] = {
            'angle': angle,
            'color': {
                'Sun': 'gold',
                'Moon': 'silver',
                'Mercury': 'gray',
                'Venus': 'lightgreen',
                'Mars': 'red',
                'Jupiter': 'orange',
                'Saturn': 'darkgoldenrod'
            }[planet],
            'size': {
                'Sun': 8,
                'Moon': 6,
                'Mercury': 5,
                'Venus': 7,
                'Mars': 6,
                'Jupiter': 10,
                'Saturn': 9
            }[planet]
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

# --- GENERATE ASTROLOGICAL EVENTS ---
def generate_astrological_events(date, event_type='intraday'):
    """Generate astrological events for any given date"""
    # This is a simplified simulation - in a real app, you would use an ephemeris
    # to calculate actual planetary positions and aspects
    
    if event_type == 'intraday':
        # Base intraday events (time of day)
        base_events = [
            {"time_offset": 0, "aspect": "Mercury square Jupiter + Void Moon", "impact": -0.5, "type": "bearish"},
            {"time_offset": 45, "aspect": "Moon trine Jupiter", "impact": 1.0, "type": "bullish"},
            {"time_offset": 135, "aspect": "Mars sextile Jupiter building", "impact": 0.3, "type": "neutral"},
            {"time_offset": 195, "aspect": "Sun in Leo (no aspects)", "impact": 0.0, "type": "neutral"},
            {"time_offset": 285, "aspect": "Moon square Saturn", "impact": -0.8, "type": "bearish"},
            {"time_offset": 345, "aspect": "Venus-Saturn opposition building", "impact": 0.2, "type": "neutral"},
            {"time_offset": 375, "aspect": "Close", "impact": 0.1, "type": "neutral"}
        ]
        
        # Create events for the selected date
        events = []
        start_time = datetime.combine(date, datetime.min.time()).replace(hour=9, minute=15)
        for event in base_events:
            event_time = start_time + timedelta(minutes=event["time_offset"])
            events.append({
                "time": event_time,
                "aspect": event["aspect"],
                "impact": event["impact"],
                "type": event["type"],
                "price": 0  # Will be calculated later
            })
        
        return events
    
    else:  # monthly
        # Base monthly events (day of month)
        base_events = [
            {"day_offset": 1, "aspect": "Mercury Retrograde starts", "impact": 0.5, "type": "neutral"},
            {"day_offset": 4, "aspect": "Venus Opposition Saturn", "impact": -1.0, "type": "bearish"},
            {"day_offset": 5, "aspect": "Moon-Jupiter trine → Moon-Saturn square", "impact": 1.2, "type": "bullish"},
            {"day_offset": 7, "aspect": "Full Moon in Aquarius", "impact": 0.8, "type": "bullish"},
            {"day_offset": 11, "aspect": "Jupiter Square Saturn", "impact": -1.5, "type": "bearish"},
            {"day_offset": 15, "aspect": "Sun enters Virgo", "impact": 0.3, "type": "neutral"},
            {"day_offset": 19, "aspect": "Mercury Direct", "impact": 1.0, "type": "bullish"},
            {"day_offset": 23, "aspect": "Venus enters Libra", "impact": 0.8, "type": "bullish"},
            {"day_offset": 27, "aspect": "Mars Trine Saturn", "impact": 0.5, "type": "neutral"},
            {"day_offset": 30, "aspect": "New Moon in Virgo", "impact": 1.3, "type": "bullish"}
        ]
        
        # Get the number of days in the selected month
        if isinstance(date, datetime):
            year = date.year
            month = date.month
        else:
            year = date.year
            month = date.month
            
        days_in_month = calendar.monthrange(year, month)[1]
        
        # Create events for the selected month
        events = []
        for event in base_events:
            # Adjust day offset if it exceeds days in month
            day = min(event["day_offset"], days_in_month)
            event_date = datetime(year, month, day)
            events.append({
                "date": event_date,
                "aspect": event["aspect"],
                "impact": event["impact"],
                "type": event["type"],
                "price": 0  # Will be calculated later
            })
        
        return events

# --- ENHANCED INTRADAY CHART ---
def generate_intraday_chart(symbol, starting_price, selected_date):
    # Convert date to datetime if it's not already
    if isinstance(selected_date, date) and not isinstance(selected_date, datetime):
        selected_date = datetime.combine(selected_date, datetime.min.time())
    
    # Create time range from 9:15 AM to 3:30 PM with 15-minute intervals
    start_time = selected_date.replace(hour=9, minute=15)
    end_time = selected_date.replace(hour=15, minute=30)
    times = pd.date_range(start=start_time, end=end_time, freq='15T')
    
    # Initialize price array
    prices = np.zeros(len(times))
    base_price = starting_price  # User-provided starting price
    
    # Generate astrological events for the selected date
    events = generate_astrological_events(selected_date, 'intraday')
    
    # Set event prices based on impact
    for event in events:
        # Calculate price based on impact and base price
        price_change = event["impact"] * base_price * 0.01  # Scale impact as percentage of base price
        event["price"] = base_price + price_change
    
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
            change = (event_influence + random_change) * base_price * 0.001  # Scale to base price
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
        y_pos = event['price'] + base_price * 0.01 if event['price'] < base_price * 1.01 else event['price'] - base_price * 0.01
        ax_main.annotate(event['aspect'], 
                        xy=(event['time'], event['price']),
                        xytext=(event['time'], y_pos),
                        arrowprops=dict(arrowstyle='->', color=marker_color, lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                        fontsize=9, ha='center')
    
    # Formatting
    ax_main.set_title(f'{symbol} Intraday Chart - {selected_date.strftime("%B %d, %Y")}\n(Astrological Transits & Aspects)', 
                     fontsize=16, pad=20)
    ax_main.set_xlabel('Time (IST)', fontsize=12)
    ax_main.set_ylabel('Price', fontsize=12)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_main.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.setp(ax_main.get_xticklabels(), rotation=45, ha='right')
    
    # Add closing price annotation
    ax_main.annotate(f'Close: {df_intraday["Price"].iloc[-1]:.2f}', 
                    xy=(df_intraday['Time'].iloc[-1], df_intraday['Price'].iloc[-1]),
                    xytext=(df_intraday['Time'].iloc[-1] - timedelta(hours=1), 
                           df_intraday['Price'].iloc[-1] + base_price * 0.01),
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
def generate_monthly_chart(symbol, starting_price, selected_month, selected_year):
    # Create date range for the selected month
    start_date = datetime(selected_year, selected_month, 1)
    days_in_month = calendar.monthrange(selected_year, selected_month)[1]
    end_date = datetime(selected_year, selected_month, days_in_month)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Initialize price array
    prices = np.zeros(len(dates))
    base_price = starting_price  # User-provided starting price
    
    # Generate astrological events for the selected month
    events = generate_astrological_events(start_date, 'monthly')
    
    # Set event prices based on impact
    for event in events:
        # Calculate price based on impact and base price
        price_change = event["impact"] * base_price * 0.01  # Scale impact as percentage of base price
        event["price"] = base_price + price_change
    
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
            change = (event_influence + random_change) * base_price * 0.002  # Scale to base price
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
        y_pos = event['price'] + base_price * 0.02 if event['price'] < base_price * 1.02 else event['price'] - base_price * 0.02
        ax_main.annotate(event['aspect'], 
                        xy=(event['date'], event['price']),
                        xytext=(event['date'], y_pos),
                        arrowprops=dict(arrowstyle='->', color=marker_color, lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                        fontsize=9, ha='center')
    
    # Formatting
    ax_main.set_title(f'{symbol} Monthly Chart - {start_date.strftime("%B %Y")}\n(Astrological Transits & Aspects)', 
                     fontsize=16, pad=20)
    ax_main.set_xlabel('Date', fontsize=12)
    ax_main.set_ylabel('Price', fontsize=12)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax_main.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, days_in_month//10)))
    plt.setp(ax_main.get_xticklabels(), rotation=45, ha='right')
    
    # Add month-end close annotation
    ax_main.annotate(f'Month Close: {df_monthly["Price"].iloc[-1]:.2f}', 
                    xy=(df_monthly['Date'].iloc[-1], df_monthly['Price'].iloc[-1]),
                    xytext=(df_monthly['Date'].iloc[-1] - timedelta(days=days_in_month//5), 
                           df_monthly['Price'].iloc[-1] + base_price * 0.02),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                    fontsize=12, fontweight='bold',
                    bbox=dict(facecolor='yellow', alpha=0.7, edgecolor='none', pad=2))
    
    # Planetary positions for key dates
    ax_planets = fig.add_subplot(gs[0, 1])
    key_dates = [
        start_date,  # Beginning of month
        start_date + timedelta(days=days_in_month//3),  # First third
        start_date + timedelta(days=2*days_in_month//3),  # Second third
        end_date  # End of month
    ]
    
    for i, date in enumerate(key_dates):
        ax_sub = fig.add_axes([0.75, 0.8-i*0.2, 0.15, 0.15])
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
    ax_calendar.set_title('Astrological Aspect Strength Throughout the Month', fontsize=12)
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
            'Mercury Retrograde', 
            'Venus Opposition Saturn', 
            'Moon-Jupiter Trine', 
            'Full Moon', 
            'Jupiter Square Saturn', 
            'Mercury Direct', 
            'Venus enters Libra', 
            'New Moon'
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
        'Typical Price Change': [
            '±1-2%', 
            '-1.5-2%', 
            '+1-2%', 
            '±0.5-1%', 
            '-2-3%', 
            '+0.5-1%', 
            '+0.5-1%', 
            '+1-2%'
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
    for change in df_aspects['Typical Price Change']:
        # Extract the percentage value
        if change.startswith('+'):
            num = float(change[1:].replace('%', ''))
        elif change.startswith('-'):
            num = -float(change[1:].replace('%', ''))
        elif change.startswith('±'):
            num = float(change[1:].replace('%', ''))
        else:
            num = 0
        price_changes.append(num)
    
    # Create color map based on impact
    colors = ['red' if 'Bearish' in impact or 'Tension' in impact or 'Volatility' in impact 
              else 'green' for impact in df_aspects['Market Impact']]
    
    bars = ax.bar(df_aspects['Aspect'], price_changes, color=colors, alpha=0.7)
    
    # Add labels
    ax.set_title('Astrological Aspect Impact on Price Changes', fontsize=14, pad=20)
    ax.set_ylabel('Typical Price Change (%)', fontsize=12)
    
    # Rotate x-axis labels and adjust alignment
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    return fig, df_aspects

# --- STREAMLIT APP ---
def main():
    st.title('🌟 Universal Astrological Trading Dashboard')
    st.write('This interactive dashboard generates charts for any symbol based on astrological transits and aspects.')
    
    # Sidebar for inputs
    st.sidebar.title('📊 Chart Configuration')
    
    # Symbol input
    symbol = st.sidebar.text_input(
        'Enter Symbol:',
        value='NIFTY',
        max_chars=10
    ).upper()
    
    # Starting price input
    starting_price = st.sidebar.number_input(
        'Enter Starting Price:',
        min_value=0.01,
        value=24620.0,
        step=1.0
    )
    
    # Chart type selection
    chart_type = st.sidebar.selectbox(
        'Choose a chart to generate:',
        ['Intraday Chart', 'Monthly Chart', 'Aspect Analysis']
    )
    
    if chart_type == 'Intraday Chart':
        st.header(f'📈 {symbol} Intraday Chart')
        st.write('Simulated intraday price movements based on astrological transits and aspects.')
        
        # Date selector
        selected_date = st.date_input(
            'Select a date:',
            value=datetime(2025, 8, 5).date(),
            min_value=datetime(2020, 1, 1).date(),
            max_value=datetime(2030, 12, 31).date()
        )
        
        # Generate chart
        fig = generate_intraday_chart(symbol, starting_price, selected_date)
        st.pyplot(fig)
        
        # Additional information
        st.subheader('🔮 Astrological Insights')
        st.info(f"""
        **Key Astrological Events for {selected_date.strftime("%B %d, %Y")}:**
        - **Mercury Retrograde**: Causes market volatility and unpredictable movements
        - **Moon-Jupiter Trine**: Creates bullish sentiment and optimism
        - **Moon-Saturn Square**: Triggers fear and selling pressure
        - **Venus-Saturn Opposition**: Creates financial stress and risk aversion
        
        **Trading Strategy:**
        - Best time to buy: Around Moon-Jupiter trine (10:00 AM)
        - Best time to sell: Before Moon-Saturn square (2:00 PM)
        - Avoid trading during Mercury retrograde periods
        """)
        
    elif chart_type == 'Monthly Chart':
        st.header(f'📊 {symbol} Monthly Chart')
        st.write('Simulated daily closing prices based on astrological transits and aspects.')
        
        # Month and year selectors
        col1, col2 = st.columns(2)
        with col1:
            selected_month = st.selectbox(
                'Select Month:',
                range(1, 13),
                format_func=lambda x: calendar.month_name[x],
                index=7  # August
            )
        with col2:
            selected_year = st.selectbox(
                'Select Year:',
                range(2020, 2031),
                index=5  # 2025
            )
        
        # Generate chart
        fig = generate_monthly_chart(symbol, starting_price, selected_month, selected_year)
        st.pyplot(fig)
        
        # Additional information
        month_name = calendar.month_name[selected_month]
        st.subheader(f'🌙 Monthly Astrological Summary - {month_name} {selected_year}')
        st.info(f"""
        **{month_name} {selected_year} Key Transits:**
        - **Mercury Retrograde**: High volatility, avoid major decisions
        - **Jupiter Square Saturn**: Major tension between optimism and restriction
        - **Mercury Direct**: Clarity returns, good for new positions
        - **New Moon**: Strong bullish close to the month
        
        **Market Phases:**
        1. **Early {month_name}**: Bearish pressure with high volatility
        2. **Mid {month_name}**: Stabilization and consolidation
        3. **Late {month_name}**: Bullish surge with financial sector rally
        """)
        
    elif chart_type == 'Aspect Analysis':
        st.header('📋 Astrological Aspect Analysis')
        st.write('Analysis of key astrological aspects and their impact on market prices.')
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
