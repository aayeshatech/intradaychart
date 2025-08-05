import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, date as date_class
import calendar
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

# --- TRADING HOURS CONFIGURATION ---
TRADING_HOURS = {
    # Indian Market Hours
    'NIFTY': {'start_hour': 9, 'start_minute': 15, 'end_hour': 15, 'end_minute': 30},
    'BANKNIFTY': {'start_hour': 9, 'start_minute': 15, 'end_hour': 15, 'end_minute': 30},
    'TCS': {'start_hour': 9, 'start_minute': 15, 'end_hour': 15, 'end_minute': 30},
    'ICICIBANK': {'start_hour': 9, 'start_minute': 15, 'end_hour': 15, 'end_minute': 30},
    'MARUTI': {'start_hour': 9, 'start_minute': 15, 'end_hour': 15, 'end_minute': 30},
    'DLF': {'start_hour': 9, 'start_minute': 15, 'end_hour': 15, 'end_minute': 30},
    'NESTLEIND': {'start_hour': 9, 'start_minute': 15, 'end_hour': 15, 'end_minute': 30},
    'RELIANCE': {'start_hour': 9, 'start_minute': 15, 'end_hour': 15, 'end_minute': 30},
    'SBI': {'start_hour': 9, 'start_minute': 15, 'end_hour': 15, 'end_minute': 30},
    'SUNPHARMA': {'start_hour': 9, 'start_minute': 15, 'end_hour': 15, 'end_minute': 30},
    'DRREDDY': {'start_hour': 9, 'start_minute': 15, 'end_hour': 15, 'end_minute': 30},
    
    # Global Market Hours (24-hour trading)
    'GOLD': {'start_hour': 5, 'start_minute': 0, 'end_hour': 23, 'end_minute': 55},
    'DOWJONES': {'start_hour': 5, 'start_minute': 0, 'end_hour': 23, 'end_minute': 55},
    'SILVER': {'start_hour': 5, 'start_minute': 0, 'end_hour': 23, 'end_minute': 55},
    'CRUDE': {'start_hour': 5, 'start_minute': 0, 'end_hour': 23, 'end_minute': 55},
    'BTC': {'start_hour': 5, 'start_minute': 0, 'end_hour': 23, 'end_minute': 55},
}

# --- SYMBOL CONFIGURATIONS ---
SYMBOL_CONFIG = {
    'NIFTY': {'name': 'Nifty 50', 'currency': '₹', 'default_price': 24620.0, 'sector': 'Index'},
    'BANKNIFTY': {'name': 'Bank Nifty', 'currency': '₹', 'default_price': 52000.0, 'sector': 'Banking Index'},
    'TCS': {'name': 'Tata Consultancy Services', 'currency': '₹', 'default_price': 4200.0, 'sector': 'Technology'},
    'ICICIBANK': {'name': 'ICICI Bank', 'currency': '₹', 'default_price': 1200.0, 'sector': 'Banking'},
    'MARUTI': {'name': 'Maruti Suzuki', 'currency': '₹', 'default_price': 12000.0, 'sector': 'Automotive'},
    'DLF': {'name': 'DLF Limited', 'currency': '₹', 'default_price': 800.0, 'sector': 'Realty'},
    'NESTLEIND': {'name': 'Nestlé India', 'currency': '₹', 'default_price': 2400.0, 'sector': 'FMCG'},
    'RELIANCE': {'name': 'Reliance Industries', 'currency': '₹', 'default_price': 3000.0, 'sector': 'Energy'},
    'SBI': {'name': 'State Bank of India', 'currency': '₹', 'default_price': 850.0, 'sector': 'PSU Banking'},
    'SUNPHARMA': {'name': 'Sun Pharma', 'currency': '₹', 'default_price': 1700.0, 'sector': 'Pharma'},
    'DRREDDY': {'name': 'Dr. Reddy Labs', 'currency': '₹', 'default_price': 6800.0, 'sector': 'Pharma'},
    
    'GOLD': {'name': 'Gold Futures', 'currency': '

# --- STOCK DATABASE ---
stock_data = {
    'Symbol': [
        'TCS', 'ICICIBANK', 'MARUTI', 'DLF', 'NESTLEIND', 
        'RELIANCE', 'SBI', 'SUNPHARMA', 'DRREDDY', 'GOLD',
        'DOWJONES', 'SILVER', 'CRUDE', 'BTC'
    ],
    'Sector': [
        'Technology', 'Banking', 'Automotive', 'Realty', 'FMCG',
        'Energy', 'PSUs', 'Pharma', 'Pharma', 'Precious Metals',
        'US Index', 'Precious Metals', 'Energy', 'Cryptocurrency'
    ],
    'MarketCap': [
        'Large', 'Large', 'Large', 'Large', 'Large',
        'Large', 'Large', 'Large', 'Large', 'Commodity',
        'Index', 'Commodity', 'Commodity', 'Crypto'
    ]
}

STOCK_DATABASE = pd.DataFrame(stock_data)

# --- SECTOR-PLANETARY MAPPINGS ---
SECTOR_PLANETARY_INFLUENCES = {
    'Technology': ['Mercury'],
    'Banking': ['Jupiter', 'Saturn'],
    'FMCG': ['Moon'],
    'Pharma': ['Neptune'],
    'Energy': ['Mars'],
    'Automotive': ['Saturn'],
    'Realty': ['Saturn'],
    'PSUs': ['Pluto'],
    'Midcaps': ['Uranus'],
    'Smallcaps': ['Pluto'],
    'Precious Metals': ['Venus', 'Jupiter'],
    'US Index': ['Jupiter', 'Saturn'],
    'Cryptocurrency': ['Uranus', 'Pluto']
}

# --- ASPECT-SECTOR IMPACT ---
ASPECT_SECTOR_IMPACTS = {
    'Square': {
        'Technology': 'Negative', 'Banking': 'Negative', 'FMCG': 'Neutral',
        'Pharma': 'Neutral', 'Energy': 'Neutral', 'Automotive': 'Neutral',
        'Realty': 'Neutral', 'PSUs': 'Neutral', 'Midcaps': 'Neutral',
        'Smallcaps': 'Neutral', 'Precious Metals': 'Positive', 'US Index': 'Negative',
        'Cryptocurrency': 'Negative'
    },
    'Opposition': {
        'Technology': 'Neutral', 'Banking': 'Neutral', 'FMCG': 'Neutral',
        'Pharma': 'Neutral', 'Energy': 'Neutral', 'Automotive': 'Negative',
        'Realty': 'Negative', 'PSUs': 'Neutral', 'Midcaps': 'Neutral',
        'Smallcaps': 'Neutral', 'Precious Metals': 'Positive', 'US Index': 'Neutral',
        'Cryptocurrency': 'Positive'
    },
    'Trine': {
        'Technology': 'Neutral', 'Banking': 'Neutral', 'FMCG': 'Positive',
        'Pharma': 'Positive', 'Energy': 'Neutral', 'Automotive': 'Neutral',
        'Realty': 'Neutral', 'PSUs': 'Neutral', 'Midcaps': 'Neutral',
        'Smallcaps': 'Neutral', 'Precious Metals': 'Positive', 'US Index': 'Positive',
        'Cryptocurrency': 'Neutral'
    },
    'Conjunction': {
        'Technology': 'Neutral', 'Banking': 'Neutral', 'FMCG': 'Neutral',
        'Pharma': 'Neutral', 'Energy': 'Positive', 'Automotive': 'Neutral',
        'Realty': 'Neutral', 'PSUs': 'Neutral', 'Midcaps': 'Negative',
        'Smallcaps': 'Neutral', 'Precious Metals': 'Positive', 'US Index': 'Neutral',
        'Cryptocurrency': 'Positive'
    },
    'Sextile': {
        'Technology': 'Neutral', 'Banking': 'Neutral', 'FMCG': 'Neutral',
        'Pharma': 'Neutral', 'Energy': 'Neutral', 'Automotive': 'Neutral',
        'Realty': 'Neutral', 'PSUs': 'Positive', 'Midcaps': 'Neutral',
        'Smallcaps': 'Negative', 'Precious Metals': 'Positive', 'US Index': 'Positive',
        'Cryptocurrency': 'Neutral'
    }
}

# --- PLANETARY POSITION VISUALIZATION ---
def draw_planetary_wheel(ax, input_date, size=0.3):
    """Draw a simplified astrological wheel showing planetary positions"""
    base_date = datetime(2025, 8, 1)
    
    if isinstance(input_date, date_class) and not isinstance(input_date, datetime):
        date_obj = datetime.combine(input_date, datetime.min.time())
    else:
        date_obj = input_date
    
    days_diff = (date_obj.date() - base_date.date()).days
    
    base_positions = {
        'Sun': 135, 'Moon': 225, 'Mercury': 120, 'Venus': 170,
        'Mars': 85, 'Jupiter': 45, 'Saturn': 315
    }
    
    daily_movement = {
        'Sun': 1.0, 'Moon': 13.2, 'Mercury': 1.5, 'Venus': 1.2,
        'Mars': 0.5, 'Jupiter': 0.08, 'Saturn': 0.03
    }
    
    planets = {}
    for planet, base_pos in base_positions.items():
        angle = (base_pos + daily_movement[planet] * days_diff) % 360
        planets[planet] = {
            'angle': angle,
            'color': {
                'Sun': 'gold', 'Moon': 'silver', 'Mercury': 'gray',
                'Venus': 'lightgreen', 'Mars': 'red', 'Jupiter': 'orange',
                'Saturn': 'darkgoldenrod'
            }[planet],
            'size': {
                'Sun': 8, 'Moon': 6, 'Mercury': 5, 'Venus': 7,
                'Mars': 6, 'Jupiter': 10, 'Saturn': 9
            }[planet]
        }
    
    zodiac = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 
              'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
    
    for i, sign in enumerate(zodiac):
        angle = i * 30
        ax.add_patch(Wedge((0, 0), size, angle, angle+30, width=size*0.8, 
                          facecolor='lightgray', edgecolor='black', alpha=0.3))
        ax.text(0.85*size * np.cos(np.radians(angle+15)), 
                0.85*size * np.sin(np.radians(angle+15)), 
                sign[:3], ha='center', va='center', fontsize=6)
    
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
    ax.set_title(f'Planetary Positions\n{date_obj.strftime("%b %d, %Y")}', fontsize=8)

# --- GET TRADING HOURS FOR SYMBOL ---
def get_trading_hours(symbol):
    """Get trading hours for a specific symbol"""
    symbol = symbol.upper()
    if symbol in TRADING_HOURS:
        return TRADING_HOURS[symbol]
    else:
        # Default to Indian market hours for unknown symbols
        return TRADING_HOURS['NIFTY']

# --- GET SYMBOL INFO ---
def get_symbol_info(symbol):
    """Get symbol configuration info"""
    symbol = symbol.upper()
    if symbol in SYMBOL_CONFIG:
        return SYMBOL_CONFIG[symbol]
    else:
        # Default configuration
        return {
            'name': symbol,
            'currency': '₹',
            'default_price': 1000.0,
            'sector': 'Unknown'
        }

# --- GENERATE ASPECTS ---
def generate_todays_aspects():
    """Generate astrological aspects for today based on the provided table"""
    base_aspects = [
        {"planets": "Mercury-Jupiter", "aspect_type": "Square", "impact": -0.7, "type": "bearish"},
        {"planets": "Venus-Saturn", "aspect_type": "Opposition", "impact": -0.8, "type": "bearish"},
        {"planets": "Moon-Neptune", "aspect_type": "Trine", "impact": 0.6, "type": "bullish"},
        {"planets": "Mars-Uranus", "aspect_type": "Conjunction", "impact": 0.9, "type": "bullish"},
        {"planets": "Sun-Pluto", "aspect_type": "Sextile", "impact": 0.5, "type": "bullish"}
    ]
    
    aspects = []
    for aspect in base_aspects:
        aspects.append({
            "planets": aspect["planets"],
            "aspect_type": aspect["aspect_type"],
            "impact": aspect["impact"],
            "type": aspect["type"]
        })
    
    return aspects

# --- CREATE SUMMARY TABLE ---
def create_summary_table(aspects):
    """Create a summary table based on the astrological aspects"""
    summary_data = {
        'Aspect': [],
        'Nifty/Bank Nifty': [],
        'Bullish Sectors/Stocks': [],
        'Bearish Sectors/Stocks': []
    }
    
    for aspect in aspects:
        planets = aspect["planets"]
        aspect_type = aspect["aspect_type"]
        
        if planets == "Mercury-Jupiter" and aspect_type == "Square":
            summary_data['Aspect'].append("Mercury-Jupiter (Square)")
            summary_data['Nifty/Bank Nifty'].append("Volatile")
            summary_data['Bullish Sectors/Stocks'].append("IT (TCS), Gold")
            summary_data['Bearish Sectors/Stocks'].append("Banking (ICICI Bank), Crypto")
        
        elif planets == "Venus-Saturn" and aspect_type == "Opposition":
            summary_data['Aspect'].append("Venus-Saturn (Opposition)")
            summary_data['Nifty/Bank Nifty'].append("Downside")
            summary_data['Bullish Sectors/Stocks'].append("Gold, Silver, BTC")
            summary_data['Bearish Sectors/Stocks'].append("Auto (Maruti), Realty (DLF)")
        
        elif planets == "Moon-Neptune" and aspect_type == "Trine":
            summary_data['Aspect'].append("Moon-Neptune (Trine)")
            summary_data['Nifty/Bank Nifty'].append("Mild Support")
            summary_data['Bullish Sectors/Stocks'].append("FMCG (Nestlé), Pharma, Gold, Dow Jones")
            summary_data['Bearish Sectors/Stocks'].append("-")
        
        elif planets == "Mars-Uranus" and aspect_type == "Conjunction":
            summary_data['Aspect'].append("Mars-Uranus (Conjunction)")
            summary_data['Nifty/Bank Nifty'].append("Sharp Moves")
            summary_data['Bullish Sectors/Stocks'].append("Energy (Reliance, Crude), Gold, BTC")
            summary_data['Bearish Sectors/Stocks'].append("Weak Midcaps")
        
        elif planets == "Sun-Pluto" and aspect_type == "Sextile":
            summary_data['Aspect'].append("Sun-Pluto (Sextile)")
            summary_data['Nifty/Bank Nifty'].append("Structural Shift")
            summary_data['Bullish Sectors/Stocks'].append("PSUs (SBI), Gold, Dow Jones")
            summary_data['Bearish Sectors/Stocks'].append("Overvalued Smallcaps")
    
    return pd.DataFrame(summary_data)

# --- FILTER STOCKS BASED ON ASPECTS ---
def filter_stocks_by_aspects(aspects, stock_database):
    """Filter stocks based on today's astrological aspects"""
    sector_impacts = {sector: 0 for sector in stock_database['Sector'].unique()}
    
    for aspect in aspects:
        planet1, planet2 = aspect["planets"].split("-")
        
        for sector, planets in SECTOR_PLANETARY_INFLUENCES.items():
            if planet1 in planets or planet2 in planets:
                if sector not in sector_impacts:
                    sector_impacts[sector] = 0
                
                aspect_impact = ASPECT_SECTOR_IMPACTS[aspect["aspect_type"]].get(sector, "Neutral")
                
                if aspect_impact == "Positive":
                    sector_impacts[sector] += abs(aspect["impact"])
                elif aspect_impact == "Negative":
                    sector_impacts[sector] -= abs(aspect["impact"])
    
    bullish_sectors = [sector for sector, impact in sector_impacts.items() if impact > 0]
    bearish_sectors = [sector for sector, impact in sector_impacts.items() if impact < 0]
    neutral_sectors = [sector for sector, impact in sector_impacts.items() if impact == 0]
    
    bullish_stocks = stock_database[stock_database['Sector'].isin(bullish_sectors)].copy()
    bearish_stocks = stock_database[stock_database['Sector'].isin(bearish_sectors)].copy()
    neutral_stocks = stock_database[stock_database['Sector'].isin(neutral_sectors)].copy()
    
    bullish_stocks['Impact Score'] = bullish_stocks['Sector'].apply(lambda x: sector_impacts[x])
    bearish_stocks['Impact Score'] = bearish_stocks['Sector'].apply(lambda x: abs(sector_impacts[x]))
    neutral_stocks['Impact Score'] = 0
    
    bullish_stocks = bullish_stocks.sort_values('Impact Score', ascending=False)
    bearish_stocks = bearish_stocks.sort_values('Impact Score', ascending=False)
    
    return {
        'bullish': bullish_stocks,
        'bearish': bearish_stocks,
        'neutral': neutral_stocks,
        'sector_impacts': sector_impacts
    }

# --- GENERATE ASTROLOGICAL EVENTS ---
def generate_astrological_events(input_date, event_type='intraday', symbol='NIFTY'):
    """Generate astrological events for any given date and symbol"""
    
    if event_type == 'intraday':
        trading_hours = get_trading_hours(symbol)
        
        # Different event patterns based on trading hours
        if trading_hours['end_hour'] > 16:  # Extended hours (global markets)
            # More events spread across longer trading day
            base_events = [
                {"time_offset": 0, "aspect": "Pre-market: Mercury square Jupiter", "impact": -0.5, "type": "bearish"},
                {"time_offset": 120, "aspect": "Asian session: Moon trine Jupiter", "impact": 0.8, "type": "bullish"},
                {"time_offset": 240, "aspect": "London open: Mars sextile Jupiter", "impact": 0.4, "type": "neutral"},
                {"time_offset": 360, "aspect": "European session: Venus opposition Saturn", "impact": -0.6, "type": "bearish"},
                {"time_offset": 480, "aspect": "NY pre-market: Sun conjunct Mercury", "impact": 0.3, "type": "neutral"},
                {"time_offset": 600, "aspect": "US open: Mars conjunct Uranus", "impact": 1.0, "type": "bullish"},
                {"time_offset": 720, "aspect": "Mid-day: Moon square Saturn", "impact": -0.4, "type": "bearish"},
                {"time_offset": 840, "aspect": "Afternoon: Jupiter trine Neptune", "impact": 0.7, "type": "bullish"},
                {"time_offset": 960, "aspect": "US close approach", "impact": 0.2, "type": "neutral"},
                {"time_offset": 1080, "aspect": "After hours: Void Moon", "impact": -0.3, "type": "bearish"},
                {"time_offset": 1135, "aspect": "Session close", "impact": 0.1, "type": "neutral"}
            ]
        else:  # Standard Indian market hours
            base_events = [
                {"time_offset": 0, "aspect": "Opening: Mercury square Jupiter + Void Moon", "impact": -0.5, "type": "bearish"},
                {"time_offset": 45, "aspect": "Early trade: Moon trine Jupiter", "impact": 1.0, "type": "bullish"},
                {"time_offset": 135, "aspect": "Mid-morning: Mars sextile Jupiter", "impact": 0.3, "type": "neutral"},
                {"time_offset": 195, "aspect": "Pre-lunch: Sun in Leo (no aspects)", "impact": 0.0, "type": "neutral"},
                {"time_offset": 285, "aspect": "Post-lunch: Moon square Saturn", "impact": -0.8, "type": "bearish"},
                {"time_offset": 345, "aspect": "Late trade: Venus-Saturn opposition", "impact": -0.6, "type": "bearish"},
                {"time_offset": 375, "aspect": "Close", "impact": 0.1, "type": "neutral"}
            ]
        
        events = []
        if isinstance(input_date, date_class) and not isinstance(input_date, datetime):
            dt = datetime.combine(input_date, datetime.min.time())
        else:
            dt = input_date
            
        start_time = dt.replace(hour=trading_hours['start_hour'], minute=trading_hours['start_minute'])
        
        for event in base_events:
            event_time = start_time + timedelta(minutes=event["time_offset"])
            events.append({
                "time": event_time,
                "aspect": event["aspect"],
                "impact": event["impact"],
                "type": event["type"],
                "price": 0
            })
        
        return events
    
    else:  # monthly events remain the same
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
        
        if isinstance(input_date, datetime):
            year, month = input_date.year, input_date.month
        else:
            year, month = input_date.year, input_date.month
            
        days_in_month = calendar.monthrange(year, month)[1]
        
        events = []
        for event in base_events:
            day = min(event["day_offset"], days_in_month)
            event_date = datetime(year, month, day)
            events.append({
                "date": event_date,
                "aspect": event["aspect"],
                "impact": event["impact"],
                "type": event["type"],
                "price": 0
            })
        
        return events

# --- ENHANCED INTRADAY CHART ---
def generate_intraday_chart(symbol, starting_price, selected_date):
    """Generate enhanced intraday chart with dynamic layout"""
    symbol_info = get_symbol_info(symbol)
    trading_hours = get_trading_hours(symbol)
    
    if isinstance(selected_date, date_class) and not isinstance(selected_date, datetime):
        selected_date = datetime.combine(selected_date, datetime.min.time())
    
    start_time = selected_date.replace(hour=trading_hours['start_hour'], minute=trading_hours['start_minute'])
    end_time = selected_date.replace(hour=trading_hours['end_hour'], minute=trading_hours['end_minute'])
    
    # Adjust interval based on trading session length
    session_hours = (end_time - start_time).total_seconds() / 3600
    if session_hours > 12:
        interval = '30T'  # 30-minute intervals for long sessions
    else:
        interval = '15T'  # 15-minute intervals for shorter sessions
    
    times = pd.date_range(start=start_time, end=end_time, freq=interval)
    
    prices = np.zeros(len(times))
    base_price = starting_price
    
    events = generate_astrological_events(selected_date, 'intraday', symbol)
    
    # Adjust event impacts based on symbol type
    symbol_multiplier = 1.0
    if symbol in ['GOLD', 'SILVER']:
        symbol_multiplier = 0.8  # Precious metals less volatile to aspects
    elif symbol in ['BTC']:
        symbol_multiplier = 2.0  # Crypto more volatile
    elif symbol in ['CRUDE']:
        symbol_multiplier = 1.5  # Energy commodities more responsive
    
    for event in events:
        price_change = event["impact"] * base_price * 0.01 * symbol_multiplier
        event["price"] = base_price + price_change
    
    # Generate price movements
    for i, time in enumerate(times):
        closest_event = min(events, key=lambda x: abs((x["time"] - time).total_seconds()))
        distance = abs((closest_event["time"] - time).total_seconds()) / 3600
        
        # Adjust volatility based on symbol
        base_volatility = 0.15 if distance < 0.5 else 0.05
        if symbol in ['BTC']:
            base_volatility *= 3.0
        elif symbol in ['GOLD', 'SILVER']:
            base_volatility *= 0.5
        elif symbol in ['CRUDE']:
            base_volatility *= 2.0
        
        random_change = np.random.normal(0, base_volatility)
        event_influence = closest_event["impact"] * np.exp(-distance) * symbol_multiplier
        
        if i == 0:
            prices[i] = base_price
        else:
            change = (event_influence + random_change) * base_price * 0.001
            prices[i] = prices[i-1] + change
    
    df_intraday = pd.DataFrame({
        'Time': times,
        'Price': prices,
        'Aspect': [min(events, key=lambda x: abs((x["time"] - t).total_seconds()))["aspect"] for t in times]
    })
    
    # Create dynamic figure layout
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, height_ratios=[4, 1, 1, 1], width_ratios=[5, 2, 1])
    
    # Main price chart
    ax_main = fig.add_subplot(gs[0, :2])
    
    for i in range(1, len(df_intraday)):
        color = 'green' if df_intraday['Price'].iloc[i] > df_intraday['Price'].iloc[i-1] else 'red'
        ax_main.plot(df_intraday['Time'].iloc[i-1:i+1], 
                    df_intraday['Price'].iloc[i-1:i+1], 
                    color=color, linewidth=2.5)
    
    # Mark key events
    for event in events:
        color_map = {'bullish': 'green', 'bearish': 'red', 'neutral': 'blue'}
        ax_main.axvline(x=event['time'], color=color_map[event['type']], 
                       linestyle='--', alpha=0.7, linewidth=2)
        
        marker_color = color_map[event['type']]
        ax_main.scatter(event['time'], event['price'], color=marker_color, 
                       s=100, zorder=5, edgecolor='black', linewidth=1)
        
        # Dynamic annotation positioning
        y_offset = base_price * 0.02 if len(str(int(base_price))) >= 4 else base_price * 0.05
        y_pos = event['price'] + y_offset if event['price'] < base_price * 1.01 else event['price'] - y_offset
        
        ax_main.annotate(event['aspect'][:30] + '...' if len(event['aspect']) > 30 else event['aspect'], 
                        xy=(event['time'], event['price']),
                        xytext=(event['time'], y_pos),
                        arrowprops=dict(arrowstyle='->', color=marker_color, lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                        fontsize=8, ha='center')
    
    # Dynamic title and formatting
    currency_symbol = symbol_info['currency']
    ax_main.set_title(f'{symbol_info["name"]} ({symbol}) - {selected_date.strftime("%B %d, %Y")}\n'
                     f'Astrological Trading Analysis | Sector: {symbol_info["sector"]}', 
                     fontsize=16, pad=20)
    ax_main.set_xlabel(f'Time ({trading_hours["start_hour"]}:00 - {trading_hours["end_hour"]}:00)', fontsize=12)
    ax_main.set_ylabel(f'Price ({currency_symbol})', fontsize=12)
    
    # Dynamic time formatting based on session length
    if session_hours > 12:
        ax_main.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    else:
        ax_main.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    plt.setp(ax_main.get_xticklabels(), rotation=45, ha='right')
    
    # Closing price annotation
    close_price = df_intraday["Price"].iloc[-1]
    price_change = close_price - base_price
    price_change_pct = (price_change / base_price) * 100
    
    ax_main.annotate(f'Close: {currency_symbol}{close_price:.2f}\n'
                    f'Change: {price_change:+.2f} ({price_change_pct:+.2f}%)', 
                    xy=(df_intraday['Time'].iloc[-1], close_price),
                    xytext=(df_intraday['Time'].iloc[-1] - timedelta(hours=session_hours*0.2), 
                           close_price + base_price * 0.02),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                    fontsize=11, fontweight='bold',
                    bbox=dict(facecolor='yellow', alpha=0.7, edgecolor='none', pad=2))
    
    # Planetary wheel
    ax_wheel = fig.add_subplot(gs[0, 2])
    draw_planetary_wheel(ax_wheel, selected_date, size=0.4)
    
    # Volume chart (simulated with realistic patterns)
    ax_volume = fig.add_subplot(gs[1, :2])
    
    # Generate more realistic volume based on symbol type
    if symbol in ['BTC']:
        base_volume = np.random.randint(50000, 200000, size=len(times))
    elif symbol in ['GOLD', 'SILVER', 'CRUDE']:
        base_volume = np.random.randint(10000, 50000, size=len(times))
    elif symbol in ['DOWJONES']:
        base_volume = np.random.randint(100000, 500000, size=len(times))
    else:  # Indian stocks
        base_volume = np.random.randint(1000, 10000, size=len(times))
    
    colors_volume = ['green' if df_intraday['Price'].iloc[i] > df_intraday['Price'].iloc[i-1] 
                    else 'red' for i in range(1, len(df_intraday))]
    colors_volume.insert(0, 'green')
    
    ax_volume.bar(df_intraday['Time'], base_volume, color=colors_volume, alpha=0.7)
    ax_volume.set_title('Volume', fontsize=12)
    ax_volume.set_ylabel('Volume', fontsize=10)
    ax_volume.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax_volume.get_xticklabels(), rotation=45, ha='right')
    
    # Technical indicators (RSI simulation)
    ax_rsi = fig.add_subplot(gs[2, :2])
    rsi_values = 50 + np.random.normal(0, 15, len(times))  # Simulated RSI
    rsi_values = np.clip(rsi_values, 0, 100)
    
    ax_rsi.plot(df_intraday['Time'], rsi_values, color='purple', linewidth=2)
    ax_rsi.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
    ax_rsi.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
    ax_rsi.fill_between(df_intraday['Time'], 30, 70, alpha=0.1, color='gray')
    ax_rsi.set_title('RSI (14)', fontsize=12)
    ax_rsi.set_ylabel('RSI', fontsize=10)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.legend(loc='upper right', fontsize=8)
    
    # Aspect strength indicator
    ax_aspect = fig.add_subplot(gs[3, :2])
    aspect_times = [event['time'] for event in events]
    aspect_strengths = [abs(event['impact']) for event in events]
    aspect_colors = [{'bullish': 'green', 'bearish': 'red', 'neutral': 'blue'}[event['type']] for event in events]
    
    ax_aspect.scatter(aspect_times, aspect_strengths, color=aspect_colors, s=100, zorder=3)
    ax_aspect.plot(aspect_times, aspect_strengths, color='gray', alpha=0.5, linestyle='--')
    ax_aspect.set_title('Astrological Aspect Strength', fontsize=12)
    ax_aspect.set_ylabel('Strength', fontsize=10)
    ax_aspect.set_ylim(0, max(aspect_strengths) * 1.2 if aspect_strengths else 1.5)
    
    # Symbol info panel
    ax_info = fig.add_subplot(gs[1:, 2])
    ax_info.axis('off')
    
    info_text = f"""
SYMBOL INFO
-----------
Name: {symbol_info['name']}
Sector: {symbol_info['sector']}
Currency: {symbol_info['currency']}

TRADING HOURS
-------------
Start: {trading_hours['start_hour']:02d}:{trading_hours['start_minute']:02d}
End: {trading_hours['end_hour']:02d}:{trading_hours['end_minute']:02d}
Session: {session_hours:.1f} hours

PRICE DATA
----------
Open: {currency_symbol}{base_price:.2f}
Close: {currency_symbol}{close_price:.2f}
Change: {price_change:+.2f}
Change%: {price_change_pct:+.2f}%

High: {currency_symbol}{max(prices):.2f}
Low: {currency_symbol}{min(prices):.2f}
Range: {currency_symbol}{max(prices)-min(prices):.2f}
"""
    
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig

# --- ENHANCED MONTHLY CHART ---
def generate_monthly_chart(symbol, starting_price, selected_month, selected_year):
    """Generate enhanced monthly chart with dynamic layout"""
    symbol_info = get_symbol_info(symbol)
    
    start_date = datetime(selected_year, selected_month, 1)
    days_in_month = calendar.monthrange(selected_year, selected_month)[1]
    end_date = datetime(selected_year, selected_month, days_in_month)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    prices = np.zeros(len(dates))
    base_price = starting_price
    
    events = generate_astrological_events(start_date, 'monthly', symbol)
    
    # Adjust event impacts based on symbol type
    symbol_multiplier = 1.0
    if symbol in ['GOLD', 'SILVER']:
        symbol_multiplier = 0.8
    elif symbol in ['BTC']:
        symbol_multiplier = 2.5
    elif symbol in ['CRUDE']:
        symbol_multiplier = 1.8
    
    for event in events:
        price_change = event["impact"] * base_price * 0.01 * symbol_multiplier
        event["price"] = base_price + price_change
    
    # Generate price movements
    for i, date in enumerate(dates):
        closest_event = min(events, key=lambda x: abs((x["date"].date() - date.date()).days))
        distance = abs((closest_event["date"].date() - date.date()).days)
        
        base_volatility = 0.3 if distance < 2 else 0.1
        if symbol in ['BTC']:
            base_volatility *= 4.0
        elif symbol in ['GOLD', 'SILVER']:
            base_volatility *= 0.6
        elif symbol in ['CRUDE']:
            base_volatility *= 2.5
        
        random_change = np.random.normal(0, base_volatility)
        event_influence = closest_event["impact"] * np.exp(-distance/2) * symbol_multiplier
        
        if i == 0:
            prices[i] = base_price
        else:
            change = (event_influence + random_change) * base_price * 0.002
            prices[i] = prices[i-1] + change
    
    df_monthly = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Aspect': [min(events, key=lambda x: abs((x["date"].date() - d.date()).days))["aspect"] for d in dates]
    })
    
    # Create dynamic figure layout
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, height_ratios=[4, 1, 1, 1], width_ratios=[5, 2, 1])
    
    # Main price chart
    ax_main = fig.add_subplot(gs[0, :2])
    
    for i in range(1, len(df_monthly)):
        color = 'green' if df_monthly['Price'].iloc[i] > df_monthly['Price'].iloc[i-1] else 'red'
        ax_main.plot(df_monthly['Date'].iloc[i-1:i+1], 
                    df_monthly['Price'].iloc[i-1:i+1], 
                    color=color, linewidth=3)
    
    # Mark key events
    for event in events:
        color_map = {'bullish': 'green', 'bearish': 'red', 'neutral': 'blue'}
        ax_main.axvline(x=event['date'], color=color_map[event['type']], 
                       linestyle='--', alpha=0.7, linewidth=2)
        
        marker_color = color_map[event['type']]
        ax_main.scatter(event['date'], event['price'], color=marker_color, 
                       s=150, zorder=5, edgecolor='black', linewidth=1.5)
        
        y_offset = base_price * 0.03
        y_pos = event['price'] + y_offset if event['price'] < base_price * 1.02 else event['price'] - y_offset
        
        ax_main.annotate(event['aspect'][:25] + '...' if len(event['aspect']) > 25 else event['aspect'], 
                        xy=(event['date'], event['price']),
                        xytext=(event['date'], y_pos),
                        arrowprops=dict(arrowstyle='->', color=marker_color, lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                        fontsize=9, ha='center')
    
    # Dynamic formatting
    currency_symbol = symbol_info['currency']
    ax_main.set_title(f'{symbol_info["name"]} ({symbol}) - {start_date.strftime("%B %Y")}\n'
                     f'Monthly Astrological Analysis | Sector: {symbol_info["sector"]}', 
                     fontsize=16, pad=20)
    ax_main.set_xlabel('Date', fontsize=12)
    ax_main.set_ylabel(f'Price ({currency_symbol})', fontsize=12)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax_main.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, days_in_month//10)))
    plt.setp(ax_main.get_xticklabels(), rotation=45, ha='right')
    
    # Monthly close annotation
    close_price = df_monthly["Price"].iloc[-1]
    price_change = close_price - base_price
    price_change_pct = (price_change / base_price) * 100
    
    ax_main.annotate(f'Month Close: {currency_symbol}{close_price:.2f}\n'
                    f'Monthly Change: {price_change:+.2f} ({price_change_pct:+.2f}%)', 
                    xy=(df_monthly['Date'].iloc[-1], close_price),
                    xytext=(df_monthly['Date'].iloc[-1] - timedelta(days=days_in_month//4), 
                           close_price + base_price * 0.03),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                    fontsize=12, fontweight='bold',
                    bbox=dict(facecolor='yellow', alpha=0.7, edgecolor='none', pad=2))
    
    # Planetary positions for key dates
    ax_planets = fig.add_subplot(gs[0, 2])
    ax_planets.set_title('Key Planetary\nPositions', fontsize=10)
    key_dates = [
        start_date,
        start_date + timedelta(days=days_in_month//3),
        start_date + timedelta(days=2*days_in_month//3),
        end_date
    ]
    
    for i, date in enumerate(key_dates):
        ax_sub = fig.add_axes([0.70, 0.8-i*0.15, 0.12, 0.12])
        draw_planetary_wheel(ax_sub, date, size=0.4)
        ax_sub.set_title(f'{date.strftime("%b %d")}', fontsize=8)
    
    # Volume chart
    ax_volume = fig.add_subplot(gs[1, :2])
    
    if symbol in ['BTC']:
        volume = np.random.randint(500000, 2000000, size=len(dates))
    elif symbol in ['GOLD', 'SILVER', 'CRUDE']:
        volume = np.random.randint(100000, 500000, size=len(dates))
    elif symbol in ['DOWJONES']:
        volume = np.random.randint(1000000, 5000000, size=len(dates))
    else:
        volume = np.random.randint(10000, 100000, size=len(dates))
    
    colors_volume = ['green' if df_monthly['Price'].iloc[i] > df_monthly['Price'].iloc[i-1] 
                    else 'red' for i in range(1, len(df_monthly))]
    colors_volume.insert(0, 'green')
    
    ax_volume.bar(df_monthly['Date'], volume, color=colors_volume, alpha=0.7)
    ax_volume.set_title('Daily Volume', fontsize=12)
    ax_volume.set_ylabel('Volume', fontsize=10)
    ax_volume.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    # Moving averages
    ax_ma = fig.add_subplot(gs[2, :2])
    ma_5 = df_monthly['Price'].rolling(window=5, min_periods=1).mean()
    ma_20 = df_monthly['Price'].rolling(window=min(20, len(df_monthly)), min_periods=1).mean()
    
    ax_ma.plot(df_monthly['Date'], ma_5, color='blue', linewidth=2, label='MA5', alpha=0.7)
    ax_ma.plot(df_monthly['Date'], ma_20, color='red', linewidth=2, label='MA20', alpha=0.7)
    ax_ma.fill_between(df_monthly['Date'], ma_5, ma_20, alpha=0.1, 
                      color='green' if ma_5.iloc[-1] > ma_20.iloc[-1] else 'red')
    ax_ma.set_title('Moving Averages', fontsize=12)
    ax_ma.set_ylabel('Price', fontsize=10)
    ax_ma.legend(loc='upper left', fontsize=10)
    
    # Aspect calendar
    ax_calendar = fig.add_subplot(gs[3, :2])
    aspect_dates = [event['date'] for event in events]
    aspect_strengths = [abs(event['impact']) for event in events]
    aspect_colors = [{'bullish': 'green', 'bearish': 'red', 'neutral': 'blue'}[event['type']] for event in events]
    
    ax_calendar.scatter(aspect_dates, aspect_strengths, color=aspect_colors, s=200, zorder=3)
    ax_calendar.plot(aspect_dates, aspect_strengths, color='gray', alpha=0.5, linestyle='--')
    ax_calendar.set_title('Monthly Astrological Event Strength', fontsize=12)
    ax_calendar.set_ylabel('Impact Strength', fontsize=10)
    ax_calendar.set_ylim(0, max(aspect_strengths) * 1.2 if aspect_strengths else 2)
    
    # Monthly summary panel
    ax_summary = fig.add_subplot(gs[1:, 2])
    ax_summary.axis('off')
    
    monthly_high = max(prices)
    monthly_low = min(prices)
    monthly_range = monthly_high - monthly_low
    avg_price = np.mean(prices)
    
    summary_text = f"""
MONTHLY SUMMARY
--------------
Symbol: {symbol}
Sector: {symbol_info['sector']}
Month: {start_date.strftime('%B %Y')}

PRICE STATISTICS
---------------
Open: {currency_symbol}{base_price:.2f}
Close: {currency_symbol}{close_price:.2f}
Change: {price_change:+.2f}
Change%: {price_change_pct:+.2f}%

High: {currency_symbol}{monthly_high:.2f}
Low: {currency_symbol}{monthly_low:.2f}
Range: {currency_symbol}{monthly_range:.2f}
Average: {currency_symbol}{avg_price:.2f}

VOLATILITY
----------
Daily Avg: {np.std(np.diff(prices)):.2f}
Monthly Vol: {(monthly_range/avg_price)*100:.1f}%

TREND ANALYSIS
--------------
Bullish Days: {sum(1 for i in range(1, len(prices)) if prices[i] > prices[i-1])}
Bearish Days: {sum(1 for i in range(1, len(prices)) if prices[i] < prices[i-1])}
Neutral Days: {sum(1 for i in range(1, len(prices)) if prices[i] == prices[i-1])}
"""
    
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes, fontsize=8,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    return fig

# --- ANALYZE ASPECTS ---
def analyze_aspects():
    """Enhanced aspect analysis with dynamic content"""
    aspects_data = {
        'Aspect': [
            'Mercury Retrograde', 'Venus Opposition Saturn', 'Moon-Jupiter Trine', 
            'Full Moon', 'Jupiter Square Saturn', 'Mercury Direct',
            'Venus enters Libra', 'New Moon', 'Mars-Uranus Conjunction',
            'Sun-Pluto Sextile'
        ],
        'Market Impact': [
            'High Volatility', 'Bearish Pressure', 'Bullish Surge', 
            'Trend Reversal', 'Major Tension', 'Clarity Returns',
            'Financial Rally', 'Strong Bullish', 'Energy Surge',
            'Structural Change'
        ],
        'Typical Price Change': [
            '±2-3%', '-1.5-2%', '+1-2%', 
            '±1-1.5%', '-2-3%', '+0.5-1%',
            '+0.5-1%', '+1-2%', '+2-4%',
            '±1-2%'
        ],
        'Sector Focus': [
            'All Sectors', 'Banking/Realty', 'Broad Market', 
            'Technology', 'Financials', 'Technology',
            'Banking/Finance', 'Broad Market', 'Energy/Commodities',
            'Infrastructure/PSUs'
        ],
        'Best Symbols': [
            'Gold, BTC', 'Gold, Silver', 'FMCG, Pharma', 
            'Tech Stocks', 'Defensive', 'Tech, Crypto',
            'Banking', 'Growth Stocks', 'Energy, Crude',
            'PSU, Infrastructure'
        ]
    }
    
    df_aspects = pd.DataFrame(aspects_data)
    
    # Create enhanced visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Price change impact chart
    price_changes = []
    for change in df_aspects['Typical Price Change']:
        clean_change = change.replace('%', '').replace('±', '')
        if '-' in clean_change and not clean_change.startswith('-'):
            num_str = clean_change.split('-')[1]  # Take higher value for impact
        else:
            num_str = clean_change.replace('+', '')
        
        try:
            num = float(num_str)
        except:
            num = 1.0
        price_changes.append(num)
    
    colors = ['red' if 'Bearish' in impact or 'Tension' in impact or 'Volatility' in impact 
              else 'orange' if 'Reversal' in impact or 'Change' in impact
              else 'green' for impact in df_aspects['Market Impact']]
    
    bars1 = ax1.bar(range(len(df_aspects)), price_changes, color=colors, alpha=0.7)
    ax1.set_title('Astrological Aspect Impact on Price Changes', fontsize=14)
    ax1.set_ylabel('Maximum Price Change (%)', fontsize=12)
    ax1.set_xticks(range(len(df_aspects)))
    ax1.set_xticklabels(df_aspects['Aspect'], rotation=45, ha='right')
    
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.annotate(f'{height}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    # Sector distribution pie chart
    sector_counts = {}
    for sectors in df_aspects['Sector Focus']:
        for sector in sectors.split('/'):
            sector = sector.strip()
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
    
    ax2.pie(sector_counts.values(), labels=sector_counts.keys(), autopct='%1.1f%%', startangle=90)
    ax2.set_title('Most Affected Sectors by Astrological Aspects', fontsize=14)
    
    # Market impact distribution
    impact_counts = {}
    for impact in df_aspects['Market Impact']:
        impact_type = 'Bullish' if any(word in impact for word in ['Bullish', 'Rally', 'Surge', 'Returns']) else \
                     'Bearish' if any(word in impact for word in ['Bearish', 'Pressure', 'Tension']) else \
                     'Neutral'
        impact_counts[impact_type] = impact_counts.get(impact_type, 0) + 1
    
    colors_impact = ['green', 'red', 'gray']
    ax3.bar(impact_counts.keys(), impact_counts.values(), color=colors_impact, alpha=0.7)
    ax3.set_title('Distribution of Market Impact Types', fontsize=14)
    ax3.set_ylabel('Number of Aspects', fontsize=12)
    
    # Best performing symbols chart
    symbol_mentions = {}
    for symbols in df_aspects['Best Symbols']:
        for symbol in symbols.split(', '):
            symbol = symbol.strip()
            symbol_mentions[symbol] = symbol_mentions.get(symbol, 0) + 1
    
    sorted_symbols = sorted(symbol_mentions.items(), key=lambda x: x[1], reverse=True)
    symbols, counts = zip(*sorted_symbols) if sorted_symbols else ([], [])
    
    ax4.barh(symbols, counts, color='gold', alpha=0.7)
    ax4.set_title('Most Favorable Symbols Across Aspects', fontsize=14)
    ax4.set_xlabel('Favorable Mentions', fontsize=12)
    
    return fig, df_aspects

# --- STREAMLIT APP ---
def main():
    # Page configuration for better responsive design
    st.set_page_config(
        page_title="🌟 Astrological Trading Dashboard",
        page_icon="🌟",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .symbol-input {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #e8f4fd;
        border-left: 5px solid #1f77b4;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🌟 Astrological Trading Dashboard</h1>
        <p>Advanced Financial Analysis through Planetary Movements & Cosmic Patterns</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for inputs with enhanced design
    with st.sidebar:
        st.markdown("### 📊 Dashboard Controls")
        
        # Dashboard section selection with better descriptions
        dashboard_section = st.selectbox(
            '🎯 Choose Analysis Section:',
            [
                'Summary Table - Market Overview',
                'Stock Filter - Sector Analysis', 
                'Aspect Analysis - Deep Insights',
                'Intraday Chart - Live Patterns',
                'Monthly Chart - Trend Analysis'
            ]
        )
        
        # Extract the main section name
        section_name = dashboard_section.split(' - ')[0]
        
        st.markdown("---")
        
        # Symbol selection with enhanced interface
        if section_name in ['Intraday Chart', 'Monthly Chart']:
            st.markdown("### 📈 Symbol Configuration")
            
            # Popular symbols with categories
            symbol_categories = {
                'Indian Indices': ['NIFTY', 'BANKNIFTY'],
                'Indian Stocks': ['TCS', 'ICICIBANK', 'MARUTI', 'DLF', 'NESTLEIND', 'RELIANCE', 'SBI', 'SUNPHARMA', 'DRREDDY'],
                'Global Markets': ['DOWJONES'],
                'Commodities': ['GOLD', 'SILVER', 'CRUDE'],
                'Cryptocurrency': ['BTC']
            }
            
            selected_category = st.selectbox('📂 Select Category:', list(symbol_categories.keys()))
            
            if selected_category:
                symbol_options = symbol_categories[selected_category]
                selected_symbol = st.selectbox('🎯 Choose Symbol:', symbol_options)
                
                # Custom symbol input
                custom_symbol = st.text_input('✏️ Or enter custom symbol:', max_chars=10)
                symbol = custom_symbol.upper() if custom_symbol else selected_symbol
                
                # Get symbol info for dynamic defaults
                symbol_info = get_symbol_info(symbol)
                trading_hours = get_trading_hours(symbol)
                
                # Display symbol information
                st.markdown(f"""
                <div class="info-box">
                    <strong>📊 {symbol_info['name']}</strong><br>
                    <small>Sector: {symbol_info['sector']}</small><br>
                    <small>Currency: {symbol_info['currency']}</small><br>
                    <small>Trading: {trading_hours['start_hour']:02d}:{trading_hours['start_minute']:02d} - {trading_hours['end_hour']:02d}:{trading_hours['end_minute']:02d}</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Price input with dynamic default
                starting_price = st.number_input(
                    f'💰 Starting Price ({symbol_info["currency"]}):',
                    min_value=0.01,
                    value=symbol_info['default_price'],
                    step=1.0 if symbol_info['default_price'] > 100 else 0.01,
                    format="%.2f"
                )
                
                # Date/time selection based on chart type
                if section_name == 'Intraday Chart':
                    selected_date = st.date_input(
                        '📅 Select Trading Date:',
                        value=datetime(2025, 8, 5).date(),
                        min_value=datetime(2020, 1, 1).date(),
                        max_value=datetime(2030, 12, 31).date()
                    )
                elif section_name == 'Monthly Chart':
                    col1, col2 = st.columns(2)
                    with col1:
                        selected_month = st.selectbox(
                            '📅 Month:',
                            range(1, 13),
                            format_func=lambda x: calendar.month_name[x],
                            index=7  # August
                        )
                    with col2:
                        selected_year = st.selectbox(
                            '📅 Year:',
                            range(2020, 2031),
                            index=5  # 2025
                        )
        
        # Trading insights
        st.markdown("---")
        st.markdown("### 🔮 Quick Insights")
        
        # Generate today's aspects for sidebar display
        aspects = generate_todays_aspects()
        bullish_count = sum(1 for aspect in aspects if aspect['type'] == 'bullish')
        bearish_count = sum(1 for aspect in aspects if aspect['type'] == 'bearish')
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🟢 Bullish Aspects", bullish_count)
        with col2:
            st.metric("🔴 Bearish Aspects", bearish_count)
        
        # Market sentiment
        if bullish_count > bearish_count:
            sentiment = "🟢 Bullish"
            sentiment_color = "green"
        elif bearish_count > bullish_count:
            sentiment = "🔴 Bearish"
            sentiment_color = "red"
        else:
            sentiment = "🟡 Neutral"
            sentiment_color = "orange"
        
        st.markdown(f"**Overall Sentiment:** <span style='color:{sentiment_color}'>{sentiment}</span>", 
                   unsafe_allow_html=True)

    # Main content area
    aspects = generate_todays_aspects()
    
    if section_name == 'Summary Table':
        st.header('📋 Market Summary & Astrological Overview')
        
        # Create columns for better layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader('📊 Today\'s Astrological Aspects Impact')
            summary_df = create_summary_table(aspects)
            
            # Style the dataframe
            styled_df = summary_df.style.apply(
                lambda x: ['background-color: #d4edda' if 'Bullish' in str(val) or '+' in str(val) 
                          else 'background-color: #f8d7da' if 'Bearish' in str(val) or 'Downside' in str(val)
                          else '' for val in x], axis=0
            )
            
            st.dataframe(styled_df, use_container_width=True)
        
        with col2:
            st.subheader('🎯 Key Metrics')
            
            # Calculate impact scores
            total_impact = sum(abs(aspect['impact']) for aspect in aspects)
            avg_impact = total_impact / len(aspects) if aspects else 0
            
            st.metric("Total Cosmic Energy", f"{total_impact:.1f}")
            st.metric("Average Impact", f"{avg_impact:.2f}")
            st.metric("Active Aspects", len(aspects))
            
            # Risk assessment
            high_risk_aspects = sum(1 for aspect in aspects if abs(aspect['impact']) > 0.7)
            risk_level = "High" if high_risk_aspects >= 3 else "Medium" if high_risk_aspects >= 1 else "Low"
            risk_color = "red" if risk_level == "High" else "orange" if risk_level == "Medium" else "green"
            
            st.markdown(f"**Risk Level:** <span style='color:{risk_color}'>{risk_level}</span>", 
                       unsafe_allow_html=True)
        
        # Detailed insights
        st.subheader('🔮 Detailed Market Insights')
        
        # Create tabs for different insights
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Strategy", "📈 Opportunities", "⚠️ Risks", "🌟 Cosmic Events"])
        
        with tab1:
            st.markdown("""
            **🎯 Recommended Trading Strategy:**
            
            **🟢 Bullish Opportunities:**
            - **Energy Sector**: Mars-Uranus conjunction favors Reliance, Crude Oil
            - **Precious Metals**: Multiple aspects support Gold and Silver
            - **FMCG & Pharma**: Moon-Neptune trine provides defensive strength
            - **PSU Stocks**: Sun-Pluto sextile indicates structural positives
            
            **🔴 Bearish Risks:**
            - **Banking Sector**: Mercury-Jupiter square creates volatility
            - **Automotive & Realty**: Venus-Saturn opposition brings pressure
            - **Technology**: Mixed signals, trade with caution
            
            **⚡ High-Impact Trades:**
            - Consider Gold positions during Venus-Saturn opposition
            - Energy stocks may see sharp moves (Mars-Uranus)
            - BTC could be volatile but trending up on global aspects
            """)
        
        with tab2:
            st.markdown("""
            **📈 Sector-wise Opportunities:**
            
            **🥇 Top Picks:**
            1. **Gold/Silver**: Multiple supportive aspects across all planetary configurations
            2. **Energy Commodities**: Mars-Uranus conjunction + global supply dynamics
            3. **Pharmaceutical**: Moon-Neptune trine supports defensive healthcare
            4. **PSU Banking**: Sun-Pluto sextile for structural transformation
            
            **🎯 Specific Symbols:**
            - **GOLD**: $2,050+ target on safe-haven demand
            - **CRUDE**: Energy transition + Mars-Uranus = volatility opportunities
            - **BTC**: Crypto favorable on Uranus-Pluto aspects
            - **SBI**: PSU transformation theme
            """)
        
        with tab3:
            st.markdown("""
            **⚠️ Risk Management:**
            
            **🔴 High-Risk Sectors:**
            - **Private Banking**: ICICI Bank under Mercury-Jupiter square pressure
            - **Automotive**: Maruti facing Venus-Saturn headwinds
            - **Real Estate**: DLF vulnerable to credit tightening aspects
            
            **📊 Risk Mitigation:**
            - Reduce position sizes during Mercury-Jupiter square (high volatility)
            - Use stop-losses 2-3% below support for Venus-Saturn affected stocks
            - Avoid leveraged positions in Midcap segment (Mars-Uranus volatility)
            
            **⏰ Timing Risks:**
            - Morning session volatility expected (Mercury aspects)
            - Post-lunch session may see pressure (Saturn influence)
            """)
        
        with tab4:
            st.markdown("""
            **🌟 Today's Cosmic Events Schedule:**
            
            **🌅 Pre-Market (Before 9:15 AM):**
            - Mercury-Jupiter square builds tension
            - Global markets influence domestic opening
            
            **🌄 Morning Session (9:15-12:00):**
            - Initial volatility from Mercury aspects
            - Energy stocks may show strength
            
            **🌞 Afternoon Session (12:00-15:30):**
            - Venus-Saturn opposition peaks
            - Defensive sectors gain relative strength
            - Banking sector under pressure
            
            **🌆 Post-Market:**
            - Global commodity movements (Gold, Crude)
            - Crypto markets reaction to day's developments
            
            **📊 Weekly Outlook:**
            - Aspects intensify mid-week
            - Weekend planetary shifts to monitor
            """)
    
    elif section_name == 'Stock Filter':
        st.header('🔍 Advanced Stock Filtering & Sector Analysis')
        
        # Create main layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader('🌟 Today\'s Astrological Configuration')
            
            # Display aspects in a nice format
            aspects_data = []
            for aspect in aspects:
                aspects_data.append({
                    'Planets': aspect['planets'],
                    'Aspect': aspect['aspect_type'],
                    'Impact': f"{aspect['impact']:+.1f}",
                    'Sentiment': aspect['type'].title(),
                    'Strength': '🔥' * min(3, int(abs(aspect['impact']) * 3))
                })
            
            aspects_df = pd.DataFrame(aspects_data)
            
            # Color code the dataframe
            def color_sentiment(val):
                if 'Bullish' in str(val):
                    return 'background-color: #d4edda; color: #155724'
                elif 'Bearish' in str(val):
                    return 'background-color: #f8d7da; color: #721c24'
                return ''
            
            styled_aspects = aspects_df.style.applymap(color_sentiment, subset=['Sentiment'])
            st.dataframe(styled_aspects, use_container_width=True)
        
        with col2:
            st.subheader('📊 Aspect Statistics')
            
            # Create a simple pie chart for aspect types
            aspect_types = {}
            for aspect in aspects:
                aspect_types[aspect['type']] = aspect_types.get(aspect['type'], 0) + 1
            
            if aspect_types:
                fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
                colors = ['green' if k == 'bullish' else 'red' if k == 'bearish' else 'gray' 
                         for k in aspect_types.keys()]
                wedges, texts, autotexts = ax_pie.pie(aspect_types.values(), 
                                                     labels=[k.title() for k in aspect_types.keys()], 
                                                     colors=colors, autopct='%1.0f%%', startangle=90)
                ax_pie.set_title('Today\'s Aspect Distribution')
                st.pyplot(fig_pie)
        
        # Sector analysis
        st.subheader('📈 Sector Impact Analysis')
        
        filtered_stocks = filter_stocks_by_aspects(aspects, STOCK_DATABASE)
        
        # Create sector impact visualization
        sector_impacts_df = pd.DataFrame({
            'Sector': list(filtered_stocks['sector_impacts'].keys()),
            'Impact Score': list(filtered_stocks['sector_impacts'].values()),
            'Recommendation': ['Strong Buy' if x > 0.5 else 'Buy' if x > 0 else 'Hold' if x == 0 
                             else 'Sell' if x > -0.5 else 'Strong Sell' 
                             for x in filtered_stocks['sector_impacts'].values()]
        })
        sector_impacts_df = sector_impacts_df.sort_values('Impact Score', ascending=False)
        
        # Enhanced bar chart
        fig_sectors, ax_sectors = plt.subplots(figsize=(14, 8))
        colors = ['darkgreen' if x > 0.5 else 'green' if x > 0 else 'gray' if x == 0 
                 else 'red' if x > -0.5 else 'darkred' 
                 for x in sector_impacts_df['Impact Score']]
        
        bars = ax_sectors.bar(sector_impacts_df['Sector'], sector_impacts_df['Impact Score'], 
                             color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax_sectors.set_title('Sector Impact Scores - Astrological Analysis', fontsize=16, pad=20)
        ax_sectors.set_ylabel('Impact Score', fontsize=12)
        ax_sectors.set_xlabel('Sector', fontsize=12)
        ax_sectors.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax_sectors.grid(True, alpha=0.3)
        
        # Add value labels and recommendations
        for i, (bar, rec) in enumerate(zip(bars, sector_impacts_df['Recommendation'])):
            height = bar.get_height()
            ax_sectors.annotate(f'{height:.2f}\n{rec}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 5 if height >= 0 else -25),
                               textcoords="offset points",
                               ha='center', va='bottom' if height >= 0 else 'top',
                               fontsize=9, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig_sectors)
        
        # Stock recommendations in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader('🟢 Bullish Stocks')
            if not filtered_stocks['bullish'].empty:
                bullish_df = filtered_stocks['bullish'][['Symbol', 'Sector', 'Impact Score']].copy()
                bullish_df['Action'] = bullish_df['Impact Score'].apply(
                    lambda x: 'Strong Buy' if x > 0.5 else 'Buy'
                )
                
                for _, row in bullish_df.iterrows():
                    with st.container():
                        st.markdown(f"""
                        **{row['Symbol']}** ({row['Sector']})  
                        Score: {row['Impact Score']:.2f} | **{row['Action']}**
                        """)
                        st.progress(min(1.0, row['Impact Score']))
            else:
                st.info("No strong bullish signals today")
        
        with col2:
            st.subheader('🔴 Bearish Stocks')
            if not filtered_stocks['bearish'].empty:
                bearish_df = filtered_stocks['bearish'][['Symbol', 'Sector', 'Impact Score']].copy()
                bearish_df['Action'] = bearish_df['Impact Score'].apply(
                    lambda x: 'Strong Sell' if x > 0.5 else 'Sell'
                )
                
                for _, row in bearish_df.iterrows():
                    with st.container():
                        st.markdown(f"""
                        **{row['Symbol']}** ({row['Sector']})  
                        Risk Score: {row['Impact Score']:.2f} | **{row['Action']}**
                        """)
                        st.progress(min(1.0, row['Impact Score']))
            else:
                st.info("No strong bearish signals today")
        
        with col3:
            st.subheader('⚪ Neutral Stocks')
            if not filtered_stocks['neutral'].empty:
                neutral_df = filtered_stocks['neutral'][['Symbol', 'Sector']].head(5)
                
                for _, row in neutral_df.iterrows():
                    st.markdown(f"**{row['Symbol']}** ({row['Sector']}) - Hold")
            else:
                st.info("All stocks showing directional bias")
    
    elif section_name == 'Aspect Analysis':
        st.header('📋 Deep Astrological Aspect Analysis')
        
        # Generate enhanced analysis
        fig, df_aspects = analyze_aspects()
        st.pyplot(fig)
        
        # Display detailed aspect table
        st.subheader('📊 Detailed Aspect Reference Table')
        
        # Add more columns for better analysis
        df_enhanced = df_aspects.copy()
        df_enhanced['Trading Action'] = df_enhanced.apply(
            lambda row: 'Hedge/Reduce' if 'Bearish' in row['Market Impact'] or 'Tension' in row['Market Impact']
            else 'Accumulate' if 'Bullish' in row['Market Impact'] or 'Rally' in row['Market Impact']
            else 'Monitor', axis=1
        )
        
        df_enhanced['Risk Level'] = df_enhanced['Typical Price Change'].apply(
            lambda x: 'High' if any(num in x for num in ['3', '4']) 
            else 'Medium' if '2' in x else 'Low'
        )
        
        # Style the enhanced dataframe
        def highlight_risk(val):
            if val == 'High':
                return 'background-color: #ffcccc'
            elif val == 'Medium':
                return 'background-color: #ffffcc'
            elif val == 'Low':
                return 'background-color: #ccffcc'
            return ''
        
        def highlight_action(val):
            if val == 'Accumulate':
                return 'background-color: #d4edda; color: #155724'
            elif val == 'Hedge/Reduce':
                return 'background-color: #f8d7da; color: #721c24'
            elif val == 'Monitor':
                return 'background-color: #fff3cd; color: #856404'
            return ''
        
        styled_enhanced = df_enhanced.style.applymap(highlight_risk, subset=['Risk Level']).applymap(highlight_action, subset=['Trading Action'])
        st.dataframe(styled_enhanced, use_container_width=True)
        
        # Aspect interpretation guide
        st.subheader('🔭 Astrological Aspect Interpretation Guide')
        
        tab1, tab2, tab3 = st.tabs(["🌟 Aspect Types", "🪐 Planetary Influences", "📈 Trading Applications"])
        
        with tab1:
            st.markdown("""
            ### Understanding Astrological Aspects
            
            **🔄 Conjunction (0°)**: 
            - *Market Effect*: Powerful combining of energies, can create sharp moves
            - *Trading*: Expect significant price action, potential breakouts
            - *Example*: Mars-Uranus conjunction = explosive energy moves
            
            **⚔️ Square (90°)**: 
            - *Market Effect*: Tension, conflict, volatility
            - *Trading*: Increased intraday swings, good for scalping
            - *Example*: Mercury-Jupiter square = communication/policy confusion
            
            **🎯 Trine (120°)**: 
            - *Market Effect*: Harmonious, easy flow of energy
            - *Trading*: Trending moves, good for position trading
            - *Example*: Moon-Neptune trine = emotional/intuitive support
            
            **⚖️ Opposition (180°)**: 
            - *Market Effect*: Polarization, requires balance
            - *Trading*: Range-bound action, reversals possible
            - *Example*: Venus-Saturn opposition = value vs. restriction
            
            **🤝 Sextile (60°)**: 
            - *Market Effect*: Opportunity aspects, mild positive
            - *Trading*: Gentle trends, good for swing trades
            - *Example*: Sun-Pluto sextile = gradual transformation
            """)
        
        with tab2:
            st.markdown("""
            ### Planetary Market Influences
            
            **☀️ Sun**: Leadership, government policy, large-cap stocks, gold
            **🌙 Moon**: Public sentiment, emotions, consumer sectors, silver
            **☿️ Mercury**: Communication, technology, volatility, news-driven moves
            **♀️ Venus**: Finance, banking, luxury goods, relationships, copper
            **♂️ Mars**: Energy, metals, defense, aggressive moves, oil
            **♃ Jupiter**: Growth, expansion, optimism, financial sector
            **♄ Saturn**: Restriction, discipline, structure, defensive sectors
            **♅ Uranus**: Innovation, technology, sudden changes, crypto
            **♆ Neptune**: Illusion, oil, pharma, confusion, speculation
            **♇ Pluto**: Transformation, power, mining, major shifts
            
            ### Sector-Planet Correlations
            - **Technology**: Mercury, Uranus
            - **Banking**: Jupiter, Venus, Saturn  
            - **Energy**: Mars, Sun, Pluto
            - **Healthcare**: Neptune, Moon
            - **Precious Metals**: Venus, Jupiter, Sun
            - **Cryptocurrency**: Uranus, Pluto
            """)
        
        with tab3:
            st.markdown("""
            ### Practical Trading Applications
            
            **📊 Intraday Trading:**
            - Use Moon aspects for sentiment shifts (2-4 hour cycles)
            - Mercury aspects for news/volatility spikes
            - Mars aspects for energy sector breakouts
            
            **📈 Swing Trading:**
            - Venus aspects for financial sector trends (3-7 days)
            - Jupiter aspects for broad market optimism
            - Saturn aspects for defensive positioning
            
            **📉 Position Trading:**
            - Outer planet aspects (Uranus, Neptune, Pluto) for long-term themes
            - Eclipse patterns for major sector rotations
            - Retrograde periods for trend reversals
            
            **⚠️ Risk Management:**
            - Increase cash during multiple challenging aspects
            - Reduce position size during Mercury retrograde
            - Use tighter stops during Mars-Saturn squares
            
            **🎯 Sector Rotation:**
            - Follow Jupiter through zodiac signs for sector leadership
            - Track Saturn aspects for value opportunities
            - Monitor Uranus for innovation themes
            """)
    
    elif section_name == 'Intraday Chart':
        st.header(f'📈 {symbol} - Intraday Astrological Analysis')
        
        # Display symbol information prominently
        symbol_info = get_symbol_info(symbol)
        trading_hours = get_trading_hours(symbol)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Symbol", symbol)
        with col2:
            st.metric("Sector", symbol_info['sector'])
        with col3:
            st.metric("Currency", symbol_info['currency'])
        with col4:
            session_length = trading_hours['end_hour'] - trading_hours['start_hour'] + \
                           (trading_hours['end_minute'] - trading_hours['start_minute']) / 60
            st.metric("Session Hours", f"{session_length:.1f}h")
        
        # Generate and display chart
        with st.spinner(f'Generating astrological analysis for {symbol}...'):
            fig = generate_intraday_chart(symbol, starting_price, selected_date)
            st.pyplot(fig)
        
        # Trading insights based on symbol
        st.subheader(f'🎯 {symbol} Trading Insights')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Technical Outlook")
            
            # Generate insights based on symbol type
            if symbol in ['GOLD', 'SILVER']:
                st.markdown("""
                **Precious Metals Analysis:**
                - Multiple planetary aspects favor safe-haven demand
                - Venus-Saturn opposition creates financial stress → Gold strength  
                - Moon-Neptune trine supports intuitive precious metal buying
                - Best trading windows: During global uncertainty aspects
                
                **Key Levels:**
                - Watch for breakouts during Mars-Uranus conjunction
                - Support likely during Moon aspects
                - Resistance at previous highs during Saturn aspects
                """)
            
            elif symbol in ['BTC']:
                st.markdown("""
                **Cryptocurrency Analysis:**
                - Uranus aspects strongly favor crypto volatility
                - Mars-Uranus conjunction = explosive price moves
                - Traditional financial stress (Venus-Saturn) → Crypto rotation
                - High volatility expected - use proper risk management
                
                **Trading Strategy:**
                - Momentum plays during Uranus aspects
                - Contrarian plays during Saturn oppositions
                - Volume spikes likely at aspect peaks
                """)
            
            elif symbol in ['CRUDE']:
                st.markdown("""
                **Energy Commodity Analysis:**
                - Mars-Uranus conjunction directly impacts energy sector
                - Global supply disruption themes (Pluto aspects)
                - Geopolitical tensions favor energy prices
                - Weather and seasonal patterns amplified by aspects
                
                **Supply-Demand Factors:**
                - Production disruptions during Mars aspects
                - Demand surges during economic aspects
                - Storage plays during Saturn aspects
                """)
            
            elif symbol in ['DOWJONES']:
                st.markdown("""
                **US Index Analysis:**
                - Jupiter aspects favor broad market optimism
                - Saturn aspects create rotation into defensive sectors
                - Mercury aspects increase intraday volatility
                - Fed policy sensitivity during Venus-Saturn opposition
                
                **Sector Rotation:**
                - Technology during Mercury aspects
                - Energy during Mars aspects  
                - Financials during Jupiter aspects
                """)
            
            else:  # Indian stocks
                st.markdown(f"""
                **{symbol_info['sector']} Sector Analysis:**
                - Domestic market influenced by global planetary patterns
                - FII/DII flows affected by Venus-Saturn aspects
                - Sector rotation based on planetary emphasis
                - Currency impacts during outer planet aspects
                
                **Indian Market Specifics:**
                - Opening gap influenced by global overnight aspects
                - Lunch hour volatility during Moon aspects
                - Closing session strength during Jupiter aspects
                """)
        
        with col2:
            st.markdown("#### ⏰ Timing Analysis")
            
            # Generate time-specific insights based on trading hours
            if trading_hours['end_hour'] > 16:  # Extended hours
                st.markdown("""
                **Extended Session Analysis:**
                
                **🌅 Asian Session (5:00-8:00):**
                - Pre-market positioning based on overnight aspects
                - Lower volumes, higher impact from aspects
                - Key economic data releases amplified
                
                **🌍 European Session (8:00-16:00):**
                - Peak liquidity and aspect impacts
                - Central bank policy influences
                - Cross-asset correlations strongest
                
                **🌎 US Session (16:00-20:00):**
                - Maximum volatility potential
                - Aspect peaks create significant moves
                - News flow interaction with cosmic patterns
                
                **🌙 After Hours (20:00-23:55):**
                - Reduced liquidity amplifies aspect effects
                - Position adjustments for next day
                - Asian preview impact
                """)
            else:  # Indian market hours
                st.markdown("""
                **Indian Session Analysis:**
                
                **🌅 Opening (9:15-10:30):**
                - Gap opening based on global aspects
                - High volatility, aspect impacts magnified
                - Initial trend direction setting
                
                **🌞 Mid-Morning (10:30-12:00):**
                - Institutional activity peaks
                - Aspect-driven sector rotation
                - News flow integration
                
                **🍽️ Lunch Hour (12:00-13:00):**
                - Reduced activity, Moon aspects dominate
                - Range-bound unless strong aspects active
                - Position consolidation period
                
                **🌆 Closing (13:00-15:30):**
                - Final institutional positioning
                - Aspect resolution for day
                - Next-day setup formation
                """)
            
            # Risk management
            st.markdown("#### ⚠️ Risk Management")
            st.markdown(f"""
            **Position Sizing:**
            - Standard position: 1-2% of capital
            - High aspect days: Reduce to 0.5-1%
            - Strong confluence: Increase to 2-3%
            
            **Stop Loss Levels:**
            - Tight stops during Mercury aspects: 1-2%
            - Normal stops during stable aspects: 2-3%
            - Wide stops during Mars aspects: 3-5%
            
            **Profit Targets:**
            - Quick scalps: 0.5-1% (15-30 minutes)
            - Swing trades: 2-5% (2-4 hours)
            - Position trades: 5-10% (1-3 days)
            
            **Volatility Adjustments:**
            - {symbol}: Expected daily range ±{2.5 if symbol in ['BTC'] else 1.5 if symbol in ['CRUDE'] else 1.0 if symbol in ['GOLD', 'SILVER'] else 0.8}%
            - Adjust position size inversely to volatility
            - Use options for high-volatility periods
            """)
    
    elif section_name == 'Monthly Chart':
        st.header(f'📊 {symbol} - Monthly Astrological Trend Analysis')
        
        # Display symbol information
        symbol_info = get_symbol_info(symbol)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Symbol", symbol)
        with col2:
            st.metric("Analysis Period", f"{calendar.month_name[selected_month]} {selected_year}")
        with col3:
            st.metric("Sector Focus", symbol_info['sector'])
        with col4:
            st.metric("Currency", symbol_info['currency'])
        
        # Generate and display chart
        with st.spinner(f'Generating monthly analysis for {symbol}...'):
            fig = generate_monthly_chart(symbol, starting_price, selected_month, selected_year)
            st.pyplot(fig)
        
        # Monthly analysis insights
        st.subheader(f'📈 {calendar.month_name[selected_month]} {selected_year} - Strategic Analysis')
        
        # Create tabs for different analysis aspects
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Monthly Outlook", "📊 Technical Analysis", "🌙 Lunar Cycles", "💼 Portfolio Strategy"])
        
        with tab1:
            month_name = calendar.month_name[selected_month]
            
            if symbol in ['GOLD', 'SILVER']:
                st.markdown(f"""
                ### {symbol} - {month_name} {selected_year} Precious Metals Outlook
                
                **🌟 Astrological Themes:**
                - **Venus-Jupiter aspects**: Strong precious metals demand from financial uncertainty
                - **Saturn transits**: Safe-haven buying during economic restrictions
                - **Moon phases**: Emotional buying patterns aligned with lunar cycles
                - **Mercury retrograde periods**: Technical analysis less reliable, fundamentals dominate
                
                **📈 Price Drivers:**
                - Central bank policy uncertainty (Saturn aspects)
                - Currency devaluation themes (Pluto aspects)
                - Geopolitical tensions (Mars aspects)
                - Inflation hedging demand (Jupiter-Saturn aspects)
                
                **🎯 Trading Strategy:**
                - **Accumulate** during New Moon phases (stronger buying interest)
                - **Profit-take** during Full Moon phases (emotional peaks)
                - **Hold through** Mercury retrograde (avoid technical trading)
                - **Scale in** during Saturn aspects (structural support)
                
                **📊 Target Levels:**
                - **Monthly High**: Expect during Jupiter-Venus trine periods
                - **Monthly Low**: Likely during Mars-Saturn square periods
                - **Breakout Potential**: Mars-Uranus conjunction periods
                - **Support Zones**: Previous month's Jupiter aspect levels
                """)
            
            elif symbol in ['BTC']:
                st.markdown(f"""
                ### {symbol} - {month_name} {selected_year} Cryptocurrency Outlook
                
                **⚡ Astrological Themes:**
                - **Uranus-Pluto aspects**: Revolutionary technology adoption waves
                - **Mercury-Uranus aspects**: Network upgrades and technical developments
                - **Mars-Uranus conjunctions**: Explosive price movements and FOMO
                - **Saturn aspects**: Regulatory clarity or restrictions
                
                **🚀 Volatility Drivers:**
                - Institutional adoption news (Jupiter aspects)
                - Regulatory developments (Saturn aspects)
                - Technical network changes (Mercury-Uranus)
                - Market manipulation concerns (Neptune aspects)
                
                **⚠️ Risk Factors:**
                - **High volatility** during Mars-Uranus aspects (±10-20% daily swings)
                - **Regulatory risks** during Saturn-Pluto aspects
                - **Technical failures** during Mercury retrograde
                - **Market manipulation** during Neptune-Mercury aspects
                
                **💡 Strategic Approach:**
                - **DCA strategy** during volatile periods
                - **Momentum trading** during Uranus aspects
                - **Risk-off** during Saturn hard aspects
                - **HODL mentality** during Jupiter-Pluto trines
                """)
            
            elif symbol in ['CRUDE']:
                st.markdown(f"""
                ### {symbol} - {month_name} {selected_year} Energy Commodity Outlook
                
                **🛢️ Astrological Themes:**
                - **Mars-Pluto aspects**: Geopolitical tensions affecting supply
                - **Jupiter-Saturn cycles**: Economic growth vs. restriction cycles
                - **Uranus aspects**: Renewable energy transition impacts
                - **Moon phases**: Seasonal demand patterns amplified
                
                **⚡ Supply-Demand Dynamics:**
                - Production disruptions (Mars-Saturn squares)
                - Economic growth spurts (Jupiter aspects)
                - Weather pattern extremes (Uranus-Neptune aspects)
                - Strategic reserve changes (Pluto aspects)
                
                **🌍 Geopolitical Factors:**
                - **OPEC decisions** aligned with Saturn aspects
                - **Pipeline disruptions** during Mars-Uranus periods
                - **Currency impacts** during Venus-Pluto aspects
                - **Seasonal patterns** enhanced by lunar cycles
                
                **📈 Trading Levels:**
                - **Resistance**: Previous Jupiter aspect highs
                - **Support**: Saturn aspect consolidation zones
                - **Breakout zones**: Mars-Uranus conjunction levels
                - **Reversal points**: Full Moon technical confluences
                """)
            
            elif symbol in ['DOWJONES']:
                st.markdown(f"""
                ### {symbol} - {month_name} {selected_year} US Index Outlook
                
                **🇺🇸 Macro Astrological Themes:**
                - **Jupiter-Saturn cycles**: Economic expansion vs. contraction
                - **Mercury-Venus aspects**: Corporate earnings and consumer spending
                - **Mars-Jupiter aspects**: Business investment and growth
                - **Outer planet aspects**: Long-term structural changes
                
                **🏛️ Federal Reserve Alignment:**
                - **Venus-Saturn aspects**: Interest rate policy changes
                - **Mercury-Jupiter aspects**: Fed communication clarity
                - **Moon phases**: Market sentiment around FOMC meetings
                - **Eclipse periods**: Major policy shift announcements
                
                **🔄 Sector Rotation Patterns:**
                - **Technology** leadership during Mercury-Uranus aspects
                - **Energy** strength during Mars-Pluto periods
                - **Financials** favor during Venus-Jupiter trines
                - **Healthcare** defensive during Saturn aspects
                
                **📊 Technical Confluence:**
                - **Monthly resistance**: Jupiter aspect previous highs
                - **Monthly support**: Saturn aspect previous lows
                - **Breakout potential**: New Moon near technical levels
                - **Reversal zones**: Full Moon at key Fibonacci levels
                """)
            
            else:  # Indian stocks
                st.markdown(f"""
                ### {symbol} ({symbol_info['sector']}) - {month_name} {selected_year} Indian Market Outlook
                
                **🇮🇳 Domestic Astrological Influences:**
                - **Jupiter transits**: Market leadership and FII flows
                - **Saturn aspects**: Regulatory changes and policy shifts
                - **Mars-Venus aspects**: Consumer spending and investment flows
                - **Moon phases**: Retail investor sentiment cycles
                
                **💹 Sector-Specific Themes:**
                - **{symbol_info['sector']} sector** influenced by specific planetary combinations
                - **Monsoon patterns** (if applicable) aligned with water sign emphasis
                - **Festival seasons** amplified by benefic planetary aspects
                - **Budget impacts** during Saturn-Jupiter aspects
                
                **🌏 Global Correlation Factors:**
                - **US Fed policy** impacts during Venus-Saturn aspects
                - **China growth** concerns during Mars-Saturn periods  
                - **Oil prices** affecting through Mars-Pluto aspects
                - **Dollar strength** impacts during Pluto aspects
                
                **📈 Monthly Strategy:**
                - **Accumulate** during Saturn aspects (value opportunities)
                - **Momentum plays** during Mars-Jupiter periods
                - **Defensive positioning** during challenging outer planet aspects
                - **Sector rotation** based on planetary emphasis shifts
                """)
        
        with tab2:
            st.markdown(f"""
            ### Technical Analysis Integration with Astrological Cycles
            
            **📊 Moving Average Alignment:**
            - **MA5 vs MA20**: Bullish when Jupiter aspects dominate
            - **Golden Cross** potential during Venus-Jupiter trines
            - **Death Cross** risk during Saturn-Mars squares
            - **MA support/resistance** stronger during lunar phases
            
            **🎯 Support & Resistance Levels:**
            - **Primary resistance**: Previous month's Jupiter aspect highs
            - **Primary support**: Saturn aspect consolidation lows
            - **Secondary levels**: Full Moon reversal points
            - **Breakout levels**: New Moon momentum points
            
            **📈 Momentum Indicators:**
            - **RSI overbought** (>70) more reliable during Full Moons
            - **RSI oversold** (<30) stronger signal during New Moons
            - **MACD divergences** amplified during Mercury aspects
            - **Volume confirmations** critical during Mars aspects
            
            **🌙 Lunar Cycle Technical Correlation:**
            - **New Moon**: Trend initiation, breakout potential
            - **Waxing Moon**: Momentum continuation, bullish bias
            - **Full Moon**: Trend exhaustion, reversal potential
            - **Waning Moon**: Correction phases, consolidation
            
            **⚡ Volatility Patterns:**
            - **Highest volatility**: Mars-Uranus aspect periods
            - **Lowest volatility**: Venus-Jupiter trine periods
            - **Unexpected moves**: Mercury-Neptune confusion aspects
            - **Gap movements**: Eclipse and outer planet aspects
            
            **🔄 Pattern Recognition:**
            - **Triangle breakouts** during Uranus aspects
            - **Flag patterns** during Mars aspects  
            - **Head & Shoulders** during Saturn aspects
            - **Double tops/bottoms** during opposition aspects
            """)
        
        with tab3:
            st.markdown(f"""
            ### Lunar Cycles & Market Psychology for {month_name} {selected_year}
            
            **🌑 New Moon Phases (Market Initiation):**
            - **Energy**: Fresh starts, new trend beginnings
            - **Psychology**: Optimism, risk-taking increases
            - **Trading**: Look for breakout setups, trend initiations
            - **Volume**: Often lower but quality moves
            - **Best for**: Opening new positions, trend following
            
            **🌓 Waxing Moon (Building Momentum):**
            - **Energy**: Growth, expansion, building confidence  
            - **Psychology**: FOMO starts building, bullish sentiment
            - **Trading**: Momentum continuation, pyramid additions
            - **Volume**: Increasing participation
            - **Best for**: Adding to winning positions
            
            **🌕 Full Moon Phases (Emotional Peaks):**
            - **Energy**: Maximum emotion, extremes, reversals
            - **Psychology**: Euphoria or panic peaks
            - **Trading**: Reversal setups, profit-taking
            - **Volume**: Often highest of cycle
            - **Best for**: Profit booking, contrarian plays
            
            **🌗 Waning Moon (Consolidation):**
            - **Energy**: Release, correction, cooling off
            - **Psychology**: Reality check, risk assessment
            - **Trading**: Consolidation patterns, value hunting
            - **Volume**: Declining, selective moves
            - **Best for**: Position adjustments, planning
            
            **🔮 {month_name} {selected_year} Specific Lunar Events:**
            
            **Key Lunar Dates to Watch:**
            - **New Moon**: Potential trend change or continuation signal
            - **First Quarter**: Momentum confirmation or failure
            - **Full Moon**: Profit-taking opportunity or reversal signal  
            - **Last Quarter**: Consolidation phase or weakness signal
            
            **Moon Sign Influences:**
            - **Fire Signs** (Aries, Leo, Sagittarius): Aggressive moves, energy sector strength
            - **Earth Signs** (Taurus, Virgo, Capricorn): Value focus, stability preference
            - **Air Signs** (Gemini, Libra, Aquarius): Communication, technology emphasis
            - **Water Signs** (Cancer, Scorpio, Pisces): Emotional decisions, defensive moves
            """)
        
        with tab4:
            st.markdown(f"""
            ### Portfolio Strategy for {month_name} {selected_year}
            
            **🎯 Strategic Asset Allocation:**
            
            **Core Holdings (50-60%):**
            - **Large Cap Stability**: Jupiter-aspected blue chips
            - **Sector Leaders**: Dominant players in favored sectors
            - **Defensive Assets**: During challenging aspect periods
            - **Currency Hedge**: If significant Pluto aspects present
            
            **Growth Opportunities (20-30%):**
            - **Momentum Plays**: Mars-Jupiter aspect beneficiaries
            - **Breakout Candidates**: Technical + astrological confluence
            - **Sector Rotation**: Following planetary emphasis shifts
            - **Emerging Themes**: Uranus aspect innovation plays
            
            **Speculative/Trading (10-20%):**
            - **High Beta Names**: For Mars-Uranus periods
            - **Volatility Plays**: Options during aspect peaks
            - **Contrarian Bets**: Against crowd during extremes
            - **Crypto Allocation**: If comfortable with high volatility
            
            **📊 Risk Management Framework:**
            
            **Position Sizing Rules:**
            - **Maximum single position**: 5% during stable periods
            - **Reduce to 3%**: During challenging aspects
            - **Increase to 7%**: During strong favorable confluences
            - **Cash levels**: 10-20% based on aspect favorability
            
            **Stop Loss Strategy:**
            - **Tight stops** (3-5%): During Mercury retrograde periods
            - **Normal stops** (5-8%): During regular market conditions
            - **Wide stops** (8-12%): During high volatility aspect periods
            - **No stops**: For long-term Jupiter-blessed positions
            
            **📅 Monthly Rebalancing Schedule:**
            
            **Week 1**: Review and adjust based on new lunar cycle
            **Week 2**: Add to momentum winners if aspects support
            **Week 3**: Prepare for Full Moon profit-taking opportunities
            **Week 4**: Position for next month's astrological themes
            
            **🔄 Sector Rotation Strategy:**
            
            **Early Month**: Follow Jupiter aspects for growth sectors
            **Mid Month**: Mars aspects may favor energy/materials
            **Late Month**: Venus aspects support financials/consumer
            **Month End**: Saturn aspects favor defensives/utilities
            
            **💡 Advanced Strategies:**
            
            **Pairs Trading**: Long favored sectors, short challenged sectors
            **Options Overlay**: Sell calls during Full Moons, buy calls during New Moons
            **Currency Hedge**: Hedge foreign exposure during Pluto aspects
            **Volatility Trading**: Long volatility before aspect peaks
            
            **📈 Performance Tracking:**
            
            **Monthly Metrics**:
            - Absolute return vs. benchmark
            - Risk-adjusted return (Sharpe ratio)
            - Maximum drawdown during challenging aspects
            - Hit rate on astrological predictions
            
            **Aspect Correlation Analysis**:
            - Track which aspects work best for {symbol}
            - Note sector rotation timing accuracy
            - Measure volatility prediction success
            - Document lunar cycle correlations
            """)
        
        # Additional insights for monthly strategy
        st.subheader('🎭 Market Psychology & Sentiment Analysis')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            #### 🧠 Psychological Drivers - {month_name}
            
            **Institutional Behavior:**
            - Month-end window dressing effects
            - Quarterly rebalancing influences  
            - Earnings season psychological impacts
            - Fed meeting anticipation/reaction
            
            **Retail Investor Patterns:**
            - Payroll cycle investment flows
            - Tax implications (if year-end)
            - Holiday season spending impacts
            - Social media sentiment amplification
            
            **Global Sentiment Factors:**
            - US-China trade relationship status
            - European economic data impacts
            - Emerging market flow dynamics
            - Cryptocurrency correlation effects
            """)
        
        with col2:
            st.markdown(f"""
            #### 📊 Sentiment Indicators to Watch
            
            **Technical Sentiment:**
            - VIX levels and term structure
            - Put/Call ratios by sector
            - High-low index readings
            - Advance-decline line trends
            
            **Fundamental Sentiment:**
            - Earnings revision trends
            - Analyst recommendation changes
            - Insider buying/selling activity
            - Share buyback announcements
            
            **Alternative Data:**
            - Google search trends
            - Social media mention analysis
            - Options flow analysis
            - Crypto correlation strength
            """)

# Footer with additional information
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <h4>🌟 Disclaimer & Important Notes</h4>
        <p><strong>Educational Purpose Only:</strong> This dashboard is for educational and research purposes. 
        Astrological analysis should be combined with fundamental and technical analysis for trading decisions.</p>
        
        <p><strong>Risk Warning:</strong> All trading involves risk. Past performance and astrological correlations 
        do not guarantee future results. Always use proper risk management and never invest more than you can afford to lose.</p>
        
        <p><strong>Data Sources:</strong> Simulated price data based on astrological aspect calculations. 
        For live trading, use real market data and professional trading platforms.</p>
        
        <p style='font-size: 12px; margin-top: 20px;'>
        🔮 <em>"The stars impel, they do not compel. Wisdom lies in using all available tools - 
        fundamental, technical, and cosmic - for informed decision making."</em>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main(), 'default_price': 2050.0, 'sector': 'Precious Metals'},
    'DOWJONES': {'name': 'Dow Jones Industrial Average', 'currency': '

# --- STOCK DATABASE ---
stock_data = {
    'Symbol': [
        'TCS', 'ICICIBANK', 'MARUTI', 'DLF', 'NESTLEIND', 
        'RELIANCE', 'SBI', 'SUNPHARMA', 'DRREDDY', 'GOLD',
        'DOWJONES', 'SILVER', 'CRUDE', 'BTC'
    ],
    'Sector': [
        'Technology', 'Banking', 'Automotive', 'Realty', 'FMCG',
        'Energy', 'PSUs', 'Pharma', 'Pharma', 'Precious Metals',
        'US Index', 'Precious Metals', 'Energy', 'Cryptocurrency'
    ],
    'MarketCap': [
        'Large', 'Large', 'Large', 'Large', 'Large',
        'Large', 'Large', 'Large', 'Large', 'Commodity',
        'Index', 'Commodity', 'Commodity', 'Crypto'
    ]
}

STOCK_DATABASE = pd.DataFrame(stock_data)

# --- SECTOR-PLANETARY MAPPINGS ---
SECTOR_PLANETARY_INFLUENCES = {
    'Technology': ['Mercury'],
    'Banking': ['Jupiter', 'Saturn'],
    'FMCG': ['Moon'],
    'Pharma': ['Neptune'],
    'Energy': ['Mars'],
    'Automotive': ['Saturn'],
    'Realty': ['Saturn'],
    'PSUs': ['Pluto'],
    'Midcaps': ['Uranus'],
    'Smallcaps': ['Pluto'],
    'Precious Metals': ['Venus', 'Jupiter'],
    'US Index': ['Jupiter', 'Saturn'],
    'Cryptocurrency': ['Uranus', 'Pluto']
}

# --- ASPECT-SECTOR IMPACT ---
ASPECT_SECTOR_IMPACTS = {
    'Square': {
        'Technology': 'Negative', 'Banking': 'Negative', 'FMCG': 'Neutral',
        'Pharma': 'Neutral', 'Energy': 'Neutral', 'Automotive': 'Neutral',
        'Realty': 'Neutral', 'PSUs': 'Neutral', 'Midcaps': 'Neutral',
        'Smallcaps': 'Neutral', 'Precious Metals': 'Positive', 'US Index': 'Negative',
        'Cryptocurrency': 'Negative'
    },
    'Opposition': {
        'Technology': 'Neutral', 'Banking': 'Neutral', 'FMCG': 'Neutral',
        'Pharma': 'Neutral', 'Energy': 'Neutral', 'Automotive': 'Negative',
        'Realty': 'Negative', 'PSUs': 'Neutral', 'Midcaps': 'Neutral',
        'Smallcaps': 'Neutral', 'Precious Metals': 'Positive', 'US Index': 'Neutral',
        'Cryptocurrency': 'Positive'
    },
    'Trine': {
        'Technology': 'Neutral', 'Banking': 'Neutral', 'FMCG': 'Positive',
        'Pharma': 'Positive', 'Energy': 'Neutral', 'Automotive': 'Neutral',
        'Realty': 'Neutral', 'PSUs': 'Neutral', 'Midcaps': 'Neutral',
        'Smallcaps': 'Neutral', 'Precious Metals': 'Positive', 'US Index': 'Positive',
        'Cryptocurrency': 'Neutral'
    },
    'Conjunction': {
        'Technology': 'Neutral', 'Banking': 'Neutral', 'FMCG': 'Neutral',
        'Pharma': 'Neutral', 'Energy': 'Positive', 'Automotive': 'Neutral',
        'Realty': 'Neutral', 'PSUs': 'Neutral', 'Midcaps': 'Negative',
        'Smallcaps': 'Neutral', 'Precious Metals': 'Positive', 'US Index': 'Neutral',
        'Cryptocurrency': 'Positive'
    },
    'Sextile': {
        'Technology': 'Neutral', 'Banking': 'Neutral', 'FMCG': 'Neutral',
        'Pharma': 'Neutral', 'Energy': 'Neutral', 'Automotive': 'Neutral',
        'Realty': 'Neutral', 'PSUs': 'Positive', 'Midcaps': 'Neutral',
        'Smallcaps': 'Negative', 'Precious Metals': 'Positive', 'US Index': 'Positive',
        'Cryptocurrency': 'Neutral'
    }
}

# --- PLANETARY POSITION VISUALIZATION ---
def draw_planetary_wheel(ax, input_date, size=0.3):
    """Draw a simplified astrological wheel showing planetary positions"""
    base_date = datetime(2025, 8, 1)
    
    if isinstance(input_date, date_class) and not isinstance(input_date, datetime):
        date_obj = datetime.combine(input_date, datetime.min.time())
    else:
        date_obj = input_date
    
    days_diff = (date_obj.date() - base_date.date()).days
    
    base_positions = {
        'Sun': 135, 'Moon': 225, 'Mercury': 120, 'Venus': 170,
        'Mars': 85, 'Jupiter': 45, 'Saturn': 315
    }
    
    daily_movement = {
        'Sun': 1.0, 'Moon': 13.2, 'Mercury': 1.5, 'Venus': 1.2,
        'Mars': 0.5, 'Jupiter': 0.08, 'Saturn': 0.03
    }
    
    planets = {}
    for planet, base_pos in base_positions.items():
        angle = (base_pos + daily_movement[planet] * days_diff) % 360
        planets[planet] = {
            'angle': angle,
            'color': {
                'Sun': 'gold', 'Moon': 'silver', 'Mercury': 'gray',
                'Venus': 'lightgreen', 'Mars': 'red', 'Jupiter': 'orange',
                'Saturn': 'darkgoldenrod'
            }[planet],
            'size': {
                'Sun': 8, 'Moon': 6, 'Mercury': 5, 'Venus': 7,
                'Mars': 6, 'Jupiter': 10, 'Saturn': 9
            }[planet]
        }
    
    zodiac = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 
              'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
    
    for i, sign in enumerate(zodiac):
        angle = i * 30
        ax.add_patch(Wedge((0, 0), size, angle, angle+30, width=size*0.8, 
                          facecolor='lightgray', edgecolor='black', alpha=0.3))
        ax.text(0.85*size * np.cos(np.radians(angle+15)), 
                0.85*size * np.sin(np.radians(angle+15)), 
                sign[:3], ha='center', va='center', fontsize=6)
    
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
    ax.set_title(f'Planetary Positions\n{date_obj.strftime("%b %d, %Y")}', fontsize=8)

# --- GET TRADING HOURS FOR SYMBOL ---
def get_trading_hours(symbol):
    """Get trading hours for a specific symbol"""
    symbol = symbol.upper()
    if symbol in TRADING_HOURS:
        return TRADING_HOURS[symbol]
    else:
        # Default to Indian market hours for unknown symbols
        return TRADING_HOURS['NIFTY']

# --- GET SYMBOL INFO ---
def get_symbol_info(symbol):
    """Get symbol configuration info"""
    symbol = symbol.upper()
    if symbol in SYMBOL_CONFIG:
        return SYMBOL_CONFIG[symbol]
    else:
        # Default configuration
        return {
            'name': symbol,
            'currency': '₹',
            'default_price': 1000.0,
            'sector': 'Unknown'
        }

# --- GENERATE ASPECTS ---
def generate_todays_aspects():
    """Generate astrological aspects for today based on the provided table"""
    base_aspects = [
        {"planets": "Mercury-Jupiter", "aspect_type": "Square", "impact": -0.7, "type": "bearish"},
        {"planets": "Venus-Saturn", "aspect_type": "Opposition", "impact": -0.8, "type": "bearish"},
        {"planets": "Moon-Neptune", "aspect_type": "Trine", "impact": 0.6, "type": "bullish"},
        {"planets": "Mars-Uranus", "aspect_type": "Conjunction", "impact": 0.9, "type": "bullish"},
        {"planets": "Sun-Pluto", "aspect_type": "Sextile", "impact": 0.5, "type": "bullish"}
    ]
    
    aspects = []
    for aspect in base_aspects:
        aspects.append({
            "planets": aspect["planets"],
            "aspect_type": aspect["aspect_type"],
            "impact": aspect["impact"],
            "type": aspect["type"]
        })
    
    return aspects

# --- CREATE SUMMARY TABLE ---
def create_summary_table(aspects):
    """Create a summary table based on the astrological aspects"""
    summary_data = {
        'Aspect': [],
        'Nifty/Bank Nifty': [],
        'Bullish Sectors/Stocks': [],
        'Bearish Sectors/Stocks': []
    }
    
    for aspect in aspects:
        planets = aspect["planets"]
        aspect_type = aspect["aspect_type"]
        
        if planets == "Mercury-Jupiter" and aspect_type == "Square":
            summary_data['Aspect'].append("Mercury-Jupiter (Square)")
            summary_data['Nifty/Bank Nifty'].append("Volatile")
            summary_data['Bullish Sectors/Stocks'].append("IT (TCS), Gold")
            summary_data['Bearish Sectors/Stocks'].append("Banking (ICICI Bank), Crypto")
        
        elif planets == "Venus-Saturn" and aspect_type == "Opposition":
            summary_data['Aspect'].append("Venus-Saturn (Opposition)")
            summary_data['Nifty/Bank Nifty'].append("Downside")
            summary_data['Bullish Sectors/Stocks'].append("Gold, Silver, BTC")
            summary_data['Bearish Sectors/Stocks'].append("Auto (Maruti), Realty (DLF)")
        
        elif planets == "Moon-Neptune" and aspect_type == "Trine":
            summary_data['Aspect'].append("Moon-Neptune (Trine)")
            summary_data['Nifty/Bank Nifty'].append("Mild Support")
            summary_data['Bullish Sectors/Stocks'].append("FMCG (Nestlé), Pharma, Gold, Dow Jones")
            summary_data['Bearish Sectors/Stocks'].append("-")
        
        elif planets == "Mars-Uranus" and aspect_type == "Conjunction":
            summary_data['Aspect'].append("Mars-Uranus (Conjunction)")
            summary_data['Nifty/Bank Nifty'].append("Sharp Moves")
            summary_data['Bullish Sectors/Stocks'].append("Energy (Reliance, Crude), Gold, BTC")
            summary_data['Bearish Sectors/Stocks'].append("Weak Midcaps")
        
        elif planets == "Sun-Pluto" and aspect_type == "Sextile":
            summary_data['Aspect'].append("Sun-Pluto (Sextile)")
            summary_data['Nifty/Bank Nifty'].append("Structural Shift")
            summary_data['Bullish Sectors/Stocks'].append("PSUs (SBI), Gold, Dow Jones")
            summary_data['Bearish Sectors/Stocks'].append("Overvalued Smallcaps")
    
    return pd.DataFrame(summary_data)

# --- FILTER STOCKS BASED ON ASPECTS ---
def filter_stocks_by_aspects(aspects, stock_database):
    """Filter stocks based on today's astrological aspects"""
    sector_impacts = {sector: 0 for sector in stock_database['Sector'].unique()}
    
    for aspect in aspects:
        planet1, planet2 = aspect["planets"].split("-")
        
        for sector, planets in SECTOR_PLANETARY_INFLUENCES.items():
            if planet1 in planets or planet2 in planets:
                if sector not in sector_impacts:
                    sector_impacts[sector] = 0
                
                aspect_impact = ASPECT_SECTOR_IMPACTS[aspect["aspect_type"]].get(sector, "Neutral")
                
                if aspect_impact == "Positive":
                    sector_impacts[sector] += abs(aspect["impact"])
                elif aspect_impact == "Negative":
                    sector_impacts[sector] -= abs(aspect["impact"])
    
    bullish_sectors = [sector for sector, impact in sector_impacts.items() if impact > 0]
    bearish_sectors = [sector for sector, impact in sector_impacts.items() if impact < 0]
    neutral_sectors = [sector for sector, impact in sector_impacts.items() if impact == 0]
    
    bullish_stocks = stock_database[stock_database['Sector'].isin(bullish_sectors)].copy()
    bearish_stocks = stock_database[stock_database['Sector'].isin(bearish_sectors)].copy()
    neutral_stocks = stock_database[stock_database['Sector'].isin(neutral_sectors)].copy()
    
    bullish_stocks['Impact Score'] = bullish_stocks['Sector'].apply(lambda x: sector_impacts[x])
    bearish_stocks['Impact Score'] = bearish_stocks['Sector'].apply(lambda x: abs(sector_impacts[x]))
    neutral_stocks['Impact Score'] = 0
    
    bullish_stocks = bullish_stocks.sort_values('Impact Score', ascending=False)
    bearish_stocks = bearish_stocks.sort_values('Impact Score', ascending=False)
    
    return {
        'bullish': bullish_stocks,
        'bearish': bearish_stocks,
        'neutral': neutral_stocks,
        'sector_impacts': sector_impacts
    }

# --- GENERATE ASTROLOGICAL EVENTS ---
def generate_astrological_events(input_date, event_type='intraday', symbol='NIFTY'):
    """Generate astrological events for any given date and symbol"""
    
    if event_type == 'intraday':
        trading_hours = get_trading_hours(symbol)
        
        # Different event patterns based on trading hours
        if trading_hours['end_hour'] > 16:  # Extended hours (global markets)
            # More events spread across longer trading day
            base_events = [
                {"time_offset": 0, "aspect": "Pre-market: Mercury square Jupiter", "impact": -0.5, "type": "bearish"},
                {"time_offset": 120, "aspect": "Asian session: Moon trine Jupiter", "impact": 0.8, "type": "bullish"},
                {"time_offset": 240, "aspect": "London open: Mars sextile Jupiter", "impact": 0.4, "type": "neutral"},
                {"time_offset": 360, "aspect": "European session: Venus opposition Saturn", "impact": -0.6, "type": "bearish"},
                {"time_offset": 480, "aspect": "NY pre-market: Sun conjunct Mercury", "impact": 0.3, "type": "neutral"},
                {"time_offset": 600, "aspect": "US open: Mars conjunct Uranus", "impact": 1.0, "type": "bullish"},
                {"time_offset": 720, "aspect": "Mid-day: Moon square Saturn", "impact": -0.4, "type": "bearish"},
                {"time_offset": 840, "aspect": "Afternoon: Jupiter trine Neptune", "impact": 0.7, "type": "bullish"},
                {"time_offset": 960, "aspect": "US close approach", "impact": 0.2, "type": "neutral"},
                {"time_offset": 1080, "aspect": "After hours: Void Moon", "impact": -0.3, "type": "bearish"},
                {"time_offset": 1135, "aspect": "Session close", "impact": 0.1, "type": "neutral"}
            ]
        else:  # Standard Indian market hours
            base_events = [
                {"time_offset": 0, "aspect": "Opening: Mercury square Jupiter + Void Moon", "impact": -0.5, "type": "bearish"},
                {"time_offset": 45, "aspect": "Early trade: Moon trine Jupiter", "impact": 1.0, "type": "bullish"},
                {"time_offset": 135, "aspect": "Mid-morning: Mars sextile Jupiter", "impact": 0.3, "type": "neutral"},
                {"time_offset": 195, "aspect": "Pre-lunch: Sun in Leo (no aspects)", "impact": 0.0, "type": "neutral"},
                {"time_offset": 285, "aspect": "Post-lunch: Moon square Saturn", "impact": -0.8, "type": "bearish"},
                {"time_offset": 345, "aspect": "Late trade: Venus-Saturn opposition", "impact": -0.6, "type": "bearish"},
                {"time_offset": 375, "aspect": "Close", "impact": 0.1, "type": "neutral"}
            ]
        
        events = []
        if isinstance(input_date, date_class) and not isinstance(input_date, datetime):
            dt = datetime.combine(input_date, datetime.min.time())
        else:
            dt = input_date
            
        start_time = dt.replace(hour=trading_hours['start_hour'], minute=trading_hours['start_minute'])
        
        for event in base_events:
            event_time = start_time + timedelta(minutes=event["time_offset"])
            events.append({
                "time": event_time,
                "aspect": event["aspect"],
                "impact": event["impact"],
                "type": event["type"],
                "price": 0
            })
        
        return events
    
    else:  # monthly events remain the same
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
        
        if isinstance(input_date, datetime):
            year, month = input_date.year, input_date.month
        else:
            year, month = input_date.year, input_date.month
            
        days_in_month = calendar.monthrange(year, month)[1]
        
        events = []
        for event in base_events:
            day = min(event["day_offset"], days_in_month)
            event_date = datetime(year, month, day)
            events.append({
                "date": event_date,
                "aspect": event["aspect"],
                "impact": event["impact"],
                "type": event["type"],
                "price": 0
            })
        
        return events

# --- ENHANCED INTRADAY CHART ---
def generate_intraday_chart(symbol, starting_price, selected_date):
    """Generate enhanced intraday chart with dynamic layout"""
    symbol_info = get_symbol_info(symbol)
    trading_hours = get_trading_hours(symbol)
    
    if isinstance(selected_date, date_class) and not isinstance(selected_date, datetime):
        selected_date = datetime.combine(selected_date, datetime.min.time())
    
    start_time = selected_date.replace(hour=trading_hours['start_hour'], minute=trading_hours['start_minute'])
    end_time = selected_date.replace(hour=trading_hours['end_hour'], minute=trading_hours['end_minute'])
    
    # Adjust interval based on trading session length
    session_hours = (end_time - start_time).total_seconds() / 3600
    if session_hours > 12:
        interval = '30T'  # 30-minute intervals for long sessions
    else:
        interval = '15T'  # 15-minute intervals for shorter sessions
    
    times = pd.date_range(start=start_time, end=end_time, freq=interval)
    
    prices = np.zeros(len(times))
    base_price = starting_price
    
    events = generate_astrological_events(selected_date, 'intraday', symbol)
    
    # Adjust event impacts based on symbol type
    symbol_multiplier = 1.0
    if symbol in ['GOLD', 'SILVER']:
        symbol_multiplier = 0.8  # Precious metals less volatile to aspects
    elif symbol in ['BTC']:
        symbol_multiplier = 2.0  # Crypto more volatile
    elif symbol in ['CRUDE']:
        symbol_multiplier = 1.5  # Energy commodities more responsive
    
    for event in events:
        price_change = event["impact"] * base_price * 0.01 * symbol_multiplier
        event["price"] = base_price + price_change
    
    # Generate price movements
    for i, time in enumerate(times):
        closest_event = min(events, key=lambda x: abs((x["time"] - time).total_seconds()))
        distance = abs((closest_event["time"] - time).total_seconds()) / 3600
        
        # Adjust volatility based on symbol
        base_volatility = 0.15 if distance < 0.5 else 0.05
        if symbol in ['BTC']:
            base_volatility *= 3.0
        elif symbol in ['GOLD', 'SILVER']:
            base_volatility *= 0.5
        elif symbol in ['CRUDE']:
            base_volatility *= 2.0
        
        random_change = np.random.normal(0, base_volatility)
        event_influence = closest_event["impact"] * np.exp(-distance) * symbol_multiplier
        
        if i == 0:
            prices[i] = base_price
        else:
            change = (event_influence + random_change) * base_price * 0.001
            prices[i] = prices[i-1] + change
    
    df_intraday = pd.DataFrame({
        'Time': times,
        'Price': prices,
        'Aspect': [min(events, key=lambda x: abs((x["time"] - t).total_seconds()))["aspect"] for t in times]
    })
    
    # Create dynamic figure layout
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, height_ratios=[4, 1, 1, 1], width_ratios=[5, 2, 1])
    
    # Main price chart
    ax_main = fig.add_subplot(gs[0, :2])
    
    for i in range(1, len(df_intraday)):
        color = 'green' if df_intraday['Price'].iloc[i] > df_intraday['Price'].iloc[i-1] else 'red'
        ax_main.plot(df_intraday['Time'].iloc[i-1:i+1], 
                    df_intraday['Price'].iloc[i-1:i+1], 
                    color=color, linewidth=2.5)
    
    # Mark key events
    for event in events:
        color_map = {'bullish': 'green', 'bearish': 'red', 'neutral': 'blue'}
        ax_main.axvline(x=event['time'], color=color_map[event['type']], 
                       linestyle='--', alpha=0.7, linewidth=2)
        
        marker_color = color_map[event['type']]
        ax_main.scatter(event['time'], event['price'], color=marker_color, 
                       s=100, zorder=5, edgecolor='black', linewidth=1)
        
        # Dynamic annotation positioning
        y_offset = base_price * 0.02 if len(str(int(base_price))) >= 4 else base_price * 0.05
        y_pos = event['price'] + y_offset if event['price'] < base_price * 1.01 else event['price'] - y_offset
        
        ax_main.annotate(event['aspect'][:30] + '...' if len(event['aspect']) > 30 else event['aspect'], 
                        xy=(event['time'], event['price']),
                        xytext=(event['time'], y_pos),
                        arrowprops=dict(arrowstyle='->', color=marker_color, lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                        fontsize=8, ha='center')
    
    # Dynamic title and formatting
    currency_symbol = symbol_info['currency']
    ax_main.set_title(f'{symbol_info["name"]} ({symbol}) - {selected_date.strftime("%B %d, %Y")}\n'
                     f'Astrological Trading Analysis | Sector: {symbol_info["sector"]}', 
                     fontsize=16, pad=20)
    ax_main.set_xlabel(f'Time ({trading_hours["start_hour"]}:00 - {trading_hours["end_hour"]}:00)', fontsize=12)
    ax_main.set_ylabel(f'Price ({currency_symbol})', fontsize=12)
    
    # Dynamic time formatting based on session length
    if session_hours > 12:
        ax_main.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    else:
        ax_main.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    plt.setp(ax_main.get_xticklabels(), rotation=45, ha='right')
    
    # Closing price annotation
    close_price = df_intraday["Price"].iloc[-1]
    price_change = close_price - base_price
    price_change_pct = (price_change / base_price) * 100
    
    ax_main.annotate(f'Close: {currency_symbol}{close_price:.2f}\n'
                    f'Change: {price_change:+.2f} ({price_change_pct:+.2f}%)', 
                    xy=(df_intraday['Time'].iloc[-1], close_price),
                    xytext=(df_intraday['Time'].iloc[-1] - timedelta(hours=session_hours*0.2), 
                           close_price + base_price * 0.02),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                    fontsize=11, fontweight='bold',
                    bbox=dict(facecolor='yellow', alpha=0.7, edgecolor='none', pad=2))
    
    # Planetary wheel
    ax_wheel = fig.add_subplot(gs[0, 2])
    draw_planetary_wheel(ax_wheel, selected_date, size=0.4)
    
    # Volume chart (simulated with realistic patterns)
    ax_volume = fig.add_subplot(gs[1, :2])
    
    # Generate more realistic volume based on symbol type
    if symbol in ['BTC']:
        base_volume = np.random.randint(50000, 200000, size=len(times))
    elif symbol in ['GOLD', 'SILVER', 'CRUDE']:
        base_volume = np.random.randint(10000, 50000, size=len(times))
    elif symbol in ['DOWJONES']:
        base_volume = np.random.randint(100000, 500000, size=len(times))
    else:  # Indian stocks
        base_volume = np.random.randint(1000, 10000, size=len(times))
    
    colors_volume = ['green' if df_intraday['Price'].iloc[i] > df_intraday['Price'].iloc[i-1] 
                    else 'red' for i in range(1, len(df_intraday))]
    colors_volume.insert(0, 'green')
    
    ax_volume.bar(df_intraday['Time'], base_volume, color=colors_volume, alpha=0.7)
    ax_volume.set_title('Volume', fontsize=12)
    ax_volume.set_ylabel('Volume', fontsize=10)
    ax_volume.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax_volume.get_xticklabels(), rotation=45, ha='right')
    
    # Technical indicators (RSI simulation)
    ax_rsi = fig.add_subplot(gs[2, :2])
    rsi_values = 50 + np.random.normal(0, 15, len(times))  # Simulated RSI
    rsi_values = np.clip(rsi_values, 0, 100)
    
    ax_rsi.plot(df_intraday['Time'], rsi_values, color='purple', linewidth=2)
    ax_rsi.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
    ax_rsi.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
    ax_rsi.fill_between(df_intraday['Time'], 30, 70, alpha=0.1, color='gray')
    ax_rsi.set_title('RSI (14)', fontsize=12)
    ax_rsi.set_ylabel('RSI', fontsize=10)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.legend(loc='upper right', fontsize=8)
    
    # Aspect strength indicator
    ax_aspect = fig.add_subplot(gs[3, :2])
    aspect_times = [event['time'] for event in events]
    aspect_strengths = [abs(event['impact']) for event in events]
    aspect_colors = [{'bullish': 'green', 'bearish': 'red', 'neutral': 'blue'}[event['type']] for event in events]
    
    ax_aspect.scatter(aspect_times, aspect_strengths, color=aspect_colors, s=100, zorder=3)
    ax_aspect.plot(aspect_times, aspect_strengths, color='gray', alpha=0.5, linestyle='--')
    ax_aspect.set_title('Astrological Aspect Strength', fontsize=12)
    ax_aspect.set_ylabel('Strength', fontsize=10)
    ax_aspect.set_ylim(0, max(aspect_strengths) * 1.2 if aspect_strengths else 1.5)
    
    # Symbol info panel
    ax_info = fig.add_subplot(gs[1:, 2])
    ax_info.axis('off')
    
    info_text = f"""
SYMBOL INFO
-----------
Name: {symbol_info['name']}
Sector: {symbol_info['sector']}
Currency: {symbol_info['currency']}

TRADING HOURS
-------------
Start: {trading_hours['start_hour']:02d}:{trading_hours['start_minute']:02d}
End: {trading_hours['end_hour']:02d}:{trading_hours['end_minute']:02d}
Session: {session_hours:.1f} hours

PRICE DATA
----------
Open: {currency_symbol}{base_price:.2f}
Close: {currency_symbol}{close_price:.2f}
Change: {price_change:+.2f}
Change%: {price_change_pct:+.2f}%

High: {currency_symbol}{max(prices):.2f}
Low: {currency_symbol}{min(prices):.2f}
Range: {currency_symbol}{max(prices)-min(prices):.2f}
"""
    
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig

# --- ENHANCED MONTHLY CHART ---
def generate_monthly_chart(symbol, starting_price, selected_month, selected_year):
    """Generate enhanced monthly chart with dynamic layout"""
    symbol_info = get_symbol_info(symbol)
    
    start_date = datetime(selected_year, selected_month, 1)
    days_in_month = calendar.monthrange(selected_year, selected_month)[1]
    end_date = datetime(selected_year, selected_month, days_in_month)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    prices = np.zeros(len(dates))
    base_price = starting_price
    
    events = generate_astrological_events(start_date, 'monthly', symbol)
    
    # Adjust event impacts based on symbol type
    symbol_multiplier = 1.0
    if symbol in ['GOLD', 'SILVER']:
        symbol_multiplier = 0.8
    elif symbol in ['BTC']:
        symbol_multiplier = 2.5
    elif symbol in ['CRUDE']:
        symbol_multiplier = 1.8
    
    for event in events:
        price_change = event["impact"] * base_price * 0.01 * symbol_multiplier
        event["price"] = base_price + price_change
    
    # Generate price movements
    for i, date in enumerate(dates):
        closest_event = min(events, key=lambda x: abs((x["date"].date() - date.date()).days))
        distance = abs((closest_event["date"].date() - date.date()).days)
        
        base_volatility = 0.3 if distance < 2 else 0.1
        if symbol in ['BTC']:
            base_volatility *= 4.0
        elif symbol in ['GOLD', 'SILVER']:
            base_volatility *= 0.6
        elif symbol in ['CRUDE']:
            base_volatility *= 2.5
        
        random_change = np.random.normal(0, base_volatility)
        event_influence = closest_event["impact"] * np.exp(-distance/2) * symbol_multiplier
        
        if i == 0:
            prices[i] = base_price
        else:
            change = (event_influence + random_change) * base_price * 0.002
            prices[i] = prices[i-1] + change
    
    df_monthly = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Aspect': [min(events, key=lambda x: abs((x["date"].date() - d.date()).days))["aspect"] for d in dates]
    })
    
    # Create dynamic figure layout
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, height_ratios=[4, 1, 1, 1], width_ratios=[5, 2, 1])
    
    # Main price chart
    ax_main = fig.add_subplot(gs[0, :2])
    
    for i in range(1, len(df_monthly)):
        color = 'green' if df_monthly['Price'].iloc[i] > df_monthly['Price'].iloc[i-1] else 'red'
        ax_main.plot(df_monthly['Date'].iloc[i-1:i+1], 
                    df_monthly['Price'].iloc[i-1:i+1], 
                    color=color, linewidth=3)
    
    # Mark key events
    for event in events:
        color_map = {'bullish': 'green', 'bearish': 'red', 'neutral': 'blue'}
        ax_main.axvline(x=event['date'], color=color_map[event['type']], 
                       linestyle='--', alpha=0.7, linewidth=2)
        
        marker_color = color_map[event['type']]
        ax_main.scatter(event['date'], event['price'], color=marker_color, 
                       s=150, zorder=5, edgecolor='black', linewidth=1.5)
        
        y_offset = base_price * 0.03
        y_pos = event['price'] + y_offset if event['price'] < base_price * 1.02 else event['price'] - y_offset
        
        ax_main.annotate(event['aspect'][:25] + '...' if len(event['aspect']) > 25 else event['aspect'], 
                        xy=(event['date'], event['price']),
                        xytext=(event['date'], y_pos),
                        arrowprops=dict(arrowstyle='->', color=marker_color, lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                        fontsize=9, ha='center')
    
    # Dynamic formatting
    currency_symbol = symbol_info['currency']
    ax_main.set_title(f'{symbol_info["name"]} ({symbol}) - {start_date.strftime("%B %Y")}\n'
                     f'Monthly Astrological Analysis | Sector: {symbol_info["sector"]}', 
                     fontsize=16, pad=20)
    ax_main.set_xlabel('Date', fontsize=12)
    ax_main.set_ylabel(f'Price ({currency_symbol})', fontsize=12)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax_main.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, days_in_month//10)))
    plt.setp(ax_main.get_xticklabels(), rotation=45, ha='right')
    
    # Monthly close annotation
    close_price = df_monthly["Price"].iloc[-1]
    price_change = close_price - base_price
    price_change_pct = (price_change / base_price) * 100
    
    ax_main.annotate(f'Month Close: {currency_symbol}{close_price:.2f}\n'
                    f'Monthly Change: {price_change:+.2f} ({price_change_pct:+.2f}%)', 
                    xy=(df_monthly['Date'].iloc[-1], close_price),
                    xytext=(df_monthly['Date'].iloc[-1] - timedelta(days=days_in_month//4), 
                           close_price + base_price * 0.03),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                    fontsize=12, fontweight='bold',
                    bbox=dict(facecolor='yellow', alpha=0.7, edgecolor='none', pad=2))
    
    # Planetary positions for key dates
    ax_planets = fig.add_subplot(gs[0, 2])
    ax_planets.set_title('Key Planetary\nPositions', fontsize=10)
    key_dates = [
        start_date,
        start_date + timedelta(days=days_in_month//3),
        start_date + timedelta(days=2*days_in_month//3),
        end_date
    ]
    
    for i, date in enumerate(key_dates):
        ax_sub = fig.add_axes([0.70, 0.8-i*0.15, 0.12, 0.12])
        draw_planetary_wheel(ax_sub, date, size=0.4)
        ax_sub.set_title(f'{date.strftime("%b %d")}', fontsize=8)
    
    # Volume chart
    ax_volume = fig.add_subplot(gs[1, :2])
    
    if symbol in ['BTC']:
        volume = np.random.randint(500000, 2000000, size=len(dates))
    elif symbol in ['GOLD', 'SILVER', 'CRUDE']:
        volume = np.random.randint(100000, 500000, size=len(dates))
    elif symbol in ['DOWJONES']:
        volume = np.random.randint(1000000, 5000000, size=len(dates))
    else:
        volume = np.random.randint(10000, 100000, size=len(dates))
    
    colors_volume = ['green' if df_monthly['Price'].iloc[i] > df_monthly['Price'].iloc[i-1] 
                    else 'red' for i in range(1, len(df_monthly))]
    colors_volume.insert(0, 'green')
    
    ax_volume.bar(df_monthly['Date'], volume, color=colors_volume, alpha=0.7)
    ax_volume.set_title('Daily Volume', fontsize=12)
    ax_volume.set_ylabel('Volume', fontsize=10)
    ax_volume.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    # Moving averages
    ax_ma = fig.add_subplot(gs[2, :2])
    ma_5 = df_monthly['Price'].rolling(window=5, min_periods=1).mean()
    ma_20 = df_monthly['Price'].rolling(window=min(20, len(df_monthly)), min_periods=1).mean()
    
    ax_ma.plot(df_monthly['Date'], ma_5, color='blue', linewidth=2, label='MA5', alpha=0.7)
    ax_ma.plot(df_monthly['Date'], ma_20, color='red', linewidth=2, label='MA20', alpha=0.7)
    ax_ma.fill_between(df_monthly['Date'], ma_5, ma_20, alpha=0.1, 
                      color='green' if ma_5.iloc[-1] > ma_20.iloc[-1] else 'red')
    ax_ma.set_title('Moving Averages', fontsize=12)
    ax_ma.set_ylabel('Price', fontsize=10)
    ax_ma.legend(loc='upper left', fontsize=10)
    
    # Aspect calendar
    ax_calendar = fig.add_subplot(gs[3, :2])
    aspect_dates = [event['date'] for event in events]
    aspect_strengths = [abs(event['impact']) for event in events]
    aspect_colors = [{'bullish': 'green', 'bearish': 'red', 'neutral': 'blue'}[event['type']] for event in events]
    
    ax_calendar.scatter(aspect_dates, aspect_strengths, color=aspect_colors, s=200, zorder=3)
    ax_calendar.plot(aspect_dates, aspect_strengths, color='gray', alpha=0.5, linestyle='--')
    ax_calendar.set_title('Monthly Astrological Event Strength', fontsize=12)
    ax_calendar.set_ylabel('Impact Strength', fontsize=10)
    ax_calendar.set_ylim(0, max(aspect_strengths) * 1.2 if aspect_strengths else 2)
    
    # Monthly summary panel
    ax_summary = fig.add_subplot(gs[1:, 2])
    ax_summary.axis('off')
    
    monthly_high = max(prices)
    monthly_low = min(prices)
    monthly_range = monthly_high - monthly_low
    avg_price = np.mean(prices)
    
    summary_text = f"""
MONTHLY SUMMARY
--------------
Symbol: {symbol}
Sector: {symbol_info['sector']}
Month: {start_date.strftime('%B %Y')}

PRICE STATISTICS
---------------
Open: {currency_symbol}{base_price:.2f}
Close: {currency_symbol}{close_price:.2f}
Change: {price_change:+.2f}
Change%: {price_change_pct:+.2f}%

High: {currency_symbol}{monthly_high:.2f}
Low: {currency_symbol}{monthly_low:.2f}
Range: {currency_symbol}{monthly_range:.2f}
Average: {currency_symbol}{avg_price:.2f}

VOLATILITY
----------
Daily Avg: {np.std(np.diff(prices)):.2f}
Monthly Vol: {(monthly_range/avg_price)*100:.1f}%

TREND ANALYSIS
--------------
Bullish Days: {sum(1 for i in range(1, len(prices)) if prices[i] > prices[i-1])}
Bearish Days: {sum(1 for i in range(1, len(prices)) if prices[i] < prices[i-1])}
Neutral Days: {sum(1 for i in range(1, len(prices)) if prices[i] == prices[i-1])}
"""
    
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes, fontsize=8,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    return fig

# --- ANALYZE ASPECTS ---
def analyze_aspects():
    """Enhanced aspect analysis with dynamic content"""
    aspects_data = {
        'Aspect': [
            'Mercury Retrograde', 'Venus Opposition Saturn', 'Moon-Jupiter Trine', 
            'Full Moon', 'Jupiter Square Saturn', 'Mercury Direct',
            'Venus enters Libra', 'New Moon', 'Mars-Uranus Conjunction',
            'Sun-Pluto Sextile'
        ],
        'Market Impact': [
            'High Volatility', 'Bearish Pressure', 'Bullish Surge', 
            'Trend Reversal', 'Major Tension', 'Clarity Returns',
            'Financial Rally', 'Strong Bullish', 'Energy Surge',
            'Structural Change'
        ],
        'Typical Price Change': [
            '±2-3%', '-1.5-2%', '+1-2%', 
            '±1-1.5%', '-2-3%', '+0.5-1%',
            '+0.5-1%', '+1-2%', '+2-4%',
            '±1-2%'
        ],
        'Sector Focus': [
            'All Sectors', 'Banking/Realty', 'Broad Market', 
            'Technology', 'Financials', 'Technology',
            'Banking/Finance', 'Broad Market', 'Energy/Commodities',
            'Infrastructure/PSUs'
        ],
        'Best Symbols': [
            'Gold, BTC', 'Gold, Silver', 'FMCG, Pharma', 
            'Tech Stocks', 'Defensive', 'Tech, Crypto',
            'Banking', 'Growth Stocks', 'Energy, Crude',
            'PSU, Infrastructure'
        ]
    }
    
    df_aspects = pd.DataFrame(aspects_data)
    
    # Create enhanced visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Price change impact chart
    price_changes = []
    for change in df_aspects['Typical Price Change']:
        clean_change = change.replace('%', '').replace('±', '')
        if '-' in clean_change and not clean_change.startswith('-'):
            num_str = clean_change.split('-')[1]  # Take higher value for impact
        else:
            num_str = clean_change.replace('+', '')
        
        try:
            num = float(num_str)
        except:
            num = 1.0
        price_changes.append(num)
    
    colors = ['red' if 'Bearish' in impact or 'Tension' in impact or 'Volatility' in impact 
              else 'orange' if 'Reversal' in impact or 'Change' in impact
              else 'green' for impact in df_aspects['Market Impact']]
    
    bars1 = ax1.bar(range(len(df_aspects)), price_changes, color=colors, alpha=0.7)
    ax1.set_title('Astrological Aspect Impact on Price Changes', fontsize=14)
    ax1.set_ylabel('Maximum Price Change (%)', fontsize=12)
    ax1.set_xticks(range(len(df_aspects)))
    ax1.set_xticklabels(df_aspects['Aspect'], rotation=45, ha='right')
    
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.annotate(f'{height}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    # Sector distribution pie chart
    sector_counts = {}
    for sectors in df_aspects['Sector Focus']:
        for sector in sectors.split('/'):
            sector = sector.strip()
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
    
    ax2.pie(sector_counts.values(), labels=sector_counts.keys(), autopct='%1.1f%%', startangle=90)
    ax2.set_title('Most Affected Sectors by Astrological Aspects', fontsize=14)
    
    # Market impact distribution
    impact_counts = {}
    for impact in df_aspects['Market Impact']:
        impact_type = 'Bullish' if any(word in impact for word in ['Bullish', 'Rally', 'Surge', 'Returns']) else \
                     'Bearish' if any(word in impact for word in ['Bearish', 'Pressure', 'Tension']) else \
                     'Neutral'
        impact_counts[impact_type] = impact_counts.get(impact_type, 0) + 1
    
    colors_impact = ['green', 'red', 'gray']
    ax3.bar(impact_counts.keys(), impact_counts.values(), color=colors_impact, alpha=0.7)
    ax3.set_title('Distribution of Market Impact Types', fontsize=14)
    ax3.set_ylabel('Number of Aspects', fontsize=12)
    
    # Best performing symbols chart
    symbol_mentions = {}
    for symbols in df_aspects['Best Symbols']:
        for symbol in symbols.split(', '):
            symbol = symbol.strip()
            symbol_mentions[symbol] = symbol_mentions.get(symbol, 0) + 1
    
    sorted_symbols = sorted(symbol_mentions.items(), key=lambda x: x[1], reverse=True)
    symbols, counts = zip(*sorted_symbols) if sorted_symbols else ([], [])
    
    ax4.barh(symbols, counts, color='gold', alpha=0.7)
    ax4.set_title('Most Favorable Symbols Across Aspects', fontsize=14)
    ax4.set_xlabel('Favorable Mentions', fontsize=12)
    
    return fig, df_aspects

# --- STREAMLIT APP ---
def main():
    # Page configuration for better responsive design
    st.set_page_config(
        page_title="🌟 Astrological Trading Dashboard",
        page_icon="🌟",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .symbol-input {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #e8f4fd;
        border-left: 5px solid #1f77b4;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🌟 Astrological Trading Dashboard</h1>
        <p>Advanced Financial Analysis through Planetary Movements & Cosmic Patterns</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for inputs with enhanced design
    with st.sidebar:
        st.markdown("### 📊 Dashboard Controls")
        
        # Dashboard section selection with better descriptions
        dashboard_section = st.selectbox(
            '🎯 Choose Analysis Section:',
            [
                'Summary Table - Market Overview',
                'Stock Filter - Sector Analysis', 
                'Aspect Analysis - Deep Insights',
                'Intraday Chart - Live Patterns',
                'Monthly Chart - Trend Analysis'
            ]
        )
        
        # Extract the main section name
        section_name = dashboard_section.split(' - ')[0]
        
        st.markdown("---")
        
        # Symbol selection with enhanced interface
        if section_name in ['Intraday Chart', 'Monthly Chart']:
            st.markdown("### 📈 Symbol Configuration")
            
            # Popular symbols with categories
            symbol_categories = {
                'Indian Indices': ['NIFTY', 'BANKNIFTY'],
                'Indian Stocks': ['TCS', 'ICICIBANK', 'MARUTI', 'DLF', 'NESTLEIND', 'RELIANCE', 'SBI', 'SUNPHARMA', 'DRREDDY'],
                'Global Markets': ['DOWJONES'],
                'Commodities': ['GOLD', 'SILVER', 'CRUDE'],
                'Cryptocurrency': ['BTC']
            }
            
            selected_category = st.selectbox('📂 Select Category:', list(symbol_categories.keys()))
            
            if selected_category:
                symbol_options = symbol_categories[selected_category]
                selected_symbol = st.selectbox('🎯 Choose Symbol:', symbol_options)
                
                # Custom symbol input
                custom_symbol = st.text_input('✏️ Or enter custom symbol:', max_chars=10)
                symbol = custom_symbol.upper() if custom_symbol else selected_symbol
                
                # Get symbol info for dynamic defaults
                symbol_info = get_symbol_info(symbol)
                trading_hours = get_trading_hours(symbol)
                
                # Display symbol information
                st.markdown(f"""
                <div class="info-box">
                    <strong>📊 {symbol_info['name']}</strong><br>
                    <small>Sector: {symbol_info['sector']}</small><br>
                    <small>Currency: {symbol_info['currency']}</small><br>
                    <small>Trading: {trading_hours['start_hour']:02d}:{trading_hours['start_minute']:02d} - {trading_hours['end_hour']:02d}:{trading_hours['end_minute']:02d}</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Price input with dynamic default
                starting_price = st.number_input(
                    f'💰 Starting Price ({symbol_info["currency"]}):',
                    min_value=0.01,
                    value=symbol_info['default_price'],
                    step=1.0 if symbol_info['default_price'] > 100 else 0.01,
                    format="%.2f"
                )
                
                # Date/time selection based on chart type
                if section_name == 'Intraday Chart':
                    selected_date = st.date_input(
                        '📅 Select Trading Date:',
                        value=datetime(2025, 8, 5).date(),
                        min_value=datetime(2020, 1, 1).date(),
                        max_value=datetime(2030, 12, 31).date()
                    )
                elif section_name == 'Monthly Chart':
                    col1, col2 = st.columns(2)
                    with col1:
                        selected_month = st.selectbox(
                            '📅 Month:',
                            range(1, 13),
                            format_func=lambda x: calendar.month_name[x],
                            index=7  # August
                        )
                    with col2:
                        selected_year = st.selectbox(
                            '📅 Year:',
                            range(2020, 2031),
                            index=5  # 2025
                        )
        
        # Trading insights
        st.markdown("---")
        st.markdown("### 🔮 Quick Insights")
        
        # Generate today's aspects for sidebar display
        aspects = generate_todays_aspects()
        bullish_count = sum(1 for aspect in aspects if aspect['type'] == 'bullish')
        bearish_count = sum(1 for aspect in aspects if aspect['type'] == 'bearish')
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🟢 Bullish Aspects", bullish_count)
        with col2:
            st.metric("🔴 Bearish Aspects", bearish_count)
        
        # Market sentiment
        if bullish_count > bearish_count:
            sentiment = "🟢 Bullish"
            sentiment_color = "green"
        elif bearish_count > bullish_count:
            sentiment = "🔴 Bearish"
            sentiment_color = "red"
        else:
            sentiment = "🟡 Neutral"
            sentiment_color = "orange"
        
        st.markdown(f"**Overall Sentiment:** <span style='color:{sentiment_color}'>{sentiment}</span>", 
                   unsafe_allow_html=True)

    # Main content area
    aspects = generate_todays_aspects()
    
    if section_name == 'Summary Table':
        st.header('📋 Market Summary & Astrological Overview')
        
        # Create columns for better layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader('📊 Today\'s Astrological Aspects Impact')
            summary_df = create_summary_table(aspects)
            
            # Style the dataframe
            styled_df = summary_df.style.apply(
                lambda x: ['background-color: #d4edda' if 'Bullish' in str(val) or '+' in str(val) 
                          else 'background-color: #f8d7da' if 'Bearish' in str(val) or 'Downside' in str(val)
                          else '' for val in x], axis=0
            )
            
            st.dataframe(styled_df, use_container_width=True)
        
        with col2:
            st.subheader('🎯 Key Metrics')
            
            # Calculate impact scores
            total_impact = sum(abs(aspect['impact']) for aspect in aspects)
            avg_impact = total_impact / len(aspects) if aspects else 0
            
            st.metric("Total Cosmic Energy", f"{total_impact:.1f}")
            st.metric("Average Impact", f"{avg_impact:.2f}")
            st.metric("Active Aspects", len(aspects))
            
            # Risk assessment
            high_risk_aspects = sum(1 for aspect in aspects if abs(aspect['impact']) > 0.7)
            risk_level = "High" if high_risk_aspects >= 3 else "Medium" if high_risk_aspects >= 1 else "Low"
            risk_color = "red" if risk_level == "High" else "orange" if risk_level == "Medium" else "green"
            
            st.markdown(f"**Risk Level:** <span style='color:{risk_color}'>{risk_level}</span>", 
                       unsafe_allow_html=True)
        
        # Detailed insights
        st.subheader('🔮 Detailed Market Insights')
        
        # Create tabs for different insights
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Strategy", "📈 Opportunities", "⚠️ Risks", "🌟 Cosmic Events"])
        
        with tab1:
            st.markdown("""
            **🎯 Recommended Trading Strategy:**
            
            **🟢 Bullish Opportunities:**
            - **Energy Sector**: Mars-Uranus conjunction favors Reliance, Crude Oil
            - **Precious Metals**: Multiple aspects support Gold and Silver
            - **FMCG & Pharma**: Moon-Neptune trine provides defensive strength
            - **PSU Stocks**: Sun-Pluto sextile indicates structural positives
            
            **🔴 Bearish Risks:**
            - **Banking Sector**: Mercury-Jupiter square creates volatility
            - **Automotive & Realty**: Venus-Saturn opposition brings pressure
            - **Technology**: Mixed signals, trade with caution
            
            **⚡ High-Impact Trades:**
            - Consider Gold positions during Venus-Saturn opposition
            - Energy stocks may see sharp moves (Mars-Uranus)
            - BTC could be volatile but trending up on global aspects
            """)
        
        with tab2:
            st.markdown("""
            **📈 Sector-wise Opportunities:**
            
            **🥇 Top Picks:**
            1. **Gold/Silver**: Multiple supportive aspects across all planetary configurations
            2. **Energy Commodities**: Mars-Uranus conjunction + global supply dynamics
            3. **Pharmaceutical**: Moon-Neptune trine supports defensive healthcare
            4. **PSU Banking**: Sun-Pluto sextile for structural transformation
            
            **🎯 Specific Symbols:**
            - **GOLD**: $2,050+ target on safe-haven demand
            - **CRUDE**: Energy transition + Mars-Uranus = volatility opportunities
            - **BTC**: Crypto favorable on Uranus-Pluto aspects
            - **SBI**: PSU transformation theme
            """)
        
        with tab3:
            st.markdown("""
            **⚠️ Risk Management:**
            
            **🔴 High-Risk Sectors:**
            - **Private Banking**: ICICI Bank under Mercury-Jupiter square pressure
            - **Automotive**: Maruti facing Venus-Saturn headwinds
            - **Real Estate**: DLF vulnerable to credit tightening aspects
            
            **📊 Risk Mitigation:**
            - Reduce position sizes during Mercury-Jupiter square (high volatility)
            - Use stop-losses 2-3% below support for Venus-Saturn affected stocks
            - Avoid leveraged positions in Midcap segment (Mars-Uranus volatility)
            
            **⏰ Timing Risks:**
            - Morning session volatility expected (Mercury aspects)
            - Post-lunch session may see pressure (Saturn influence)
            """)
        
        with tab4:
            st.markdown("""
            **🌟 Today's Cosmic Events Schedule:**
            
            **🌅 Pre-Market (Before 9:15 AM):**
            - Mercury-Jupiter square builds tension
            - Global markets influence domestic opening
            
            **🌄 Morning Session (9:15-12:00):**
            - Initial volatility from Mercury aspects
            - Energy stocks may show strength
            
            **🌞 Afternoon Session (12:00-15:30):**
            - Venus-Saturn opposition peaks
            - Defensive sectors gain relative strength
            - Banking sector under pressure
            
            **🌆 Post-Market:**
            - Global commodity movements (Gold, Crude)
            - Crypto markets reaction to day's developments
            
            **📊 Weekly Outlook:**
            - Aspects intensify mid-week
            - Weekend planetary shifts to monitor
            """)
    
    elif section_name == 'Stock Filter':
        st.header('🔍 Advanced Stock Filtering & Sector Analysis')
        
        # Create main layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader('🌟 Today\'s Astrological Configuration')
            
            # Display aspects in a nice format
            aspects_data = []
            for aspect in aspects:
                aspects_data.append({
                    'Planets': aspect['planets'],
                    'Aspect': aspect['aspect_type'],
                    'Impact': f"{aspect['impact']:+.1f}",
                    'Sentiment': aspect['type'].title(),
                    'Strength': '🔥' * min(3, int(abs(aspect['impact']) * 3))
                })
            
            aspects_df = pd.DataFrame(aspects_data)
            
            # Color code the dataframe
            def color_sentiment(val):
                if 'Bullish' in str(val):
                    return 'background-color: #d4edda; color: #155724'
                elif 'Bearish' in str(val):
                    return 'background-color: #f8d7da; color: #721c24'
                return ''
            
            styled_aspects = aspects_df.style.applymap(color_sentiment, subset=['Sentiment'])
            st.dataframe(styled_aspects, use_container_width=True)
        
        with col2:
            st.subheader('📊 Aspect Statistics')
            
            # Create a simple pie chart for aspect types
            aspect_types = {}
            for aspect in aspects:
                aspect_types[aspect['type']] = aspect_types.get(aspect['type'], 0) + 1
            
            if aspect_types:
                fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
                colors = ['green' if k == 'bullish' else 'red' if k == 'bearish' else 'gray' 
                         for k in aspect_types.keys()]
                wedges, texts, autotexts = ax_pie.pie(aspect_types.values(), 
                                                     labels=[k.title() for k in aspect_types.keys()], 
                                                     colors=colors, autopct='%1.0f%%', startangle=90)
                ax_pie.set_title('Today\'s Aspect Distribution')
                st.pyplot(fig_pie)
        
        # Sector analysis
        st.subheader('📈 Sector Impact Analysis')
        
        filtered_stocks = filter_stocks_by_aspects(aspects, STOCK_DATABASE)
        
        # Create sector impact visualization
        sector_impacts_df = pd.DataFrame({
            'Sector': list(filtered_stocks['sector_impacts'].keys()),
            'Impact Score': list(filtered_stocks['sector_impacts'].values()),
            'Recommendation': ['Strong Buy' if x > 0.5 else 'Buy' if x > 0 else 'Hold' if x == 0 
                             else 'Sell' if x > -0.5 else 'Strong Sell' 
                             for x in filtered_stocks['sector_impacts'].values()]
        })
        sector_impacts_df = sector_impacts_df.sort_values('Impact Score', ascending=False)
        
        # Enhanced bar chart
        fig_sectors, ax_sectors = plt.subplots(figsize=(14, 8))
        colors = ['darkgreen' if x > 0.5 else 'green' if x > 0 else 'gray' if x == 0 
                 else 'red' if x > -0.5 else 'darkred' 
                 for x in sector_impacts_df['Impact Score']]
        
        bars = ax_sectors.bar(sector_impacts_df['Sector'], sector_impacts_df['Impact Score'], 
                             color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax_sectors.set_title('Sector Impact Scores - Astrological Analysis', fontsize=16, pad=20)
        ax_sectors.set_ylabel('Impact Score', fontsize=12)
        ax_sectors.set_xlabel('Sector', fontsize=12)
        ax_sectors.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax_sectors.grid(True, alpha=0.3)
        
        # Add value labels and recommendations
        for i, (bar, rec) in enumerate(zip(bars, sector_impacts_df['Recommendation'])):
            height = bar.get_height()
            ax_sectors.annotate(f'{height:.2f}\n{rec}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 5 if height >= 0 else -25),
                               textcoords="offset points",
                               ha='center', va='bottom' if height >= 0 else 'top',
                               fontsize=9, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig_sectors)
        
        # Stock recommendations in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader('🟢 Bullish Stocks')
            if not filtered_stocks['bullish'].empty:
                bullish_df = filtered_stocks['bullish'][['Symbol', 'Sector', 'Impact Score']].copy()
                bullish_df['Action'] = bullish_df['Impact Score'].apply(
                    lambda x: 'Strong Buy' if x > 0.5 else 'Buy'
                )
                
                for _, row in bullish_df.iterrows():
                    with st.container():
                        st.markdown(f"""
                        **{row['Symbol']}** ({row['Sector']})  
                        Score: {row['Impact Score']:.2f} | **{row['Action']}**
                        """)
                        st.progress(min(1.0, row['Impact Score']))
            else:
                st.info("No strong bullish signals today")
        
        with col2:
            st.subheader('🔴 Bearish Stocks')
            if not filtered_stocks['bearish'].empty:
                bearish_df = filtered_stocks['bearish'][['Symbol', 'Sector', 'Impact Score']].copy()
                bearish_df['Action'] = bearish_df['Impact Score'].apply(
                    lambda x: 'Strong Sell' if x > 0.5 else 'Sell'
                )
                
                for _, row in bearish_df.iterrows():
                    with st.container():
                        st.markdown(f"""
                        **{row['Symbol']}** ({row['Sector']})  
                        Risk Score: {row['Impact Score']:.2f} | **{row['Action']}**
                        """)
                        st.progress(min(1.0, row['Impact Score']))
            else:
                st.info("No strong bearish signals today")
        
        with col3:
            st.subheader('⚪ Neutral Stocks')
            if not filtered_stocks['neutral'].empty:
                neutral_df = filtered_stocks['neutral'][['Symbol', 'Sector']].head(5)
                
                for _, row in neutral_df.iterrows():
                    st.markdown(f"**{row['Symbol']}** ({row['Sector']}) - Hold")
            else:
                st.info("All stocks showing directional bias")
    
    elif section_name == 'Aspect Analysis':
        st.header('📋 Deep Astrological Aspect Analysis')
        
        # Generate enhanced analysis
        fig, df_aspects = analyze_aspects()
        st.pyplot(fig)
        
        # Display detailed aspect table
        st.subheader('📊 Detailed Aspect Reference Table')
        
        # Add more columns for better analysis
        df_enhanced = df_aspects.copy()
        df_enhanced['Trading Action'] = df_enhanced.apply(
            lambda row: 'Hedge/Reduce' if 'Bearish' in row['Market Impact'] or 'Tension' in row['Market Impact']
            else 'Accumulate' if 'Bullish' in row['Market Impact'] or 'Rally' in row['Market Impact']
            else 'Monitor', axis=1
        )
        
        df_enhanced['Risk Level'] = df_enhanced['Typical Price Change'].apply(
            lambda x: 'High' if any(num in x for num in ['3', '4']) 
            else 'Medium' if '2' in x else 'Low'
        )
        
        # Style the enhanced dataframe
        def highlight_risk(val):
            if val == 'High':
                return 'background-color: #ffcccc'
            elif val == 'Medium':
                return 'background-color: #ffffcc'
            elif val == 'Low':
                return 'background-color: #ccffcc'
            return ''
        
        def highlight_action(val):
            if val == 'Accumulate':
                return 'background-color: #d4edda; color: #155724'
            elif val == 'Hedge/Reduce':
                return 'background-color: #f8d7da; color: #721c24'
            elif val == 'Monitor':
                return 'background-color: #fff3cd; color: #856404'
            return ''
        
        styled_enhanced = df_enhanced.style.applymap(highlight_risk, subset=['Risk Level']).applymap(highlight_action, subset=['Trading Action'])
        st.dataframe(styled_enhanced, use_container_width=True)
        
        # Aspect interpretation guide
        st.subheader('🔭 Astrological Aspect Interpretation Guide')
        
        tab1, tab2, tab3 = st.tabs(["🌟 Aspect Types", "🪐 Planetary Influences", "📈 Trading Applications"])
        
        with tab1:
            st.markdown("""
            ### Understanding Astrological Aspects
            
            **🔄 Conjunction (0°)**: 
            - *Market Effect*: Powerful combining of energies, can create sharp moves
            - *Trading*: Expect significant price action, potential breakouts
            - *Example*: Mars-Uranus conjunction = explosive energy moves
            
            **⚔️ Square (90°)**: 
            - *Market Effect*: Tension, conflict, volatility
            - *Trading*: Increased intraday swings, good for scalping
            - *Example*: Mercury-Jupiter square = communication/policy confusion
            
            **🎯 Trine (120°)**: 
            - *Market Effect*: Harmonious, easy flow of energy
            - *Trading*: Trending moves, good for position trading
            - *Example*: Moon-Neptune trine = emotional/intuitive support
            
            **⚖️ Opposition (180°)**: 
            - *Market Effect*: Polarization, requires balance
            - *Trading*: Range-bound action, reversals possible
            - *Example*: Venus-Saturn opposition = value vs. restriction
            
            **🤝 Sextile (60°)**: 
            - *Market Effect*: Opportunity aspects, mild positive
            - *Trading*: Gentle trends, good for swing trades
            - *Example*: Sun-Pluto sextile = gradual transformation
            """)
        
        with tab2:
            st.markdown("""
            ### Planetary Market Influences
            
            **☀️ Sun**: Leadership, government policy, large-cap stocks, gold
            **🌙 Moon**: Public sentiment, emotions, consumer sectors, silver
            **☿️ Mercury**: Communication, technology, volatility, news-driven moves
            **♀️ Venus**: Finance, banking, luxury goods, relationships, copper
            **♂️ Mars**: Energy, metals, defense, aggressive moves, oil
            **♃ Jupiter**: Growth, expansion, optimism, financial sector
            **♄ Saturn**: Restriction, discipline, structure, defensive sectors
            **♅ Uranus**: Innovation, technology, sudden changes, crypto
            **♆ Neptune**: Illusion, oil, pharma, confusion, speculation
            **♇ Pluto**: Transformation, power, mining, major shifts
            
            ### Sector-Planet Correlations
            - **Technology**: Mercury, Uranus
            - **Banking**: Jupiter, Venus, Saturn  
            - **Energy**: Mars, Sun, Pluto
            - **Healthcare**: Neptune, Moon
            - **Precious Metals**: Venus, Jupiter, Sun
            - **Cryptocurrency**: Uranus, Pluto
            """)
        
        with tab3:
            st.markdown("""
            ### Practical Trading Applications
            
            **📊 Intraday Trading:**
            - Use Moon aspects for sentiment shifts (2-4 hour cycles)
            - Mercury aspects for news/volatility spikes
            - Mars aspects for energy sector breakouts
            
            **📈 Swing Trading:**
            - Venus aspects for financial sector trends (3-7 days)
            - Jupiter aspects for broad market optimism
            - Saturn aspects for defensive positioning
            
            **📉 Position Trading:**
            - Outer planet aspects (Uranus, Neptune, Pluto) for long-term themes
            - Eclipse patterns for major sector rotations
            - Retrograde periods for trend reversals
            
            **⚠️ Risk Management:**
            - Increase cash during multiple challenging aspects
            - Reduce position size during Mercury retrograde
            - Use tighter stops during Mars-Saturn squares
            
            **🎯 Sector Rotation:**
            - Follow Jupiter through zodiac signs for sector leadership
            - Track Saturn aspects for value opportunities
            - Monitor Uranus for innovation themes
            """)
    
    elif section_name == 'Intraday Chart':
        st.header(f'📈 {symbol} - Intraday Astrological Analysis')
        
        # Display symbol information prominently
        symbol_info = get_symbol_info(symbol)
        trading_hours = get_trading_hours(symbol)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Symbol", symbol)
        with col2:
            st.metric("Sector", symbol_info['sector'])
        with col3:
            st.metric("Currency", symbol_info['currency'])
        with col4:
            session_length = trading_hours['end_hour'] - trading_hours['start_hour'] + \
                           (trading_hours['end_minute'] - trading_hours['start_minute']) / 60
            st.metric("Session Hours", f"{session_length:.1f}h")
        
        # Generate and display chart
        with st.spinner(f'Generating astrological analysis for {symbol}...'):
            fig = generate_intraday_chart(symbol, starting_price, selected_date)
            st.pyplot(fig)
        
        # Trading insights based on symbol
        st.subheader(f'🎯 {symbol} Trading Insights')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Technical Outlook")
            
            # Generate insights based on symbol type
            if symbol in ['GOLD', 'SILVER']:
                st.markdown("""
                **Precious Metals Analysis:**
                - Multiple planetary aspects favor safe-haven demand
                - Venus-Saturn opposition creates financial stress → Gold strength  
                - Moon-Neptune trine supports intuitive precious metal buying
                - Best trading windows: During global uncertainty aspects
                
                **Key Levels:**
                - Watch for breakouts during Mars-Uranus conjunction
                - Support likely during Moon aspects
                - Resistance at previous highs during Saturn aspects
                """)
            
            elif symbol in ['BTC']:
                st.markdown("""
                **Cryptocurrency Analysis:**
                - Uranus aspects strongly favor crypto volatility
                - Mars-Uranus conjunction = explosive price moves
                - Traditional financial stress (Venus-Saturn) → Crypto rotation
                - High volatility expected - use proper risk management
                
                **Trading Strategy:**
                - Momentum plays during Uranus aspects
                - Contrarian plays during Saturn oppositions
                - Volume spikes likely at aspect peaks
                """)
            
            elif symbol in ['CRUDE']:
                st.markdown("""
                **Energy Commodity Analysis:**
                - Mars-Uranus conjunction directly impacts energy sector
                - Global supply disruption themes (Pluto aspects)
                - Geopolitical tensions favor energy prices
                - Weather and seasonal patterns amplified by aspects
                
                **Supply-Demand Factors:**
                - Production disruptions during Mars aspects
                - Demand surges during economic aspects
                - Storage plays during Saturn aspects
                """)
            
            elif symbol in ['DOWJONES']:
                st.markdown("""
                **US Index Analysis:**
                - Jupiter aspects favor broad market optimism
                - Saturn aspects create rotation into defensive sectors
                - Mercury aspects increase intraday volatility
                - Fed policy sensitivity during Venus-Saturn opposition
                
                **Sector Rotation:**
                - Technology during Mercury aspects
                - Energy during Mars aspects  
                - Financials during Jupiter aspects
                """)
            
            else:  # Indian stocks
                st.markdown(f"""
                **{symbol_info['sector']} Sector Analysis:**
                - Domestic market influenced by global planetary patterns
                - FII/DII flows affected by Venus-Saturn aspects
                - Sector rotation based on planetary emphasis
                - Currency impacts during outer planet aspects
                
                **Indian Market Specifics:**
                - Opening gap influenced by global overnight aspects
                - Lunch hour volatility during Moon aspects
                - Closing session strength during Jupiter aspects
                """)
        
        with col2:
            st.markdown("#### ⏰ Timing Analysis")
            
            # Generate time-specific insights based on trading hours
            if trading_hours['end_hour'] > 16:  # Extended hours
                st.markdown("""
                **Extended Session Analysis:**
                
                **🌅 Asian Session (5:00-8:00):**
                - Pre-market positioning based on overnight aspects
                - Lower volumes, higher impact from aspects
                - Key economic data releases amplified
                
                **🌍 European Session (8:00-16:00):**
                - Peak liquidity and aspect impacts
                - Central bank policy influences
                - Cross-asset correlations strongest
                
                **🌎 US Session (16:00-20:00):**
                - Maximum volatility potential
                - Aspect peaks create significant moves
                - News flow interaction with cosmic patterns
                
                **🌙 After Hours (20:00-23:55):**
                - Reduced liquidity amplifies aspect effects
                - Position adjustments for next day
                - Asian preview impact
                """)
            else:  # Indian market hours
                st.markdown("""
                **Indian Session Analysis:**
                
                **🌅 Opening (9:15-10:30):**
                - Gap opening based on global aspects
                - High volatility, aspect impacts magnified
                - Initial trend direction setting
                
                **🌞 Mid-Morning (10:30-12:00):**
                - Institutional activity peaks
                - Aspect-driven sector rotation
                - News flow integration
                
                **🍽️ Lunch Hour (12:00-13:00):**
                - Reduced activity, Moon aspects dominate
                - Range-bound unless strong aspects active
                - Position consolidation period
                
                **🌆 Closing (13:00-15:30):**
                - Final institutional positioning
                - Aspect resolution for day
                - Next-day setup formation
                """)
            
            # Risk management
            st.markdown("#### ⚠️ Risk Management")
            st.markdown(f"""
            **Position Sizing:**
            - Standard position: 1-2% of capital
            - High aspect days: Reduce to 0.5-1%
            - Strong confluence: Increase to 2-3%
            
            **Stop Loss Levels:**
            - Tight stops during Mercury aspects: 1-2%
            - Normal stops during stable aspects: 2-3%
            - Wide stops during Mars aspects: 3-5%
            
            **Profit Targets:**
            - Quick scalps: 0.5-1% (15-30 minutes)
            - Swing trades: 2-5% (2-4 hours)
            - Position trades: 5-10% (1-3 days)
            
            **Volatility Adjustments:**
            - {symbol}: Expected daily range ±{2.5 if symbol in ['BTC'] else 1.5 if symbol in ['CRUDE'] else 1.0 if symbol in ['GOLD', 'SILVER'] else 0.8}%
            - Adjust position size inversely to volatility
            - Use options for high-volatility periods
            """)
    
    elif section_name == 'Monthly Chart':
        st.header(f'📊 {symbol} - Monthly Astrological Trend Analysis')
        
        # Display symbol information
        symbol_info = get_symbol_info(symbol)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Symbol", symbol)
        with col2:
            st.metric("Analysis Period", f"{calendar.month_name[selected_month]} {selected_year}")
        with col3:
            st.metric("Sector Focus", symbol_info['sector'])
        with col4:
            st.metric("Currency", symbol_info['currency'])
        
        # Generate and display chart
        with st.spinner(f'Generating monthly analysis for {symbol}...'):
            fig = generate_monthly_chart(symbol, starting_price, selected_month, selected_year)
            st.pyplot(fig)
        
        # Monthly analysis insights
        st.subheader(f'📈 {calendar.month_name[selected_month]} {selected_year} - Strategic Analysis')
        
        # Create tabs for different analysis aspects
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Monthly Outlook", "📊 Technical Analysis", "🌙 Lunar Cycles", "💼 Portfolio Strategy"])
        
        with tab1:
            month_name = calendar.month_name[selected_month]
            
            if symbol in ['GOLD', 'SILVER']:
                st.markdown(f"""
                ### {symbol} - {month_name} {selected_year} Precious Metals Outlook
                
                **🌟 Astrological Themes:**
                - **Venus-Jupiter aspects**: Strong precious metals demand from financial uncertainty
                - **Saturn transits**: Safe-haven buying during economic restrictions
                - **Moon phases**: Emotional buying patterns aligned with lunar cycles
                - **Mercury retrograde periods**: Technical analysis less reliable, fundamentals dominate
                
                **📈 Price Drivers:**
                - Central bank policy uncertainty (Saturn aspects)
                - Currency devaluation themes (Pluto aspects)
                - Geopolitical tensions (Mars aspects)
                - Inflation hedging demand (Jupiter-Saturn aspects)
                
                **🎯 Trading Strategy:**
                - **Accumulate** during New Moon phases (stronger buying interest)
                - **Profit-take** during Full Moon phases (emotional peaks)
                - **Hold through** Mercury retrograde (avoid technical trading)
                - **Scale in** during Saturn aspects (structural support)
                
                **📊 Target Levels:**
                - **Monthly High**: Expect during Jupiter-Venus trine periods
                - **Monthly Low**: Likely during Mars-Saturn square periods
                - **Breakout Potential**: Mars-Uranus conjunction periods
                - **Support Zones**: Previous month's Jupiter aspect levels
                """)
            
            elif symbol in ['BTC']:
                st.markdown(f"""
                ### {symbol} - {month_name} {selected_year} Cryptocurrency Outlook
                
                **⚡ Astrological Themes:**
                - **Uranus-Pluto aspects**: Revolutionary technology adoption waves
                - **Mercury-Uranus aspects**: Network upgrades and technical developments
                - **Mars-Uranus conjunctions**: Explosive price movements and FOMO
                - **Saturn aspects**: Regulatory clarity or restrictions
                
                **🚀 Volatility Drivers:**
                - Institutional adoption news (Jupiter aspects)
                - Regulatory developments (Saturn aspects)
                - Technical network changes (Mercury-Uranus)
                - Market manipulation concerns (Neptune aspects)
                
                **⚠️ Risk Factors:**
                - **High volatility** during Mars-Uranus aspects (±10-20% daily swings)
                - **Regulatory risks** during Saturn-Pluto aspects
                - **Technical failures** during Mercury retrograde
                - **Market manipulation** during Neptune-Mercury aspects
                
                **💡 Strategic Approach:**
                - **DCA strategy** during volatile periods
                - **Momentum trading** during Uranus aspects
                - **Risk-off** during Saturn hard aspects
                - **HODL mentality** during Jupiter-Pluto trines
                """)
            
            elif symbol in ['CRUDE']:
                st.markdown(f"""
                ### {symbol} - {month_name} {selected_year} Energy Commodity Outlook
                
                **🛢️ Astrological Themes:**
                - **Mars-Pluto aspects**: Geopolitical tensions affecting supply
                - **Jupiter-Saturn cycles**: Economic growth vs. restriction cycles
                - **Uranus aspects**: Renewable energy transition impacts
                - **Moon phases**: Seasonal demand patterns amplified
                
                **⚡ Supply-Demand Dynamics:**
                - Production disruptions (Mars-Saturn squares)
                - Economic growth spurts (Jupiter aspects)
                - Weather pattern extremes (Uranus-Neptune aspects)
                - Strategic reserve changes (Pluto aspects)
                
                **🌍 Geopolitical Factors:**
                - **OPEC decisions** aligned with Saturn aspects
                - **Pipeline disruptions** during Mars-Uranus periods
                - **Currency impacts** during Venus-Pluto aspects
                - **Seasonal patterns** enhanced by lunar cycles
                
                **📈 Trading Levels:**
                - **Resistance**: Previous Jupiter aspect highs
                - **Support**: Saturn aspect consolidation zones
                - **Breakout zones**: Mars-Uranus conjunction levels
                - **Reversal points**: Full Moon technical confluences
                """)
            
            elif symbol in ['DOWJONES']:
                st.markdown(f"""
                ### {symbol} - {month_name} {selected_year} US Index Outlook
                
                **🇺🇸 Macro Astrological Themes:**
                - **Jupiter-Saturn cycles**: Economic expansion vs. contraction
                - **Mercury-Venus aspects**: Corporate earnings and consumer spending
                - **Mars-Jupiter aspects**: Business investment and growth
                - **Outer planet aspects**: Long-term structural changes
                
                **🏛️ Federal Reserve Alignment:**
                - **Venus-Saturn aspects**: Interest rate policy changes
                - **Mercury-Jupiter aspects**: Fed communication clarity
                - **Moon phases**: Market sentiment around FOMC meetings
                - **Eclipse periods**: Major policy shift announcements
                
                **🔄 Sector Rotation Patterns:**
                - **Technology** leadership during Mercury-Uranus aspects
                - **Energy** strength during Mars-Pluto periods
                - **Financials** favor during Venus-Jupiter trines
                - **Healthcare** defensive during Saturn aspects
                
                **📊 Technical Confluence:**
                - **Monthly resistance**: Jupiter aspect previous highs
                - **Monthly support**: Saturn aspect previous lows
                - **Breakout potential**: New Moon near technical levels
                - **Reversal zones**: Full Moon at key Fibonacci levels
                """)
            
            else:  # Indian stocks
                st.markdown(f"""
                ### {symbol} ({symbol_info['sector']}) - {month_name} {selected_year} Indian Market Outlook
                
                **🇮🇳 Domestic Astrological Influences:**
                - **Jupiter transits**: Market leadership and FII flows
                - **Saturn aspects**: Regulatory changes and policy shifts
                - **Mars-Venus aspects**: Consumer spending and investment flows
                - **Moon phases**: Retail investor sentiment cycles
                
                **💹 Sector-Specific Themes:**
                - **{symbol_info['sector']} sector** influenced by specific planetary combinations
                - **Monsoon patterns** (if applicable) aligned with water sign emphasis
                - **Festival seasons** amplified by benefic planetary aspects
                - **Budget impacts** during Saturn-Jupiter aspects
                
                **🌏 Global Correlation Factors:**
                - **US Fed policy** impacts during Venus-Saturn aspects
                - **China growth** concerns during Mars-Saturn periods  
                - **Oil prices** affecting through Mars-Pluto aspects
                - **Dollar strength** impacts during Pluto aspects
                
                **📈 Monthly Strategy:**
                - **Accumulate** during Saturn aspects (value opportunities)
                - **Momentum plays** during Mars-Jupiter periods
                - **Defensive positioning** during challenging outer planet aspects
                - **Sector rotation** based on planetary emphasis shifts
                """)
        
        with tab2:
            st.markdown(f"""
            ### Technical Analysis Integration with Astrological Cycles
            
            **📊 Moving Average Alignment:**
            - **MA5 vs MA20**: Bullish when Jupiter aspects dominate
            - **Golden Cross** potential during Venus-Jupiter trines
            - **Death Cross** risk during Saturn-Mars squares
            - **MA support/resistance** stronger during lunar phases
            
            **🎯 Support & Resistance Levels:**
            - **Primary resistance**: Previous month's Jupiter aspect highs
            - **Primary support**: Saturn aspect consolidation lows
            - **Secondary levels**: Full Moon reversal points
            - **Breakout levels**: New Moon momentum points
            
            **📈 Momentum Indicators:**
            - **RSI overbought** (>70) more reliable during Full Moons
            - **RSI oversold** (<30) stronger signal during New Moons
            - **MACD divergences** amplified during Mercury aspects
            - **Volume confirmations** critical during Mars aspects
            
            **🌙 Lunar Cycle Technical Correlation:**
            - **New Moon**: Trend initiation, breakout potential
            - **Waxing Moon**: Momentum continuation, bullish bias
            - **Full Moon**: Trend exhaustion, reversal potential
            - **Waning Moon**: Correction phases, consolidation
            
            **⚡ Volatility Patterns:**
            - **Highest volatility**: Mars-Uranus aspect periods
            - **Lowest volatility**: Venus-Jupiter trine periods
            - **Unexpected moves**: Mercury-Neptune confusion aspects
            - **Gap movements**: Eclipse and outer planet aspects
            
            **🔄 Pattern Recognition:**
            - **Triangle breakouts** during Uranus aspects
            - **Flag patterns** during Mars aspects  
            - **Head & Shoulders** during Saturn aspects
            - **Double tops/bottoms** during opposition aspects
            """)
        
        with tab3:
            st.markdown(f"""
            ### Lunar Cycles & Market Psychology for {month_name} {selected_year}
            
            **🌑 New Moon Phases (Market Initiation):**
            - **Energy**: Fresh starts, new trend beginnings
            - **Psychology**: Optimism, risk-taking increases
            - **Trading**: Look for breakout setups, trend initiations
            - **Volume**: Often lower but quality moves
            - **Best for**: Opening new positions, trend following
            
            **🌓 Waxing Moon (Building Momentum):**
            - **Energy**: Growth, expansion, building confidence  
            - **Psychology**: FOMO starts building, bullish sentiment
            - **Trading**: Momentum continuation, pyramid additions
            - **Volume**: Increasing participation
            - **Best for**: Adding to winning positions
            
            **🌕 Full Moon Phases (Emotional Peaks):**
            - **Energy**: Maximum emotion, extremes, reversals
            - **Psychology**: Euphoria or panic peaks
            - **Trading**: Reversal setups, profit-taking
            - **Volume**: Often highest of cycle
            - **Best for**: Profit booking, contrarian plays
            
            **🌗 Waning Moon (Consolidation):**
            - **Energy**: Release, correction, cooling off
            - **Psychology**: Reality check, risk assessment
            - **Trading**: Consolidation patterns, value hunting
            - **Volume**: Declining, selective moves
            - **Best for**: Position adjustments, planning
            
            **🔮 {month_name} {selected_year} Specific Lunar Events:**
            
            **Key Lunar Dates to Watch:**
            - **New Moon**: Potential trend change or continuation signal
            - **First Quarter**: Momentum confirmation or failure
            - **Full Moon**: Profit-taking opportunity or reversal signal  
            - **Last Quarter**: Consolidation phase or weakness signal
            
            **Moon Sign Influences:**
            - **Fire Signs** (Aries, Leo, Sagittarius): Aggressive moves, energy sector strength
            - **Earth Signs** (Taurus, Virgo, Capricorn): Value focus, stability preference
            - **Air Signs** (Gemini, Libra, Aquarius): Communication, technology emphasis
            - **Water Signs** (Cancer, Scorpio, Pisces): Emotional decisions, defensive moves
            """)
        
        with tab4:
            st.markdown(f"""
            ### Portfolio Strategy for {month_name} {selected_year}
            
            **🎯 Strategic Asset Allocation:**
            
            **Core Holdings (50-60%):**
            - **Large Cap Stability**: Jupiter-aspected blue chips
            - **Sector Leaders**: Dominant players in favored sectors
            - **Defensive Assets**: During challenging aspect periods
            - **Currency Hedge**: If significant Pluto aspects present
            
            **Growth Opportunities (20-30%):**
            - **Momentum Plays**: Mars-Jupiter aspect beneficiaries
            - **Breakout Candidates**: Technical + astrological confluence
            - **Sector Rotation**: Following planetary emphasis shifts
            - **Emerging Themes**: Uranus aspect innovation plays
            
            **Speculative/Trading (10-20%):**
            - **High Beta Names**: For Mars-Uranus periods
            - **Volatility Plays**: Options during aspect peaks
            - **Contrarian Bets**: Against crowd during extremes
            - **Crypto Allocation**: If comfortable with high volatility
            
            **📊 Risk Management Framework:**
            
            **Position Sizing Rules:**
            - **Maximum single position**: 5% during stable periods
            - **Reduce to 3%**: During challenging aspects
            - **Increase to 7%**: During strong favorable confluences
            - **Cash levels**: 10-20% based on aspect favorability
            
            **Stop Loss Strategy:**
            - **Tight stops** (3-5%): During Mercury retrograde periods
            - **Normal stops** (5-8%): During regular market conditions
            - **Wide stops** (8-12%): During high volatility aspect periods
            - **No stops**: For long-term Jupiter-blessed positions
            
            **📅 Monthly Rebalancing Schedule:**
            
            **Week 1**: Review and adjust based on new lunar cycle
            **Week 2**: Add to momentum winners if aspects support
            **Week 3**: Prepare for Full Moon profit-taking opportunities
            **Week 4**: Position for next month's astrological themes
            
            **🔄 Sector Rotation Strategy:**
            
            **Early Month**: Follow Jupiter aspects for growth sectors
            **Mid Month**: Mars aspects may favor energy/materials
            **Late Month**: Venus aspects support financials/consumer
            **Month End**: Saturn aspects favor defensives/utilities
            
            **💡 Advanced Strategies:**
            
            **Pairs Trading**: Long favored sectors, short challenged sectors
            **Options Overlay**: Sell calls during Full Moons, buy calls during New Moons
            **Currency Hedge**: Hedge foreign exposure during Pluto aspects
            **Volatility Trading**: Long volatility before aspect peaks
            
            **📈 Performance Tracking:**
            
            **Monthly Metrics**:
            - Absolute return vs. benchmark
            - Risk-adjusted return (Sharpe ratio)
            - Maximum drawdown during challenging aspects
            - Hit rate on astrological predictions
            
            **Aspect Correlation Analysis**:
            - Track which aspects work best for {symbol}
            - Note sector rotation timing accuracy
            - Measure volatility prediction success
            - Document lunar cycle correlations
            """)
        
        # Additional insights for monthly strategy
        st.subheader('🎭 Market Psychology & Sentiment Analysis')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            #### 🧠 Psychological Drivers - {month_name}
            
            **Institutional Behavior:**
            - Month-end window dressing effects
            - Quarterly rebalancing influences  
            - Earnings season psychological impacts
            - Fed meeting anticipation/reaction
            
            **Retail Investor Patterns:**
            - Payroll cycle investment flows
            - Tax implications (if year-end)
            - Holiday season spending impacts
            - Social media sentiment amplification
            
            **Global Sentiment Factors:**
            - US-China trade relationship status
            - European economic data impacts
            - Emerging market flow dynamics
            - Cryptocurrency correlation effects
            """)
        
        with col2:
            st.markdown(f"""
            #### 📊 Sentiment Indicators to Watch
            
            **Technical Sentiment:**
            - VIX levels and term structure
            - Put/Call ratios by sector
            - High-low index readings
            - Advance-decline line trends
            
            **Fundamental Sentiment:**
            - Earnings revision trends
            - Analyst recommendation changes
            - Insider buying/selling activity
            - Share buyback announcements
            
            **Alternative Data:**
            - Google search trends
            - Social media mention analysis
            - Options flow analysis
            - Crypto correlation strength
            """)

# Footer with additional information
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <h4>🌟 Disclaimer & Important Notes</h4>
        <p><strong>Educational Purpose Only:</strong> This dashboard is for educational and research purposes. 
        Astrological analysis should be combined with fundamental and technical analysis for trading decisions.</p>
        
        <p><strong>Risk Warning:</strong> All trading involves risk. Past performance and astrological correlations 
        do not guarantee future results. Always use proper risk management and never invest more than you can afford to lose.</p>
        
        <p><strong>Data Sources:</strong> Simulated price data based on astrological aspect calculations. 
        For live trading, use real market data and professional trading platforms.</p>
        
        <p style='font-size: 12px; margin-top: 20px;'>
        🔮 <em>"The stars impel, they do not compel. Wisdom lies in using all available tools - 
        fundamental, technical, and cosmic - for informed decision making."</em>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main(), 'default_price': 35000.0, 'sector': 'US Index'},
    'SILVER': {'name': 'Silver Futures', 'currency': '

# --- STOCK DATABASE ---
stock_data = {
    'Symbol': [
        'TCS', 'ICICIBANK', 'MARUTI', 'DLF', 'NESTLEIND', 
        'RELIANCE', 'SBI', 'SUNPHARMA', 'DRREDDY', 'GOLD',
        'DOWJONES', 'SILVER', 'CRUDE', 'BTC'
    ],
    'Sector': [
        'Technology', 'Banking', 'Automotive', 'Realty', 'FMCG',
        'Energy', 'PSUs', 'Pharma', 'Pharma', 'Precious Metals',
        'US Index', 'Precious Metals', 'Energy', 'Cryptocurrency'
    ],
    'MarketCap': [
        'Large', 'Large', 'Large', 'Large', 'Large',
        'Large', 'Large', 'Large', 'Large', 'Commodity',
        'Index', 'Commodity', 'Commodity', 'Crypto'
    ]
}

STOCK_DATABASE = pd.DataFrame(stock_data)

# --- SECTOR-PLANETARY MAPPINGS ---
SECTOR_PLANETARY_INFLUENCES = {
    'Technology': ['Mercury'],
    'Banking': ['Jupiter', 'Saturn'],
    'FMCG': ['Moon'],
    'Pharma': ['Neptune'],
    'Energy': ['Mars'],
    'Automotive': ['Saturn'],
    'Realty': ['Saturn'],
    'PSUs': ['Pluto'],
    'Midcaps': ['Uranus'],
    'Smallcaps': ['Pluto'],
    'Precious Metals': ['Venus', 'Jupiter'],
    'US Index': ['Jupiter', 'Saturn'],
    'Cryptocurrency': ['Uranus', 'Pluto']
}

# --- ASPECT-SECTOR IMPACT ---
ASPECT_SECTOR_IMPACTS = {
    'Square': {
        'Technology': 'Negative', 'Banking': 'Negative', 'FMCG': 'Neutral',
        'Pharma': 'Neutral', 'Energy': 'Neutral', 'Automotive': 'Neutral',
        'Realty': 'Neutral', 'PSUs': 'Neutral', 'Midcaps': 'Neutral',
        'Smallcaps': 'Neutral', 'Precious Metals': 'Positive', 'US Index': 'Negative',
        'Cryptocurrency': 'Negative'
    },
    'Opposition': {
        'Technology': 'Neutral', 'Banking': 'Neutral', 'FMCG': 'Neutral',
        'Pharma': 'Neutral', 'Energy': 'Neutral', 'Automotive': 'Negative',
        'Realty': 'Negative', 'PSUs': 'Neutral', 'Midcaps': 'Neutral',
        'Smallcaps': 'Neutral', 'Precious Metals': 'Positive', 'US Index': 'Neutral',
        'Cryptocurrency': 'Positive'
    },
    'Trine': {
        'Technology': 'Neutral', 'Banking': 'Neutral', 'FMCG': 'Positive',
        'Pharma': 'Positive', 'Energy': 'Neutral', 'Automotive': 'Neutral',
        'Realty': 'Neutral', 'PSUs': 'Neutral', 'Midcaps': 'Neutral',
        'Smallcaps': 'Neutral', 'Precious Metals': 'Positive', 'US Index': 'Positive',
        'Cryptocurrency': 'Neutral'
    },
    'Conjunction': {
        'Technology': 'Neutral', 'Banking': 'Neutral', 'FMCG': 'Neutral',
        'Pharma': 'Neutral', 'Energy': 'Positive', 'Automotive': 'Neutral',
        'Realty': 'Neutral', 'PSUs': 'Neutral', 'Midcaps': 'Negative',
        'Smallcaps': 'Neutral', 'Precious Metals': 'Positive', 'US Index': 'Neutral',
        'Cryptocurrency': 'Positive'
    },
    'Sextile': {
        'Technology': 'Neutral', 'Banking': 'Neutral', 'FMCG': 'Neutral',
        'Pharma': 'Neutral', 'Energy': 'Neutral', 'Automotive': 'Neutral',
        'Realty': 'Neutral', 'PSUs': 'Positive', 'Midcaps': 'Neutral',
        'Smallcaps': 'Negative', 'Precious Metals': 'Positive', 'US Index': 'Positive',
        'Cryptocurrency': 'Neutral'
    }
}

# --- PLANETARY POSITION VISUALIZATION ---
def draw_planetary_wheel(ax, input_date, size=0.3):
    """Draw a simplified astrological wheel showing planetary positions"""
    base_date = datetime(2025, 8, 1)
    
    if isinstance(input_date, date_class) and not isinstance(input_date, datetime):
        date_obj = datetime.combine(input_date, datetime.min.time())
    else:
        date_obj = input_date
    
    days_diff = (date_obj.date() - base_date.date()).days
    
    base_positions = {
        'Sun': 135, 'Moon': 225, 'Mercury': 120, 'Venus': 170,
        'Mars': 85, 'Jupiter': 45, 'Saturn': 315
    }
    
    daily_movement = {
        'Sun': 1.0, 'Moon': 13.2, 'Mercury': 1.5, 'Venus': 1.2,
        'Mars': 0.5, 'Jupiter': 0.08, 'Saturn': 0.03
    }
    
    planets = {}
    for planet, base_pos in base_positions.items():
        angle = (base_pos + daily_movement[planet] * days_diff) % 360
        planets[planet] = {
            'angle': angle,
            'color': {
                'Sun': 'gold', 'Moon': 'silver', 'Mercury': 'gray',
                'Venus': 'lightgreen', 'Mars': 'red', 'Jupiter': 'orange',
                'Saturn': 'darkgoldenrod'
            }[planet],
            'size': {
                'Sun': 8, 'Moon': 6, 'Mercury': 5, 'Venus': 7,
                'Mars': 6, 'Jupiter': 10, 'Saturn': 9
            }[planet]
        }
    
    zodiac = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 
              'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
    
    for i, sign in enumerate(zodiac):
        angle = i * 30
        ax.add_patch(Wedge((0, 0), size, angle, angle+30, width=size*0.8, 
                          facecolor='lightgray', edgecolor='black', alpha=0.3))
        ax.text(0.85*size * np.cos(np.radians(angle+15)), 
                0.85*size * np.sin(np.radians(angle+15)), 
                sign[:3], ha='center', va='center', fontsize=6)
    
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
    ax.set_title(f'Planetary Positions\n{date_obj.strftime("%b %d, %Y")}', fontsize=8)

# --- GET TRADING HOURS FOR SYMBOL ---
def get_trading_hours(symbol):
    """Get trading hours for a specific symbol"""
    symbol = symbol.upper()
    if symbol in TRADING_HOURS:
        return TRADING_HOURS[symbol]
    else:
        # Default to Indian market hours for unknown symbols
        return TRADING_HOURS['NIFTY']

# --- GET SYMBOL INFO ---
def get_symbol_info(symbol):
    """Get symbol configuration info"""
    symbol = symbol.upper()
    if symbol in SYMBOL_CONFIG:
        return SYMBOL_CONFIG[symbol]
    else:
        # Default configuration
        return {
            'name': symbol,
            'currency': '₹',
            'default_price': 1000.0,
            'sector': 'Unknown'
        }

# --- GENERATE ASPECTS ---
def generate_todays_aspects():
    """Generate astrological aspects for today based on the provided table"""
    base_aspects = [
        {"planets": "Mercury-Jupiter", "aspect_type": "Square", "impact": -0.7, "type": "bearish"},
        {"planets": "Venus-Saturn", "aspect_type": "Opposition", "impact": -0.8, "type": "bearish"},
        {"planets": "Moon-Neptune", "aspect_type": "Trine", "impact": 0.6, "type": "bullish"},
        {"planets": "Mars-Uranus", "aspect_type": "Conjunction", "impact": 0.9, "type": "bullish"},
        {"planets": "Sun-Pluto", "aspect_type": "Sextile", "impact": 0.5, "type": "bullish"}
    ]
    
    aspects = []
    for aspect in base_aspects:
        aspects.append({
            "planets": aspect["planets"],
            "aspect_type": aspect["aspect_type"],
            "impact": aspect["impact"],
            "type": aspect["type"]
        })
    
    return aspects

# --- CREATE SUMMARY TABLE ---
def create_summary_table(aspects):
    """Create a summary table based on the astrological aspects"""
    summary_data = {
        'Aspect': [],
        'Nifty/Bank Nifty': [],
        'Bullish Sectors/Stocks': [],
        'Bearish Sectors/Stocks': []
    }
    
    for aspect in aspects:
        planets = aspect["planets"]
        aspect_type = aspect["aspect_type"]
        
        if planets == "Mercury-Jupiter" and aspect_type == "Square":
            summary_data['Aspect'].append("Mercury-Jupiter (Square)")
            summary_data['Nifty/Bank Nifty'].append("Volatile")
            summary_data['Bullish Sectors/Stocks'].append("IT (TCS), Gold")
            summary_data['Bearish Sectors/Stocks'].append("Banking (ICICI Bank), Crypto")
        
        elif planets == "Venus-Saturn" and aspect_type == "Opposition":
            summary_data['Aspect'].append("Venus-Saturn (Opposition)")
            summary_data['Nifty/Bank Nifty'].append("Downside")
            summary_data['Bullish Sectors/Stocks'].append("Gold, Silver, BTC")
            summary_data['Bearish Sectors/Stocks'].append("Auto (Maruti), Realty (DLF)")
        
        elif planets == "Moon-Neptune" and aspect_type == "Trine":
            summary_data['Aspect'].append("Moon-Neptune (Trine)")
            summary_data['Nifty/Bank Nifty'].append("Mild Support")
            summary_data['Bullish Sectors/Stocks'].append("FMCG (Nestlé), Pharma, Gold, Dow Jones")
            summary_data['Bearish Sectors/Stocks'].append("-")
        
        elif planets == "Mars-Uranus" and aspect_type == "Conjunction":
            summary_data['Aspect'].append("Mars-Uranus (Conjunction)")
            summary_data['Nifty/Bank Nifty'].append("Sharp Moves")
            summary_data['Bullish Sectors/Stocks'].append("Energy (Reliance, Crude), Gold, BTC")
            summary_data['Bearish Sectors/Stocks'].append("Weak Midcaps")
        
        elif planets == "Sun-Pluto" and aspect_type == "Sextile":
            summary_data['Aspect'].append("Sun-Pluto (Sextile)")
            summary_data['Nifty/Bank Nifty'].append("Structural Shift")
            summary_data['Bullish Sectors/Stocks'].append("PSUs (SBI), Gold, Dow Jones")
            summary_data['Bearish Sectors/Stocks'].append("Overvalued Smallcaps")
    
    return pd.DataFrame(summary_data)

# --- FILTER STOCKS BASED ON ASPECTS ---
def filter_stocks_by_aspects(aspects, stock_database):
    """Filter stocks based on today's astrological aspects"""
    sector_impacts = {sector: 0 for sector in stock_database['Sector'].unique()}
    
    for aspect in aspects:
        planet1, planet2 = aspect["planets"].split("-")
        
        for sector, planets in SECTOR_PLANETARY_INFLUENCES.items():
            if planet1 in planets or planet2 in planets:
                if sector not in sector_impacts:
                    sector_impacts[sector] = 0
                
                aspect_impact = ASPECT_SECTOR_IMPACTS[aspect["aspect_type"]].get(sector, "Neutral")
                
                if aspect_impact == "Positive":
                    sector_impacts[sector] += abs(aspect["impact"])
                elif aspect_impact == "Negative":
                    sector_impacts[sector] -= abs(aspect["impact"])
    
    bullish_sectors = [sector for sector, impact in sector_impacts.items() if impact > 0]
    bearish_sectors = [sector for sector, impact in sector_impacts.items() if impact < 0]
    neutral_sectors = [sector for sector, impact in sector_impacts.items() if impact == 0]
    
    bullish_stocks = stock_database[stock_database['Sector'].isin(bullish_sectors)].copy()
    bearish_stocks = stock_database[stock_database['Sector'].isin(bearish_sectors)].copy()
    neutral_stocks = stock_database[stock_database['Sector'].isin(neutral_sectors)].copy()
    
    bullish_stocks['Impact Score'] = bullish_stocks['Sector'].apply(lambda x: sector_impacts[x])
    bearish_stocks['Impact Score'] = bearish_stocks['Sector'].apply(lambda x: abs(sector_impacts[x]))
    neutral_stocks['Impact Score'] = 0
    
    bullish_stocks = bullish_stocks.sort_values('Impact Score', ascending=False)
    bearish_stocks = bearish_stocks.sort_values('Impact Score', ascending=False)
    
    return {
        'bullish': bullish_stocks,
        'bearish': bearish_stocks,
        'neutral': neutral_stocks,
        'sector_impacts': sector_impacts
    }

# --- GENERATE ASTROLOGICAL EVENTS ---
def generate_astrological_events(input_date, event_type='intraday', symbol='NIFTY'):
    """Generate astrological events for any given date and symbol"""
    
    if event_type == 'intraday':
        trading_hours = get_trading_hours(symbol)
        
        # Different event patterns based on trading hours
        if trading_hours['end_hour'] > 16:  # Extended hours (global markets)
            # More events spread across longer trading day
            base_events = [
                {"time_offset": 0, "aspect": "Pre-market: Mercury square Jupiter", "impact": -0.5, "type": "bearish"},
                {"time_offset": 120, "aspect": "Asian session: Moon trine Jupiter", "impact": 0.8, "type": "bullish"},
                {"time_offset": 240, "aspect": "London open: Mars sextile Jupiter", "impact": 0.4, "type": "neutral"},
                {"time_offset": 360, "aspect": "European session: Venus opposition Saturn", "impact": -0.6, "type": "bearish"},
                {"time_offset": 480, "aspect": "NY pre-market: Sun conjunct Mercury", "impact": 0.3, "type": "neutral"},
                {"time_offset": 600, "aspect": "US open: Mars conjunct Uranus", "impact": 1.0, "type": "bullish"},
                {"time_offset": 720, "aspect": "Mid-day: Moon square Saturn", "impact": -0.4, "type": "bearish"},
                {"time_offset": 840, "aspect": "Afternoon: Jupiter trine Neptune", "impact": 0.7, "type": "bullish"},
                {"time_offset": 960, "aspect": "US close approach", "impact": 0.2, "type": "neutral"},
                {"time_offset": 1080, "aspect": "After hours: Void Moon", "impact": -0.3, "type": "bearish"},
                {"time_offset": 1135, "aspect": "Session close", "impact": 0.1, "type": "neutral"}
            ]
        else:  # Standard Indian market hours
            base_events = [
                {"time_offset": 0, "aspect": "Opening: Mercury square Jupiter + Void Moon", "impact": -0.5, "type": "bearish"},
                {"time_offset": 45, "aspect": "Early trade: Moon trine Jupiter", "impact": 1.0, "type": "bullish"},
                {"time_offset": 135, "aspect": "Mid-morning: Mars sextile Jupiter", "impact": 0.3, "type": "neutral"},
                {"time_offset": 195, "aspect": "Pre-lunch: Sun in Leo (no aspects)", "impact": 0.0, "type": "neutral"},
                {"time_offset": 285, "aspect": "Post-lunch: Moon square Saturn", "impact": -0.8, "type": "bearish"},
                {"time_offset": 345, "aspect": "Late trade: Venus-Saturn opposition", "impact": -0.6, "type": "bearish"},
                {"time_offset": 375, "aspect": "Close", "impact": 0.1, "type": "neutral"}
            ]
        
        events = []
        if isinstance(input_date, date_class) and not isinstance(input_date, datetime):
            dt = datetime.combine(input_date, datetime.min.time())
        else:
            dt = input_date
            
        start_time = dt.replace(hour=trading_hours['start_hour'], minute=trading_hours['start_minute'])
        
        for event in base_events:
            event_time = start_time + timedelta(minutes=event["time_offset"])
            events.append({
                "time": event_time,
                "aspect": event["aspect"],
                "impact": event["impact"],
                "type": event["type"],
                "price": 0
            })
        
        return events
    
    else:  # monthly events remain the same
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
        
        if isinstance(input_date, datetime):
            year, month = input_date.year, input_date.month
        else:
            year, month = input_date.year, input_date.month
            
        days_in_month = calendar.monthrange(year, month)[1]
        
        events = []
        for event in base_events:
            day = min(event["day_offset"], days_in_month)
            event_date = datetime(year, month, day)
            events.append({
                "date": event_date,
                "aspect": event["aspect"],
                "impact": event["impact"],
                "type": event["type"],
                "price": 0
            })
        
        return events

# --- ENHANCED INTRADAY CHART ---
def generate_intraday_chart(symbol, starting_price, selected_date):
    """Generate enhanced intraday chart with dynamic layout"""
    symbol_info = get_symbol_info(symbol)
    trading_hours = get_trading_hours(symbol)
    
    if isinstance(selected_date, date_class) and not isinstance(selected_date, datetime):
        selected_date = datetime.combine(selected_date, datetime.min.time())
    
    start_time = selected_date.replace(hour=trading_hours['start_hour'], minute=trading_hours['start_minute'])
    end_time = selected_date.replace(hour=trading_hours['end_hour'], minute=trading_hours['end_minute'])
    
    # Adjust interval based on trading session length
    session_hours = (end_time - start_time).total_seconds() / 3600
    if session_hours > 12:
        interval = '30T'  # 30-minute intervals for long sessions
    else:
        interval = '15T'  # 15-minute intervals for shorter sessions
    
    times = pd.date_range(start=start_time, end=end_time, freq=interval)
    
    prices = np.zeros(len(times))
    base_price = starting_price
    
    events = generate_astrological_events(selected_date, 'intraday', symbol)
    
    # Adjust event impacts based on symbol type
    symbol_multiplier = 1.0
    if symbol in ['GOLD', 'SILVER']:
        symbol_multiplier = 0.8  # Precious metals less volatile to aspects
    elif symbol in ['BTC']:
        symbol_multiplier = 2.0  # Crypto more volatile
    elif symbol in ['CRUDE']:
        symbol_multiplier = 1.5  # Energy commodities more responsive
    
    for event in events:
        price_change = event["impact"] * base_price * 0.01 * symbol_multiplier
        event["price"] = base_price + price_change
    
    # Generate price movements
    for i, time in enumerate(times):
        closest_event = min(events, key=lambda x: abs((x["time"] - time).total_seconds()))
        distance = abs((closest_event["time"] - time).total_seconds()) / 3600
        
        # Adjust volatility based on symbol
        base_volatility = 0.15 if distance < 0.5 else 0.05
        if symbol in ['BTC']:
            base_volatility *= 3.0
        elif symbol in ['GOLD', 'SILVER']:
            base_volatility *= 0.5
        elif symbol in ['CRUDE']:
            base_volatility *= 2.0
        
        random_change = np.random.normal(0, base_volatility)
        event_influence = closest_event["impact"] * np.exp(-distance) * symbol_multiplier
        
        if i == 0:
            prices[i] = base_price
        else:
            change = (event_influence + random_change) * base_price * 0.001
            prices[i] = prices[i-1] + change
    
    df_intraday = pd.DataFrame({
        'Time': times,
        'Price': prices,
        'Aspect': [min(events, key=lambda x: abs((x["time"] - t).total_seconds()))["aspect"] for t in times]
    })
    
    # Create dynamic figure layout
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, height_ratios=[4, 1, 1, 1], width_ratios=[5, 2, 1])
    
    # Main price chart
    ax_main = fig.add_subplot(gs[0, :2])
    
    for i in range(1, len(df_intraday)):
        color = 'green' if df_intraday['Price'].iloc[i] > df_intraday['Price'].iloc[i-1] else 'red'
        ax_main.plot(df_intraday['Time'].iloc[i-1:i+1], 
                    df_intraday['Price'].iloc[i-1:i+1], 
                    color=color, linewidth=2.5)
    
    # Mark key events
    for event in events:
        color_map = {'bullish': 'green', 'bearish': 'red', 'neutral': 'blue'}
        ax_main.axvline(x=event['time'], color=color_map[event['type']], 
                       linestyle='--', alpha=0.7, linewidth=2)
        
        marker_color = color_map[event['type']]
        ax_main.scatter(event['time'], event['price'], color=marker_color, 
                       s=100, zorder=5, edgecolor='black', linewidth=1)
        
        # Dynamic annotation positioning
        y_offset = base_price * 0.02 if len(str(int(base_price))) >= 4 else base_price * 0.05
        y_pos = event['price'] + y_offset if event['price'] < base_price * 1.01 else event['price'] - y_offset
        
        ax_main.annotate(event['aspect'][:30] + '...' if len(event['aspect']) > 30 else event['aspect'], 
                        xy=(event['time'], event['price']),
                        xytext=(event['time'], y_pos),
                        arrowprops=dict(arrowstyle='->', color=marker_color, lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                        fontsize=8, ha='center')
    
    # Dynamic title and formatting
    currency_symbol = symbol_info['currency']
    ax_main.set_title(f'{symbol_info["name"]} ({symbol}) - {selected_date.strftime("%B %d, %Y")}\n'
                     f'Astrological Trading Analysis | Sector: {symbol_info["sector"]}', 
                     fontsize=16, pad=20)
    ax_main.set_xlabel(f'Time ({trading_hours["start_hour"]}:00 - {trading_hours["end_hour"]}:00)', fontsize=12)
    ax_main.set_ylabel(f'Price ({currency_symbol})', fontsize=12)
    
    # Dynamic time formatting based on session length
    if session_hours > 12:
        ax_main.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    else:
        ax_main.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    plt.setp(ax_main.get_xticklabels(), rotation=45, ha='right')
    
    # Closing price annotation
    close_price = df_intraday["Price"].iloc[-1]
    price_change = close_price - base_price
    price_change_pct = (price_change / base_price) * 100
    
    ax_main.annotate(f'Close: {currency_symbol}{close_price:.2f}\n'
                    f'Change: {price_change:+.2f} ({price_change_pct:+.2f}%)', 
                    xy=(df_intraday['Time'].iloc[-1], close_price),
                    xytext=(df_intraday['Time'].iloc[-1] - timedelta(hours=session_hours*0.2), 
                           close_price + base_price * 0.02),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                    fontsize=11, fontweight='bold',
                    bbox=dict(facecolor='yellow', alpha=0.7, edgecolor='none', pad=2))
    
    # Planetary wheel
    ax_wheel = fig.add_subplot(gs[0, 2])
    draw_planetary_wheel(ax_wheel, selected_date, size=0.4)
    
    # Volume chart (simulated with realistic patterns)
    ax_volume = fig.add_subplot(gs[1, :2])
    
    # Generate more realistic volume based on symbol type
    if symbol in ['BTC']:
        base_volume = np.random.randint(50000, 200000, size=len(times))
    elif symbol in ['GOLD', 'SILVER', 'CRUDE']:
        base_volume = np.random.randint(10000, 50000, size=len(times))
    elif symbol in ['DOWJONES']:
        base_volume = np.random.randint(100000, 500000, size=len(times))
    else:  # Indian stocks
        base_volume = np.random.randint(1000, 10000, size=len(times))
    
    colors_volume = ['green' if df_intraday['Price'].iloc[i] > df_intraday['Price'].iloc[i-1] 
                    else 'red' for i in range(1, len(df_intraday))]
    colors_volume.insert(0, 'green')
    
    ax_volume.bar(df_intraday['Time'], base_volume, color=colors_volume, alpha=0.7)
    ax_volume.set_title('Volume', fontsize=12)
    ax_volume.set_ylabel('Volume', fontsize=10)
    ax_volume.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax_volume.get_xticklabels(), rotation=45, ha='right')
    
    # Technical indicators (RSI simulation)
    ax_rsi = fig.add_subplot(gs[2, :2])
    rsi_values = 50 + np.random.normal(0, 15, len(times))  # Simulated RSI
    rsi_values = np.clip(rsi_values, 0, 100)
    
    ax_rsi.plot(df_intraday['Time'], rsi_values, color='purple', linewidth=2)
    ax_rsi.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
    ax_rsi.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
    ax_rsi.fill_between(df_intraday['Time'], 30, 70, alpha=0.1, color='gray')
    ax_rsi.set_title('RSI (14)', fontsize=12)
    ax_rsi.set_ylabel('RSI', fontsize=10)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.legend(loc='upper right', fontsize=8)
    
    # Aspect strength indicator
    ax_aspect = fig.add_subplot(gs[3, :2])
    aspect_times = [event['time'] for event in events]
    aspect_strengths = [abs(event['impact']) for event in events]
    aspect_colors = [{'bullish': 'green', 'bearish': 'red', 'neutral': 'blue'}[event['type']] for event in events]
    
    ax_aspect.scatter(aspect_times, aspect_strengths, color=aspect_colors, s=100, zorder=3)
    ax_aspect.plot(aspect_times, aspect_strengths, color='gray', alpha=0.5, linestyle='--')
    ax_aspect.set_title('Astrological Aspect Strength', fontsize=12)
    ax_aspect.set_ylabel('Strength', fontsize=10)
    ax_aspect.set_ylim(0, max(aspect_strengths) * 1.2 if aspect_strengths else 1.5)
    
    # Symbol info panel
    ax_info = fig.add_subplot(gs[1:, 2])
    ax_info.axis('off')
    
    info_text = f"""
SYMBOL INFO
-----------
Name: {symbol_info['name']}
Sector: {symbol_info['sector']}
Currency: {symbol_info['currency']}

TRADING HOURS
-------------
Start: {trading_hours['start_hour']:02d}:{trading_hours['start_minute']:02d}
End: {trading_hours['end_hour']:02d}:{trading_hours['end_minute']:02d}
Session: {session_hours:.1f} hours

PRICE DATA
----------
Open: {currency_symbol}{base_price:.2f}
Close: {currency_symbol}{close_price:.2f}
Change: {price_change:+.2f}
Change%: {price_change_pct:+.2f}%

High: {currency_symbol}{max(prices):.2f}
Low: {currency_symbol}{min(prices):.2f}
Range: {currency_symbol}{max(prices)-min(prices):.2f}
"""
    
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig

# --- ENHANCED MONTHLY CHART ---
def generate_monthly_chart(symbol, starting_price, selected_month, selected_year):
    """Generate enhanced monthly chart with dynamic layout"""
    symbol_info = get_symbol_info(symbol)
    
    start_date = datetime(selected_year, selected_month, 1)
    days_in_month = calendar.monthrange(selected_year, selected_month)[1]
    end_date = datetime(selected_year, selected_month, days_in_month)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    prices = np.zeros(len(dates))
    base_price = starting_price
    
    events = generate_astrological_events(start_date, 'monthly', symbol)
    
    # Adjust event impacts based on symbol type
    symbol_multiplier = 1.0
    if symbol in ['GOLD', 'SILVER']:
        symbol_multiplier = 0.8
    elif symbol in ['BTC']:
        symbol_multiplier = 2.5
    elif symbol in ['CRUDE']:
        symbol_multiplier = 1.8
    
    for event in events:
        price_change = event["impact"] * base_price * 0.01 * symbol_multiplier
        event["price"] = base_price + price_change
    
    # Generate price movements
    for i, date in enumerate(dates):
        closest_event = min(events, key=lambda x: abs((x["date"].date() - date.date()).days))
        distance = abs((closest_event["date"].date() - date.date()).days)
        
        base_volatility = 0.3 if distance < 2 else 0.1
        if symbol in ['BTC']:
            base_volatility *= 4.0
        elif symbol in ['GOLD', 'SILVER']:
            base_volatility *= 0.6
        elif symbol in ['CRUDE']:
            base_volatility *= 2.5
        
        random_change = np.random.normal(0, base_volatility)
        event_influence = closest_event["impact"] * np.exp(-distance/2) * symbol_multiplier
        
        if i == 0:
            prices[i] = base_price
        else:
            change = (event_influence + random_change) * base_price * 0.002
            prices[i] = prices[i-1] + change
    
    df_monthly = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Aspect': [min(events, key=lambda x: abs((x["date"].date() - d.date()).days))["aspect"] for d in dates]
    })
    
    # Create dynamic figure layout
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, height_ratios=[4, 1, 1, 1], width_ratios=[5, 2, 1])
    
    # Main price chart
    ax_main = fig.add_subplot(gs[0, :2])
    
    for i in range(1, len(df_monthly)):
        color = 'green' if df_monthly['Price'].iloc[i] > df_monthly['Price'].iloc[i-1] else 'red'
        ax_main.plot(df_monthly['Date'].iloc[i-1:i+1], 
                    df_monthly['Price'].iloc[i-1:i+1], 
                    color=color, linewidth=3)
    
    # Mark key events
    for event in events:
        color_map = {'bullish': 'green', 'bearish': 'red', 'neutral': 'blue'}
        ax_main.axvline(x=event['date'], color=color_map[event['type']], 
                       linestyle='--', alpha=0.7, linewidth=2)
        
        marker_color = color_map[event['type']]
        ax_main.scatter(event['date'], event['price'], color=marker_color, 
                       s=150, zorder=5, edgecolor='black', linewidth=1.5)
        
        y_offset = base_price * 0.03
        y_pos = event['price'] + y_offset if event['price'] < base_price * 1.02 else event['price'] - y_offset
        
        ax_main.annotate(event['aspect'][:25] + '...' if len(event['aspect']) > 25 else event['aspect'], 
                        xy=(event['date'], event['price']),
                        xytext=(event['date'], y_pos),
                        arrowprops=dict(arrowstyle='->', color=marker_color, lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                        fontsize=9, ha='center')
    
    # Dynamic formatting
    currency_symbol = symbol_info['currency']
    ax_main.set_title(f'{symbol_info["name"]} ({symbol}) - {start_date.strftime("%B %Y")}\n'
                     f'Monthly Astrological Analysis | Sector: {symbol_info["sector"]}', 
                     fontsize=16, pad=20)
    ax_main.set_xlabel('Date', fontsize=12)
    ax_main.set_ylabel(f'Price ({currency_symbol})', fontsize=12)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax_main.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, days_in_month//10)))
    plt.setp(ax_main.get_xticklabels(), rotation=45, ha='right')
    
    # Monthly close annotation
    close_price = df_monthly["Price"].iloc[-1]
    price_change = close_price - base_price
    price_change_pct = (price_change / base_price) * 100
    
    ax_main.annotate(f'Month Close: {currency_symbol}{close_price:.2f}\n'
                    f'Monthly Change: {price_change:+.2f} ({price_change_pct:+.2f}%)', 
                    xy=(df_monthly['Date'].iloc[-1], close_price),
                    xytext=(df_monthly['Date'].iloc[-1] - timedelta(days=days_in_month//4), 
                           close_price + base_price * 0.03),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                    fontsize=12, fontweight='bold',
                    bbox=dict(facecolor='yellow', alpha=0.7, edgecolor='none', pad=2))
    
    # Planetary positions for key dates
    ax_planets = fig.add_subplot(gs[0, 2])
    ax_planets.set_title('Key Planetary\nPositions', fontsize=10)
    key_dates = [
        start_date,
        start_date + timedelta(days=days_in_month//3),
        start_date + timedelta(days=2*days_in_month//3),
        end_date
    ]
    
    for i, date in enumerate(key_dates):
        ax_sub = fig.add_axes([0.70, 0.8-i*0.15, 0.12, 0.12])
        draw_planetary_wheel(ax_sub, date, size=0.4)
        ax_sub.set_title(f'{date.strftime("%b %d")}', fontsize=8)
    
    # Volume chart
    ax_volume = fig.add_subplot(gs[1, :2])
    
    if symbol in ['BTC']:
        volume = np.random.randint(500000, 2000000, size=len(dates))
    elif symbol in ['GOLD', 'SILVER', 'CRUDE']:
        volume = np.random.randint(100000, 500000, size=len(dates))
    elif symbol in ['DOWJONES']:
        volume = np.random.randint(1000000, 5000000, size=len(dates))
    else:
        volume = np.random.randint(10000, 100000, size=len(dates))
    
    colors_volume = ['green' if df_monthly['Price'].iloc[i] > df_monthly['Price'].iloc[i-1] 
                    else 'red' for i in range(1, len(df_monthly))]
    colors_volume.insert(0, 'green')
    
    ax_volume.bar(df_monthly['Date'], volume, color=colors_volume, alpha=0.7)
    ax_volume.set_title('Daily Volume', fontsize=12)
    ax_volume.set_ylabel('Volume', fontsize=10)
    ax_volume.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    # Moving averages
    ax_ma = fig.add_subplot(gs[2, :2])
    ma_5 = df_monthly['Price'].rolling(window=5, min_periods=1).mean()
    ma_20 = df_monthly['Price'].rolling(window=min(20, len(df_monthly)), min_periods=1).mean()
    
    ax_ma.plot(df_monthly['Date'], ma_5, color='blue', linewidth=2, label='MA5', alpha=0.7)
    ax_ma.plot(df_monthly['Date'], ma_20, color='red', linewidth=2, label='MA20', alpha=0.7)
    ax_ma.fill_between(df_monthly['Date'], ma_5, ma_20, alpha=0.1, 
                      color='green' if ma_5.iloc[-1] > ma_20.iloc[-1] else 'red')
    ax_ma.set_title('Moving Averages', fontsize=12)
    ax_ma.set_ylabel('Price', fontsize=10)
    ax_ma.legend(loc='upper left', fontsize=10)
    
    # Aspect calendar
    ax_calendar = fig.add_subplot(gs[3, :2])
    aspect_dates = [event['date'] for event in events]
    aspect_strengths = [abs(event['impact']) for event in events]
    aspect_colors = [{'bullish': 'green', 'bearish': 'red', 'neutral': 'blue'}[event['type']] for event in events]
    
    ax_calendar.scatter(aspect_dates, aspect_strengths, color=aspect_colors, s=200, zorder=3)
    ax_calendar.plot(aspect_dates, aspect_strengths, color='gray', alpha=0.5, linestyle='--')
    ax_calendar.set_title('Monthly Astrological Event Strength', fontsize=12)
    ax_calendar.set_ylabel('Impact Strength', fontsize=10)
    ax_calendar.set_ylim(0, max(aspect_strengths) * 1.2 if aspect_strengths else 2)
    
    # Monthly summary panel
    ax_summary = fig.add_subplot(gs[1:, 2])
    ax_summary.axis('off')
    
    monthly_high = max(prices)
    monthly_low = min(prices)
    monthly_range = monthly_high - monthly_low
    avg_price = np.mean(prices)
    
    summary_text = f"""
MONTHLY SUMMARY
--------------
Symbol: {symbol}
Sector: {symbol_info['sector']}
Month: {start_date.strftime('%B %Y')}

PRICE STATISTICS
---------------
Open: {currency_symbol}{base_price:.2f}
Close: {currency_symbol}{close_price:.2f}
Change: {price_change:+.2f}
Change%: {price_change_pct:+.2f}%

High: {currency_symbol}{monthly_high:.2f}
Low: {currency_symbol}{monthly_low:.2f}
Range: {currency_symbol}{monthly_range:.2f}
Average: {currency_symbol}{avg_price:.2f}

VOLATILITY
----------
Daily Avg: {np.std(np.diff(prices)):.2f}
Monthly Vol: {(monthly_range/avg_price)*100:.1f}%

TREND ANALYSIS
--------------
Bullish Days: {sum(1 for i in range(1, len(prices)) if prices[i] > prices[i-1])}
Bearish Days: {sum(1 for i in range(1, len(prices)) if prices[i] < prices[i-1])}
Neutral Days: {sum(1 for i in range(1, len(prices)) if prices[i] == prices[i-1])}
"""
    
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes, fontsize=8,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    return fig

# --- ANALYZE ASPECTS ---
def analyze_aspects():
    """Enhanced aspect analysis with dynamic content"""
    aspects_data = {
        'Aspect': [
            'Mercury Retrograde', 'Venus Opposition Saturn', 'Moon-Jupiter Trine', 
            'Full Moon', 'Jupiter Square Saturn', 'Mercury Direct',
            'Venus enters Libra', 'New Moon', 'Mars-Uranus Conjunction',
            'Sun-Pluto Sextile'
        ],
        'Market Impact': [
            'High Volatility', 'Bearish Pressure', 'Bullish Surge', 
            'Trend Reversal', 'Major Tension', 'Clarity Returns',
            'Financial Rally', 'Strong Bullish', 'Energy Surge',
            'Structural Change'
        ],
        'Typical Price Change': [
            '±2-3%', '-1.5-2%', '+1-2%', 
            '±1-1.5%', '-2-3%', '+0.5-1%',
            '+0.5-1%', '+1-2%', '+2-4%',
            '±1-2%'
        ],
        'Sector Focus': [
            'All Sectors', 'Banking/Realty', 'Broad Market', 
            'Technology', 'Financials', 'Technology',
            'Banking/Finance', 'Broad Market', 'Energy/Commodities',
            'Infrastructure/PSUs'
        ],
        'Best Symbols': [
            'Gold, BTC', 'Gold, Silver', 'FMCG, Pharma', 
            'Tech Stocks', 'Defensive', 'Tech, Crypto',
            'Banking', 'Growth Stocks', 'Energy, Crude',
            'PSU, Infrastructure'
        ]
    }
    
    df_aspects = pd.DataFrame(aspects_data)
    
    # Create enhanced visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Price change impact chart
    price_changes = []
    for change in df_aspects['Typical Price Change']:
        clean_change = change.replace('%', '').replace('±', '')
        if '-' in clean_change and not clean_change.startswith('-'):
            num_str = clean_change.split('-')[1]  # Take higher value for impact
        else:
            num_str = clean_change.replace('+', '')
        
        try:
            num = float(num_str)
        except:
            num = 1.0
        price_changes.append(num)
    
    colors = ['red' if 'Bearish' in impact or 'Tension' in impact or 'Volatility' in impact 
              else 'orange' if 'Reversal' in impact or 'Change' in impact
              else 'green' for impact in df_aspects['Market Impact']]
    
    bars1 = ax1.bar(range(len(df_aspects)), price_changes, color=colors, alpha=0.7)
    ax1.set_title('Astrological Aspect Impact on Price Changes', fontsize=14)
    ax1.set_ylabel('Maximum Price Change (%)', fontsize=12)
    ax1.set_xticks(range(len(df_aspects)))
    ax1.set_xticklabels(df_aspects['Aspect'], rotation=45, ha='right')
    
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.annotate(f'{height}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    # Sector distribution pie chart
    sector_counts = {}
    for sectors in df_aspects['Sector Focus']:
        for sector in sectors.split('/'):
            sector = sector.strip()
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
    
    ax2.pie(sector_counts.values(), labels=sector_counts.keys(), autopct='%1.1f%%', startangle=90)
    ax2.set_title('Most Affected Sectors by Astrological Aspects', fontsize=14)
    
    # Market impact distribution
    impact_counts = {}
    for impact in df_aspects['Market Impact']:
        impact_type = 'Bullish' if any(word in impact for word in ['Bullish', 'Rally', 'Surge', 'Returns']) else \
                     'Bearish' if any(word in impact for word in ['Bearish', 'Pressure', 'Tension']) else \
                     'Neutral'
        impact_counts[impact_type] = impact_counts.get(impact_type, 0) + 1
    
    colors_impact = ['green', 'red', 'gray']
    ax3.bar(impact_counts.keys(), impact_counts.values(), color=colors_impact, alpha=0.7)
    ax3.set_title('Distribution of Market Impact Types', fontsize=14)
    ax3.set_ylabel('Number of Aspects', fontsize=12)
    
    # Best performing symbols chart
    symbol_mentions = {}
    for symbols in df_aspects['Best Symbols']:
        for symbol in symbols.split(', '):
            symbol = symbol.strip()
            symbol_mentions[symbol] = symbol_mentions.get(symbol, 0) + 1
    
    sorted_symbols = sorted(symbol_mentions.items(), key=lambda x: x[1], reverse=True)
    symbols, counts = zip(*sorted_symbols) if sorted_symbols else ([], [])
    
    ax4.barh(symbols, counts, color='gold', alpha=0.7)
    ax4.set_title('Most Favorable Symbols Across Aspects', fontsize=14)
    ax4.set_xlabel('Favorable Mentions', fontsize=12)
    
    return fig, df_aspects

# --- STREAMLIT APP ---
def main():
    # Page configuration for better responsive design
    st.set_page_config(
        page_title="🌟 Astrological Trading Dashboard",
        page_icon="🌟",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .symbol-input {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #e8f4fd;
        border-left: 5px solid #1f77b4;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🌟 Astrological Trading Dashboard</h1>
        <p>Advanced Financial Analysis through Planetary Movements & Cosmic Patterns</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for inputs with enhanced design
    with st.sidebar:
        st.markdown("### 📊 Dashboard Controls")
        
        # Dashboard section selection with better descriptions
        dashboard_section = st.selectbox(
            '🎯 Choose Analysis Section:',
            [
                'Summary Table - Market Overview',
                'Stock Filter - Sector Analysis', 
                'Aspect Analysis - Deep Insights',
                'Intraday Chart - Live Patterns',
                'Monthly Chart - Trend Analysis'
            ]
        )
        
        # Extract the main section name
        section_name = dashboard_section.split(' - ')[0]
        
        st.markdown("---")
        
        # Symbol selection with enhanced interface
        if section_name in ['Intraday Chart', 'Monthly Chart']:
            st.markdown("### 📈 Symbol Configuration")
            
            # Popular symbols with categories
            symbol_categories = {
                'Indian Indices': ['NIFTY', 'BANKNIFTY'],
                'Indian Stocks': ['TCS', 'ICICIBANK', 'MARUTI', 'DLF', 'NESTLEIND', 'RELIANCE', 'SBI', 'SUNPHARMA', 'DRREDDY'],
                'Global Markets': ['DOWJONES'],
                'Commodities': ['GOLD', 'SILVER', 'CRUDE'],
                'Cryptocurrency': ['BTC']
            }
            
            selected_category = st.selectbox('📂 Select Category:', list(symbol_categories.keys()))
            
            if selected_category:
                symbol_options = symbol_categories[selected_category]
                selected_symbol = st.selectbox('🎯 Choose Symbol:', symbol_options)
                
                # Custom symbol input
                custom_symbol = st.text_input('✏️ Or enter custom symbol:', max_chars=10)
                symbol = custom_symbol.upper() if custom_symbol else selected_symbol
                
                # Get symbol info for dynamic defaults
                symbol_info = get_symbol_info(symbol)
                trading_hours = get_trading_hours(symbol)
                
                # Display symbol information
                st.markdown(f"""
                <div class="info-box">
                    <strong>📊 {symbol_info['name']}</strong><br>
                    <small>Sector: {symbol_info['sector']}</small><br>
                    <small>Currency: {symbol_info['currency']}</small><br>
                    <small>Trading: {trading_hours['start_hour']:02d}:{trading_hours['start_minute']:02d} - {trading_hours['end_hour']:02d}:{trading_hours['end_minute']:02d}</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Price input with dynamic default
                starting_price = st.number_input(
                    f'💰 Starting Price ({symbol_info["currency"]}):',
                    min_value=0.01,
                    value=symbol_info['default_price'],
                    step=1.0 if symbol_info['default_price'] > 100 else 0.01,
                    format="%.2f"
                )
                
                # Date/time selection based on chart type
                if section_name == 'Intraday Chart':
                    selected_date = st.date_input(
                        '📅 Select Trading Date:',
                        value=datetime(2025, 8, 5).date(),
                        min_value=datetime(2020, 1, 1).date(),
                        max_value=datetime(2030, 12, 31).date()
                    )
                elif section_name == 'Monthly Chart':
                    col1, col2 = st.columns(2)
                    with col1:
                        selected_month = st.selectbox(
                            '📅 Month:',
                            range(1, 13),
                            format_func=lambda x: calendar.month_name[x],
                            index=7  # August
                        )
                    with col2:
                        selected_year = st.selectbox(
                            '📅 Year:',
                            range(2020, 2031),
                            index=5  # 2025
                        )
        
        # Trading insights
        st.markdown("---")
        st.markdown("### 🔮 Quick Insights")
        
        # Generate today's aspects for sidebar display
        aspects = generate_todays_aspects()
        bullish_count = sum(1 for aspect in aspects if aspect['type'] == 'bullish')
        bearish_count = sum(1 for aspect in aspects if aspect['type'] == 'bearish')
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🟢 Bullish Aspects", bullish_count)
        with col2:
            st.metric("🔴 Bearish Aspects", bearish_count)
        
        # Market sentiment
        if bullish_count > bearish_count:
            sentiment = "🟢 Bullish"
            sentiment_color = "green"
        elif bearish_count > bullish_count:
            sentiment = "🔴 Bearish"
            sentiment_color = "red"
        else:
            sentiment = "🟡 Neutral"
            sentiment_color = "orange"
        
        st.markdown(f"**Overall Sentiment:** <span style='color:{sentiment_color}'>{sentiment}</span>", 
                   unsafe_allow_html=True)

    # Main content area
    aspects = generate_todays_aspects()
    
    if section_name == 'Summary Table':
        st.header('📋 Market Summary & Astrological Overview')
        
        # Create columns for better layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader('📊 Today\'s Astrological Aspects Impact')
            summary_df = create_summary_table(aspects)
            
            # Style the dataframe
            styled_df = summary_df.style.apply(
                lambda x: ['background-color: #d4edda' if 'Bullish' in str(val) or '+' in str(val) 
                          else 'background-color: #f8d7da' if 'Bearish' in str(val) or 'Downside' in str(val)
                          else '' for val in x], axis=0
            )
            
            st.dataframe(styled_df, use_container_width=True)
        
        with col2:
            st.subheader('🎯 Key Metrics')
            
            # Calculate impact scores
            total_impact = sum(abs(aspect['impact']) for aspect in aspects)
            avg_impact = total_impact / len(aspects) if aspects else 0
            
            st.metric("Total Cosmic Energy", f"{total_impact:.1f}")
            st.metric("Average Impact", f"{avg_impact:.2f}")
            st.metric("Active Aspects", len(aspects))
            
            # Risk assessment
            high_risk_aspects = sum(1 for aspect in aspects if abs(aspect['impact']) > 0.7)
            risk_level = "High" if high_risk_aspects >= 3 else "Medium" if high_risk_aspects >= 1 else "Low"
            risk_color = "red" if risk_level == "High" else "orange" if risk_level == "Medium" else "green"
            
            st.markdown(f"**Risk Level:** <span style='color:{risk_color}'>{risk_level}</span>", 
                       unsafe_allow_html=True)
        
        # Detailed insights
        st.subheader('🔮 Detailed Market Insights')
        
        # Create tabs for different insights
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Strategy", "📈 Opportunities", "⚠️ Risks", "🌟 Cosmic Events"])
        
        with tab1:
            st.markdown("""
            **🎯 Recommended Trading Strategy:**
            
            **🟢 Bullish Opportunities:**
            - **Energy Sector**: Mars-Uranus conjunction favors Reliance, Crude Oil
            - **Precious Metals**: Multiple aspects support Gold and Silver
            - **FMCG & Pharma**: Moon-Neptune trine provides defensive strength
            - **PSU Stocks**: Sun-Pluto sextile indicates structural positives
            
            **🔴 Bearish Risks:**
            - **Banking Sector**: Mercury-Jupiter square creates volatility
            - **Automotive & Realty**: Venus-Saturn opposition brings pressure
            - **Technology**: Mixed signals, trade with caution
            
            **⚡ High-Impact Trades:**
            - Consider Gold positions during Venus-Saturn opposition
            - Energy stocks may see sharp moves (Mars-Uranus)
            - BTC could be volatile but trending up on global aspects
            """)
        
        with tab2:
            st.markdown("""
            **📈 Sector-wise Opportunities:**
            
            **🥇 Top Picks:**
            1. **Gold/Silver**: Multiple supportive aspects across all planetary configurations
            2. **Energy Commodities**: Mars-Uranus conjunction + global supply dynamics
            3. **Pharmaceutical**: Moon-Neptune trine supports defensive healthcare
            4. **PSU Banking**: Sun-Pluto sextile for structural transformation
            
            **🎯 Specific Symbols:**
            - **GOLD**: $2,050+ target on safe-haven demand
            - **CRUDE**: Energy transition + Mars-Uranus = volatility opportunities
            - **BTC**: Crypto favorable on Uranus-Pluto aspects
            - **SBI**: PSU transformation theme
            """)
        
        with tab3:
            st.markdown("""
            **⚠️ Risk Management:**
            
            **🔴 High-Risk Sectors:**
            - **Private Banking**: ICICI Bank under Mercury-Jupiter square pressure
            - **Automotive**: Maruti facing Venus-Saturn headwinds
            - **Real Estate**: DLF vulnerable to credit tightening aspects
            
            **📊 Risk Mitigation:**
            - Reduce position sizes during Mercury-Jupiter square (high volatility)
            - Use stop-losses 2-3% below support for Venus-Saturn affected stocks
            - Avoid leveraged positions in Midcap segment (Mars-Uranus volatility)
            
            **⏰ Timing Risks:**
            - Morning session volatility expected (Mercury aspects)
            - Post-lunch session may see pressure (Saturn influence)
            """)
        
        with tab4:
            st.markdown("""
            **🌟 Today's Cosmic Events Schedule:**
            
            **🌅 Pre-Market (Before 9:15 AM):**
            - Mercury-Jupiter square builds tension
            - Global markets influence domestic opening
            
            **🌄 Morning Session (9:15-12:00):**
            - Initial volatility from Mercury aspects
            - Energy stocks may show strength
            
            **🌞 Afternoon Session (12:00-15:30):**
            - Venus-Saturn opposition peaks
            - Defensive sectors gain relative strength
            - Banking sector under pressure
            
            **🌆 Post-Market:**
            - Global commodity movements (Gold, Crude)
            - Crypto markets reaction to day's developments
            
            **📊 Weekly Outlook:**
            - Aspects intensify mid-week
            - Weekend planetary shifts to monitor
            """)
    
    elif section_name == 'Stock Filter':
        st.header('🔍 Advanced Stock Filtering & Sector Analysis')
        
        # Create main layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader('🌟 Today\'s Astrological Configuration')
            
            # Display aspects in a nice format
            aspects_data = []
            for aspect in aspects:
                aspects_data.append({
                    'Planets': aspect['planets'],
                    'Aspect': aspect['aspect_type'],
                    'Impact': f"{aspect['impact']:+.1f}",
                    'Sentiment': aspect['type'].title(),
                    'Strength': '🔥' * min(3, int(abs(aspect['impact']) * 3))
                })
            
            aspects_df = pd.DataFrame(aspects_data)
            
            # Color code the dataframe
            def color_sentiment(val):
                if 'Bullish' in str(val):
                    return 'background-color: #d4edda; color: #155724'
                elif 'Bearish' in str(val):
                    return 'background-color: #f8d7da; color: #721c24'
                return ''
            
            styled_aspects = aspects_df.style.applymap(color_sentiment, subset=['Sentiment'])
            st.dataframe(styled_aspects, use_container_width=True)
        
        with col2:
            st.subheader('📊 Aspect Statistics')
            
            # Create a simple pie chart for aspect types
            aspect_types = {}
            for aspect in aspects:
                aspect_types[aspect['type']] = aspect_types.get(aspect['type'], 0) + 1
            
            if aspect_types:
                fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
                colors = ['green' if k == 'bullish' else 'red' if k == 'bearish' else 'gray' 
                         for k in aspect_types.keys()]
                wedges, texts, autotexts = ax_pie.pie(aspect_types.values(), 
                                                     labels=[k.title() for k in aspect_types.keys()], 
                                                     colors=colors, autopct='%1.0f%%', startangle=90)
                ax_pie.set_title('Today\'s Aspect Distribution')
                st.pyplot(fig_pie)
        
        # Sector analysis
        st.subheader('📈 Sector Impact Analysis')
        
        filtered_stocks = filter_stocks_by_aspects(aspects, STOCK_DATABASE)
        
        # Create sector impact visualization
        sector_impacts_df = pd.DataFrame({
            'Sector': list(filtered_stocks['sector_impacts'].keys()),
            'Impact Score': list(filtered_stocks['sector_impacts'].values()),
            'Recommendation': ['Strong Buy' if x > 0.5 else 'Buy' if x > 0 else 'Hold' if x == 0 
                             else 'Sell' if x > -0.5 else 'Strong Sell' 
                             for x in filtered_stocks['sector_impacts'].values()]
        })
        sector_impacts_df = sector_impacts_df.sort_values('Impact Score', ascending=False)
        
        # Enhanced bar chart
        fig_sectors, ax_sectors = plt.subplots(figsize=(14, 8))
        colors = ['darkgreen' if x > 0.5 else 'green' if x > 0 else 'gray' if x == 0 
                 else 'red' if x > -0.5 else 'darkred' 
                 for x in sector_impacts_df['Impact Score']]
        
        bars = ax_sectors.bar(sector_impacts_df['Sector'], sector_impacts_df['Impact Score'], 
                             color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax_sectors.set_title('Sector Impact Scores - Astrological Analysis', fontsize=16, pad=20)
        ax_sectors.set_ylabel('Impact Score', fontsize=12)
        ax_sectors.set_xlabel('Sector', fontsize=12)
        ax_sectors.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax_sectors.grid(True, alpha=0.3)
        
        # Add value labels and recommendations
        for i, (bar, rec) in enumerate(zip(bars, sector_impacts_df['Recommendation'])):
            height = bar.get_height()
            ax_sectors.annotate(f'{height:.2f}\n{rec}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 5 if height >= 0 else -25),
                               textcoords="offset points",
                               ha='center', va='bottom' if height >= 0 else 'top',
                               fontsize=9, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig_sectors)
        
        # Stock recommendations in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader('🟢 Bullish Stocks')
            if not filtered_stocks['bullish'].empty:
                bullish_df = filtered_stocks['bullish'][['Symbol', 'Sector', 'Impact Score']].copy()
                bullish_df['Action'] = bullish_df['Impact Score'].apply(
                    lambda x: 'Strong Buy' if x > 0.5 else 'Buy'
                )
                
                for _, row in bullish_df.iterrows():
                    with st.container():
                        st.markdown(f"""
                        **{row['Symbol']}** ({row['Sector']})  
                        Score: {row['Impact Score']:.2f} | **{row['Action']}**
                        """)
                        st.progress(min(1.0, row['Impact Score']))
            else:
                st.info("No strong bullish signals today")
        
        with col2:
            st.subheader('🔴 Bearish Stocks')
            if not filtered_stocks['bearish'].empty:
                bearish_df = filtered_stocks['bearish'][['Symbol', 'Sector', 'Impact Score']].copy()
                bearish_df['Action'] = bearish_df['Impact Score'].apply(
                    lambda x: 'Strong Sell' if x > 0.5 else 'Sell'
                )
                
                for _, row in bearish_df.iterrows():
                    with st.container():
                        st.markdown(f"""
                        **{row['Symbol']}** ({row['Sector']})  
                        Risk Score: {row['Impact Score']:.2f} | **{row['Action']}**
                        """)
                        st.progress(min(1.0, row['Impact Score']))
            else:
                st.info("No strong bearish signals today")
        
        with col3:
            st.subheader('⚪ Neutral Stocks')
            if not filtered_stocks['neutral'].empty:
                neutral_df = filtered_stocks['neutral'][['Symbol', 'Sector']].head(5)
                
                for _, row in neutral_df.iterrows():
                    st.markdown(f"**{row['Symbol']}** ({row['Sector']}) - Hold")
            else:
                st.info("All stocks showing directional bias")
    
    elif section_name == 'Aspect Analysis':
        st.header('📋 Deep Astrological Aspect Analysis')
        
        # Generate enhanced analysis
        fig, df_aspects = analyze_aspects()
        st.pyplot(fig)
        
        # Display detailed aspect table
        st.subheader('📊 Detailed Aspect Reference Table')
        
        # Add more columns for better analysis
        df_enhanced = df_aspects.copy()
        df_enhanced['Trading Action'] = df_enhanced.apply(
            lambda row: 'Hedge/Reduce' if 'Bearish' in row['Market Impact'] or 'Tension' in row['Market Impact']
            else 'Accumulate' if 'Bullish' in row['Market Impact'] or 'Rally' in row['Market Impact']
            else 'Monitor', axis=1
        )
        
        df_enhanced['Risk Level'] = df_enhanced['Typical Price Change'].apply(
            lambda x: 'High' if any(num in x for num in ['3', '4']) 
            else 'Medium' if '2' in x else 'Low'
        )
        
        # Style the enhanced dataframe
        def highlight_risk(val):
            if val == 'High':
                return 'background-color: #ffcccc'
            elif val == 'Medium':
                return 'background-color: #ffffcc'
            elif val == 'Low':
                return 'background-color: #ccffcc'
            return ''
        
        def highlight_action(val):
            if val == 'Accumulate':
                return 'background-color: #d4edda; color: #155724'
            elif val == 'Hedge/Reduce':
                return 'background-color: #f8d7da; color: #721c24'
            elif val == 'Monitor':
                return 'background-color: #fff3cd; color: #856404'
            return ''
        
        styled_enhanced = df_enhanced.style.applymap(highlight_risk, subset=['Risk Level']).applymap(highlight_action, subset=['Trading Action'])
        st.dataframe(styled_enhanced, use_container_width=True)
        
        # Aspect interpretation guide
        st.subheader('🔭 Astrological Aspect Interpretation Guide')
        
        tab1, tab2, tab3 = st.tabs(["🌟 Aspect Types", "🪐 Planetary Influences", "📈 Trading Applications"])
        
        with tab1:
            st.markdown("""
            ### Understanding Astrological Aspects
            
            **🔄 Conjunction (0°)**: 
            - *Market Effect*: Powerful combining of energies, can create sharp moves
            - *Trading*: Expect significant price action, potential breakouts
            - *Example*: Mars-Uranus conjunction = explosive energy moves
            
            **⚔️ Square (90°)**: 
            - *Market Effect*: Tension, conflict, volatility
            - *Trading*: Increased intraday swings, good for scalping
            - *Example*: Mercury-Jupiter square = communication/policy confusion
            
            **🎯 Trine (120°)**: 
            - *Market Effect*: Harmonious, easy flow of energy
            - *Trading*: Trending moves, good for position trading
            - *Example*: Moon-Neptune trine = emotional/intuitive support
            
            **⚖️ Opposition (180°)**: 
            - *Market Effect*: Polarization, requires balance
            - *Trading*: Range-bound action, reversals possible
            - *Example*: Venus-Saturn opposition = value vs. restriction
            
            **🤝 Sextile (60°)**: 
            - *Market Effect*: Opportunity aspects, mild positive
            - *Trading*: Gentle trends, good for swing trades
            - *Example*: Sun-Pluto sextile = gradual transformation
            """)
        
        with tab2:
            st.markdown("""
            ### Planetary Market Influences
            
            **☀️ Sun**: Leadership, government policy, large-cap stocks, gold
            **🌙 Moon**: Public sentiment, emotions, consumer sectors, silver
            **☿️ Mercury**: Communication, technology, volatility, news-driven moves
            **♀️ Venus**: Finance, banking, luxury goods, relationships, copper
            **♂️ Mars**: Energy, metals, defense, aggressive moves, oil
            **♃ Jupiter**: Growth, expansion, optimism, financial sector
            **♄ Saturn**: Restriction, discipline, structure, defensive sectors
            **♅ Uranus**: Innovation, technology, sudden changes, crypto
            **♆ Neptune**: Illusion, oil, pharma, confusion, speculation
            **♇ Pluto**: Transformation, power, mining, major shifts
            
            ### Sector-Planet Correlations
            - **Technology**: Mercury, Uranus
            - **Banking**: Jupiter, Venus, Saturn  
            - **Energy**: Mars, Sun, Pluto
            - **Healthcare**: Neptune, Moon
            - **Precious Metals**: Venus, Jupiter, Sun
            - **Cryptocurrency**: Uranus, Pluto
            """)
        
        with tab3:
            st.markdown("""
            ### Practical Trading Applications
            
            **📊 Intraday Trading:**
            - Use Moon aspects for sentiment shifts (2-4 hour cycles)
            - Mercury aspects for news/volatility spikes
            - Mars aspects for energy sector breakouts
            
            **📈 Swing Trading:**
            - Venus aspects for financial sector trends (3-7 days)
            - Jupiter aspects for broad market optimism
            - Saturn aspects for defensive positioning
            
            **📉 Position Trading:**
            - Outer planet aspects (Uranus, Neptune, Pluto) for long-term themes
            - Eclipse patterns for major sector rotations
            - Retrograde periods for trend reversals
            
            **⚠️ Risk Management:**
            - Increase cash during multiple challenging aspects
            - Reduce position size during Mercury retrograde
            - Use tighter stops during Mars-Saturn squares
            
            **🎯 Sector Rotation:**
            - Follow Jupiter through zodiac signs for sector leadership
            - Track Saturn aspects for value opportunities
            - Monitor Uranus for innovation themes
            """)
    
    elif section_name == 'Intraday Chart':
        st.header(f'📈 {symbol} - Intraday Astrological Analysis')
        
        # Display symbol information prominently
        symbol_info = get_symbol_info(symbol)
        trading_hours = get_trading_hours(symbol)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Symbol", symbol)
        with col2:
            st.metric("Sector", symbol_info['sector'])
        with col3:
            st.metric("Currency", symbol_info['currency'])
        with col4:
            session_length = trading_hours['end_hour'] - trading_hours['start_hour'] + \
                           (trading_hours['end_minute'] - trading_hours['start_minute']) / 60
            st.metric("Session Hours", f"{session_length:.1f}h")
        
        # Generate and display chart
        with st.spinner(f'Generating astrological analysis for {symbol}...'):
            fig = generate_intraday_chart(symbol, starting_price, selected_date)
            st.pyplot(fig)
        
        # Trading insights based on symbol
        st.subheader(f'🎯 {symbol} Trading Insights')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Technical Outlook")
            
            # Generate insights based on symbol type
            if symbol in ['GOLD', 'SILVER']:
                st.markdown("""
                **Precious Metals Analysis:**
                - Multiple planetary aspects favor safe-haven demand
                - Venus-Saturn opposition creates financial stress → Gold strength  
                - Moon-Neptune trine supports intuitive precious metal buying
                - Best trading windows: During global uncertainty aspects
                
                **Key Levels:**
                - Watch for breakouts during Mars-Uranus conjunction
                - Support likely during Moon aspects
                - Resistance at previous highs during Saturn aspects
                """)
            
            elif symbol in ['BTC']:
                st.markdown("""
                **Cryptocurrency Analysis:**
                - Uranus aspects strongly favor crypto volatility
                - Mars-Uranus conjunction = explosive price moves
                - Traditional financial stress (Venus-Saturn) → Crypto rotation
                - High volatility expected - use proper risk management
                
                **Trading Strategy:**
                - Momentum plays during Uranus aspects
                - Contrarian plays during Saturn oppositions
                - Volume spikes likely at aspect peaks
                """)
            
            elif symbol in ['CRUDE']:
                st.markdown("""
                **Energy Commodity Analysis:**
                - Mars-Uranus conjunction directly impacts energy sector
                - Global supply disruption themes (Pluto aspects)
                - Geopolitical tensions favor energy prices
                - Weather and seasonal patterns amplified by aspects
                
                **Supply-Demand Factors:**
                - Production disruptions during Mars aspects
                - Demand surges during economic aspects
                - Storage plays during Saturn aspects
                """)
            
            elif symbol in ['DOWJONES']:
                st.markdown("""
                **US Index Analysis:**
                - Jupiter aspects favor broad market optimism
                - Saturn aspects create rotation into defensive sectors
                - Mercury aspects increase intraday volatility
                - Fed policy sensitivity during Venus-Saturn opposition
                
                **Sector Rotation:**
                - Technology during Mercury aspects
                - Energy during Mars aspects  
                - Financials during Jupiter aspects
                """)
            
            else:  # Indian stocks
                st.markdown(f"""
                **{symbol_info['sector']} Sector Analysis:**
                - Domestic market influenced by global planetary patterns
                - FII/DII flows affected by Venus-Saturn aspects
                - Sector rotation based on planetary emphasis
                - Currency impacts during outer planet aspects
                
                **Indian Market Specifics:**
                - Opening gap influenced by global overnight aspects
                - Lunch hour volatility during Moon aspects
                - Closing session strength during Jupiter aspects
                """)
        
        with col2:
            st.markdown("#### ⏰ Timing Analysis")
            
            # Generate time-specific insights based on trading hours
            if trading_hours['end_hour'] > 16:  # Extended hours
                st.markdown("""
                **Extended Session Analysis:**
                
                **🌅 Asian Session (5:00-8:00):**
                - Pre-market positioning based on overnight aspects
                - Lower volumes, higher impact from aspects
                - Key economic data releases amplified
                
                **🌍 European Session (8:00-16:00):**
                - Peak liquidity and aspect impacts
                - Central bank policy influences
                - Cross-asset correlations strongest
                
                **🌎 US Session (16:00-20:00):**
                - Maximum volatility potential
                - Aspect peaks create significant moves
                - News flow interaction with cosmic patterns
                
                **🌙 After Hours (20:00-23:55):**
                - Reduced liquidity amplifies aspect effects
                - Position adjustments for next day
                - Asian preview impact
                """)
            else:  # Indian market hours
                st.markdown("""
                **Indian Session Analysis:**
                
                **🌅 Opening (9:15-10:30):**
                - Gap opening based on global aspects
                - High volatility, aspect impacts magnified
                - Initial trend direction setting
                
                **🌞 Mid-Morning (10:30-12:00):**
                - Institutional activity peaks
                - Aspect-driven sector rotation
                - News flow integration
                
                **🍽️ Lunch Hour (12:00-13:00):**
                - Reduced activity, Moon aspects dominate
                - Range-bound unless strong aspects active
                - Position consolidation period
                
                **🌆 Closing (13:00-15:30):**
                - Final institutional positioning
                - Aspect resolution for day
                - Next-day setup formation
                """)
            
            # Risk management
            st.markdown("#### ⚠️ Risk Management")
            st.markdown(f"""
            **Position Sizing:**
            - Standard position: 1-2% of capital
            - High aspect days: Reduce to 0.5-1%
            - Strong confluence: Increase to 2-3%
            
            **Stop Loss Levels:**
            - Tight stops during Mercury aspects: 1-2%
            - Normal stops during stable aspects: 2-3%
            - Wide stops during Mars aspects: 3-5%
            
            **Profit Targets:**
            - Quick scalps: 0.5-1% (15-30 minutes)
            - Swing trades: 2-5% (2-4 hours)
            - Position trades: 5-10% (1-3 days)
            
            **Volatility Adjustments:**
            - {symbol}: Expected daily range ±{2.5 if symbol in ['BTC'] else 1.5 if symbol in ['CRUDE'] else 1.0 if symbol in ['GOLD', 'SILVER'] else 0.8}%
            - Adjust position size inversely to volatility
            - Use options for high-volatility periods
            """)
    
    elif section_name == 'Monthly Chart':
        st.header(f'📊 {symbol} - Monthly Astrological Trend Analysis')
        
        # Display symbol information
        symbol_info = get_symbol_info(symbol)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Symbol", symbol)
        with col2:
            st.metric("Analysis Period", f"{calendar.month_name[selected_month]} {selected_year}")
        with col3:
            st.metric("Sector Focus", symbol_info['sector'])
        with col4:
            st.metric("Currency", symbol_info['currency'])
        
        # Generate and display chart
        with st.spinner(f'Generating monthly analysis for {symbol}...'):
            fig = generate_monthly_chart(symbol, starting_price, selected_month, selected_year)
            st.pyplot(fig)
        
        # Monthly analysis insights
        st.subheader(f'📈 {calendar.month_name[selected_month]} {selected_year} - Strategic Analysis')
        
        # Create tabs for different analysis aspects
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Monthly Outlook", "📊 Technical Analysis", "🌙 Lunar Cycles", "💼 Portfolio Strategy"])
        
        with tab1:
            month_name = calendar.month_name[selected_month]
            
            if symbol in ['GOLD', 'SILVER']:
                st.markdown(f"""
                ### {symbol} - {month_name} {selected_year} Precious Metals Outlook
                
                **🌟 Astrological Themes:**
                - **Venus-Jupiter aspects**: Strong precious metals demand from financial uncertainty
                - **Saturn transits**: Safe-haven buying during economic restrictions
                - **Moon phases**: Emotional buying patterns aligned with lunar cycles
                - **Mercury retrograde periods**: Technical analysis less reliable, fundamentals dominate
                
                **📈 Price Drivers:**
                - Central bank policy uncertainty (Saturn aspects)
                - Currency devaluation themes (Pluto aspects)
                - Geopolitical tensions (Mars aspects)
                - Inflation hedging demand (Jupiter-Saturn aspects)
                
                **🎯 Trading Strategy:**
                - **Accumulate** during New Moon phases (stronger buying interest)
                - **Profit-take** during Full Moon phases (emotional peaks)
                - **Hold through** Mercury retrograde (avoid technical trading)
                - **Scale in** during Saturn aspects (structural support)
                
                **📊 Target Levels:**
                - **Monthly High**: Expect during Jupiter-Venus trine periods
                - **Monthly Low**: Likely during Mars-Saturn square periods
                - **Breakout Potential**: Mars-Uranus conjunction periods
                - **Support Zones**: Previous month's Jupiter aspect levels
                """)
            
            elif symbol in ['BTC']:
                st.markdown(f"""
                ### {symbol} - {month_name} {selected_year} Cryptocurrency Outlook
                
                **⚡ Astrological Themes:**
                - **Uranus-Pluto aspects**: Revolutionary technology adoption waves
                - **Mercury-Uranus aspects**: Network upgrades and technical developments
                - **Mars-Uranus conjunctions**: Explosive price movements and FOMO
                - **Saturn aspects**: Regulatory clarity or restrictions
                
                **🚀 Volatility Drivers:**
                - Institutional adoption news (Jupiter aspects)
                - Regulatory developments (Saturn aspects)
                - Technical network changes (Mercury-Uranus)
                - Market manipulation concerns (Neptune aspects)
                
                **⚠️ Risk Factors:**
                - **High volatility** during Mars-Uranus aspects (±10-20% daily swings)
                - **Regulatory risks** during Saturn-Pluto aspects
                - **Technical failures** during Mercury retrograde
                - **Market manipulation** during Neptune-Mercury aspects
                
                **💡 Strategic Approach:**
                - **DCA strategy** during volatile periods
                - **Momentum trading** during Uranus aspects
                - **Risk-off** during Saturn hard aspects
                - **HODL mentality** during Jupiter-Pluto trines
                """)
            
            elif symbol in ['CRUDE']:
                st.markdown(f"""
                ### {symbol} - {month_name} {selected_year} Energy Commodity Outlook
                
                **🛢️ Astrological Themes:**
                - **Mars-Pluto aspects**: Geopolitical tensions affecting supply
                - **Jupiter-Saturn cycles**: Economic growth vs. restriction cycles
                - **Uranus aspects**: Renewable energy transition impacts
                - **Moon phases**: Seasonal demand patterns amplified
                
                **⚡ Supply-Demand Dynamics:**
                - Production disruptions (Mars-Saturn squares)
                - Economic growth spurts (Jupiter aspects)
                - Weather pattern extremes (Uranus-Neptune aspects)
                - Strategic reserve changes (Pluto aspects)
                
                **🌍 Geopolitical Factors:**
                - **OPEC decisions** aligned with Saturn aspects
                - **Pipeline disruptions** during Mars-Uranus periods
                - **Currency impacts** during Venus-Pluto aspects
                - **Seasonal patterns** enhanced by lunar cycles
                
                **📈 Trading Levels:**
                - **Resistance**: Previous Jupiter aspect highs
                - **Support**: Saturn aspect consolidation zones
                - **Breakout zones**: Mars-Uranus conjunction levels
                - **Reversal points**: Full Moon technical confluences
                """)
            
            elif symbol in ['DOWJONES']:
                st.markdown(f"""
                ### {symbol} - {month_name} {selected_year} US Index Outlook
                
                **🇺🇸 Macro Astrological Themes:**
                - **Jupiter-Saturn cycles**: Economic expansion vs. contraction
                - **Mercury-Venus aspects**: Corporate earnings and consumer spending
                - **Mars-Jupiter aspects**: Business investment and growth
                - **Outer planet aspects**: Long-term structural changes
                
                **🏛️ Federal Reserve Alignment:**
                - **Venus-Saturn aspects**: Interest rate policy changes
                - **Mercury-Jupiter aspects**: Fed communication clarity
                - **Moon phases**: Market sentiment around FOMC meetings
                - **Eclipse periods**: Major policy shift announcements
                
                **🔄 Sector Rotation Patterns:**
                - **Technology** leadership during Mercury-Uranus aspects
                - **Energy** strength during Mars-Pluto periods
                - **Financials** favor during Venus-Jupiter trines
                - **Healthcare** defensive during Saturn aspects
                
                **📊 Technical Confluence:**
                - **Monthly resistance**: Jupiter aspect previous highs
                - **Monthly support**: Saturn aspect previous lows
                - **Breakout potential**: New Moon near technical levels
                - **Reversal zones**: Full Moon at key Fibonacci levels
                """)
            
            else:  # Indian stocks
                st.markdown(f"""
                ### {symbol} ({symbol_info['sector']}) - {month_name} {selected_year} Indian Market Outlook
                
                **🇮🇳 Domestic Astrological Influences:**
                - **Jupiter transits**: Market leadership and FII flows
                - **Saturn aspects**: Regulatory changes and policy shifts
                - **Mars-Venus aspects**: Consumer spending and investment flows
                - **Moon phases**: Retail investor sentiment cycles
                
                **💹 Sector-Specific Themes:**
                - **{symbol_info['sector']} sector** influenced by specific planetary combinations
                - **Monsoon patterns** (if applicable) aligned with water sign emphasis
                - **Festival seasons** amplified by benefic planetary aspects
                - **Budget impacts** during Saturn-Jupiter aspects
                
                **🌏 Global Correlation Factors:**
                - **US Fed policy** impacts during Venus-Saturn aspects
                - **China growth** concerns during Mars-Saturn periods  
                - **Oil prices** affecting through Mars-Pluto aspects
                - **Dollar strength** impacts during Pluto aspects
                
                **📈 Monthly Strategy:**
                - **Accumulate** during Saturn aspects (value opportunities)
                - **Momentum plays** during Mars-Jupiter periods
                - **Defensive positioning** during challenging outer planet aspects
                - **Sector rotation** based on planetary emphasis shifts
                """)
        
        with tab2:
            st.markdown(f"""
            ### Technical Analysis Integration with Astrological Cycles
            
            **📊 Moving Average Alignment:**
            - **MA5 vs MA20**: Bullish when Jupiter aspects dominate
            - **Golden Cross** potential during Venus-Jupiter trines
            - **Death Cross** risk during Saturn-Mars squares
            - **MA support/resistance** stronger during lunar phases
            
            **🎯 Support & Resistance Levels:**
            - **Primary resistance**: Previous month's Jupiter aspect highs
            - **Primary support**: Saturn aspect consolidation lows
            - **Secondary levels**: Full Moon reversal points
            - **Breakout levels**: New Moon momentum points
            
            **📈 Momentum Indicators:**
            - **RSI overbought** (>70) more reliable during Full Moons
            - **RSI oversold** (<30) stronger signal during New Moons
            - **MACD divergences** amplified during Mercury aspects
            - **Volume confirmations** critical during Mars aspects
            
            **🌙 Lunar Cycle Technical Correlation:**
            - **New Moon**: Trend initiation, breakout potential
            - **Waxing Moon**: Momentum continuation, bullish bias
            - **Full Moon**: Trend exhaustion, reversal potential
            - **Waning Moon**: Correction phases, consolidation
            
            **⚡ Volatility Patterns:**
            - **Highest volatility**: Mars-Uranus aspect periods
            - **Lowest volatility**: Venus-Jupiter trine periods
            - **Unexpected moves**: Mercury-Neptune confusion aspects
            - **Gap movements**: Eclipse and outer planet aspects
            
            **🔄 Pattern Recognition:**
            - **Triangle breakouts** during Uranus aspects
            - **Flag patterns** during Mars aspects  
            - **Head & Shoulders** during Saturn aspects
            - **Double tops/bottoms** during opposition aspects
            """)
        
        with tab3:
            st.markdown(f"""
            ### Lunar Cycles & Market Psychology for {month_name} {selected_year}
            
            **🌑 New Moon Phases (Market Initiation):**
            - **Energy**: Fresh starts, new trend beginnings
            - **Psychology**: Optimism, risk-taking increases
            - **Trading**: Look for breakout setups, trend initiations
            - **Volume**: Often lower but quality moves
            - **Best for**: Opening new positions, trend following
            
            **🌓 Waxing Moon (Building Momentum):**
            - **Energy**: Growth, expansion, building confidence  
            - **Psychology**: FOMO starts building, bullish sentiment
            - **Trading**: Momentum continuation, pyramid additions
            - **Volume**: Increasing participation
            - **Best for**: Adding to winning positions
            
            **🌕 Full Moon Phases (Emotional Peaks):**
            - **Energy**: Maximum emotion, extremes, reversals
            - **Psychology**: Euphoria or panic peaks
            - **Trading**: Reversal setups, profit-taking
            - **Volume**: Often highest of cycle
            - **Best for**: Profit booking, contrarian plays
            
            **🌗 Waning Moon (Consolidation):**
            - **Energy**: Release, correction, cooling off
            - **Psychology**: Reality check, risk assessment
            - **Trading**: Consolidation patterns, value hunting
            - **Volume**: Declining, selective moves
            - **Best for**: Position adjustments, planning
            
            **🔮 {month_name} {selected_year} Specific Lunar Events:**
            
            **Key Lunar Dates to Watch:**
            - **New Moon**: Potential trend change or continuation signal
            - **First Quarter**: Momentum confirmation or failure
            - **Full Moon**: Profit-taking opportunity or reversal signal  
            - **Last Quarter**: Consolidation phase or weakness signal
            
            **Moon Sign Influences:**
            - **Fire Signs** (Aries, Leo, Sagittarius): Aggressive moves, energy sector strength
            - **Earth Signs** (Taurus, Virgo, Capricorn): Value focus, stability preference
            - **Air Signs** (Gemini, Libra, Aquarius): Communication, technology emphasis
            - **Water Signs** (Cancer, Scorpio, Pisces): Emotional decisions, defensive moves
            """)
        
        with tab4:
            st.markdown(f"""
            ### Portfolio Strategy for {month_name} {selected_year}
            
            **🎯 Strategic Asset Allocation:**
            
            **Core Holdings (50-60%):**
            - **Large Cap Stability**: Jupiter-aspected blue chips
            - **Sector Leaders**: Dominant players in favored sectors
            - **Defensive Assets**: During challenging aspect periods
            - **Currency Hedge**: If significant Pluto aspects present
            
            **Growth Opportunities (20-30%):**
            - **Momentum Plays**: Mars-Jupiter aspect beneficiaries
            - **Breakout Candidates**: Technical + astrological confluence
            - **Sector Rotation**: Following planetary emphasis shifts
            - **Emerging Themes**: Uranus aspect innovation plays
            
            **Speculative/Trading (10-20%):**
            - **High Beta Names**: For Mars-Uranus periods
            - **Volatility Plays**: Options during aspect peaks
            - **Contrarian Bets**: Against crowd during extremes
            - **Crypto Allocation**: If comfortable with high volatility
            
            **📊 Risk Management Framework:**
            
            **Position Sizing Rules:**
            - **Maximum single position**: 5% during stable periods
            - **Reduce to 3%**: During challenging aspects
            - **Increase to 7%**: During strong favorable confluences
            - **Cash levels**: 10-20% based on aspect favorability
            
            **Stop Loss Strategy:**
            - **Tight stops** (3-5%): During Mercury retrograde periods
            - **Normal stops** (5-8%): During regular market conditions
            - **Wide stops** (8-12%): During high volatility aspect periods
            - **No stops**: For long-term Jupiter-blessed positions
            
            **📅 Monthly Rebalancing Schedule:**
            
            **Week 1**: Review and adjust based on new lunar cycle
            **Week 2**: Add to momentum winners if aspects support
            **Week 3**: Prepare for Full Moon profit-taking opportunities
            **Week 4**: Position for next month's astrological themes
            
            **🔄 Sector Rotation Strategy:**
            
            **Early Month**: Follow Jupiter aspects for growth sectors
            **Mid Month**: Mars aspects may favor energy/materials
            **Late Month**: Venus aspects support financials/consumer
            **Month End**: Saturn aspects favor defensives/utilities
            
            **💡 Advanced Strategies:**
            
            **Pairs Trading**: Long favored sectors, short challenged sectors
            **Options Overlay**: Sell calls during Full Moons, buy calls during New Moons
            **Currency Hedge**: Hedge foreign exposure during Pluto aspects
            **Volatility Trading**: Long volatility before aspect peaks
            
            **📈 Performance Tracking:**
            
            **Monthly Metrics**:
            - Absolute return vs. benchmark
            - Risk-adjusted return (Sharpe ratio)
            - Maximum drawdown during challenging aspects
            - Hit rate on astrological predictions
            
            **Aspect Correlation Analysis**:
            - Track which aspects work best for {symbol}
            - Note sector rotation timing accuracy
            - Measure volatility prediction success
            - Document lunar cycle correlations
            """)
        
        # Additional insights for monthly strategy
        st.subheader('🎭 Market Psychology & Sentiment Analysis')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            #### 🧠 Psychological Drivers - {month_name}
            
            **Institutional Behavior:**
            - Month-end window dressing effects
            - Quarterly rebalancing influences  
            - Earnings season psychological impacts
            - Fed meeting anticipation/reaction
            
            **Retail Investor Patterns:**
            - Payroll cycle investment flows
            - Tax implications (if year-end)
            - Holiday season spending impacts
            - Social media sentiment amplification
            
            **Global Sentiment Factors:**
            - US-China trade relationship status
            - European economic data impacts
            - Emerging market flow dynamics
            - Cryptocurrency correlation effects
            """)
        
        with col2:
            st.markdown(f"""
            #### 📊 Sentiment Indicators to Watch
            
            **Technical Sentiment:**
            - VIX levels and term structure
            - Put/Call ratios by sector
            - High-low index readings
            - Advance-decline line trends
            
            **Fundamental Sentiment:**
            - Earnings revision trends
            - Analyst recommendation changes
            - Insider buying/selling activity
            - Share buyback announcements
            
            **Alternative Data:**
            - Google search trends
            - Social media mention analysis
            - Options flow analysis
            - Crypto correlation strength
            """)

# Footer with additional information
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <h4>🌟 Disclaimer & Important Notes</h4>
        <p><strong>Educational Purpose Only:</strong> This dashboard is for educational and research purposes. 
        Astrological analysis should be combined with fundamental and technical analysis for trading decisions.</p>
        
        <p><strong>Risk Warning:</strong> All trading involves risk. Past performance and astrological correlations 
        do not guarantee future results. Always use proper risk management and never invest more than you can afford to lose.</p>
        
        <p><strong>Data Sources:</strong> Simulated price data based on astrological aspect calculations. 
        For live trading, use real market data and professional trading platforms.</p>
        
        <p style='font-size: 12px; margin-top: 20px;'>
        🔮 <em>"The stars impel, they do not compel. Wisdom lies in using all available tools - 
        fundamental, technical, and cosmic - for informed decision making."</em>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main(), 'default_price': 25.50, 'sector': 'Precious Metals'},
    'CRUDE': {'name': 'Crude Oil WTI', 'currency': '

# --- STOCK DATABASE ---
stock_data = {
    'Symbol': [
        'TCS', 'ICICIBANK', 'MARUTI', 'DLF', 'NESTLEIND', 
        'RELIANCE', 'SBI', 'SUNPHARMA', 'DRREDDY', 'GOLD',
        'DOWJONES', 'SILVER', 'CRUDE', 'BTC'
    ],
    'Sector': [
        'Technology', 'Banking', 'Automotive', 'Realty', 'FMCG',
        'Energy', 'PSUs', 'Pharma', 'Pharma', 'Precious Metals',
        'US Index', 'Precious Metals', 'Energy', 'Cryptocurrency'
    ],
    'MarketCap': [
        'Large', 'Large', 'Large', 'Large', 'Large',
        'Large', 'Large', 'Large', 'Large', 'Commodity',
        'Index', 'Commodity', 'Commodity', 'Crypto'
    ]
}

STOCK_DATABASE = pd.DataFrame(stock_data)

# --- SECTOR-PLANETARY MAPPINGS ---
SECTOR_PLANETARY_INFLUENCES = {
    'Technology': ['Mercury'],
    'Banking': ['Jupiter', 'Saturn'],
    'FMCG': ['Moon'],
    'Pharma': ['Neptune'],
    'Energy': ['Mars'],
    'Automotive': ['Saturn'],
    'Realty': ['Saturn'],
    'PSUs': ['Pluto'],
    'Midcaps': ['Uranus'],
    'Smallcaps': ['Pluto'],
    'Precious Metals': ['Venus', 'Jupiter'],
    'US Index': ['Jupiter', 'Saturn'],
    'Cryptocurrency': ['Uranus', 'Pluto']
}

# --- ASPECT-SECTOR IMPACT ---
ASPECT_SECTOR_IMPACTS = {
    'Square': {
        'Technology': 'Negative', 'Banking': 'Negative', 'FMCG': 'Neutral',
        'Pharma': 'Neutral', 'Energy': 'Neutral', 'Automotive': 'Neutral',
        'Realty': 'Neutral', 'PSUs': 'Neutral', 'Midcaps': 'Neutral',
        'Smallcaps': 'Neutral', 'Precious Metals': 'Positive', 'US Index': 'Negative',
        'Cryptocurrency': 'Negative'
    },
    'Opposition': {
        'Technology': 'Neutral', 'Banking': 'Neutral', 'FMCG': 'Neutral',
        'Pharma': 'Neutral', 'Energy': 'Neutral', 'Automotive': 'Negative',
        'Realty': 'Negative', 'PSUs': 'Neutral', 'Midcaps': 'Neutral',
        'Smallcaps': 'Neutral', 'Precious Metals': 'Positive', 'US Index': 'Neutral',
        'Cryptocurrency': 'Positive'
    },
    'Trine': {
        'Technology': 'Neutral', 'Banking': 'Neutral', 'FMCG': 'Positive',
        'Pharma': 'Positive', 'Energy': 'Neutral', 'Automotive': 'Neutral',
        'Realty': 'Neutral', 'PSUs': 'Neutral', 'Midcaps': 'Neutral',
        'Smallcaps': 'Neutral', 'Precious Metals': 'Positive', 'US Index': 'Positive',
        'Cryptocurrency': 'Neutral'
    },
    'Conjunction': {
        'Technology': 'Neutral', 'Banking': 'Neutral', 'FMCG': 'Neutral',
        'Pharma': 'Neutral', 'Energy': 'Positive', 'Automotive': 'Neutral',
        'Realty': 'Neutral', 'PSUs': 'Neutral', 'Midcaps': 'Negative',
        'Smallcaps': 'Neutral', 'Precious Metals': 'Positive', 'US Index': 'Neutral',
        'Cryptocurrency': 'Positive'
    },
    'Sextile': {
        'Technology': 'Neutral', 'Banking': 'Neutral', 'FMCG': 'Neutral',
        'Pharma': 'Neutral', 'Energy': 'Neutral', 'Automotive': 'Neutral',
        'Realty': 'Neutral', 'PSUs': 'Positive', 'Midcaps': 'Neutral',
        'Smallcaps': 'Negative', 'Precious Metals': 'Positive', 'US Index': 'Positive',
        'Cryptocurrency': 'Neutral'
    }
}

# --- PLANETARY POSITION VISUALIZATION ---
def draw_planetary_wheel(ax, input_date, size=0.3):
    """Draw a simplified astrological wheel showing planetary positions"""
    base_date = datetime(2025, 8, 1)
    
    if isinstance(input_date, date_class) and not isinstance(input_date, datetime):
        date_obj = datetime.combine(input_date, datetime.min.time())
    else:
        date_obj = input_date
    
    days_diff = (date_obj.date() - base_date.date()).days
    
    base_positions = {
        'Sun': 135, 'Moon': 225, 'Mercury': 120, 'Venus': 170,
        'Mars': 85, 'Jupiter': 45, 'Saturn': 315
    }
    
    daily_movement = {
        'Sun': 1.0, 'Moon': 13.2, 'Mercury': 1.5, 'Venus': 1.2,
        'Mars': 0.5, 'Jupiter': 0.08, 'Saturn': 0.03
    }
    
    planets = {}
    for planet, base_pos in base_positions.items():
        angle = (base_pos + daily_movement[planet] * days_diff) % 360
        planets[planet] = {
            'angle': angle,
            'color': {
                'Sun': 'gold', 'Moon': 'silver', 'Mercury': 'gray',
                'Venus': 'lightgreen', 'Mars': 'red', 'Jupiter': 'orange',
                'Saturn': 'darkgoldenrod'
            }[planet],
            'size': {
                'Sun': 8, 'Moon': 6, 'Mercury': 5, 'Venus': 7,
                'Mars': 6, 'Jupiter': 10, 'Saturn': 9
            }[planet]
        }
    
    zodiac = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 
              'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
    
    for i, sign in enumerate(zodiac):
        angle = i * 30
        ax.add_patch(Wedge((0, 0), size, angle, angle+30, width=size*0.8, 
                          facecolor='lightgray', edgecolor='black', alpha=0.3))
        ax.text(0.85*size * np.cos(np.radians(angle+15)), 
                0.85*size * np.sin(np.radians(angle+15)), 
                sign[:3], ha='center', va='center', fontsize=6)
    
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
    ax.set_title(f'Planetary Positions\n{date_obj.strftime("%b %d, %Y")}', fontsize=8)

# --- GET TRADING HOURS FOR SYMBOL ---
def get_trading_hours(symbol):
    """Get trading hours for a specific symbol"""
    symbol = symbol.upper()
    if symbol in TRADING_HOURS:
        return TRADING_HOURS[symbol]
    else:
        # Default to Indian market hours for unknown symbols
        return TRADING_HOURS['NIFTY']

# --- GET SYMBOL INFO ---
def get_symbol_info(symbol):
    """Get symbol configuration info"""
    symbol = symbol.upper()
    if symbol in SYMBOL_CONFIG:
        return SYMBOL_CONFIG[symbol]
    else:
        # Default configuration
        return {
            'name': symbol,
            'currency': '₹',
            'default_price': 1000.0,
            'sector': 'Unknown'
        }

# --- GENERATE ASPECTS ---
def generate_todays_aspects():
    """Generate astrological aspects for today based on the provided table"""
    base_aspects = [
        {"planets": "Mercury-Jupiter", "aspect_type": "Square", "impact": -0.7, "type": "bearish"},
        {"planets": "Venus-Saturn", "aspect_type": "Opposition", "impact": -0.8, "type": "bearish"},
        {"planets": "Moon-Neptune", "aspect_type": "Trine", "impact": 0.6, "type": "bullish"},
        {"planets": "Mars-Uranus", "aspect_type": "Conjunction", "impact": 0.9, "type": "bullish"},
        {"planets": "Sun-Pluto", "aspect_type": "Sextile", "impact": 0.5, "type": "bullish"}
    ]
    
    aspects = []
    for aspect in base_aspects:
        aspects.append({
            "planets": aspect["planets"],
            "aspect_type": aspect["aspect_type"],
            "impact": aspect["impact"],
            "type": aspect["type"]
        })
    
    return aspects

# --- CREATE SUMMARY TABLE ---
def create_summary_table(aspects):
    """Create a summary table based on the astrological aspects"""
    summary_data = {
        'Aspect': [],
        'Nifty/Bank Nifty': [],
        'Bullish Sectors/Stocks': [],
        'Bearish Sectors/Stocks': []
    }
    
    for aspect in aspects:
        planets = aspect["planets"]
        aspect_type = aspect["aspect_type"]
        
        if planets == "Mercury-Jupiter" and aspect_type == "Square":
            summary_data['Aspect'].append("Mercury-Jupiter (Square)")
            summary_data['Nifty/Bank Nifty'].append("Volatile")
            summary_data['Bullish Sectors/Stocks'].append("IT (TCS), Gold")
            summary_data['Bearish Sectors/Stocks'].append("Banking (ICICI Bank), Crypto")
        
        elif planets == "Venus-Saturn" and aspect_type == "Opposition":
            summary_data['Aspect'].append("Venus-Saturn (Opposition)")
            summary_data['Nifty/Bank Nifty'].append("Downside")
            summary_data['Bullish Sectors/Stocks'].append("Gold, Silver, BTC")
            summary_data['Bearish Sectors/Stocks'].append("Auto (Maruti), Realty (DLF)")
        
        elif planets == "Moon-Neptune" and aspect_type == "Trine":
            summary_data['Aspect'].append("Moon-Neptune (Trine)")
            summary_data['Nifty/Bank Nifty'].append("Mild Support")
            summary_data['Bullish Sectors/Stocks'].append("FMCG (Nestlé), Pharma, Gold, Dow Jones")
            summary_data['Bearish Sectors/Stocks'].append("-")
        
        elif planets == "Mars-Uranus" and aspect_type == "Conjunction":
            summary_data['Aspect'].append("Mars-Uranus (Conjunction)")
            summary_data['Nifty/Bank Nifty'].append("Sharp Moves")
            summary_data['Bullish Sectors/Stocks'].append("Energy (Reliance, Crude), Gold, BTC")
            summary_data['Bearish Sectors/Stocks'].append("Weak Midcaps")
        
        elif planets == "Sun-Pluto" and aspect_type == "Sextile":
            summary_data['Aspect'].append("Sun-Pluto (Sextile)")
            summary_data['Nifty/Bank Nifty'].append("Structural Shift")
            summary_data['Bullish Sectors/Stocks'].append("PSUs (SBI), Gold, Dow Jones")
            summary_data['Bearish Sectors/Stocks'].append("Overvalued Smallcaps")
    
    return pd.DataFrame(summary_data)

# --- FILTER STOCKS BASED ON ASPECTS ---
def filter_stocks_by_aspects(aspects, stock_database):
    """Filter stocks based on today's astrological aspects"""
    sector_impacts = {sector: 0 for sector in stock_database['Sector'].unique()}
    
    for aspect in aspects:
        planet1, planet2 = aspect["planets"].split("-")
        
        for sector, planets in SECTOR_PLANETARY_INFLUENCES.items():
            if planet1 in planets or planet2 in planets:
                if sector not in sector_impacts:
                    sector_impacts[sector] = 0
                
                aspect_impact = ASPECT_SECTOR_IMPACTS[aspect["aspect_type"]].get(sector, "Neutral")
                
                if aspect_impact == "Positive":
                    sector_impacts[sector] += abs(aspect["impact"])
                elif aspect_impact == "Negative":
                    sector_impacts[sector] -= abs(aspect["impact"])
    
    bullish_sectors = [sector for sector, impact in sector_impacts.items() if impact > 0]
    bearish_sectors = [sector for sector, impact in sector_impacts.items() if impact < 0]
    neutral_sectors = [sector for sector, impact in sector_impacts.items() if impact == 0]
    
    bullish_stocks = stock_database[stock_database['Sector'].isin(bullish_sectors)].copy()
    bearish_stocks = stock_database[stock_database['Sector'].isin(bearish_sectors)].copy()
    neutral_stocks = stock_database[stock_database['Sector'].isin(neutral_sectors)].copy()
    
    bullish_stocks['Impact Score'] = bullish_stocks['Sector'].apply(lambda x: sector_impacts[x])
    bearish_stocks['Impact Score'] = bearish_stocks['Sector'].apply(lambda x: abs(sector_impacts[x]))
    neutral_stocks['Impact Score'] = 0
    
    bullish_stocks = bullish_stocks.sort_values('Impact Score', ascending=False)
    bearish_stocks = bearish_stocks.sort_values('Impact Score', ascending=False)
    
    return {
        'bullish': bullish_stocks,
        'bearish': bearish_stocks,
        'neutral': neutral_stocks,
        'sector_impacts': sector_impacts
    }

# --- GENERATE ASTROLOGICAL EVENTS ---
def generate_astrological_events(input_date, event_type='intraday', symbol='NIFTY'):
    """Generate astrological events for any given date and symbol"""
    
    if event_type == 'intraday':
        trading_hours = get_trading_hours(symbol)
        
        # Different event patterns based on trading hours
        if trading_hours['end_hour'] > 16:  # Extended hours (global markets)
            # More events spread across longer trading day
            base_events = [
                {"time_offset": 0, "aspect": "Pre-market: Mercury square Jupiter", "impact": -0.5, "type": "bearish"},
                {"time_offset": 120, "aspect": "Asian session: Moon trine Jupiter", "impact": 0.8, "type": "bullish"},
                {"time_offset": 240, "aspect": "London open: Mars sextile Jupiter", "impact": 0.4, "type": "neutral"},
                {"time_offset": 360, "aspect": "European session: Venus opposition Saturn", "impact": -0.6, "type": "bearish"},
                {"time_offset": 480, "aspect": "NY pre-market: Sun conjunct Mercury", "impact": 0.3, "type": "neutral"},
                {"time_offset": 600, "aspect": "US open: Mars conjunct Uranus", "impact": 1.0, "type": "bullish"},
                {"time_offset": 720, "aspect": "Mid-day: Moon square Saturn", "impact": -0.4, "type": "bearish"},
                {"time_offset": 840, "aspect": "Afternoon: Jupiter trine Neptune", "impact": 0.7, "type": "bullish"},
                {"time_offset": 960, "aspect": "US close approach", "impact": 0.2, "type": "neutral"},
                {"time_offset": 1080, "aspect": "After hours: Void Moon", "impact": -0.3, "type": "bearish"},
                {"time_offset": 1135, "aspect": "Session close", "impact": 0.1, "type": "neutral"}
            ]
        else:  # Standard Indian market hours
            base_events = [
                {"time_offset": 0, "aspect": "Opening: Mercury square Jupiter + Void Moon", "impact": -0.5, "type": "bearish"},
                {"time_offset": 45, "aspect": "Early trade: Moon trine Jupiter", "impact": 1.0, "type": "bullish"},
                {"time_offset": 135, "aspect": "Mid-morning: Mars sextile Jupiter", "impact": 0.3, "type": "neutral"},
                {"time_offset": 195, "aspect": "Pre-lunch: Sun in Leo (no aspects)", "impact": 0.0, "type": "neutral"},
                {"time_offset": 285, "aspect": "Post-lunch: Moon square Saturn", "impact": -0.8, "type": "bearish"},
                {"time_offset": 345, "aspect": "Late trade: Venus-Saturn opposition", "impact": -0.6, "type": "bearish"},
                {"time_offset": 375, "aspect": "Close", "impact": 0.1, "type": "neutral"}
            ]
        
        events = []
        if isinstance(input_date, date_class) and not isinstance(input_date, datetime):
            dt = datetime.combine(input_date, datetime.min.time())
        else:
            dt = input_date
            
        start_time = dt.replace(hour=trading_hours['start_hour'], minute=trading_hours['start_minute'])
        
        for event in base_events:
            event_time = start_time + timedelta(minutes=event["time_offset"])
            events.append({
                "time": event_time,
                "aspect": event["aspect"],
                "impact": event["impact"],
                "type": event["type"],
                "price": 0
            })
        
        return events
    
    else:  # monthly events remain the same
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
        
        if isinstance(input_date, datetime):
            year, month = input_date.year, input_date.month
        else:
            year, month = input_date.year, input_date.month
            
        days_in_month = calendar.monthrange(year, month)[1]
        
        events = []
        for event in base_events:
            day = min(event["day_offset"], days_in_month)
            event_date = datetime(year, month, day)
            events.append({
                "date": event_date,
                "aspect": event["aspect"],
                "impact": event["impact"],
                "type": event["type"],
                "price": 0
            })
        
        return events

# --- ENHANCED INTRADAY CHART ---
def generate_intraday_chart(symbol, starting_price, selected_date):
    """Generate enhanced intraday chart with dynamic layout"""
    symbol_info = get_symbol_info(symbol)
    trading_hours = get_trading_hours(symbol)
    
    if isinstance(selected_date, date_class) and not isinstance(selected_date, datetime):
        selected_date = datetime.combine(selected_date, datetime.min.time())
    
    start_time = selected_date.replace(hour=trading_hours['start_hour'], minute=trading_hours['start_minute'])
    end_time = selected_date.replace(hour=trading_hours['end_hour'], minute=trading_hours['end_minute'])
    
    # Adjust interval based on trading session length
    session_hours = (end_time - start_time).total_seconds() / 3600
    if session_hours > 12:
        interval = '30T'  # 30-minute intervals for long sessions
    else:
        interval = '15T'  # 15-minute intervals for shorter sessions
    
    times = pd.date_range(start=start_time, end=end_time, freq=interval)
    
    prices = np.zeros(len(times))
    base_price = starting_price
    
    events = generate_astrological_events(selected_date, 'intraday', symbol)
    
    # Adjust event impacts based on symbol type
    symbol_multiplier = 1.0
    if symbol in ['GOLD', 'SILVER']:
        symbol_multiplier = 0.8  # Precious metals less volatile to aspects
    elif symbol in ['BTC']:
        symbol_multiplier = 2.0  # Crypto more volatile
    elif symbol in ['CRUDE']:
        symbol_multiplier = 1.5  # Energy commodities more responsive
    
    for event in events:
        price_change = event["impact"] * base_price * 0.01 * symbol_multiplier
        event["price"] = base_price + price_change
    
    # Generate price movements
    for i, time in enumerate(times):
        closest_event = min(events, key=lambda x: abs((x["time"] - time).total_seconds()))
        distance = abs((closest_event["time"] - time).total_seconds()) / 3600
        
        # Adjust volatility based on symbol
        base_volatility = 0.15 if distance < 0.5 else 0.05
        if symbol in ['BTC']:
            base_volatility *= 3.0
        elif symbol in ['GOLD', 'SILVER']:
            base_volatility *= 0.5
        elif symbol in ['CRUDE']:
            base_volatility *= 2.0
        
        random_change = np.random.normal(0, base_volatility)
        event_influence = closest_event["impact"] * np.exp(-distance) * symbol_multiplier
        
        if i == 0:
            prices[i] = base_price
        else:
            change = (event_influence + random_change) * base_price * 0.001
            prices[i] = prices[i-1] + change
    
    df_intraday = pd.DataFrame({
        'Time': times,
        'Price': prices,
        'Aspect': [min(events, key=lambda x: abs((x["time"] - t).total_seconds()))["aspect"] for t in times]
    })
    
    # Create dynamic figure layout
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, height_ratios=[4, 1, 1, 1], width_ratios=[5, 2, 1])
    
    # Main price chart
    ax_main = fig.add_subplot(gs[0, :2])
    
    for i in range(1, len(df_intraday)):
        color = 'green' if df_intraday['Price'].iloc[i] > df_intraday['Price'].iloc[i-1] else 'red'
        ax_main.plot(df_intraday['Time'].iloc[i-1:i+1], 
                    df_intraday['Price'].iloc[i-1:i+1], 
                    color=color, linewidth=2.5)
    
    # Mark key events
    for event in events:
        color_map = {'bullish': 'green', 'bearish': 'red', 'neutral': 'blue'}
        ax_main.axvline(x=event['time'], color=color_map[event['type']], 
                       linestyle='--', alpha=0.7, linewidth=2)
        
        marker_color = color_map[event['type']]
        ax_main.scatter(event['time'], event['price'], color=marker_color, 
                       s=100, zorder=5, edgecolor='black', linewidth=1)
        
        # Dynamic annotation positioning
        y_offset = base_price * 0.02 if len(str(int(base_price))) >= 4 else base_price * 0.05
        y_pos = event['price'] + y_offset if event['price'] < base_price * 1.01 else event['price'] - y_offset
        
        ax_main.annotate(event['aspect'][:30] + '...' if len(event['aspect']) > 30 else event['aspect'], 
                        xy=(event['time'], event['price']),
                        xytext=(event['time'], y_pos),
                        arrowprops=dict(arrowstyle='->', color=marker_color, lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                        fontsize=8, ha='center')
    
    # Dynamic title and formatting
    currency_symbol = symbol_info['currency']
    ax_main.set_title(f'{symbol_info["name"]} ({symbol}) - {selected_date.strftime("%B %d, %Y")}\n'
                     f'Astrological Trading Analysis | Sector: {symbol_info["sector"]}', 
                     fontsize=16, pad=20)
    ax_main.set_xlabel(f'Time ({trading_hours["start_hour"]}:00 - {trading_hours["end_hour"]}:00)', fontsize=12)
    ax_main.set_ylabel(f'Price ({currency_symbol})', fontsize=12)
    
    # Dynamic time formatting based on session length
    if session_hours > 12:
        ax_main.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    else:
        ax_main.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    plt.setp(ax_main.get_xticklabels(), rotation=45, ha='right')
    
    # Closing price annotation
    close_price = df_intraday["Price"].iloc[-1]
    price_change = close_price - base_price
    price_change_pct = (price_change / base_price) * 100
    
    ax_main.annotate(f'Close: {currency_symbol}{close_price:.2f}\n'
                    f'Change: {price_change:+.2f} ({price_change_pct:+.2f}%)', 
                    xy=(df_intraday['Time'].iloc[-1], close_price),
                    xytext=(df_intraday['Time'].iloc[-1] - timedelta(hours=session_hours*0.2), 
                           close_price + base_price * 0.02),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                    fontsize=11, fontweight='bold',
                    bbox=dict(facecolor='yellow', alpha=0.7, edgecolor='none', pad=2))
    
    # Planetary wheel
    ax_wheel = fig.add_subplot(gs[0, 2])
    draw_planetary_wheel(ax_wheel, selected_date, size=0.4)
    
    # Volume chart (simulated with realistic patterns)
    ax_volume = fig.add_subplot(gs[1, :2])
    
    # Generate more realistic volume based on symbol type
    if symbol in ['BTC']:
        base_volume = np.random.randint(50000, 200000, size=len(times))
    elif symbol in ['GOLD', 'SILVER', 'CRUDE']:
        base_volume = np.random.randint(10000, 50000, size=len(times))
    elif symbol in ['DOWJONES']:
        base_volume = np.random.randint(100000, 500000, size=len(times))
    else:  # Indian stocks
        base_volume = np.random.randint(1000, 10000, size=len(times))
    
    colors_volume = ['green' if df_intraday['Price'].iloc[i] > df_intraday['Price'].iloc[i-1] 
                    else 'red' for i in range(1, len(df_intraday))]
    colors_volume.insert(0, 'green')
    
    ax_volume.bar(df_intraday['Time'], base_volume, color=colors_volume, alpha=0.7)
    ax_volume.set_title('Volume', fontsize=12)
    ax_volume.set_ylabel('Volume', fontsize=10)
    ax_volume.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax_volume.get_xticklabels(), rotation=45, ha='right')
    
    # Technical indicators (RSI simulation)
    ax_rsi = fig.add_subplot(gs[2, :2])
    rsi_values = 50 + np.random.normal(0, 15, len(times))  # Simulated RSI
    rsi_values = np.clip(rsi_values, 0, 100)
    
    ax_rsi.plot(df_intraday['Time'], rsi_values, color='purple', linewidth=2)
    ax_rsi.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
    ax_rsi.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
    ax_rsi.fill_between(df_intraday['Time'], 30, 70, alpha=0.1, color='gray')
    ax_rsi.set_title('RSI (14)', fontsize=12)
    ax_rsi.set_ylabel('RSI', fontsize=10)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.legend(loc='upper right', fontsize=8)
    
    # Aspect strength indicator
    ax_aspect = fig.add_subplot(gs[3, :2])
    aspect_times = [event['time'] for event in events]
    aspect_strengths = [abs(event['impact']) for event in events]
    aspect_colors = [{'bullish': 'green', 'bearish': 'red', 'neutral': 'blue'}[event['type']] for event in events]
    
    ax_aspect.scatter(aspect_times, aspect_strengths, color=aspect_colors, s=100, zorder=3)
    ax_aspect.plot(aspect_times, aspect_strengths, color='gray', alpha=0.5, linestyle='--')
    ax_aspect.set_title('Astrological Aspect Strength', fontsize=12)
    ax_aspect.set_ylabel('Strength', fontsize=10)
    ax_aspect.set_ylim(0, max(aspect_strengths) * 1.2 if aspect_strengths else 1.5)
    
    # Symbol info panel
    ax_info = fig.add_subplot(gs[1:, 2])
    ax_info.axis('off')
    
    info_text = f"""
SYMBOL INFO
-----------
Name: {symbol_info['name']}
Sector: {symbol_info['sector']}
Currency: {symbol_info['currency']}

TRADING HOURS
-------------
Start: {trading_hours['start_hour']:02d}:{trading_hours['start_minute']:02d}
End: {trading_hours['end_hour']:02d}:{trading_hours['end_minute']:02d}
Session: {session_hours:.1f} hours

PRICE DATA
----------
Open: {currency_symbol}{base_price:.2f}
Close: {currency_symbol}{close_price:.2f}
Change: {price_change:+.2f}
Change%: {price_change_pct:+.2f}%

High: {currency_symbol}{max(prices):.2f}
Low: {currency_symbol}{min(prices):.2f}
Range: {currency_symbol}{max(prices)-min(prices):.2f}
"""
    
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig

# --- ENHANCED MONTHLY CHART ---
def generate_monthly_chart(symbol, starting_price, selected_month, selected_year):
    """Generate enhanced monthly chart with dynamic layout"""
    symbol_info = get_symbol_info(symbol)
    
    start_date = datetime(selected_year, selected_month, 1)
    days_in_month = calendar.monthrange(selected_year, selected_month)[1]
    end_date = datetime(selected_year, selected_month, days_in_month)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    prices = np.zeros(len(dates))
    base_price = starting_price
    
    events = generate_astrological_events(start_date, 'monthly', symbol)
    
    # Adjust event impacts based on symbol type
    symbol_multiplier = 1.0
    if symbol in ['GOLD', 'SILVER']:
        symbol_multiplier = 0.8
    elif symbol in ['BTC']:
        symbol_multiplier = 2.5
    elif symbol in ['CRUDE']:
        symbol_multiplier = 1.8
    
    for event in events:
        price_change = event["impact"] * base_price * 0.01 * symbol_multiplier
        event["price"] = base_price + price_change
    
    # Generate price movements
    for i, date in enumerate(dates):
        closest_event = min(events, key=lambda x: abs((x["date"].date() - date.date()).days))
        distance = abs((closest_event["date"].date() - date.date()).days)
        
        base_volatility = 0.3 if distance < 2 else 0.1
        if symbol in ['BTC']:
            base_volatility *= 4.0
        elif symbol in ['GOLD', 'SILVER']:
            base_volatility *= 0.6
        elif symbol in ['CRUDE']:
            base_volatility *= 2.5
        
        random_change = np.random.normal(0, base_volatility)
        event_influence = closest_event["impact"] * np.exp(-distance/2) * symbol_multiplier
        
        if i == 0:
            prices[i] = base_price
        else:
            change = (event_influence + random_change) * base_price * 0.002
            prices[i] = prices[i-1] + change
    
    df_monthly = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Aspect': [min(events, key=lambda x: abs((x["date"].date() - d.date()).days))["aspect"] for d in dates]
    })
    
    # Create dynamic figure layout
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, height_ratios=[4, 1, 1, 1], width_ratios=[5, 2, 1])
    
    # Main price chart
    ax_main = fig.add_subplot(gs[0, :2])
    
    for i in range(1, len(df_monthly)):
        color = 'green' if df_monthly['Price'].iloc[i] > df_monthly['Price'].iloc[i-1] else 'red'
        ax_main.plot(df_monthly['Date'].iloc[i-1:i+1], 
                    df_monthly['Price'].iloc[i-1:i+1], 
                    color=color, linewidth=3)
    
    # Mark key events
    for event in events:
        color_map = {'bullish': 'green', 'bearish': 'red', 'neutral': 'blue'}
        ax_main.axvline(x=event['date'], color=color_map[event['type']], 
                       linestyle='--', alpha=0.7, linewidth=2)
        
        marker_color = color_map[event['type']]
        ax_main.scatter(event['date'], event['price'], color=marker_color, 
                       s=150, zorder=5, edgecolor='black', linewidth=1.5)
        
        y_offset = base_price * 0.03
        y_pos = event['price'] + y_offset if event['price'] < base_price * 1.02 else event['price'] - y_offset
        
        ax_main.annotate(event['aspect'][:25] + '...' if len(event['aspect']) > 25 else event['aspect'], 
                        xy=(event['date'], event['price']),
                        xytext=(event['date'], y_pos),
                        arrowprops=dict(arrowstyle='->', color=marker_color, lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                        fontsize=9, ha='center')
    
    # Dynamic formatting
    currency_symbol = symbol_info['currency']
    ax_main.set_title(f'{symbol_info["name"]} ({symbol}) - {start_date.strftime("%B %Y")}\n'
                     f'Monthly Astrological Analysis | Sector: {symbol_info["sector"]}', 
                     fontsize=16, pad=20)
    ax_main.set_xlabel('Date', fontsize=12)
    ax_main.set_ylabel(f'Price ({currency_symbol})', fontsize=12)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax_main.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, days_in_month//10)))
    plt.setp(ax_main.get_xticklabels(), rotation=45, ha='right')
    
    # Monthly close annotation
    close_price = df_monthly["Price"].iloc[-1]
    price_change = close_price - base_price
    price_change_pct = (price_change / base_price) * 100
    
    ax_main.annotate(f'Month Close: {currency_symbol}{close_price:.2f}\n'
                    f'Monthly Change: {price_change:+.2f} ({price_change_pct:+.2f}%)', 
                    xy=(df_monthly['Date'].iloc[-1], close_price),
                    xytext=(df_monthly['Date'].iloc[-1] - timedelta(days=days_in_month//4), 
                           close_price + base_price * 0.03),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                    fontsize=12, fontweight='bold',
                    bbox=dict(facecolor='yellow', alpha=0.7, edgecolor='none', pad=2))
    
    # Planetary positions for key dates
    ax_planets = fig.add_subplot(gs[0, 2])
    ax_planets.set_title('Key Planetary\nPositions', fontsize=10)
    key_dates = [
        start_date,
        start_date + timedelta(days=days_in_month//3),
        start_date + timedelta(days=2*days_in_month//3),
        end_date
    ]
    
    for i, date in enumerate(key_dates):
        ax_sub = fig.add_axes([0.70, 0.8-i*0.15, 0.12, 0.12])
        draw_planetary_wheel(ax_sub, date, size=0.4)
        ax_sub.set_title(f'{date.strftime("%b %d")}', fontsize=8)
    
    # Volume chart
    ax_volume = fig.add_subplot(gs[1, :2])
    
    if symbol in ['BTC']:
        volume = np.random.randint(500000, 2000000, size=len(dates))
    elif symbol in ['GOLD', 'SILVER', 'CRUDE']:
        volume = np.random.randint(100000, 500000, size=len(dates))
    elif symbol in ['DOWJONES']:
        volume = np.random.randint(1000000, 5000000, size=len(dates))
    else:
        volume = np.random.randint(10000, 100000, size=len(dates))
    
    colors_volume = ['green' if df_monthly['Price'].iloc[i] > df_monthly['Price'].iloc[i-1] 
                    else 'red' for i in range(1, len(df_monthly))]
    colors_volume.insert(0, 'green')
    
    ax_volume.bar(df_monthly['Date'], volume, color=colors_volume, alpha=0.7)
    ax_volume.set_title('Daily Volume', fontsize=12)
    ax_volume.set_ylabel('Volume', fontsize=10)
    ax_volume.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    # Moving averages
    ax_ma = fig.add_subplot(gs[2, :2])
    ma_5 = df_monthly['Price'].rolling(window=5, min_periods=1).mean()
    ma_20 = df_monthly['Price'].rolling(window=min(20, len(df_monthly)), min_periods=1).mean()
    
    ax_ma.plot(df_monthly['Date'], ma_5, color='blue', linewidth=2, label='MA5', alpha=0.7)
    ax_ma.plot(df_monthly['Date'], ma_20, color='red', linewidth=2, label='MA20', alpha=0.7)
    ax_ma.fill_between(df_monthly['Date'], ma_5, ma_20, alpha=0.1, 
                      color='green' if ma_5.iloc[-1] > ma_20.iloc[-1] else 'red')
    ax_ma.set_title('Moving Averages', fontsize=12)
    ax_ma.set_ylabel('Price', fontsize=10)
    ax_ma.legend(loc='upper left', fontsize=10)
    
    # Aspect calendar
    ax_calendar = fig.add_subplot(gs[3, :2])
    aspect_dates = [event['date'] for event in events]
    aspect_strengths = [abs(event['impact']) for event in events]
    aspect_colors = [{'bullish': 'green', 'bearish': 'red', 'neutral': 'blue'}[event['type']] for event in events]
    
    ax_calendar.scatter(aspect_dates, aspect_strengths, color=aspect_colors, s=200, zorder=3)
    ax_calendar.plot(aspect_dates, aspect_strengths, color='gray', alpha=0.5, linestyle='--')
    ax_calendar.set_title('Monthly Astrological Event Strength', fontsize=12)
    ax_calendar.set_ylabel('Impact Strength', fontsize=10)
    ax_calendar.set_ylim(0, max(aspect_strengths) * 1.2 if aspect_strengths else 2)
    
    # Monthly summary panel
    ax_summary = fig.add_subplot(gs[1:, 2])
    ax_summary.axis('off')
    
    monthly_high = max(prices)
    monthly_low = min(prices)
    monthly_range = monthly_high - monthly_low
    avg_price = np.mean(prices)
    
    summary_text = f"""
MONTHLY SUMMARY
--------------
Symbol: {symbol}
Sector: {symbol_info['sector']}
Month: {start_date.strftime('%B %Y')}

PRICE STATISTICS
---------------
Open: {currency_symbol}{base_price:.2f}
Close: {currency_symbol}{close_price:.2f}
Change: {price_change:+.2f}
Change%: {price_change_pct:+.2f}%

High: {currency_symbol}{monthly_high:.2f}
Low: {currency_symbol}{monthly_low:.2f}
Range: {currency_symbol}{monthly_range:.2f}
Average: {currency_symbol}{avg_price:.2f}

VOLATILITY
----------
Daily Avg: {np.std(np.diff(prices)):.2f}
Monthly Vol: {(monthly_range/avg_price)*100:.1f}%

TREND ANALYSIS
--------------
Bullish Days: {sum(1 for i in range(1, len(prices)) if prices[i] > prices[i-1])}
Bearish Days: {sum(1 for i in range(1, len(prices)) if prices[i] < prices[i-1])}
Neutral Days: {sum(1 for i in range(1, len(prices)) if prices[i] == prices[i-1])}
"""
    
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes, fontsize=8,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    return fig

# --- ANALYZE ASPECTS ---
def analyze_aspects():
    """Enhanced aspect analysis with dynamic content"""
    aspects_data = {
        'Aspect': [
            'Mercury Retrograde', 'Venus Opposition Saturn', 'Moon-Jupiter Trine', 
            'Full Moon', 'Jupiter Square Saturn', 'Mercury Direct',
            'Venus enters Libra', 'New Moon', 'Mars-Uranus Conjunction',
            'Sun-Pluto Sextile'
        ],
        'Market Impact': [
            'High Volatility', 'Bearish Pressure', 'Bullish Surge', 
            'Trend Reversal', 'Major Tension', 'Clarity Returns',
            'Financial Rally', 'Strong Bullish', 'Energy Surge',
            'Structural Change'
        ],
        'Typical Price Change': [
            '±2-3%', '-1.5-2%', '+1-2%', 
            '±1-1.5%', '-2-3%', '+0.5-1%',
            '+0.5-1%', '+1-2%', '+2-4%',
            '±1-2%'
        ],
        'Sector Focus': [
            'All Sectors', 'Banking/Realty', 'Broad Market', 
            'Technology', 'Financials', 'Technology',
            'Banking/Finance', 'Broad Market', 'Energy/Commodities',
            'Infrastructure/PSUs'
        ],
        'Best Symbols': [
            'Gold, BTC', 'Gold, Silver', 'FMCG, Pharma', 
            'Tech Stocks', 'Defensive', 'Tech, Crypto',
            'Banking', 'Growth Stocks', 'Energy, Crude',
            'PSU, Infrastructure'
        ]
    }
    
    df_aspects = pd.DataFrame(aspects_data)
    
    # Create enhanced visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Price change impact chart
    price_changes = []
    for change in df_aspects['Typical Price Change']:
        clean_change = change.replace('%', '').replace('±', '')
        if '-' in clean_change and not clean_change.startswith('-'):
            num_str = clean_change.split('-')[1]  # Take higher value for impact
        else:
            num_str = clean_change.replace('+', '')
        
        try:
            num = float(num_str)
        except:
            num = 1.0
        price_changes.append(num)
    
    colors = ['red' if 'Bearish' in impact or 'Tension' in impact or 'Volatility' in impact 
              else 'orange' if 'Reversal' in impact or 'Change' in impact
              else 'green' for impact in df_aspects['Market Impact']]
    
    bars1 = ax1.bar(range(len(df_aspects)), price_changes, color=colors, alpha=0.7)
    ax1.set_title('Astrological Aspect Impact on Price Changes', fontsize=14)
    ax1.set_ylabel('Maximum Price Change (%)', fontsize=12)
    ax1.set_xticks(range(len(df_aspects)))
    ax1.set_xticklabels(df_aspects['Aspect'], rotation=45, ha='right')
    
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.annotate(f'{height}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    # Sector distribution pie chart
    sector_counts = {}
    for sectors in df_aspects['Sector Focus']:
        for sector in sectors.split('/'):
            sector = sector.strip()
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
    
    ax2.pie(sector_counts.values(), labels=sector_counts.keys(), autopct='%1.1f%%', startangle=90)
    ax2.set_title('Most Affected Sectors by Astrological Aspects', fontsize=14)
    
    # Market impact distribution
    impact_counts = {}
    for impact in df_aspects['Market Impact']:
        impact_type = 'Bullish' if any(word in impact for word in ['Bullish', 'Rally', 'Surge', 'Returns']) else \
                     'Bearish' if any(word in impact for word in ['Bearish', 'Pressure', 'Tension']) else \
                     'Neutral'
        impact_counts[impact_type] = impact_counts.get(impact_type, 0) + 1
    
    colors_impact = ['green', 'red', 'gray']
    ax3.bar(impact_counts.keys(), impact_counts.values(), color=colors_impact, alpha=0.7)
    ax3.set_title('Distribution of Market Impact Types', fontsize=14)
    ax3.set_ylabel('Number of Aspects', fontsize=12)
    
    # Best performing symbols chart
    symbol_mentions = {}
    for symbols in df_aspects['Best Symbols']:
        for symbol in symbols.split(', '):
            symbol = symbol.strip()
            symbol_mentions[symbol] = symbol_mentions.get(symbol, 0) + 1
    
    sorted_symbols = sorted(symbol_mentions.items(), key=lambda x: x[1], reverse=True)
    symbols, counts = zip(*sorted_symbols) if sorted_symbols else ([], [])
    
    ax4.barh(symbols, counts, color='gold', alpha=0.7)
    ax4.set_title('Most Favorable Symbols Across Aspects', fontsize=14)
    ax4.set_xlabel('Favorable Mentions', fontsize=12)
    
    return fig, df_aspects

# --- STREAMLIT APP ---
def main():
    # Page configuration for better responsive design
    st.set_page_config(
        page_title="🌟 Astrological Trading Dashboard",
        page_icon="🌟",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .symbol-input {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #e8f4fd;
        border-left: 5px solid #1f77b4;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🌟 Astrological Trading Dashboard</h1>
        <p>Advanced Financial Analysis through Planetary Movements & Cosmic Patterns</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for inputs with enhanced design
    with st.sidebar:
        st.markdown("### 📊 Dashboard Controls")
        
        # Dashboard section selection with better descriptions
        dashboard_section = st.selectbox(
            '🎯 Choose Analysis Section:',
            [
                'Summary Table - Market Overview',
                'Stock Filter - Sector Analysis', 
                'Aspect Analysis - Deep Insights',
                'Intraday Chart - Live Patterns',
                'Monthly Chart - Trend Analysis'
            ]
        )
        
        # Extract the main section name
        section_name = dashboard_section.split(' - ')[0]
        
        st.markdown("---")
        
        # Symbol selection with enhanced interface
        if section_name in ['Intraday Chart', 'Monthly Chart']:
            st.markdown("### 📈 Symbol Configuration")
            
            # Popular symbols with categories
            symbol_categories = {
                'Indian Indices': ['NIFTY', 'BANKNIFTY'],
                'Indian Stocks': ['TCS', 'ICICIBANK', 'MARUTI', 'DLF', 'NESTLEIND', 'RELIANCE', 'SBI', 'SUNPHARMA', 'DRREDDY'],
                'Global Markets': ['DOWJONES'],
                'Commodities': ['GOLD', 'SILVER', 'CRUDE'],
                'Cryptocurrency': ['BTC']
            }
            
            selected_category = st.selectbox('📂 Select Category:', list(symbol_categories.keys()))
            
            if selected_category:
                symbol_options = symbol_categories[selected_category]
                selected_symbol = st.selectbox('🎯 Choose Symbol:', symbol_options)
                
                # Custom symbol input
                custom_symbol = st.text_input('✏️ Or enter custom symbol:', max_chars=10)
                symbol = custom_symbol.upper() if custom_symbol else selected_symbol
                
                # Get symbol info for dynamic defaults
                symbol_info = get_symbol_info(symbol)
                trading_hours = get_trading_hours(symbol)
                
                # Display symbol information
                st.markdown(f"""
                <div class="info-box">
                    <strong>📊 {symbol_info['name']}</strong><br>
                    <small>Sector: {symbol_info['sector']}</small><br>
                    <small>Currency: {symbol_info['currency']}</small><br>
                    <small>Trading: {trading_hours['start_hour']:02d}:{trading_hours['start_minute']:02d} - {trading_hours['end_hour']:02d}:{trading_hours['end_minute']:02d}</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Price input with dynamic default
                starting_price = st.number_input(
                    f'💰 Starting Price ({symbol_info["currency"]}):',
                    min_value=0.01,
                    value=symbol_info['default_price'],
                    step=1.0 if symbol_info['default_price'] > 100 else 0.01,
                    format="%.2f"
                )
                
                # Date/time selection based on chart type
                if section_name == 'Intraday Chart':
                    selected_date = st.date_input(
                        '📅 Select Trading Date:',
                        value=datetime(2025, 8, 5).date(),
                        min_value=datetime(2020, 1, 1).date(),
                        max_value=datetime(2030, 12, 31).date()
                    )
                elif section_name == 'Monthly Chart':
                    col1, col2 = st.columns(2)
                    with col1:
                        selected_month = st.selectbox(
                            '📅 Month:',
                            range(1, 13),
                            format_func=lambda x: calendar.month_name[x],
                            index=7  # August
                        )
                    with col2:
                        selected_year = st.selectbox(
                            '📅 Year:',
                            range(2020, 2031),
                            index=5  # 2025
                        )
        
        # Trading insights
        st.markdown("---")
        st.markdown("### 🔮 Quick Insights")
        
        # Generate today's aspects for sidebar display
        aspects = generate_todays_aspects()
        bullish_count = sum(1 for aspect in aspects if aspect['type'] == 'bullish')
        bearish_count = sum(1 for aspect in aspects if aspect['type'] == 'bearish')
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🟢 Bullish Aspects", bullish_count)
        with col2:
            st.metric("🔴 Bearish Aspects", bearish_count)
        
        # Market sentiment
        if bullish_count > bearish_count:
            sentiment = "🟢 Bullish"
            sentiment_color = "green"
        elif bearish_count > bullish_count:
            sentiment = "🔴 Bearish"
            sentiment_color = "red"
        else:
            sentiment = "🟡 Neutral"
            sentiment_color = "orange"
        
        st.markdown(f"**Overall Sentiment:** <span style='color:{sentiment_color}'>{sentiment}</span>", 
                   unsafe_allow_html=True)

    # Main content area
    aspects = generate_todays_aspects()
    
    if section_name == 'Summary Table':
        st.header('📋 Market Summary & Astrological Overview')
        
        # Create columns for better layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader('📊 Today\'s Astrological Aspects Impact')
            summary_df = create_summary_table(aspects)
            
            # Style the dataframe
            styled_df = summary_df.style.apply(
                lambda x: ['background-color: #d4edda' if 'Bullish' in str(val) or '+' in str(val) 
                          else 'background-color: #f8d7da' if 'Bearish' in str(val) or 'Downside' in str(val)
                          else '' for val in x], axis=0
            )
            
            st.dataframe(styled_df, use_container_width=True)
        
        with col2:
            st.subheader('🎯 Key Metrics')
            
            # Calculate impact scores
            total_impact = sum(abs(aspect['impact']) for aspect in aspects)
            avg_impact = total_impact / len(aspects) if aspects else 0
            
            st.metric("Total Cosmic Energy", f"{total_impact:.1f}")
            st.metric("Average Impact", f"{avg_impact:.2f}")
            st.metric("Active Aspects", len(aspects))
            
            # Risk assessment
            high_risk_aspects = sum(1 for aspect in aspects if abs(aspect['impact']) > 0.7)
            risk_level = "High" if high_risk_aspects >= 3 else "Medium" if high_risk_aspects >= 1 else "Low"
            risk_color = "red" if risk_level == "High" else "orange" if risk_level == "Medium" else "green"
            
            st.markdown(f"**Risk Level:** <span style='color:{risk_color}'>{risk_level}</span>", 
                       unsafe_allow_html=True)
        
        # Detailed insights
        st.subheader('🔮 Detailed Market Insights')
        
        # Create tabs for different insights
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Strategy", "📈 Opportunities", "⚠️ Risks", "🌟 Cosmic Events"])
        
        with tab1:
            st.markdown("""
            **🎯 Recommended Trading Strategy:**
            
            **🟢 Bullish Opportunities:**
            - **Energy Sector**: Mars-Uranus conjunction favors Reliance, Crude Oil
            - **Precious Metals**: Multiple aspects support Gold and Silver
            - **FMCG & Pharma**: Moon-Neptune trine provides defensive strength
            - **PSU Stocks**: Sun-Pluto sextile indicates structural positives
            
            **🔴 Bearish Risks:**
            - **Banking Sector**: Mercury-Jupiter square creates volatility
            - **Automotive & Realty**: Venus-Saturn opposition brings pressure
            - **Technology**: Mixed signals, trade with caution
            
            **⚡ High-Impact Trades:**
            - Consider Gold positions during Venus-Saturn opposition
            - Energy stocks may see sharp moves (Mars-Uranus)
            - BTC could be volatile but trending up on global aspects
            """)
        
        with tab2:
            st.markdown("""
            **📈 Sector-wise Opportunities:**
            
            **🥇 Top Picks:**
            1. **Gold/Silver**: Multiple supportive aspects across all planetary configurations
            2. **Energy Commodities**: Mars-Uranus conjunction + global supply dynamics
            3. **Pharmaceutical**: Moon-Neptune trine supports defensive healthcare
            4. **PSU Banking**: Sun-Pluto sextile for structural transformation
            
            **🎯 Specific Symbols:**
            - **GOLD**: $2,050+ target on safe-haven demand
            - **CRUDE**: Energy transition + Mars-Uranus = volatility opportunities
            - **BTC**: Crypto favorable on Uranus-Pluto aspects
            - **SBI**: PSU transformation theme
            """)
        
        with tab3:
            st.markdown("""
            **⚠️ Risk Management:**
            
            **🔴 High-Risk Sectors:**
            - **Private Banking**: ICICI Bank under Mercury-Jupiter square pressure
            - **Automotive**: Maruti facing Venus-Saturn headwinds
            - **Real Estate**: DLF vulnerable to credit tightening aspects
            
            **📊 Risk Mitigation:**
            - Reduce position sizes during Mercury-Jupiter square (high volatility)
            - Use stop-losses 2-3% below support for Venus-Saturn affected stocks
            - Avoid leveraged positions in Midcap segment (Mars-Uranus volatility)
            
            **⏰ Timing Risks:**
            - Morning session volatility expected (Mercury aspects)
            - Post-lunch session may see pressure (Saturn influence)
            """)
        
        with tab4:
            st.markdown("""
            **🌟 Today's Cosmic Events Schedule:**
            
            **🌅 Pre-Market (Before 9:15 AM):**
            - Mercury-Jupiter square builds tension
            - Global markets influence domestic opening
            
            **🌄 Morning Session (9:15-12:00):**
            - Initial volatility from Mercury aspects
            - Energy stocks may show strength
            
            **🌞 Afternoon Session (12:00-15:30):**
            - Venus-Saturn opposition peaks
            - Defensive sectors gain relative strength
            - Banking sector under pressure
            
            **🌆 Post-Market:**
            - Global commodity movements (Gold, Crude)
            - Crypto markets reaction to day's developments
            
            **📊 Weekly Outlook:**
            - Aspects intensify mid-week
            - Weekend planetary shifts to monitor
            """)
    
    elif section_name == 'Stock Filter':
        st.header('🔍 Advanced Stock Filtering & Sector Analysis')
        
        # Create main layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader('🌟 Today\'s Astrological Configuration')
            
            # Display aspects in a nice format
            aspects_data = []
            for aspect in aspects:
                aspects_data.append({
                    'Planets': aspect['planets'],
                    'Aspect': aspect['aspect_type'],
                    'Impact': f"{aspect['impact']:+.1f}",
                    'Sentiment': aspect['type'].title(),
                    'Strength': '🔥' * min(3, int(abs(aspect['impact']) * 3))
                })
            
            aspects_df = pd.DataFrame(aspects_data)
            
            # Color code the dataframe
            def color_sentiment(val):
                if 'Bullish' in str(val):
                    return 'background-color: #d4edda; color: #155724'
                elif 'Bearish' in str(val):
                    return 'background-color: #f8d7da; color: #721c24'
                return ''
            
            styled_aspects = aspects_df.style.applymap(color_sentiment, subset=['Sentiment'])
            st.dataframe(styled_aspects, use_container_width=True)
        
        with col2:
            st.subheader('📊 Aspect Statistics')
            
            # Create a simple pie chart for aspect types
            aspect_types = {}
            for aspect in aspects:
                aspect_types[aspect['type']] = aspect_types.get(aspect['type'], 0) + 1
            
            if aspect_types:
                fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
                colors = ['green' if k == 'bullish' else 'red' if k == 'bearish' else 'gray' 
                         for k in aspect_types.keys()]
                wedges, texts, autotexts = ax_pie.pie(aspect_types.values(), 
                                                     labels=[k.title() for k in aspect_types.keys()], 
                                                     colors=colors, autopct='%1.0f%%', startangle=90)
                ax_pie.set_title('Today\'s Aspect Distribution')
                st.pyplot(fig_pie)
        
        # Sector analysis
        st.subheader('📈 Sector Impact Analysis')
        
        filtered_stocks = filter_stocks_by_aspects(aspects, STOCK_DATABASE)
        
        # Create sector impact visualization
        sector_impacts_df = pd.DataFrame({
            'Sector': list(filtered_stocks['sector_impacts'].keys()),
            'Impact Score': list(filtered_stocks['sector_impacts'].values()),
            'Recommendation': ['Strong Buy' if x > 0.5 else 'Buy' if x > 0 else 'Hold' if x == 0 
                             else 'Sell' if x > -0.5 else 'Strong Sell' 
                             for x in filtered_stocks['sector_impacts'].values()]
        })
        sector_impacts_df = sector_impacts_df.sort_values('Impact Score', ascending=False)
        
        # Enhanced bar chart
        fig_sectors, ax_sectors = plt.subplots(figsize=(14, 8))
        colors = ['darkgreen' if x > 0.5 else 'green' if x > 0 else 'gray' if x == 0 
                 else 'red' if x > -0.5 else 'darkred' 
                 for x in sector_impacts_df['Impact Score']]
        
        bars = ax_sectors.bar(sector_impacts_df['Sector'], sector_impacts_df['Impact Score'], 
                             color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax_sectors.set_title('Sector Impact Scores - Astrological Analysis', fontsize=16, pad=20)
        ax_sectors.set_ylabel('Impact Score', fontsize=12)
        ax_sectors.set_xlabel('Sector', fontsize=12)
        ax_sectors.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax_sectors.grid(True, alpha=0.3)
        
        # Add value labels and recommendations
        for i, (bar, rec) in enumerate(zip(bars, sector_impacts_df['Recommendation'])):
            height = bar.get_height()
            ax_sectors.annotate(f'{height:.2f}\n{rec}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 5 if height >= 0 else -25),
                               textcoords="offset points",
                               ha='center', va='bottom' if height >= 0 else 'top',
                               fontsize=9, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig_sectors)
        
        # Stock recommendations in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader('🟢 Bullish Stocks')
            if not filtered_stocks['bullish'].empty:
                bullish_df = filtered_stocks['bullish'][['Symbol', 'Sector', 'Impact Score']].copy()
                bullish_df['Action'] = bullish_df['Impact Score'].apply(
                    lambda x: 'Strong Buy' if x > 0.5 else 'Buy'
                )
                
                for _, row in bullish_df.iterrows():
                    with st.container():
                        st.markdown(f"""
                        **{row['Symbol']}** ({row['Sector']})  
                        Score: {row['Impact Score']:.2f} | **{row['Action']}**
                        """)
                        st.progress(min(1.0, row['Impact Score']))
            else:
                st.info("No strong bullish signals today")
        
        with col2:
            st.subheader('🔴 Bearish Stocks')
            if not filtered_stocks['bearish'].empty:
                bearish_df = filtered_stocks['bearish'][['Symbol', 'Sector', 'Impact Score']].copy()
                bearish_df['Action'] = bearish_df['Impact Score'].apply(
                    lambda x: 'Strong Sell' if x > 0.5 else 'Sell'
                )
                
                for _, row in bearish_df.iterrows():
                    with st.container():
                        st.markdown(f"""
                        **{row['Symbol']}** ({row['Sector']})  
                        Risk Score: {row['Impact Score']:.2f} | **{row['Action']}**
                        """)
                        st.progress(min(1.0, row['Impact Score']))
            else:
                st.info("No strong bearish signals today")
        
        with col3:
            st.subheader('⚪ Neutral Stocks')
            if not filtered_stocks['neutral'].empty:
                neutral_df = filtered_stocks['neutral'][['Symbol', 'Sector']].head(5)
                
                for _, row in neutral_df.iterrows():
                    st.markdown(f"**{row['Symbol']}** ({row['Sector']}) - Hold")
            else:
                st.info("All stocks showing directional bias")
    
    elif section_name == 'Aspect Analysis':
        st.header('📋 Deep Astrological Aspect Analysis')
        
        # Generate enhanced analysis
        fig, df_aspects = analyze_aspects()
        st.pyplot(fig)
        
        # Display detailed aspect table
        st.subheader('📊 Detailed Aspect Reference Table')
        
        # Add more columns for better analysis
        df_enhanced = df_aspects.copy()
        df_enhanced['Trading Action'] = df_enhanced.apply(
            lambda row: 'Hedge/Reduce' if 'Bearish' in row['Market Impact'] or 'Tension' in row['Market Impact']
            else 'Accumulate' if 'Bullish' in row['Market Impact'] or 'Rally' in row['Market Impact']
            else 'Monitor', axis=1
        )
        
        df_enhanced['Risk Level'] = df_enhanced['Typical Price Change'].apply(
            lambda x: 'High' if any(num in x for num in ['3', '4']) 
            else 'Medium' if '2' in x else 'Low'
        )
        
        # Style the enhanced dataframe
        def highlight_risk(val):
            if val == 'High':
                return 'background-color: #ffcccc'
            elif val == 'Medium':
                return 'background-color: #ffffcc'
            elif val == 'Low':
                return 'background-color: #ccffcc'
            return ''
        
        def highlight_action(val):
            if val == 'Accumulate':
                return 'background-color: #d4edda; color: #155724'
            elif val == 'Hedge/Reduce':
                return 'background-color: #f8d7da; color: #721c24'
            elif val == 'Monitor':
                return 'background-color: #fff3cd; color: #856404'
            return ''
        
        styled_enhanced = df_enhanced.style.applymap(highlight_risk, subset=['Risk Level']).applymap(highlight_action, subset=['Trading Action'])
        st.dataframe(styled_enhanced, use_container_width=True)
        
        # Aspect interpretation guide
        st.subheader('🔭 Astrological Aspect Interpretation Guide')
        
        tab1, tab2, tab3 = st.tabs(["🌟 Aspect Types", "🪐 Planetary Influences", "📈 Trading Applications"])
        
        with tab1:
            st.markdown("""
            ### Understanding Astrological Aspects
            
            **🔄 Conjunction (0°)**: 
            - *Market Effect*: Powerful combining of energies, can create sharp moves
            - *Trading*: Expect significant price action, potential breakouts
            - *Example*: Mars-Uranus conjunction = explosive energy moves
            
            **⚔️ Square (90°)**: 
            - *Market Effect*: Tension, conflict, volatility
            - *Trading*: Increased intraday swings, good for scalping
            - *Example*: Mercury-Jupiter square = communication/policy confusion
            
            **🎯 Trine (120°)**: 
            - *Market Effect*: Harmonious, easy flow of energy
            - *Trading*: Trending moves, good for position trading
            - *Example*: Moon-Neptune trine = emotional/intuitive support
            
            **⚖️ Opposition (180°)**: 
            - *Market Effect*: Polarization, requires balance
            - *Trading*: Range-bound action, reversals possible
            - *Example*: Venus-Saturn opposition = value vs. restriction
            
            **🤝 Sextile (60°)**: 
            - *Market Effect*: Opportunity aspects, mild positive
            - *Trading*: Gentle trends, good for swing trades
            - *Example*: Sun-Pluto sextile = gradual transformation
            """)
        
        with tab2:
            st.markdown("""
            ### Planetary Market Influences
            
            **☀️ Sun**: Leadership, government policy, large-cap stocks, gold
            **🌙 Moon**: Public sentiment, emotions, consumer sectors, silver
            **☿️ Mercury**: Communication, technology, volatility, news-driven moves
            **♀️ Venus**: Finance, banking, luxury goods, relationships, copper
            **♂️ Mars**: Energy, metals, defense, aggressive moves, oil
            **♃ Jupiter**: Growth, expansion, optimism, financial sector
            **♄ Saturn**: Restriction, discipline, structure, defensive sectors
            **♅ Uranus**: Innovation, technology, sudden changes, crypto
            **♆ Neptune**: Illusion, oil, pharma, confusion, speculation
            **♇ Pluto**: Transformation, power, mining, major shifts
            
            ### Sector-Planet Correlations
            - **Technology**: Mercury, Uranus
            - **Banking**: Jupiter, Venus, Saturn  
            - **Energy**: Mars, Sun, Pluto
            - **Healthcare**: Neptune, Moon
            - **Precious Metals**: Venus, Jupiter, Sun
            - **Cryptocurrency**: Uranus, Pluto
            """)
        
        with tab3:
            st.markdown("""
            ### Practical Trading Applications
            
            **📊 Intraday Trading:**
            - Use Moon aspects for sentiment shifts (2-4 hour cycles)
            - Mercury aspects for news/volatility spikes
            - Mars aspects for energy sector breakouts
            
            **📈 Swing Trading:**
            - Venus aspects for financial sector trends (3-7 days)
            - Jupiter aspects for broad market optimism
            - Saturn aspects for defensive positioning
            
            **📉 Position Trading:**
            - Outer planet aspects (Uranus, Neptune, Pluto) for long-term themes
            - Eclipse patterns for major sector rotations
            - Retrograde periods for trend reversals
            
            **⚠️ Risk Management:**
            - Increase cash during multiple challenging aspects
            - Reduce position size during Mercury retrograde
            - Use tighter stops during Mars-Saturn squares
            
            **🎯 Sector Rotation:**
            - Follow Jupiter through zodiac signs for sector leadership
            - Track Saturn aspects for value opportunities
            - Monitor Uranus for innovation themes
            """)
    
    elif section_name == 'Intraday Chart':
        st.header(f'📈 {symbol} - Intraday Astrological Analysis')
        
        # Display symbol information prominently
        symbol_info = get_symbol_info(symbol)
        trading_hours = get_trading_hours(symbol)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Symbol", symbol)
        with col2:
            st.metric("Sector", symbol_info['sector'])
        with col3:
            st.metric("Currency", symbol_info['currency'])
        with col4:
            session_length = trading_hours['end_hour'] - trading_hours['start_hour'] + \
                           (trading_hours['end_minute'] - trading_hours['start_minute']) / 60
            st.metric("Session Hours", f"{session_length:.1f}h")
        
        # Generate and display chart
        with st.spinner(f'Generating astrological analysis for {symbol}...'):
            fig = generate_intraday_chart(symbol, starting_price, selected_date)
            st.pyplot(fig)
        
        # Trading insights based on symbol
        st.subheader(f'🎯 {symbol} Trading Insights')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Technical Outlook")
            
            # Generate insights based on symbol type
            if symbol in ['GOLD', 'SILVER']:
                st.markdown("""
                **Precious Metals Analysis:**
                - Multiple planetary aspects favor safe-haven demand
                - Venus-Saturn opposition creates financial stress → Gold strength  
                - Moon-Neptune trine supports intuitive precious metal buying
                - Best trading windows: During global uncertainty aspects
                
                **Key Levels:**
                - Watch for breakouts during Mars-Uranus conjunction
                - Support likely during Moon aspects
                - Resistance at previous highs during Saturn aspects
                """)
            
            elif symbol in ['BTC']:
                st.markdown("""
                **Cryptocurrency Analysis:**
                - Uranus aspects strongly favor crypto volatility
                - Mars-Uranus conjunction = explosive price moves
                - Traditional financial stress (Venus-Saturn) → Crypto rotation
                - High volatility expected - use proper risk management
                
                **Trading Strategy:**
                - Momentum plays during Uranus aspects
                - Contrarian plays during Saturn oppositions
                - Volume spikes likely at aspect peaks
                """)
            
            elif symbol in ['CRUDE']:
                st.markdown("""
                **Energy Commodity Analysis:**
                - Mars-Uranus conjunction directly impacts energy sector
                - Global supply disruption themes (Pluto aspects)
                - Geopolitical tensions favor energy prices
                - Weather and seasonal patterns amplified by aspects
                
                **Supply-Demand Factors:**
                - Production disruptions during Mars aspects
                - Demand surges during economic aspects
                - Storage plays during Saturn aspects
                """)
            
            elif symbol in ['DOWJONES']:
                st.markdown("""
                **US Index Analysis:**
                - Jupiter aspects favor broad market optimism
                - Saturn aspects create rotation into defensive sectors
                - Mercury aspects increase intraday volatility
                - Fed policy sensitivity during Venus-Saturn opposition
                
                **Sector Rotation:**
                - Technology during Mercury aspects
                - Energy during Mars aspects  
                - Financials during Jupiter aspects
                """)
            
            else:  # Indian stocks
                st.markdown(f"""
                **{symbol_info['sector']} Sector Analysis:**
                - Domestic market influenced by global planetary patterns
                - FII/DII flows affected by Venus-Saturn aspects
                - Sector rotation based on planetary emphasis
                - Currency impacts during outer planet aspects
                
                **Indian Market Specifics:**
                - Opening gap influenced by global overnight aspects
                - Lunch hour volatility during Moon aspects
                - Closing session strength during Jupiter aspects
                """)
        
        with col2:
            st.markdown("#### ⏰ Timing Analysis")
            
            # Generate time-specific insights based on trading hours
            if trading_hours['end_hour'] > 16:  # Extended hours
                st.markdown("""
                **Extended Session Analysis:**
                
                **🌅 Asian Session (5:00-8:00):**
                - Pre-market positioning based on overnight aspects
                - Lower volumes, higher impact from aspects
                - Key economic data releases amplified
                
                **🌍 European Session (8:00-16:00):**
                - Peak liquidity and aspect impacts
                - Central bank policy influences
                - Cross-asset correlations strongest
                
                **🌎 US Session (16:00-20:00):**
                - Maximum volatility potential
                - Aspect peaks create significant moves
                - News flow interaction with cosmic patterns
                
                **🌙 After Hours (20:00-23:55):**
                - Reduced liquidity amplifies aspect effects
                - Position adjustments for next day
                - Asian preview impact
                """)
            else:  # Indian market hours
                st.markdown("""
                **Indian Session Analysis:**
                
                **🌅 Opening (9:15-10:30):**
                - Gap opening based on global aspects
                - High volatility, aspect impacts magnified
                - Initial trend direction setting
                
                **🌞 Mid-Morning (10:30-12:00):**
                - Institutional activity peaks
                - Aspect-driven sector rotation
                - News flow integration
                
                **🍽️ Lunch Hour (12:00-13:00):**
                - Reduced activity, Moon aspects dominate
                - Range-bound unless strong aspects active
                - Position consolidation period
                
                **🌆 Closing (13:00-15:30):**
                - Final institutional positioning
                - Aspect resolution for day
                - Next-day setup formation
                """)
            
            # Risk management
            st.markdown("#### ⚠️ Risk Management")
            st.markdown(f"""
            **Position Sizing:**
            - Standard position: 1-2% of capital
            - High aspect days: Reduce to 0.5-1%
            - Strong confluence: Increase to 2-3%
            
            **Stop Loss Levels:**
            - Tight stops during Mercury aspects: 1-2%
            - Normal stops during stable aspects: 2-3%
            - Wide stops during Mars aspects: 3-5%
            
            **Profit Targets:**
            - Quick scalps: 0.5-1% (15-30 minutes)
            - Swing trades: 2-5% (2-4 hours)
            - Position trades: 5-10% (1-3 days)
            
            **Volatility Adjustments:**
            - {symbol}: Expected daily range ±{2.5 if symbol in ['BTC'] else 1.5 if symbol in ['CRUDE'] else 1.0 if symbol in ['GOLD', 'SILVER'] else 0.8}%
            - Adjust position size inversely to volatility
            - Use options for high-volatility periods
            """)
    
    elif section_name == 'Monthly Chart':
        st.header(f'📊 {symbol} - Monthly Astrological Trend Analysis')
        
        # Display symbol information
        symbol_info = get_symbol_info(symbol)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Symbol", symbol)
        with col2:
            st.metric("Analysis Period", f"{calendar.month_name[selected_month]} {selected_year}")
        with col3:
            st.metric("Sector Focus", symbol_info['sector'])
        with col4:
            st.metric("Currency", symbol_info['currency'])
        
        # Generate and display chart
        with st.spinner(f'Generating monthly analysis for {symbol}...'):
            fig = generate_monthly_chart(symbol, starting_price, selected_month, selected_year)
            st.pyplot(fig)
        
        # Monthly analysis insights
        st.subheader(f'📈 {calendar.month_name[selected_month]} {selected_year} - Strategic Analysis')
        
        # Create tabs for different analysis aspects
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Monthly Outlook", "📊 Technical Analysis", "🌙 Lunar Cycles", "💼 Portfolio Strategy"])
        
        with tab1:
            month_name = calendar.month_name[selected_month]
            
            if symbol in ['GOLD', 'SILVER']:
                st.markdown(f"""
                ### {symbol} - {month_name} {selected_year} Precious Metals Outlook
                
                **🌟 Astrological Themes:**
                - **Venus-Jupiter aspects**: Strong precious metals demand from financial uncertainty
                - **Saturn transits**: Safe-haven buying during economic restrictions
                - **Moon phases**: Emotional buying patterns aligned with lunar cycles
                - **Mercury retrograde periods**: Technical analysis less reliable, fundamentals dominate
                
                **📈 Price Drivers:**
                - Central bank policy uncertainty (Saturn aspects)
                - Currency devaluation themes (Pluto aspects)
                - Geopolitical tensions (Mars aspects)
                - Inflation hedging demand (Jupiter-Saturn aspects)
                
                **🎯 Trading Strategy:**
                - **Accumulate** during New Moon phases (stronger buying interest)
                - **Profit-take** during Full Moon phases (emotional peaks)
                - **Hold through** Mercury retrograde (avoid technical trading)
                - **Scale in** during Saturn aspects (structural support)
                
                **📊 Target Levels:**
                - **Monthly High**: Expect during Jupiter-Venus trine periods
                - **Monthly Low**: Likely during Mars-Saturn square periods
                - **Breakout Potential**: Mars-Uranus conjunction periods
                - **Support Zones**: Previous month's Jupiter aspect levels
                """)
            
            elif symbol in ['BTC']:
                st.markdown(f"""
                ### {symbol} - {month_name} {selected_year} Cryptocurrency Outlook
                
                **⚡ Astrological Themes:**
                - **Uranus-Pluto aspects**: Revolutionary technology adoption waves
                - **Mercury-Uranus aspects**: Network upgrades and technical developments
                - **Mars-Uranus conjunctions**: Explosive price movements and FOMO
                - **Saturn aspects**: Regulatory clarity or restrictions
                
                **🚀 Volatility Drivers:**
                - Institutional adoption news (Jupiter aspects)
                - Regulatory developments (Saturn aspects)
                - Technical network changes (Mercury-Uranus)
                - Market manipulation concerns (Neptune aspects)
                
                **⚠️ Risk Factors:**
                - **High volatility** during Mars-Uranus aspects (±10-20% daily swings)
                - **Regulatory risks** during Saturn-Pluto aspects
                - **Technical failures** during Mercury retrograde
                - **Market manipulation** during Neptune-Mercury aspects
                
                **💡 Strategic Approach:**
                - **DCA strategy** during volatile periods
                - **Momentum trading** during Uranus aspects
                - **Risk-off** during Saturn hard aspects
                - **HODL mentality** during Jupiter-Pluto trines
                """)
            
            elif symbol in ['CRUDE']:
                st.markdown(f"""
                ### {symbol} - {month_name} {selected_year} Energy Commodity Outlook
                
                **🛢️ Astrological Themes:**
                - **Mars-Pluto aspects**: Geopolitical tensions affecting supply
                - **Jupiter-Saturn cycles**: Economic growth vs. restriction cycles
                - **Uranus aspects**: Renewable energy transition impacts
                - **Moon phases**: Seasonal demand patterns amplified
                
                **⚡ Supply-Demand Dynamics:**
                - Production disruptions (Mars-Saturn squares)
                - Economic growth spurts (Jupiter aspects)
                - Weather pattern extremes (Uranus-Neptune aspects)
                - Strategic reserve changes (Pluto aspects)
                
                **🌍 Geopolitical Factors:**
                - **OPEC decisions** aligned with Saturn aspects
                - **Pipeline disruptions** during Mars-Uranus periods
                - **Currency impacts** during Venus-Pluto aspects
                - **Seasonal patterns** enhanced by lunar cycles
                
                **📈 Trading Levels:**
                - **Resistance**: Previous Jupiter aspect highs
                - **Support**: Saturn aspect consolidation zones
                - **Breakout zones**: Mars-Uranus conjunction levels
                - **Reversal points**: Full Moon technical confluences
                """)
            
            elif symbol in ['DOWJONES']:
                st.markdown(f"""
                ### {symbol} - {month_name} {selected_year} US Index Outlook
                
                **🇺🇸 Macro Astrological Themes:**
                - **Jupiter-Saturn cycles**: Economic expansion vs. contraction
                - **Mercury-Venus aspects**: Corporate earnings and consumer spending
                - **Mars-Jupiter aspects**: Business investment and growth
                - **Outer planet aspects**: Long-term structural changes
                
                **🏛️ Federal Reserve Alignment:**
                - **Venus-Saturn aspects**: Interest rate policy changes
                - **Mercury-Jupiter aspects**: Fed communication clarity
                - **Moon phases**: Market sentiment around FOMC meetings
                - **Eclipse periods**: Major policy shift announcements
                
                **🔄 Sector Rotation Patterns:**
                - **Technology** leadership during Mercury-Uranus aspects
                - **Energy** strength during Mars-Pluto periods
                - **Financials** favor during Venus-Jupiter trines
                - **Healthcare** defensive during Saturn aspects
                
                **📊 Technical Confluence:**
                - **Monthly resistance**: Jupiter aspect previous highs
                - **Monthly support**: Saturn aspect previous lows
                - **Breakout potential**: New Moon near technical levels
                - **Reversal zones**: Full Moon at key Fibonacci levels
                """)
            
            else:  # Indian stocks
                st.markdown(f"""
                ### {symbol} ({symbol_info['sector']}) - {month_name} {selected_year} Indian Market Outlook
                
                **🇮🇳 Domestic Astrological Influences:**
                - **Jupiter transits**: Market leadership and FII flows
                - **Saturn aspects**: Regulatory changes and policy shifts
                - **Mars-Venus aspects**: Consumer spending and investment flows
                - **Moon phases**: Retail investor sentiment cycles
                
                **💹 Sector-Specific Themes:**
                - **{symbol_info['sector']} sector** influenced by specific planetary combinations
                - **Monsoon patterns** (if applicable) aligned with water sign emphasis
                - **Festival seasons** amplified by benefic planetary aspects
                - **Budget impacts** during Saturn-Jupiter aspects
                
                **🌏 Global Correlation Factors:**
                - **US Fed policy** impacts during Venus-Saturn aspects
                - **China growth** concerns during Mars-Saturn periods  
                - **Oil prices** affecting through Mars-Pluto aspects
                - **Dollar strength** impacts during Pluto aspects
                
                **📈 Monthly Strategy:**
                - **Accumulate** during Saturn aspects (value opportunities)
                - **Momentum plays** during Mars-Jupiter periods
                - **Defensive positioning** during challenging outer planet aspects
                - **Sector rotation** based on planetary emphasis shifts
                """)
        
        with tab2:
            st.markdown(f"""
            ### Technical Analysis Integration with Astrological Cycles
            
            **📊 Moving Average Alignment:**
            - **MA5 vs MA20**: Bullish when Jupiter aspects dominate
            - **Golden Cross** potential during Venus-Jupiter trines
            - **Death Cross** risk during Saturn-Mars squares
            - **MA support/resistance** stronger during lunar phases
            
            **🎯 Support & Resistance Levels:**
            - **Primary resistance**: Previous month's Jupiter aspect highs
            - **Primary support**: Saturn aspect consolidation lows
            - **Secondary levels**: Full Moon reversal points
            - **Breakout levels**: New Moon momentum points
            
            **📈 Momentum Indicators:**
            - **RSI overbought** (>70) more reliable during Full Moons
            - **RSI oversold** (<30) stronger signal during New Moons
            - **MACD divergences** amplified during Mercury aspects
            - **Volume confirmations** critical during Mars aspects
            
            **🌙 Lunar Cycle Technical Correlation:**
            - **New Moon**: Trend initiation, breakout potential
            - **Waxing Moon**: Momentum continuation, bullish bias
            - **Full Moon**: Trend exhaustion, reversal potential
            - **Waning Moon**: Correction phases, consolidation
            
            **⚡ Volatility Patterns:**
            - **Highest volatility**: Mars-Uranus aspect periods
            - **Lowest volatility**: Venus-Jupiter trine periods
            - **Unexpected moves**: Mercury-Neptune confusion aspects
            - **Gap movements**: Eclipse and outer planet aspects
            
            **🔄 Pattern Recognition:**
            - **Triangle breakouts** during Uranus aspects
            - **Flag patterns** during Mars aspects  
            - **Head & Shoulders** during Saturn aspects
            - **Double tops/bottoms** during opposition aspects
            """)
        
        with tab3:
            st.markdown(f"""
            ### Lunar Cycles & Market Psychology for {month_name} {selected_year}
            
            **🌑 New Moon Phases (Market Initiation):**
            - **Energy**: Fresh starts, new trend beginnings
            - **Psychology**: Optimism, risk-taking increases
            - **Trading**: Look for breakout setups, trend initiations
            - **Volume**: Often lower but quality moves
            - **Best for**: Opening new positions, trend following
            
            **🌓 Waxing Moon (Building Momentum):**
            - **Energy**: Growth, expansion, building confidence  
            - **Psychology**: FOMO starts building, bullish sentiment
            - **Trading**: Momentum continuation, pyramid additions
            - **Volume**: Increasing participation
            - **Best for**: Adding to winning positions
            
            **🌕 Full Moon Phases (Emotional Peaks):**
            - **Energy**: Maximum emotion, extremes, reversals
            - **Psychology**: Euphoria or panic peaks
            - **Trading**: Reversal setups, profit-taking
            - **Volume**: Often highest of cycle
            - **Best for**: Profit booking, contrarian plays
            
            **🌗 Waning Moon (Consolidation):**
            - **Energy**: Release, correction, cooling off
            - **Psychology**: Reality check, risk assessment
            - **Trading**: Consolidation patterns, value hunting
            - **Volume**: Declining, selective moves
            - **Best for**: Position adjustments, planning
            
            **🔮 {month_name} {selected_year} Specific Lunar Events:**
            
            **Key Lunar Dates to Watch:**
            - **New Moon**: Potential trend change or continuation signal
            - **First Quarter**: Momentum confirmation or failure
            - **Full Moon**: Profit-taking opportunity or reversal signal  
            - **Last Quarter**: Consolidation phase or weakness signal
            
            **Moon Sign Influences:**
            - **Fire Signs** (Aries, Leo, Sagittarius): Aggressive moves, energy sector strength
            - **Earth Signs** (Taurus, Virgo, Capricorn): Value focus, stability preference
            - **Air Signs** (Gemini, Libra, Aquarius): Communication, technology emphasis
            - **Water Signs** (Cancer, Scorpio, Pisces): Emotional decisions, defensive moves
            """)
        
        with tab4:
            st.markdown(f"""
            ### Portfolio Strategy for {month_name} {selected_year}
            
            **🎯 Strategic Asset Allocation:**
            
            **Core Holdings (50-60%):**
            - **Large Cap Stability**: Jupiter-aspected blue chips
            - **Sector Leaders**: Dominant players in favored sectors
            - **Defensive Assets**: During challenging aspect periods
            - **Currency Hedge**: If significant Pluto aspects present
            
            **Growth Opportunities (20-30%):**
            - **Momentum Plays**: Mars-Jupiter aspect beneficiaries
            - **Breakout Candidates**: Technical + astrological confluence
            - **Sector Rotation**: Following planetary emphasis shifts
            - **Emerging Themes**: Uranus aspect innovation plays
            
            **Speculative/Trading (10-20%):**
            - **High Beta Names**: For Mars-Uranus periods
            - **Volatility Plays**: Options during aspect peaks
            - **Contrarian Bets**: Against crowd during extremes
            - **Crypto Allocation**: If comfortable with high volatility
            
            **📊 Risk Management Framework:**
            
            **Position Sizing Rules:**
            - **Maximum single position**: 5% during stable periods
            - **Reduce to 3%**: During challenging aspects
            - **Increase to 7%**: During strong favorable confluences
            - **Cash levels**: 10-20% based on aspect favorability
            
            **Stop Loss Strategy:**
            - **Tight stops** (3-5%): During Mercury retrograde periods
            - **Normal stops** (5-8%): During regular market conditions
            - **Wide stops** (8-12%): During high volatility aspect periods
            - **No stops**: For long-term Jupiter-blessed positions
            
            **📅 Monthly Rebalancing Schedule:**
            
            **Week 1**: Review and adjust based on new lunar cycle
            **Week 2**: Add to momentum winners if aspects support
            **Week 3**: Prepare for Full Moon profit-taking opportunities
            **Week 4**: Position for next month's astrological themes
            
            **🔄 Sector Rotation Strategy:**
            
            **Early Month**: Follow Jupiter aspects for growth sectors
            **Mid Month**: Mars aspects may favor energy/materials
            **Late Month**: Venus aspects support financials/consumer
            **Month End**: Saturn aspects favor defensives/utilities
            
            **💡 Advanced Strategies:**
            
            **Pairs Trading**: Long favored sectors, short challenged sectors
            **Options Overlay**: Sell calls during Full Moons, buy calls during New Moons
            **Currency Hedge**: Hedge foreign exposure during Pluto aspects
            **Volatility Trading**: Long volatility before aspect peaks
            
            **📈 Performance Tracking:**
            
            **Monthly Metrics**:
            - Absolute return vs. benchmark
            - Risk-adjusted return (Sharpe ratio)
            - Maximum drawdown during challenging aspects
            - Hit rate on astrological predictions
            
            **Aspect Correlation Analysis**:
            - Track which aspects work best for {symbol}
            - Note sector rotation timing accuracy
            - Measure volatility prediction success
            - Document lunar cycle correlations
            """)
        
        # Additional insights for monthly strategy
        st.subheader('🎭 Market Psychology & Sentiment Analysis')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            #### 🧠 Psychological Drivers - {month_name}
            
            **Institutional Behavior:**
            - Month-end window dressing effects
            - Quarterly rebalancing influences  
            - Earnings season psychological impacts
            - Fed meeting anticipation/reaction
            
            **Retail Investor Patterns:**
            - Payroll cycle investment flows
            - Tax implications (if year-end)
            - Holiday season spending impacts
            - Social media sentiment amplification
            
            **Global Sentiment Factors:**
            - US-China trade relationship status
            - European economic data impacts
            - Emerging market flow dynamics
            - Cryptocurrency correlation effects
            """)
        
        with col2:
            st.markdown(f"""
            #### 📊 Sentiment Indicators to Watch
            
            **Technical Sentiment:**
            - VIX levels and term structure
            - Put/Call ratios by sector
            - High-low index readings
            - Advance-decline line trends
            
            **Fundamental Sentiment:**
            - Earnings revision trends
            - Analyst recommendation changes
            - Insider buying/selling activity
            - Share buyback announcements
            
            **Alternative Data:**
            - Google search trends
            - Social media mention analysis
            - Options flow analysis
            - Crypto correlation strength
            """)

# Footer with additional information
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <h4>🌟 Disclaimer & Important Notes</h4>
        <p><strong>Educational Purpose Only:</strong> This dashboard is for educational and research purposes. 
        Astrological analysis should be combined with fundamental and technical analysis for trading decisions.</p>
        
        <p><strong>Risk Warning:</strong> All trading involves risk. Past performance and astrological correlations 
        do not guarantee future results. Always use proper risk management and never invest more than you can afford to lose.</p>
        
        <p><strong>Data Sources:</strong> Simulated price data based on astrological aspect calculations. 
        For live trading, use real market data and professional trading platforms.</p>
        
        <p style='font-size: 12px; margin-top: 20px;'>
        🔮 <em>"The stars impel, they do not compel. Wisdom lies in using all available tools - 
        fundamental, technical, and cosmic - for informed decision making."</em>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main(), 'default_price': 82.50, 'sector': 'Energy'},
    'BTC': {'name': 'Bitcoin', 'currency': '

# --- STOCK DATABASE ---
stock_data = {
    'Symbol': [
        'TCS', 'ICICIBANK', 'MARUTI', 'DLF', 'NESTLEIND', 
        'RELIANCE', 'SBI', 'SUNPHARMA', 'DRREDDY', 'GOLD',
        'DOWJONES', 'SILVER', 'CRUDE', 'BTC'
    ],
    'Sector': [
        'Technology', 'Banking', 'Automotive', 'Realty', 'FMCG',
        'Energy', 'PSUs', 'Pharma', 'Pharma', 'Precious Metals',
        'US Index', 'Precious Metals', 'Energy', 'Cryptocurrency'
    ],
    'MarketCap': [
        'Large', 'Large', 'Large', 'Large', 'Large',
        'Large', 'Large', 'Large', 'Large', 'Commodity',
        'Index', 'Commodity', 'Commodity', 'Crypto'
    ]
}

STOCK_DATABASE = pd.DataFrame(stock_data)

# --- SECTOR-PLANETARY MAPPINGS ---
SECTOR_PLANETARY_INFLUENCES = {
    'Technology': ['Mercury'],
    'Banking': ['Jupiter', 'Saturn'],
    'FMCG': ['Moon'],
    'Pharma': ['Neptune'],
    'Energy': ['Mars'],
    'Automotive': ['Saturn'],
    'Realty': ['Saturn'],
    'PSUs': ['Pluto'],
    'Midcaps': ['Uranus'],
    'Smallcaps': ['Pluto'],
    'Precious Metals': ['Venus', 'Jupiter'],
    'US Index': ['Jupiter', 'Saturn'],
    'Cryptocurrency': ['Uranus', 'Pluto']
}

# --- ASPECT-SECTOR IMPACT ---
ASPECT_SECTOR_IMPACTS = {
    'Square': {
        'Technology': 'Negative', 'Banking': 'Negative', 'FMCG': 'Neutral',
        'Pharma': 'Neutral', 'Energy': 'Neutral', 'Automotive': 'Neutral',
        'Realty': 'Neutral', 'PSUs': 'Neutral', 'Midcaps': 'Neutral',
        'Smallcaps': 'Neutral', 'Precious Metals': 'Positive', 'US Index': 'Negative',
        'Cryptocurrency': 'Negative'
    },
    'Opposition': {
        'Technology': 'Neutral', 'Banking': 'Neutral', 'FMCG': 'Neutral',
        'Pharma': 'Neutral', 'Energy': 'Neutral', 'Automotive': 'Negative',
        'Realty': 'Negative', 'PSUs': 'Neutral', 'Midcaps': 'Neutral',
        'Smallcaps': 'Neutral', 'Precious Metals': 'Positive', 'US Index': 'Neutral',
        'Cryptocurrency': 'Positive'
    },
    'Trine': {
        'Technology': 'Neutral', 'Banking': 'Neutral', 'FMCG': 'Positive',
        'Pharma': 'Positive', 'Energy': 'Neutral', 'Automotive': 'Neutral',
        'Realty': 'Neutral', 'PSUs': 'Neutral', 'Midcaps': 'Neutral',
        'Smallcaps': 'Neutral', 'Precious Metals': 'Positive', 'US Index': 'Positive',
        'Cryptocurrency': 'Neutral'
    },
    'Conjunction': {
        'Technology': 'Neutral', 'Banking': 'Neutral', 'FMCG': 'Neutral',
        'Pharma': 'Neutral', 'Energy': 'Positive', 'Automotive': 'Neutral',
        'Realty': 'Neutral', 'PSUs': 'Neutral', 'Midcaps': 'Negative',
        'Smallcaps': 'Neutral', 'Precious Metals': 'Positive', 'US Index': 'Neutral',
        'Cryptocurrency': 'Positive'
    },
    'Sextile': {
        'Technology': 'Neutral', 'Banking': 'Neutral', 'FMCG': 'Neutral',
        'Pharma': 'Neutral', 'Energy': 'Neutral', 'Automotive': 'Neutral',
        'Realty': 'Neutral', 'PSUs': 'Positive', 'Midcaps': 'Neutral',
        'Smallcaps': 'Negative', 'Precious Metals': 'Positive', 'US Index': 'Positive',
        'Cryptocurrency': 'Neutral'
    }
}

# --- PLANETARY POSITION VISUALIZATION ---
def draw_planetary_wheel(ax, input_date, size=0.3):
    """Draw a simplified astrological wheel showing planetary positions"""
    base_date = datetime(2025, 8, 1)
    
    if isinstance(input_date, date_class) and not isinstance(input_date, datetime):
        date_obj = datetime.combine(input_date, datetime.min.time())
    else:
        date_obj = input_date
    
    days_diff = (date_obj.date() - base_date.date()).days
    
    base_positions = {
        'Sun': 135, 'Moon': 225, 'Mercury': 120, 'Venus': 170,
        'Mars': 85, 'Jupiter': 45, 'Saturn': 315
    }
    
    daily_movement = {
        'Sun': 1.0, 'Moon': 13.2, 'Mercury': 1.5, 'Venus': 1.2,
        'Mars': 0.5, 'Jupiter': 0.08, 'Saturn': 0.03
    }
    
    planets = {}
    for planet, base_pos in base_positions.items():
        angle = (base_pos + daily_movement[planet] * days_diff) % 360
        planets[planet] = {
            'angle': angle,
            'color': {
                'Sun': 'gold', 'Moon': 'silver', 'Mercury': 'gray',
                'Venus': 'lightgreen', 'Mars': 'red', 'Jupiter': 'orange',
                'Saturn': 'darkgoldenrod'
            }[planet],
            'size': {
                'Sun': 8, 'Moon': 6, 'Mercury': 5, 'Venus': 7,
                'Mars': 6, 'Jupiter': 10, 'Saturn': 9
            }[planet]
        }
    
    zodiac = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 
              'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
    
    for i, sign in enumerate(zodiac):
        angle = i * 30
        ax.add_patch(Wedge((0, 0), size, angle, angle+30, width=size*0.8, 
                          facecolor='lightgray', edgecolor='black', alpha=0.3))
        ax.text(0.85*size * np.cos(np.radians(angle+15)), 
                0.85*size * np.sin(np.radians(angle+15)), 
                sign[:3], ha='center', va='center', fontsize=6)
    
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
    ax.set_title(f'Planetary Positions\n{date_obj.strftime("%b %d, %Y")}', fontsize=8)

# --- GET TRADING HOURS FOR SYMBOL ---
def get_trading_hours(symbol):
    """Get trading hours for a specific symbol"""
    symbol = symbol.upper()
    if symbol in TRADING_HOURS:
        return TRADING_HOURS[symbol]
    else:
        # Default to Indian market hours for unknown symbols
        return TRADING_HOURS['NIFTY']

# --- GET SYMBOL INFO ---
def get_symbol_info(symbol):
    """Get symbol configuration info"""
    symbol = symbol.upper()
    if symbol in SYMBOL_CONFIG:
        return SYMBOL_CONFIG[symbol]
    else:
        # Default configuration
        return {
            'name': symbol,
            'currency': '₹',
            'default_price': 1000.0,
            'sector': 'Unknown'
        }

# --- GENERATE ASPECTS ---
def generate_todays_aspects():
    """Generate astrological aspects for today based on the provided table"""
    base_aspects = [
        {"planets": "Mercury-Jupiter", "aspect_type": "Square", "impact": -0.7, "type": "bearish"},
        {"planets": "Venus-Saturn", "aspect_type": "Opposition", "impact": -0.8, "type": "bearish"},
        {"planets": "Moon-Neptune", "aspect_type": "Trine", "impact": 0.6, "type": "bullish"},
        {"planets": "Mars-Uranus", "aspect_type": "Conjunction", "impact": 0.9, "type": "bullish"},
        {"planets": "Sun-Pluto", "aspect_type": "Sextile", "impact": 0.5, "type": "bullish"}
    ]
    
    aspects = []
    for aspect in base_aspects:
        aspects.append({
            "planets": aspect["planets"],
            "aspect_type": aspect["aspect_type"],
            "impact": aspect["impact"],
            "type": aspect["type"]
        })
    
    return aspects

# --- CREATE SUMMARY TABLE ---
def create_summary_table(aspects):
    """Create a summary table based on the astrological aspects"""
    summary_data = {
        'Aspect': [],
        'Nifty/Bank Nifty': [],
        'Bullish Sectors/Stocks': [],
        'Bearish Sectors/Stocks': []
    }
    
    for aspect in aspects:
        planets = aspect["planets"]
        aspect_type = aspect["aspect_type"]
        
        if planets == "Mercury-Jupiter" and aspect_type == "Square":
            summary_data['Aspect'].append("Mercury-Jupiter (Square)")
            summary_data['Nifty/Bank Nifty'].append("Volatile")
            summary_data['Bullish Sectors/Stocks'].append("IT (TCS), Gold")
            summary_data['Bearish Sectors/Stocks'].append("Banking (ICICI Bank), Crypto")
        
        elif planets == "Venus-Saturn" and aspect_type == "Opposition":
            summary_data['Aspect'].append("Venus-Saturn (Opposition)")
            summary_data['Nifty/Bank Nifty'].append("Downside")
            summary_data['Bullish Sectors/Stocks'].append("Gold, Silver, BTC")
            summary_data['Bearish Sectors/Stocks'].append("Auto (Maruti), Realty (DLF)")
        
        elif planets == "Moon-Neptune" and aspect_type == "Trine":
            summary_data['Aspect'].append("Moon-Neptune (Trine)")
            summary_data['Nifty/Bank Nifty'].append("Mild Support")
            summary_data['Bullish Sectors/Stocks'].append("FMCG (Nestlé), Pharma, Gold, Dow Jones")
            summary_data['Bearish Sectors/Stocks'].append("-")
        
        elif planets == "Mars-Uranus" and aspect_type == "Conjunction":
            summary_data['Aspect'].append("Mars-Uranus (Conjunction)")
            summary_data['Nifty/Bank Nifty'].append("Sharp Moves")
            summary_data['Bullish Sectors/Stocks'].append("Energy (Reliance, Crude), Gold, BTC")
            summary_data['Bearish Sectors/Stocks'].append("Weak Midcaps")
        
        elif planets == "Sun-Pluto" and aspect_type == "Sextile":
            summary_data['Aspect'].append("Sun-Pluto (Sextile)")
            summary_data['Nifty/Bank Nifty'].append("Structural Shift")
            summary_data['Bullish Sectors/Stocks'].append("PSUs (SBI), Gold, Dow Jones")
            summary_data['Bearish Sectors/Stocks'].append("Overvalued Smallcaps")
    
    return pd.DataFrame(summary_data)

# --- FILTER STOCKS BASED ON ASPECTS ---
def filter_stocks_by_aspects(aspects, stock_database):
    """Filter stocks based on today's astrological aspects"""
    sector_impacts = {sector: 0 for sector in stock_database['Sector'].unique()}
    
    for aspect in aspects:
        planet1, planet2 = aspect["planets"].split("-")
        
        for sector, planets in SECTOR_PLANETARY_INFLUENCES.items():
            if planet1 in planets or planet2 in planets:
                if sector not in sector_impacts:
                    sector_impacts[sector] = 0
                
                aspect_impact = ASPECT_SECTOR_IMPACTS[aspect["aspect_type"]].get(sector, "Neutral")
                
                if aspect_impact == "Positive":
                    sector_impacts[sector] += abs(aspect["impact"])
                elif aspect_impact == "Negative":
                    sector_impacts[sector] -= abs(aspect["impact"])
    
    bullish_sectors = [sector for sector, impact in sector_impacts.items() if impact > 0]
    bearish_sectors = [sector for sector, impact in sector_impacts.items() if impact < 0]
    neutral_sectors = [sector for sector, impact in sector_impacts.items() if impact == 0]
    
    bullish_stocks = stock_database[stock_database['Sector'].isin(bullish_sectors)].copy()
    bearish_stocks = stock_database[stock_database['Sector'].isin(bearish_sectors)].copy()
    neutral_stocks = stock_database[stock_database['Sector'].isin(neutral_sectors)].copy()
    
    bullish_stocks['Impact Score'] = bullish_stocks['Sector'].apply(lambda x: sector_impacts[x])
    bearish_stocks['Impact Score'] = bearish_stocks['Sector'].apply(lambda x: abs(sector_impacts[x]))
    neutral_stocks['Impact Score'] = 0
    
    bullish_stocks = bullish_stocks.sort_values('Impact Score', ascending=False)
    bearish_stocks = bearish_stocks.sort_values('Impact Score', ascending=False)
    
    return {
        'bullish': bullish_stocks,
        'bearish': bearish_stocks,
        'neutral': neutral_stocks,
        'sector_impacts': sector_impacts
    }

# --- GENERATE ASTROLOGICAL EVENTS ---
def generate_astrological_events(input_date, event_type='intraday', symbol='NIFTY'):
    """Generate astrological events for any given date and symbol"""
    
    if event_type == 'intraday':
        trading_hours = get_trading_hours(symbol)
        
        # Different event patterns based on trading hours
        if trading_hours['end_hour'] > 16:  # Extended hours (global markets)
            # More events spread across longer trading day
            base_events = [
                {"time_offset": 0, "aspect": "Pre-market: Mercury square Jupiter", "impact": -0.5, "type": "bearish"},
                {"time_offset": 120, "aspect": "Asian session: Moon trine Jupiter", "impact": 0.8, "type": "bullish"},
                {"time_offset": 240, "aspect": "London open: Mars sextile Jupiter", "impact": 0.4, "type": "neutral"},
                {"time_offset": 360, "aspect": "European session: Venus opposition Saturn", "impact": -0.6, "type": "bearish"},
                {"time_offset": 480, "aspect": "NY pre-market: Sun conjunct Mercury", "impact": 0.3, "type": "neutral"},
                {"time_offset": 600, "aspect": "US open: Mars conjunct Uranus", "impact": 1.0, "type": "bullish"},
                {"time_offset": 720, "aspect": "Mid-day: Moon square Saturn", "impact": -0.4, "type": "bearish"},
                {"time_offset": 840, "aspect": "Afternoon: Jupiter trine Neptune", "impact": 0.7, "type": "bullish"},
                {"time_offset": 960, "aspect": "US close approach", "impact": 0.2, "type": "neutral"},
                {"time_offset": 1080, "aspect": "After hours: Void Moon", "impact": -0.3, "type": "bearish"},
                {"time_offset": 1135, "aspect": "Session close", "impact": 0.1, "type": "neutral"}
            ]
        else:  # Standard Indian market hours
            base_events = [
                {"time_offset": 0, "aspect": "Opening: Mercury square Jupiter + Void Moon", "impact": -0.5, "type": "bearish"},
                {"time_offset": 45, "aspect": "Early trade: Moon trine Jupiter", "impact": 1.0, "type": "bullish"},
                {"time_offset": 135, "aspect": "Mid-morning: Mars sextile Jupiter", "impact": 0.3, "type": "neutral"},
                {"time_offset": 195, "aspect": "Pre-lunch: Sun in Leo (no aspects)", "impact": 0.0, "type": "neutral"},
                {"time_offset": 285, "aspect": "Post-lunch: Moon square Saturn", "impact": -0.8, "type": "bearish"},
                {"time_offset": 345, "aspect": "Late trade: Venus-Saturn opposition", "impact": -0.6, "type": "bearish"},
                {"time_offset": 375, "aspect": "Close", "impact": 0.1, "type": "neutral"}
            ]
        
        events = []
        if isinstance(input_date, date_class) and not isinstance(input_date, datetime):
            dt = datetime.combine(input_date, datetime.min.time())
        else:
            dt = input_date
            
        start_time = dt.replace(hour=trading_hours['start_hour'], minute=trading_hours['start_minute'])
        
        for event in base_events:
            event_time = start_time + timedelta(minutes=event["time_offset"])
            events.append({
                "time": event_time,
                "aspect": event["aspect"],
                "impact": event["impact"],
                "type": event["type"],
                "price": 0
            })
        
        return events
    
    else:  # monthly events remain the same
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
        
        if isinstance(input_date, datetime):
            year, month = input_date.year, input_date.month
        else:
            year, month = input_date.year, input_date.month
            
        days_in_month = calendar.monthrange(year, month)[1]
        
        events = []
        for event in base_events:
            day = min(event["day_offset"], days_in_month)
            event_date = datetime(year, month, day)
            events.append({
                "date": event_date,
                "aspect": event["aspect"],
                "impact": event["impact"],
                "type": event["type"],
                "price": 0
            })
        
        return events

# --- ENHANCED INTRADAY CHART ---
def generate_intraday_chart(symbol, starting_price, selected_date):
    """Generate enhanced intraday chart with dynamic layout"""
    symbol_info = get_symbol_info(symbol)
    trading_hours = get_trading_hours(symbol)
    
    if isinstance(selected_date, date_class) and not isinstance(selected_date, datetime):
        selected_date = datetime.combine(selected_date, datetime.min.time())
    
    start_time = selected_date.replace(hour=trading_hours['start_hour'], minute=trading_hours['start_minute'])
    end_time = selected_date.replace(hour=trading_hours['end_hour'], minute=trading_hours['end_minute'])
    
    # Adjust interval based on trading session length
    session_hours = (end_time - start_time).total_seconds() / 3600
    if session_hours > 12:
        interval = '30T'  # 30-minute intervals for long sessions
    else:
        interval = '15T'  # 15-minute intervals for shorter sessions
    
    times = pd.date_range(start=start_time, end=end_time, freq=interval)
    
    prices = np.zeros(len(times))
    base_price = starting_price
    
    events = generate_astrological_events(selected_date, 'intraday', symbol)
    
    # Adjust event impacts based on symbol type
    symbol_multiplier = 1.0
    if symbol in ['GOLD', 'SILVER']:
        symbol_multiplier = 0.8  # Precious metals less volatile to aspects
    elif symbol in ['BTC']:
        symbol_multiplier = 2.0  # Crypto more volatile
    elif symbol in ['CRUDE']:
        symbol_multiplier = 1.5  # Energy commodities more responsive
    
    for event in events:
        price_change = event["impact"] * base_price * 0.01 * symbol_multiplier
        event["price"] = base_price + price_change
    
    # Generate price movements
    for i, time in enumerate(times):
        closest_event = min(events, key=lambda x: abs((x["time"] - time).total_seconds()))
        distance = abs((closest_event["time"] - time).total_seconds()) / 3600
        
        # Adjust volatility based on symbol
        base_volatility = 0.15 if distance < 0.5 else 0.05
        if symbol in ['BTC']:
            base_volatility *= 3.0
        elif symbol in ['GOLD', 'SILVER']:
            base_volatility *= 0.5
        elif symbol in ['CRUDE']:
            base_volatility *= 2.0
        
        random_change = np.random.normal(0, base_volatility)
        event_influence = closest_event["impact"] * np.exp(-distance) * symbol_multiplier
        
        if i == 0:
            prices[i] = base_price
        else:
            change = (event_influence + random_change) * base_price * 0.001
            prices[i] = prices[i-1] + change
    
    df_intraday = pd.DataFrame({
        'Time': times,
        'Price': prices,
        'Aspect': [min(events, key=lambda x: abs((x["time"] - t).total_seconds()))["aspect"] for t in times]
    })
    
    # Create dynamic figure layout
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, height_ratios=[4, 1, 1, 1], width_ratios=[5, 2, 1])
    
    # Main price chart
    ax_main = fig.add_subplot(gs[0, :2])
    
    for i in range(1, len(df_intraday)):
        color = 'green' if df_intraday['Price'].iloc[i] > df_intraday['Price'].iloc[i-1] else 'red'
        ax_main.plot(df_intraday['Time'].iloc[i-1:i+1], 
                    df_intraday['Price'].iloc[i-1:i+1], 
                    color=color, linewidth=2.5)
    
    # Mark key events
    for event in events:
        color_map = {'bullish': 'green', 'bearish': 'red', 'neutral': 'blue'}
        ax_main.axvline(x=event['time'], color=color_map[event['type']], 
                       linestyle='--', alpha=0.7, linewidth=2)
        
        marker_color = color_map[event['type']]
        ax_main.scatter(event['time'], event['price'], color=marker_color, 
                       s=100, zorder=5, edgecolor='black', linewidth=1)
        
        # Dynamic annotation positioning
        y_offset = base_price * 0.02 if len(str(int(base_price))) >= 4 else base_price * 0.05
        y_pos = event['price'] + y_offset if event['price'] < base_price * 1.01 else event['price'] - y_offset
        
        ax_main.annotate(event['aspect'][:30] + '...' if len(event['aspect']) > 30 else event['aspect'], 
                        xy=(event['time'], event['price']),
                        xytext=(event['time'], y_pos),
                        arrowprops=dict(arrowstyle='->', color=marker_color, lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                        fontsize=8, ha='center')
    
    # Dynamic title and formatting
    currency_symbol = symbol_info['currency']
    ax_main.set_title(f'{symbol_info["name"]} ({symbol}) - {selected_date.strftime("%B %d, %Y")}\n'
                     f'Astrological Trading Analysis | Sector: {symbol_info["sector"]}', 
                     fontsize=16, pad=20)
    ax_main.set_xlabel(f'Time ({trading_hours["start_hour"]}:00 - {trading_hours["end_hour"]}:00)', fontsize=12)
    ax_main.set_ylabel(f'Price ({currency_symbol})', fontsize=12)
    
    # Dynamic time formatting based on session length
    if session_hours > 12:
        ax_main.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    else:
        ax_main.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    plt.setp(ax_main.get_xticklabels(), rotation=45, ha='right')
    
    # Closing price annotation
    close_price = df_intraday["Price"].iloc[-1]
    price_change = close_price - base_price
    price_change_pct = (price_change / base_price) * 100
    
    ax_main.annotate(f'Close: {currency_symbol}{close_price:.2f}\n'
                    f'Change: {price_change:+.2f} ({price_change_pct:+.2f}%)', 
                    xy=(df_intraday['Time'].iloc[-1], close_price),
                    xytext=(df_intraday['Time'].iloc[-1] - timedelta(hours=session_hours*0.2), 
                           close_price + base_price * 0.02),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                    fontsize=11, fontweight='bold',
                    bbox=dict(facecolor='yellow', alpha=0.7, edgecolor='none', pad=2))
    
    # Planetary wheel
    ax_wheel = fig.add_subplot(gs[0, 2])
    draw_planetary_wheel(ax_wheel, selected_date, size=0.4)
    
    # Volume chart (simulated with realistic patterns)
    ax_volume = fig.add_subplot(gs[1, :2])
    
    # Generate more realistic volume based on symbol type
    if symbol in ['BTC']:
        base_volume = np.random.randint(50000, 200000, size=len(times))
    elif symbol in ['GOLD', 'SILVER', 'CRUDE']:
        base_volume = np.random.randint(10000, 50000, size=len(times))
    elif symbol in ['DOWJONES']:
        base_volume = np.random.randint(100000, 500000, size=len(times))
    else:  # Indian stocks
        base_volume = np.random.randint(1000, 10000, size=len(times))
    
    colors_volume = ['green' if df_intraday['Price'].iloc[i] > df_intraday['Price'].iloc[i-1] 
                    else 'red' for i in range(1, len(df_intraday))]
    colors_volume.insert(0, 'green')
    
    ax_volume.bar(df_intraday['Time'], base_volume, color=colors_volume, alpha=0.7)
    ax_volume.set_title('Volume', fontsize=12)
    ax_volume.set_ylabel('Volume', fontsize=10)
    ax_volume.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax_volume.get_xticklabels(), rotation=45, ha='right')
    
    # Technical indicators (RSI simulation)
    ax_rsi = fig.add_subplot(gs[2, :2])
    rsi_values = 50 + np.random.normal(0, 15, len(times))  # Simulated RSI
    rsi_values = np.clip(rsi_values, 0, 100)
    
    ax_rsi.plot(df_intraday['Time'], rsi_values, color='purple', linewidth=2)
    ax_rsi.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
    ax_rsi.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
    ax_rsi.fill_between(df_intraday['Time'], 30, 70, alpha=0.1, color='gray')
    ax_rsi.set_title('RSI (14)', fontsize=12)
    ax_rsi.set_ylabel('RSI', fontsize=10)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.legend(loc='upper right', fontsize=8)
    
    # Aspect strength indicator
    ax_aspect = fig.add_subplot(gs[3, :2])
    aspect_times = [event['time'] for event in events]
    aspect_strengths = [abs(event['impact']) for event in events]
    aspect_colors = [{'bullish': 'green', 'bearish': 'red', 'neutral': 'blue'}[event['type']] for event in events]
    
    ax_aspect.scatter(aspect_times, aspect_strengths, color=aspect_colors, s=100, zorder=3)
    ax_aspect.plot(aspect_times, aspect_strengths, color='gray', alpha=0.5, linestyle='--')
    ax_aspect.set_title('Astrological Aspect Strength', fontsize=12)
    ax_aspect.set_ylabel('Strength', fontsize=10)
    ax_aspect.set_ylim(0, max(aspect_strengths) * 1.2 if aspect_strengths else 1.5)
    
    # Symbol info panel
    ax_info = fig.add_subplot(gs[1:, 2])
    ax_info.axis('off')
    
    info_text = f"""
SYMBOL INFO
-----------
Name: {symbol_info['name']}
Sector: {symbol_info['sector']}
Currency: {symbol_info['currency']}

TRADING HOURS
-------------
Start: {trading_hours['start_hour']:02d}:{trading_hours['start_minute']:02d}
End: {trading_hours['end_hour']:02d}:{trading_hours['end_minute']:02d}
Session: {session_hours:.1f} hours

PRICE DATA
----------
Open: {currency_symbol}{base_price:.2f}
Close: {currency_symbol}{close_price:.2f}
Change: {price_change:+.2f}
Change%: {price_change_pct:+.2f}%

High: {currency_symbol}{max(prices):.2f}
Low: {currency_symbol}{min(prices):.2f}
Range: {currency_symbol}{max(prices)-min(prices):.2f}
"""
    
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig

# --- ENHANCED MONTHLY CHART ---
def generate_monthly_chart(symbol, starting_price, selected_month, selected_year):
    """Generate enhanced monthly chart with dynamic layout"""
    symbol_info = get_symbol_info(symbol)
    
    start_date = datetime(selected_year, selected_month, 1)
    days_in_month = calendar.monthrange(selected_year, selected_month)[1]
    end_date = datetime(selected_year, selected_month, days_in_month)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    prices = np.zeros(len(dates))
    base_price = starting_price
    
    events = generate_astrological_events(start_date, 'monthly', symbol)
    
    # Adjust event impacts based on symbol type
    symbol_multiplier = 1.0
    if symbol in ['GOLD', 'SILVER']:
        symbol_multiplier = 0.8
    elif symbol in ['BTC']:
        symbol_multiplier = 2.5
    elif symbol in ['CRUDE']:
        symbol_multiplier = 1.8
    
    for event in events:
        price_change = event["impact"] * base_price * 0.01 * symbol_multiplier
        event["price"] = base_price + price_change
    
    # Generate price movements
    for i, date in enumerate(dates):
        closest_event = min(events, key=lambda x: abs((x["date"].date() - date.date()).days))
        distance = abs((closest_event["date"].date() - date.date()).days)
        
        base_volatility = 0.3 if distance < 2 else 0.1
        if symbol in ['BTC']:
            base_volatility *= 4.0
        elif symbol in ['GOLD', 'SILVER']:
            base_volatility *= 0.6
        elif symbol in ['CRUDE']:
            base_volatility *= 2.5
        
        random_change = np.random.normal(0, base_volatility)
        event_influence = closest_event["impact"] * np.exp(-distance/2) * symbol_multiplier
        
        if i == 0:
            prices[i] = base_price
        else:
            change = (event_influence + random_change) * base_price * 0.002
            prices[i] = prices[i-1] + change
    
    df_monthly = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Aspect': [min(events, key=lambda x: abs((x["date"].date() - d.date()).days))["aspect"] for d in dates]
    })
    
    # Create dynamic figure layout
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, height_ratios=[4, 1, 1, 1], width_ratios=[5, 2, 1])
    
    # Main price chart
    ax_main = fig.add_subplot(gs[0, :2])
    
    for i in range(1, len(df_monthly)):
        color = 'green' if df_monthly['Price'].iloc[i] > df_monthly['Price'].iloc[i-1] else 'red'
        ax_main.plot(df_monthly['Date'].iloc[i-1:i+1], 
                    df_monthly['Price'].iloc[i-1:i+1], 
                    color=color, linewidth=3)
    
    # Mark key events
    for event in events:
        color_map = {'bullish': 'green', 'bearish': 'red', 'neutral': 'blue'}
        ax_main.axvline(x=event['date'], color=color_map[event['type']], 
                       linestyle='--', alpha=0.7, linewidth=2)
        
        marker_color = color_map[event['type']]
        ax_main.scatter(event['date'], event['price'], color=marker_color, 
                       s=150, zorder=5, edgecolor='black', linewidth=1.5)
        
        y_offset = base_price * 0.03
        y_pos = event['price'] + y_offset if event['price'] < base_price * 1.02 else event['price'] - y_offset
        
        ax_main.annotate(event['aspect'][:25] + '...' if len(event['aspect']) > 25 else event['aspect'], 
                        xy=(event['date'], event['price']),
                        xytext=(event['date'], y_pos),
                        arrowprops=dict(arrowstyle='->', color=marker_color, lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                        fontsize=9, ha='center')
    
    # Dynamic formatting
    currency_symbol = symbol_info['currency']
    ax_main.set_title(f'{symbol_info["name"]} ({symbol}) - {start_date.strftime("%B %Y")}\n'
                     f'Monthly Astrological Analysis | Sector: {symbol_info["sector"]}', 
                     fontsize=16, pad=20)
    ax_main.set_xlabel('Date', fontsize=12)
    ax_main.set_ylabel(f'Price ({currency_symbol})', fontsize=12)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax_main.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, days_in_month//10)))
    plt.setp(ax_main.get_xticklabels(), rotation=45, ha='right')
    
    # Monthly close annotation
    close_price = df_monthly["Price"].iloc[-1]
    price_change = close_price - base_price
    price_change_pct = (price_change / base_price) * 100
    
    ax_main.annotate(f'Month Close: {currency_symbol}{close_price:.2f}\n'
                    f'Monthly Change: {price_change:+.2f} ({price_change_pct:+.2f}%)', 
                    xy=(df_monthly['Date'].iloc[-1], close_price),
                    xytext=(df_monthly['Date'].iloc[-1] - timedelta(days=days_in_month//4), 
                           close_price + base_price * 0.03),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                    fontsize=12, fontweight='bold',
                    bbox=dict(facecolor='yellow', alpha=0.7, edgecolor='none', pad=2))
    
    # Planetary positions for key dates
    ax_planets = fig.add_subplot(gs[0, 2])
    ax_planets.set_title('Key Planetary\nPositions', fontsize=10)
    key_dates = [
        start_date,
        start_date + timedelta(days=days_in_month//3),
        start_date + timedelta(days=2*days_in_month//3),
        end_date
    ]
    
    for i, date in enumerate(key_dates):
        ax_sub = fig.add_axes([0.70, 0.8-i*0.15, 0.12, 0.12])
        draw_planetary_wheel(ax_sub, date, size=0.4)
        ax_sub.set_title(f'{date.strftime("%b %d")}', fontsize=8)
    
    # Volume chart
    ax_volume = fig.add_subplot(gs[1, :2])
    
    if symbol in ['BTC']:
        volume = np.random.randint(500000, 2000000, size=len(dates))
    elif symbol in ['GOLD', 'SILVER', 'CRUDE']:
        volume = np.random.randint(100000, 500000, size=len(dates))
    elif symbol in ['DOWJONES']:
        volume = np.random.randint(1000000, 5000000, size=len(dates))
    else:
        volume = np.random.randint(10000, 100000, size=len(dates))
    
    colors_volume = ['green' if df_monthly['Price'].iloc[i] > df_monthly['Price'].iloc[i-1] 
                    else 'red' for i in range(1, len(df_monthly))]
    colors_volume.insert(0, 'green')
    
    ax_volume.bar(df_monthly['Date'], volume, color=colors_volume, alpha=0.7)
    ax_volume.set_title('Daily Volume', fontsize=12)
    ax_volume.set_ylabel('Volume', fontsize=10)
    ax_volume.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    # Moving averages
    ax_ma = fig.add_subplot(gs[2, :2])
    ma_5 = df_monthly['Price'].rolling(window=5, min_periods=1).mean()
    ma_20 = df_monthly['Price'].rolling(window=min(20, len(df_monthly)), min_periods=1).mean()
    
    ax_ma.plot(df_monthly['Date'], ma_5, color='blue', linewidth=2, label='MA5', alpha=0.7)
    ax_ma.plot(df_monthly['Date'], ma_20, color='red', linewidth=2, label='MA20', alpha=0.7)
    ax_ma.fill_between(df_monthly['Date'], ma_5, ma_20, alpha=0.1, 
                      color='green' if ma_5.iloc[-1] > ma_20.iloc[-1] else 'red')
    ax_ma.set_title('Moving Averages', fontsize=12)
    ax_ma.set_ylabel('Price', fontsize=10)
    ax_ma.legend(loc='upper left', fontsize=10)
    
    # Aspect calendar
    ax_calendar = fig.add_subplot(gs[3, :2])
    aspect_dates = [event['date'] for event in events]
    aspect_strengths = [abs(event['impact']) for event in events]
    aspect_colors = [{'bullish': 'green', 'bearish': 'red', 'neutral': 'blue'}[event['type']] for event in events]
    
    ax_calendar.scatter(aspect_dates, aspect_strengths, color=aspect_colors, s=200, zorder=3)
    ax_calendar.plot(aspect_dates, aspect_strengths, color='gray', alpha=0.5, linestyle='--')
    ax_calendar.set_title('Monthly Astrological Event Strength', fontsize=12)
    ax_calendar.set_ylabel('Impact Strength', fontsize=10)
    ax_calendar.set_ylim(0, max(aspect_strengths) * 1.2 if aspect_strengths else 2)
    
    # Monthly summary panel
    ax_summary = fig.add_subplot(gs[1:, 2])
    ax_summary.axis('off')
    
    monthly_high = max(prices)
    monthly_low = min(prices)
    monthly_range = monthly_high - monthly_low
    avg_price = np.mean(prices)
    
    summary_text = f"""
MONTHLY SUMMARY
--------------
Symbol: {symbol}
Sector: {symbol_info['sector']}
Month: {start_date.strftime('%B %Y')}

PRICE STATISTICS
---------------
Open: {currency_symbol}{base_price:.2f}
Close: {currency_symbol}{close_price:.2f}
Change: {price_change:+.2f}
Change%: {price_change_pct:+.2f}%

High: {currency_symbol}{monthly_high:.2f}
Low: {currency_symbol}{monthly_low:.2f}
Range: {currency_symbol}{monthly_range:.2f}
Average: {currency_symbol}{avg_price:.2f}

VOLATILITY
----------
Daily Avg: {np.std(np.diff(prices)):.2f}
Monthly Vol: {(monthly_range/avg_price)*100:.1f}%

TREND ANALYSIS
--------------
Bullish Days: {sum(1 for i in range(1, len(prices)) if prices[i] > prices[i-1])}
Bearish Days: {sum(1 for i in range(1, len(prices)) if prices[i] < prices[i-1])}
Neutral Days: {sum(1 for i in range(1, len(prices)) if prices[i] == prices[i-1])}
"""
    
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes, fontsize=8,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    return fig

# --- ANALYZE ASPECTS ---
def analyze_aspects():
    """Enhanced aspect analysis with dynamic content"""
    aspects_data = {
        'Aspect': [
            'Mercury Retrograde', 'Venus Opposition Saturn', 'Moon-Jupiter Trine', 
            'Full Moon', 'Jupiter Square Saturn', 'Mercury Direct',
            'Venus enters Libra', 'New Moon', 'Mars-Uranus Conjunction',
            'Sun-Pluto Sextile'
        ],
        'Market Impact': [
            'High Volatility', 'Bearish Pressure', 'Bullish Surge', 
            'Trend Reversal', 'Major Tension', 'Clarity Returns',
            'Financial Rally', 'Strong Bullish', 'Energy Surge',
            'Structural Change'
        ],
        'Typical Price Change': [
            '±2-3%', '-1.5-2%', '+1-2%', 
            '±1-1.5%', '-2-3%', '+0.5-1%',
            '+0.5-1%', '+1-2%', '+2-4%',
            '±1-2%'
        ],
        'Sector Focus': [
            'All Sectors', 'Banking/Realty', 'Broad Market', 
            'Technology', 'Financials', 'Technology',
            'Banking/Finance', 'Broad Market', 'Energy/Commodities',
            'Infrastructure/PSUs'
        ],
        'Best Symbols': [
            'Gold, BTC', 'Gold, Silver', 'FMCG, Pharma', 
            'Tech Stocks', 'Defensive', 'Tech, Crypto',
            'Banking', 'Growth Stocks', 'Energy, Crude',
            'PSU, Infrastructure'
        ]
    }
    
    df_aspects = pd.DataFrame(aspects_data)
    
    # Create enhanced visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Price change impact chart
    price_changes = []
    for change in df_aspects['Typical Price Change']:
        clean_change = change.replace('%', '').replace('±', '')
        if '-' in clean_change and not clean_change.startswith('-'):
            num_str = clean_change.split('-')[1]  # Take higher value for impact
        else:
            num_str = clean_change.replace('+', '')
        
        try:
            num = float(num_str)
        except:
            num = 1.0
        price_changes.append(num)
    
    colors = ['red' if 'Bearish' in impact or 'Tension' in impact or 'Volatility' in impact 
              else 'orange' if 'Reversal' in impact or 'Change' in impact
              else 'green' for impact in df_aspects['Market Impact']]
    
    bars1 = ax1.bar(range(len(df_aspects)), price_changes, color=colors, alpha=0.7)
    ax1.set_title('Astrological Aspect Impact on Price Changes', fontsize=14)
    ax1.set_ylabel('Maximum Price Change (%)', fontsize=12)
    ax1.set_xticks(range(len(df_aspects)))
    ax1.set_xticklabels(df_aspects['Aspect'], rotation=45, ha='right')
    
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.annotate(f'{height}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    # Sector distribution pie chart
    sector_counts = {}
    for sectors in df_aspects['Sector Focus']:
        for sector in sectors.split('/'):
            sector = sector.strip()
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
    
    ax2.pie(sector_counts.values(), labels=sector_counts.keys(), autopct='%1.1f%%', startangle=90)
    ax2.set_title('Most Affected Sectors by Astrological Aspects', fontsize=14)
    
    # Market impact distribution
    impact_counts = {}
    for impact in df_aspects['Market Impact']:
        impact_type = 'Bullish' if any(word in impact for word in ['Bullish', 'Rally', 'Surge', 'Returns']) else \
                     'Bearish' if any(word in impact for word in ['Bearish', 'Pressure', 'Tension']) else \
                     'Neutral'
        impact_counts[impact_type] = impact_counts.get(impact_type, 0) + 1
    
    colors_impact = ['green', 'red', 'gray']
    ax3.bar(impact_counts.keys(), impact_counts.values(), color=colors_impact, alpha=0.7)
    ax3.set_title('Distribution of Market Impact Types', fontsize=14)
    ax3.set_ylabel('Number of Aspects', fontsize=12)
    
    # Best performing symbols chart
    symbol_mentions = {}
    for symbols in df_aspects['Best Symbols']:
        for symbol in symbols.split(', '):
            symbol = symbol.strip()
            symbol_mentions[symbol] = symbol_mentions.get(symbol, 0) + 1
    
    sorted_symbols = sorted(symbol_mentions.items(), key=lambda x: x[1], reverse=True)
    symbols, counts = zip(*sorted_symbols) if sorted_symbols else ([], [])
    
    ax4.barh(symbols, counts, color='gold', alpha=0.7)
    ax4.set_title('Most Favorable Symbols Across Aspects', fontsize=14)
    ax4.set_xlabel('Favorable Mentions', fontsize=12)
    
    return fig, df_aspects

# --- STREAMLIT APP ---
def main():
    # Page configuration for better responsive design
    st.set_page_config(
        page_title="🌟 Astrological Trading Dashboard",
        page_icon="🌟",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .symbol-input {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #e8f4fd;
        border-left: 5px solid #1f77b4;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🌟 Astrological Trading Dashboard</h1>
        <p>Advanced Financial Analysis through Planetary Movements & Cosmic Patterns</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for inputs with enhanced design
    with st.sidebar:
        st.markdown("### 📊 Dashboard Controls")
        
        # Dashboard section selection with better descriptions
        dashboard_section = st.selectbox(
            '🎯 Choose Analysis Section:',
            [
                'Summary Table - Market Overview',
                'Stock Filter - Sector Analysis', 
                'Aspect Analysis - Deep Insights',
                'Intraday Chart - Live Patterns',
                'Monthly Chart - Trend Analysis'
            ]
        )
        
        # Extract the main section name
        section_name = dashboard_section.split(' - ')[0]
        
        st.markdown("---")
        
        # Symbol selection with enhanced interface
        if section_name in ['Intraday Chart', 'Monthly Chart']:
            st.markdown("### 📈 Symbol Configuration")
            
            # Popular symbols with categories
            symbol_categories = {
                'Indian Indices': ['NIFTY', 'BANKNIFTY'],
                'Indian Stocks': ['TCS', 'ICICIBANK', 'MARUTI', 'DLF', 'NESTLEIND', 'RELIANCE', 'SBI', 'SUNPHARMA', 'DRREDDY'],
                'Global Markets': ['DOWJONES'],
                'Commodities': ['GOLD', 'SILVER', 'CRUDE'],
                'Cryptocurrency': ['BTC']
            }
            
            selected_category = st.selectbox('📂 Select Category:', list(symbol_categories.keys()))
            
            if selected_category:
                symbol_options = symbol_categories[selected_category]
                selected_symbol = st.selectbox('🎯 Choose Symbol:', symbol_options)
                
                # Custom symbol input
                custom_symbol = st.text_input('✏️ Or enter custom symbol:', max_chars=10)
                symbol = custom_symbol.upper() if custom_symbol else selected_symbol
                
                # Get symbol info for dynamic defaults
                symbol_info = get_symbol_info(symbol)
                trading_hours = get_trading_hours(symbol)
                
                # Display symbol information
                st.markdown(f"""
                <div class="info-box">
                    <strong>📊 {symbol_info['name']}</strong><br>
                    <small>Sector: {symbol_info['sector']}</small><br>
                    <small>Currency: {symbol_info['currency']}</small><br>
                    <small>Trading: {trading_hours['start_hour']:02d}:{trading_hours['start_minute']:02d} - {trading_hours['end_hour']:02d}:{trading_hours['end_minute']:02d}</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Price input with dynamic default
                starting_price = st.number_input(
                    f'💰 Starting Price ({symbol_info["currency"]}):',
                    min_value=0.01,
                    value=symbol_info['default_price'],
                    step=1.0 if symbol_info['default_price'] > 100 else 0.01,
                    format="%.2f"
                )
                
                # Date/time selection based on chart type
                if section_name == 'Intraday Chart':
                    selected_date = st.date_input(
                        '📅 Select Trading Date:',
                        value=datetime(2025, 8, 5).date(),
                        min_value=datetime(2020, 1, 1).date(),
                        max_value=datetime(2030, 12, 31).date()
                    )
                elif section_name == 'Monthly Chart':
                    col1, col2 = st.columns(2)
                    with col1:
                        selected_month = st.selectbox(
                            '📅 Month:',
                            range(1, 13),
                            format_func=lambda x: calendar.month_name[x],
                            index=7  # August
                        )
                    with col2:
                        selected_year = st.selectbox(
                            '📅 Year:',
                            range(2020, 2031),
                            index=5  # 2025
                        )
        
        # Trading insights
        st.markdown("---")
        st.markdown("### 🔮 Quick Insights")
        
        # Generate today's aspects for sidebar display
        aspects = generate_todays_aspects()
        bullish_count = sum(1 for aspect in aspects if aspect['type'] == 'bullish')
        bearish_count = sum(1 for aspect in aspects if aspect['type'] == 'bearish')
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🟢 Bullish Aspects", bullish_count)
        with col2:
            st.metric("🔴 Bearish Aspects", bearish_count)
        
        # Market sentiment
        if bullish_count > bearish_count:
            sentiment = "🟢 Bullish"
            sentiment_color = "green"
        elif bearish_count > bullish_count:
            sentiment = "🔴 Bearish"
            sentiment_color = "red"
        else:
            sentiment = "🟡 Neutral"
            sentiment_color = "orange"
        
        st.markdown(f"**Overall Sentiment:** <span style='color:{sentiment_color}'>{sentiment}</span>", 
                   unsafe_allow_html=True)

    # Main content area
    aspects = generate_todays_aspects()
    
    if section_name == 'Summary Table':
        st.header('📋 Market Summary & Astrological Overview')
        
        # Create columns for better layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader('📊 Today\'s Astrological Aspects Impact')
            summary_df = create_summary_table(aspects)
            
            # Style the dataframe
            styled_df = summary_df.style.apply(
                lambda x: ['background-color: #d4edda' if 'Bullish' in str(val) or '+' in str(val) 
                          else 'background-color: #f8d7da' if 'Bearish' in str(val) or 'Downside' in str(val)
                          else '' for val in x], axis=0
            )
            
            st.dataframe(styled_df, use_container_width=True)
        
        with col2:
            st.subheader('🎯 Key Metrics')
            
            # Calculate impact scores
            total_impact = sum(abs(aspect['impact']) for aspect in aspects)
            avg_impact = total_impact / len(aspects) if aspects else 0
            
            st.metric("Total Cosmic Energy", f"{total_impact:.1f}")
            st.metric("Average Impact", f"{avg_impact:.2f}")
            st.metric("Active Aspects", len(aspects))
            
            # Risk assessment
            high_risk_aspects = sum(1 for aspect in aspects if abs(aspect['impact']) > 0.7)
            risk_level = "High" if high_risk_aspects >= 3 else "Medium" if high_risk_aspects >= 1 else "Low"
            risk_color = "red" if risk_level == "High" else "orange" if risk_level == "Medium" else "green"
            
            st.markdown(f"**Risk Level:** <span style='color:{risk_color}'>{risk_level}</span>", 
                       unsafe_allow_html=True)
        
        # Detailed insights
        st.subheader('🔮 Detailed Market Insights')
        
        # Create tabs for different insights
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Strategy", "📈 Opportunities", "⚠️ Risks", "🌟 Cosmic Events"])
        
        with tab1:
            st.markdown("""
            **🎯 Recommended Trading Strategy:**
            
            **🟢 Bullish Opportunities:**
            - **Energy Sector**: Mars-Uranus conjunction favors Reliance, Crude Oil
            - **Precious Metals**: Multiple aspects support Gold and Silver
            - **FMCG & Pharma**: Moon-Neptune trine provides defensive strength
            - **PSU Stocks**: Sun-Pluto sextile indicates structural positives
            
            **🔴 Bearish Risks:**
            - **Banking Sector**: Mercury-Jupiter square creates volatility
            - **Automotive & Realty**: Venus-Saturn opposition brings pressure
            - **Technology**: Mixed signals, trade with caution
            
            **⚡ High-Impact Trades:**
            - Consider Gold positions during Venus-Saturn opposition
            - Energy stocks may see sharp moves (Mars-Uranus)
            - BTC could be volatile but trending up on global aspects
            """)
        
        with tab2:
            st.markdown("""
            **📈 Sector-wise Opportunities:**
            
            **🥇 Top Picks:**
            1. **Gold/Silver**: Multiple supportive aspects across all planetary configurations
            2. **Energy Commodities**: Mars-Uranus conjunction + global supply dynamics
            3. **Pharmaceutical**: Moon-Neptune trine supports defensive healthcare
            4. **PSU Banking**: Sun-Pluto sextile for structural transformation
            
            **🎯 Specific Symbols:**
            - **GOLD**: $2,050+ target on safe-haven demand
            - **CRUDE**: Energy transition + Mars-Uranus = volatility opportunities
            - **BTC**: Crypto favorable on Uranus-Pluto aspects
            - **SBI**: PSU transformation theme
            """)
        
        with tab3:
            st.markdown("""
            **⚠️ Risk Management:**
            
            **🔴 High-Risk Sectors:**
            - **Private Banking**: ICICI Bank under Mercury-Jupiter square pressure
            - **Automotive**: Maruti facing Venus-Saturn headwinds
            - **Real Estate**: DLF vulnerable to credit tightening aspects
            
            **📊 Risk Mitigation:**
            - Reduce position sizes during Mercury-Jupiter square (high volatility)
            - Use stop-losses 2-3% below support for Venus-Saturn affected stocks
            - Avoid leveraged positions in Midcap segment (Mars-Uranus volatility)
            
            **⏰ Timing Risks:**
            - Morning session volatility expected (Mercury aspects)
            - Post-lunch session may see pressure (Saturn influence)
            """)
        
        with tab4:
            st.markdown("""
            **🌟 Today's Cosmic Events Schedule:**
            
            **🌅 Pre-Market (Before 9:15 AM):**
            - Mercury-Jupiter square builds tension
            - Global markets influence domestic opening
            
            **🌄 Morning Session (9:15-12:00):**
            - Initial volatility from Mercury aspects
            - Energy stocks may show strength
            
            **🌞 Afternoon Session (12:00-15:30):**
            - Venus-Saturn opposition peaks
            - Defensive sectors gain relative strength
            - Banking sector under pressure
            
            **🌆 Post-Market:**
            - Global commodity movements (Gold, Crude)
            - Crypto markets reaction to day's developments
            
            **📊 Weekly Outlook:**
            - Aspects intensify mid-week
            - Weekend planetary shifts to monitor
            """)
    
    elif section_name == 'Stock Filter':
        st.header('🔍 Advanced Stock Filtering & Sector Analysis')
        
        # Create main layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader('🌟 Today\'s Astrological Configuration')
            
            # Display aspects in a nice format
            aspects_data = []
            for aspect in aspects:
                aspects_data.append({
                    'Planets': aspect['planets'],
                    'Aspect': aspect['aspect_type'],
                    'Impact': f"{aspect['impact']:+.1f}",
                    'Sentiment': aspect['type'].title(),
                    'Strength': '🔥' * min(3, int(abs(aspect['impact']) * 3))
                })
            
            aspects_df = pd.DataFrame(aspects_data)
            
            # Color code the dataframe
            def color_sentiment(val):
                if 'Bullish' in str(val):
                    return 'background-color: #d4edda; color: #155724'
                elif 'Bearish' in str(val):
                    return 'background-color: #f8d7da; color: #721c24'
                return ''
            
            styled_aspects = aspects_df.style.applymap(color_sentiment, subset=['Sentiment'])
            st.dataframe(styled_aspects, use_container_width=True)
        
        with col2:
            st.subheader('📊 Aspect Statistics')
            
            # Create a simple pie chart for aspect types
            aspect_types = {}
            for aspect in aspects:
                aspect_types[aspect['type']] = aspect_types.get(aspect['type'], 0) + 1
            
            if aspect_types:
                fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
                colors = ['green' if k == 'bullish' else 'red' if k == 'bearish' else 'gray' 
                         for k in aspect_types.keys()]
                wedges, texts, autotexts = ax_pie.pie(aspect_types.values(), 
                                                     labels=[k.title() for k in aspect_types.keys()], 
                                                     colors=colors, autopct='%1.0f%%', startangle=90)
                ax_pie.set_title('Today\'s Aspect Distribution')
                st.pyplot(fig_pie)
        
        # Sector analysis
        st.subheader('📈 Sector Impact Analysis')
        
        filtered_stocks = filter_stocks_by_aspects(aspects, STOCK_DATABASE)
        
        # Create sector impact visualization
        sector_impacts_df = pd.DataFrame({
            'Sector': list(filtered_stocks['sector_impacts'].keys()),
            'Impact Score': list(filtered_stocks['sector_impacts'].values()),
            'Recommendation': ['Strong Buy' if x > 0.5 else 'Buy' if x > 0 else 'Hold' if x == 0 
                             else 'Sell' if x > -0.5 else 'Strong Sell' 
                             for x in filtered_stocks['sector_impacts'].values()]
        })
        sector_impacts_df = sector_impacts_df.sort_values('Impact Score', ascending=False)
        
        # Enhanced bar chart
        fig_sectors, ax_sectors = plt.subplots(figsize=(14, 8))
        colors = ['darkgreen' if x > 0.5 else 'green' if x > 0 else 'gray' if x == 0 
                 else 'red' if x > -0.5 else 'darkred' 
                 for x in sector_impacts_df['Impact Score']]
        
        bars = ax_sectors.bar(sector_impacts_df['Sector'], sector_impacts_df['Impact Score'], 
                             color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax_sectors.set_title('Sector Impact Scores - Astrological Analysis', fontsize=16, pad=20)
        ax_sectors.set_ylabel('Impact Score', fontsize=12)
        ax_sectors.set_xlabel('Sector', fontsize=12)
        ax_sectors.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax_sectors.grid(True, alpha=0.3)
        
        # Add value labels and recommendations
        for i, (bar, rec) in enumerate(zip(bars, sector_impacts_df['Recommendation'])):
            height = bar.get_height()
            ax_sectors.annotate(f'{height:.2f}\n{rec}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 5 if height >= 0 else -25),
                               textcoords="offset points",
                               ha='center', va='bottom' if height >= 0 else 'top',
                               fontsize=9, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig_sectors)
        
        # Stock recommendations in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader('🟢 Bullish Stocks')
            if not filtered_stocks['bullish'].empty:
                bullish_df = filtered_stocks['bullish'][['Symbol', 'Sector', 'Impact Score']].copy()
                bullish_df['Action'] = bullish_df['Impact Score'].apply(
                    lambda x: 'Strong Buy' if x > 0.5 else 'Buy'
                )
                
                for _, row in bullish_df.iterrows():
                    with st.container():
                        st.markdown(f"""
                        **{row['Symbol']}** ({row['Sector']})  
                        Score: {row['Impact Score']:.2f} | **{row['Action']}**
                        """)
                        st.progress(min(1.0, row['Impact Score']))
            else:
                st.info("No strong bullish signals today")
        
        with col2:
            st.subheader('🔴 Bearish Stocks')
            if not filtered_stocks['bearish'].empty:
                bearish_df = filtered_stocks['bearish'][['Symbol', 'Sector', 'Impact Score']].copy()
                bearish_df['Action'] = bearish_df['Impact Score'].apply(
                    lambda x: 'Strong Sell' if x > 0.5 else 'Sell'
                )
                
                for _, row in bearish_df.iterrows():
                    with st.container():
                        st.markdown(f"""
                        **{row['Symbol']}** ({row['Sector']})  
                        Risk Score: {row['Impact Score']:.2f} | **{row['Action']}**
                        """)
                        st.progress(min(1.0, row['Impact Score']))
            else:
                st.info("No strong bearish signals today")
        
        with col3:
            st.subheader('⚪ Neutral Stocks')
            if not filtered_stocks['neutral'].empty:
                neutral_df = filtered_stocks['neutral'][['Symbol', 'Sector']].head(5)
                
                for _, row in neutral_df.iterrows():
                    st.markdown(f"**{row['Symbol']}** ({row['Sector']}) - Hold")
            else:
                st.info("All stocks showing directional bias")
    
    elif section_name == 'Aspect Analysis':
        st.header('📋 Deep Astrological Aspect Analysis')
        
        # Generate enhanced analysis
        fig, df_aspects = analyze_aspects()
        st.pyplot(fig)
        
        # Display detailed aspect table
        st.subheader('📊 Detailed Aspect Reference Table')
        
        # Add more columns for better analysis
        df_enhanced = df_aspects.copy()
        df_enhanced['Trading Action'] = df_enhanced.apply(
            lambda row: 'Hedge/Reduce' if 'Bearish' in row['Market Impact'] or 'Tension' in row['Market Impact']
            else 'Accumulate' if 'Bullish' in row['Market Impact'] or 'Rally' in row['Market Impact']
            else 'Monitor', axis=1
        )
        
        df_enhanced['Risk Level'] = df_enhanced['Typical Price Change'].apply(
            lambda x: 'High' if any(num in x for num in ['3', '4']) 
            else 'Medium' if '2' in x else 'Low'
        )
        
        # Style the enhanced dataframe
        def highlight_risk(val):
            if val == 'High':
                return 'background-color: #ffcccc'
            elif val == 'Medium':
                return 'background-color: #ffffcc'
            elif val == 'Low':
                return 'background-color: #ccffcc'
            return ''
        
        def highlight_action(val):
            if val == 'Accumulate':
                return 'background-color: #d4edda; color: #155724'
            elif val == 'Hedge/Reduce':
                return 'background-color: #f8d7da; color: #721c24'
            elif val == 'Monitor':
                return 'background-color: #fff3cd; color: #856404'
            return ''
        
        styled_enhanced = df_enhanced.style.applymap(highlight_risk, subset=['Risk Level']).applymap(highlight_action, subset=['Trading Action'])
        st.dataframe(styled_enhanced, use_container_width=True)
        
        # Aspect interpretation guide
        st.subheader('🔭 Astrological Aspect Interpretation Guide')
        
        tab1, tab2, tab3 = st.tabs(["🌟 Aspect Types", "🪐 Planetary Influences", "📈 Trading Applications"])
        
        with tab1:
            st.markdown("""
            ### Understanding Astrological Aspects
            
            **🔄 Conjunction (0°)**: 
            - *Market Effect*: Powerful combining of energies, can create sharp moves
            - *Trading*: Expect significant price action, potential breakouts
            - *Example*: Mars-Uranus conjunction = explosive energy moves
            
            **⚔️ Square (90°)**: 
            - *Market Effect*: Tension, conflict, volatility
            - *Trading*: Increased intraday swings, good for scalping
            - *Example*: Mercury-Jupiter square = communication/policy confusion
            
            **🎯 Trine (120°)**: 
            - *Market Effect*: Harmonious, easy flow of energy
            - *Trading*: Trending moves, good for position trading
            - *Example*: Moon-Neptune trine = emotional/intuitive support
            
            **⚖️ Opposition (180°)**: 
            - *Market Effect*: Polarization, requires balance
            - *Trading*: Range-bound action, reversals possible
            - *Example*: Venus-Saturn opposition = value vs. restriction
            
            **🤝 Sextile (60°)**: 
            - *Market Effect*: Opportunity aspects, mild positive
            - *Trading*: Gentle trends, good for swing trades
            - *Example*: Sun-Pluto sextile = gradual transformation
            """)
        
        with tab2:
            st.markdown("""
            ### Planetary Market Influences
            
            **☀️ Sun**: Leadership, government policy, large-cap stocks, gold
            **🌙 Moon**: Public sentiment, emotions, consumer sectors, silver
            **☿️ Mercury**: Communication, technology, volatility, news-driven moves
            **♀️ Venus**: Finance, banking, luxury goods, relationships, copper
            **♂️ Mars**: Energy, metals, defense, aggressive moves, oil
            **♃ Jupiter**: Growth, expansion, optimism, financial sector
            **♄ Saturn**: Restriction, discipline, structure, defensive sectors
            **♅ Uranus**: Innovation, technology, sudden changes, crypto
            **♆ Neptune**: Illusion, oil, pharma, confusion, speculation
            **♇ Pluto**: Transformation, power, mining, major shifts
            
            ### Sector-Planet Correlations
            - **Technology**: Mercury, Uranus
            - **Banking**: Jupiter, Venus, Saturn  
            - **Energy**: Mars, Sun, Pluto
            - **Healthcare**: Neptune, Moon
            - **Precious Metals**: Venus, Jupiter, Sun
            - **Cryptocurrency**: Uranus, Pluto
            """)
        
        with tab3:
            st.markdown("""
            ### Practical Trading Applications
            
            **📊 Intraday Trading:**
            - Use Moon aspects for sentiment shifts (2-4 hour cycles)
            - Mercury aspects for news/volatility spikes
            - Mars aspects for energy sector breakouts
            
            **📈 Swing Trading:**
            - Venus aspects for financial sector trends (3-7 days)
            - Jupiter aspects for broad market optimism
            - Saturn aspects for defensive positioning
            
            **📉 Position Trading:**
            - Outer planet aspects (Uranus, Neptune, Pluto) for long-term themes
            - Eclipse patterns for major sector rotations
            - Retrograde periods for trend reversals
            
            **⚠️ Risk Management:**
            - Increase cash during multiple challenging aspects
            - Reduce position size during Mercury retrograde
            - Use tighter stops during Mars-Saturn squares
            
            **🎯 Sector Rotation:**
            - Follow Jupiter through zodiac signs for sector leadership
            - Track Saturn aspects for value opportunities
            - Monitor Uranus for innovation themes
            """)
    
    elif section_name == 'Intraday Chart':
        st.header(f'📈 {symbol} - Intraday Astrological Analysis')
        
        # Display symbol information prominently
        symbol_info = get_symbol_info(symbol)
        trading_hours = get_trading_hours(symbol)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Symbol", symbol)
        with col2:
            st.metric("Sector", symbol_info['sector'])
        with col3:
            st.metric("Currency", symbol_info['currency'])
        with col4:
            session_length = trading_hours['end_hour'] - trading_hours['start_hour'] + \
                           (trading_hours['end_minute'] - trading_hours['start_minute']) / 60
            st.metric("Session Hours", f"{session_length:.1f}h")
        
        # Generate and display chart
        with st.spinner(f'Generating astrological analysis for {symbol}...'):
            fig = generate_intraday_chart(symbol, starting_price, selected_date)
            st.pyplot(fig)
        
        # Trading insights based on symbol
        st.subheader(f'🎯 {symbol} Trading Insights')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Technical Outlook")
            
            # Generate insights based on symbol type
            if symbol in ['GOLD', 'SILVER']:
                st.markdown("""
                **Precious Metals Analysis:**
                - Multiple planetary aspects favor safe-haven demand
                - Venus-Saturn opposition creates financial stress → Gold strength  
                - Moon-Neptune trine supports intuitive precious metal buying
                - Best trading windows: During global uncertainty aspects
                
                **Key Levels:**
                - Watch for breakouts during Mars-Uranus conjunction
                - Support likely during Moon aspects
                - Resistance at previous highs during Saturn aspects
                """)
            
            elif symbol in ['BTC']:
                st.markdown("""
                **Cryptocurrency Analysis:**
                - Uranus aspects strongly favor crypto volatility
                - Mars-Uranus conjunction = explosive price moves
                - Traditional financial stress (Venus-Saturn) → Crypto rotation
                - High volatility expected - use proper risk management
                
                **Trading Strategy:**
                - Momentum plays during Uranus aspects
                - Contrarian plays during Saturn oppositions
                - Volume spikes likely at aspect peaks
                """)
            
            elif symbol in ['CRUDE']:
                st.markdown("""
                **Energy Commodity Analysis:**
                - Mars-Uranus conjunction directly impacts energy sector
                - Global supply disruption themes (Pluto aspects)
                - Geopolitical tensions favor energy prices
                - Weather and seasonal patterns amplified by aspects
                
                **Supply-Demand Factors:**
                - Production disruptions during Mars aspects
                - Demand surges during economic aspects
                - Storage plays during Saturn aspects
                """)
            
            elif symbol in ['DOWJONES']:
                st.markdown("""
                **US Index Analysis:**
                - Jupiter aspects favor broad market optimism
                - Saturn aspects create rotation into defensive sectors
                - Mercury aspects increase intraday volatility
                - Fed policy sensitivity during Venus-Saturn opposition
                
                **Sector Rotation:**
                - Technology during Mercury aspects
                - Energy during Mars aspects  
                - Financials during Jupiter aspects
                """)
            
            else:  # Indian stocks
                st.markdown(f"""
                **{symbol_info['sector']} Sector Analysis:**
                - Domestic market influenced by global planetary patterns
                - FII/DII flows affected by Venus-Saturn aspects
                - Sector rotation based on planetary emphasis
                - Currency impacts during outer planet aspects
                
                **Indian Market Specifics:**
                - Opening gap influenced by global overnight aspects
                - Lunch hour volatility during Moon aspects
                - Closing session strength during Jupiter aspects
                """)
        
        with col2:
            st.markdown("#### ⏰ Timing Analysis")
            
            # Generate time-specific insights based on trading hours
            if trading_hours['end_hour'] > 16:  # Extended hours
                st.markdown("""
                **Extended Session Analysis:**
                
                **🌅 Asian Session (5:00-8:00):**
                - Pre-market positioning based on overnight aspects
                - Lower volumes, higher impact from aspects
                - Key economic data releases amplified
                
                **🌍 European Session (8:00-16:00):**
                - Peak liquidity and aspect impacts
                - Central bank policy influences
                - Cross-asset correlations strongest
                
                **🌎 US Session (16:00-20:00):**
                - Maximum volatility potential
                - Aspect peaks create significant moves
                - News flow interaction with cosmic patterns
                
                **🌙 After Hours (20:00-23:55):**
                - Reduced liquidity amplifies aspect effects
                - Position adjustments for next day
                - Asian preview impact
                """)
            else:  # Indian market hours
                st.markdown("""
                **Indian Session Analysis:**
                
                **🌅 Opening (9:15-10:30):**
                - Gap opening based on global aspects
                - High volatility, aspect impacts magnified
                - Initial trend direction setting
                
                **🌞 Mid-Morning (10:30-12:00):**
                - Institutional activity peaks
                - Aspect-driven sector rotation
                - News flow integration
                
                **🍽️ Lunch Hour (12:00-13:00):**
                - Reduced activity, Moon aspects dominate
                - Range-bound unless strong aspects active
                - Position consolidation period
                
                **🌆 Closing (13:00-15:30):**
                - Final institutional positioning
                - Aspect resolution for day
                - Next-day setup formation
                """)
            
            # Risk management
            st.markdown("#### ⚠️ Risk Management")
            st.markdown(f"""
            **Position Sizing:**
            - Standard position: 1-2% of capital
            - High aspect days: Reduce to 0.5-1%
            - Strong confluence: Increase to 2-3%
            
            **Stop Loss Levels:**
            - Tight stops during Mercury aspects: 1-2%
            - Normal stops during stable aspects: 2-3%
            - Wide stops during Mars aspects: 3-5%
            
            **Profit Targets:**
            - Quick scalps: 0.5-1% (15-30 minutes)
            - Swing trades: 2-5% (2-4 hours)
            - Position trades: 5-10% (1-3 days)
            
            **Volatility Adjustments:**
            - {symbol}: Expected daily range ±{2.5 if symbol in ['BTC'] else 1.5 if symbol in ['CRUDE'] else 1.0 if symbol in ['GOLD', 'SILVER'] else 0.8}%
            - Adjust position size inversely to volatility
            - Use options for high-volatility periods
            """)
    
    elif section_name == 'Monthly Chart':
        st.header(f'📊 {symbol} - Monthly Astrological Trend Analysis')
        
        # Display symbol information
        symbol_info = get_symbol_info(symbol)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Symbol", symbol)
        with col2:
            st.metric("Analysis Period", f"{calendar.month_name[selected_month]} {selected_year}")
        with col3:
            st.metric("Sector Focus", symbol_info['sector'])
        with col4:
            st.metric("Currency", symbol_info['currency'])
        
        # Generate and display chart
        with st.spinner(f'Generating monthly analysis for {symbol}...'):
            fig = generate_monthly_chart(symbol, starting_price, selected_month, selected_year)
            st.pyplot(fig)
        
        # Monthly analysis insights
        st.subheader(f'📈 {calendar.month_name[selected_month]} {selected_year} - Strategic Analysis')
        
        # Create tabs for different analysis aspects
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Monthly Outlook", "📊 Technical Analysis", "🌙 Lunar Cycles", "💼 Portfolio Strategy"])
        
        with tab1:
            month_name = calendar.month_name[selected_month]
            
            if symbol in ['GOLD', 'SILVER']:
                st.markdown(f"""
                ### {symbol} - {month_name} {selected_year} Precious Metals Outlook
                
                **🌟 Astrological Themes:**
                - **Venus-Jupiter aspects**: Strong precious metals demand from financial uncertainty
                - **Saturn transits**: Safe-haven buying during economic restrictions
                - **Moon phases**: Emotional buying patterns aligned with lunar cycles
                - **Mercury retrograde periods**: Technical analysis less reliable, fundamentals dominate
                
                **📈 Price Drivers:**
                - Central bank policy uncertainty (Saturn aspects)
                - Currency devaluation themes (Pluto aspects)
                - Geopolitical tensions (Mars aspects)
                - Inflation hedging demand (Jupiter-Saturn aspects)
                
                **🎯 Trading Strategy:**
                - **Accumulate** during New Moon phases (stronger buying interest)
                - **Profit-take** during Full Moon phases (emotional peaks)
                - **Hold through** Mercury retrograde (avoid technical trading)
                - **Scale in** during Saturn aspects (structural support)
                
                **📊 Target Levels:**
                - **Monthly High**: Expect during Jupiter-Venus trine periods
                - **Monthly Low**: Likely during Mars-Saturn square periods
                - **Breakout Potential**: Mars-Uranus conjunction periods
                - **Support Zones**: Previous month's Jupiter aspect levels
                """)
            
            elif symbol in ['BTC']:
                st.markdown(f"""
                ### {symbol} - {month_name} {selected_year} Cryptocurrency Outlook
                
                **⚡ Astrological Themes:**
                - **Uranus-Pluto aspects**: Revolutionary technology adoption waves
                - **Mercury-Uranus aspects**: Network upgrades and technical developments
                - **Mars-Uranus conjunctions**: Explosive price movements and FOMO
                - **Saturn aspects**: Regulatory clarity or restrictions
                
                **🚀 Volatility Drivers:**
                - Institutional adoption news (Jupiter aspects)
                - Regulatory developments (Saturn aspects)
                - Technical network changes (Mercury-Uranus)
                - Market manipulation concerns (Neptune aspects)
                
                **⚠️ Risk Factors:**
                - **High volatility** during Mars-Uranus aspects (±10-20% daily swings)
                - **Regulatory risks** during Saturn-Pluto aspects
                - **Technical failures** during Mercury retrograde
                - **Market manipulation** during Neptune-Mercury aspects
                
                **💡 Strategic Approach:**
                - **DCA strategy** during volatile periods
                - **Momentum trading** during Uranus aspects
                - **Risk-off** during Saturn hard aspects
                - **HODL mentality** during Jupiter-Pluto trines
                """)
            
            elif symbol in ['CRUDE']:
                st.markdown(f"""
                ### {symbol} - {month_name} {selected_year} Energy Commodity Outlook
                
                **🛢️ Astrological Themes:**
                - **Mars-Pluto aspects**: Geopolitical tensions affecting supply
                - **Jupiter-Saturn cycles**: Economic growth vs. restriction cycles
                - **Uranus aspects**: Renewable energy transition impacts
                - **Moon phases**: Seasonal demand patterns amplified
                
                **⚡ Supply-Demand Dynamics:**
                - Production disruptions (Mars-Saturn squares)
                - Economic growth spurts (Jupiter aspects)
                - Weather pattern extremes (Uranus-Neptune aspects)
                - Strategic reserve changes (Pluto aspects)
                
                **🌍 Geopolitical Factors:**
                - **OPEC decisions** aligned with Saturn aspects
                - **Pipeline disruptions** during Mars-Uranus periods
                - **Currency impacts** during Venus-Pluto aspects
                - **Seasonal patterns** enhanced by lunar cycles
                
                **📈 Trading Levels:**
                - **Resistance**: Previous Jupiter aspect highs
                - **Support**: Saturn aspect consolidation zones
                - **Breakout zones**: Mars-Uranus conjunction levels
                - **Reversal points**: Full Moon technical confluences
                """)
            
            elif symbol in ['DOWJONES']:
                st.markdown(f"""
                ### {symbol} - {month_name} {selected_year} US Index Outlook
                
                **🇺🇸 Macro Astrological Themes:**
                - **Jupiter-Saturn cycles**: Economic expansion vs. contraction
                - **Mercury-Venus aspects**: Corporate earnings and consumer spending
                - **Mars-Jupiter aspects**: Business investment and growth
                - **Outer planet aspects**: Long-term structural changes
                
                **🏛️ Federal Reserve Alignment:**
                - **Venus-Saturn aspects**: Interest rate policy changes
                - **Mercury-Jupiter aspects**: Fed communication clarity
                - **Moon phases**: Market sentiment around FOMC meetings
                - **Eclipse periods**: Major policy shift announcements
                
                **🔄 Sector Rotation Patterns:**
                - **Technology** leadership during Mercury-Uranus aspects
                - **Energy** strength during Mars-Pluto periods
                - **Financials** favor during Venus-Jupiter trines
                - **Healthcare** defensive during Saturn aspects
                
                **📊 Technical Confluence:**
                - **Monthly resistance**: Jupiter aspect previous highs
                - **Monthly support**: Saturn aspect previous lows
                - **Breakout potential**: New Moon near technical levels
                - **Reversal zones**: Full Moon at key Fibonacci levels
                """)
            
            else:  # Indian stocks
                st.markdown(f"""
                ### {symbol} ({symbol_info['sector']}) - {month_name} {selected_year} Indian Market Outlook
                
                **🇮🇳 Domestic Astrological Influences:**
                - **Jupiter transits**: Market leadership and FII flows
                - **Saturn aspects**: Regulatory changes and policy shifts
                - **Mars-Venus aspects**: Consumer spending and investment flows
                - **Moon phases**: Retail investor sentiment cycles
                
                **💹 Sector-Specific Themes:**
                - **{symbol_info['sector']} sector** influenced by specific planetary combinations
                - **Monsoon patterns** (if applicable) aligned with water sign emphasis
                - **Festival seasons** amplified by benefic planetary aspects
                - **Budget impacts** during Saturn-Jupiter aspects
                
                **🌏 Global Correlation Factors:**
                - **US Fed policy** impacts during Venus-Saturn aspects
                - **China growth** concerns during Mars-Saturn periods  
                - **Oil prices** affecting through Mars-Pluto aspects
                - **Dollar strength** impacts during Pluto aspects
                
                **📈 Monthly Strategy:**
                - **Accumulate** during Saturn aspects (value opportunities)
                - **Momentum plays** during Mars-Jupiter periods
                - **Defensive positioning** during challenging outer planet aspects
                - **Sector rotation** based on planetary emphasis shifts
                """)
        
        with tab2:
            st.markdown(f"""
            ### Technical Analysis Integration with Astrological Cycles
            
            **📊 Moving Average Alignment:**
            - **MA5 vs MA20**: Bullish when Jupiter aspects dominate
            - **Golden Cross** potential during Venus-Jupiter trines
            - **Death Cross** risk during Saturn-Mars squares
            - **MA support/resistance** stronger during lunar phases
            
            **🎯 Support & Resistance Levels:**
            - **Primary resistance**: Previous month's Jupiter aspect highs
            - **Primary support**: Saturn aspect consolidation lows
            - **Secondary levels**: Full Moon reversal points
            - **Breakout levels**: New Moon momentum points
            
            **📈 Momentum Indicators:**
            - **RSI overbought** (>70) more reliable during Full Moons
            - **RSI oversold** (<30) stronger signal during New Moons
            - **MACD divergences** amplified during Mercury aspects
            - **Volume confirmations** critical during Mars aspects
            
            **🌙 Lunar Cycle Technical Correlation:**
            - **New Moon**: Trend initiation, breakout potential
            - **Waxing Moon**: Momentum continuation, bullish bias
            - **Full Moon**: Trend exhaustion, reversal potential
            - **Waning Moon**: Correction phases, consolidation
            
            **⚡ Volatility Patterns:**
            - **Highest volatility**: Mars-Uranus aspect periods
            - **Lowest volatility**: Venus-Jupiter trine periods
            - **Unexpected moves**: Mercury-Neptune confusion aspects
            - **Gap movements**: Eclipse and outer planet aspects
            
            **🔄 Pattern Recognition:**
            - **Triangle breakouts** during Uranus aspects
            - **Flag patterns** during Mars aspects  
            - **Head & Shoulders** during Saturn aspects
            - **Double tops/bottoms** during opposition aspects
            """)
        
        with tab3:
            st.markdown(f"""
            ### Lunar Cycles & Market Psychology for {month_name} {selected_year}
            
            **🌑 New Moon Phases (Market Initiation):**
            - **Energy**: Fresh starts, new trend beginnings
            - **Psychology**: Optimism, risk-taking increases
            - **Trading**: Look for breakout setups, trend initiations
            - **Volume**: Often lower but quality moves
            - **Best for**: Opening new positions, trend following
            
            **🌓 Waxing Moon (Building Momentum):**
            - **Energy**: Growth, expansion, building confidence  
            - **Psychology**: FOMO starts building, bullish sentiment
            - **Trading**: Momentum continuation, pyramid additions
            - **Volume**: Increasing participation
            - **Best for**: Adding to winning positions
            
            **🌕 Full Moon Phases (Emotional Peaks):**
            - **Energy**: Maximum emotion, extremes, reversals
            - **Psychology**: Euphoria or panic peaks
            - **Trading**: Reversal setups, profit-taking
            - **Volume**: Often highest of cycle
            - **Best for**: Profit booking, contrarian plays
            
            **🌗 Waning Moon (Consolidation):**
            - **Energy**: Release, correction, cooling off
            - **Psychology**: Reality check, risk assessment
            - **Trading**: Consolidation patterns, value hunting
            - **Volume**: Declining, selective moves
            - **Best for**: Position adjustments, planning
            
            **🔮 {month_name} {selected_year} Specific Lunar Events:**
            
            **Key Lunar Dates to Watch:**
            - **New Moon**: Potential trend change or continuation signal
            - **First Quarter**: Momentum confirmation or failure
            - **Full Moon**: Profit-taking opportunity or reversal signal  
            - **Last Quarter**: Consolidation phase or weakness signal
            
            **Moon Sign Influences:**
            - **Fire Signs** (Aries, Leo, Sagittarius): Aggressive moves, energy sector strength
            - **Earth Signs** (Taurus, Virgo, Capricorn): Value focus, stability preference
            - **Air Signs** (Gemini, Libra, Aquarius): Communication, technology emphasis
            - **Water Signs** (Cancer, Scorpio, Pisces): Emotional decisions, defensive moves
            """)
        
        with tab4:
            st.markdown(f"""
            ### Portfolio Strategy for {month_name} {selected_year}
            
            **🎯 Strategic Asset Allocation:**
            
            **Core Holdings (50-60%):**
            - **Large Cap Stability**: Jupiter-aspected blue chips
            - **Sector Leaders**: Dominant players in favored sectors
            - **Defensive Assets**: During challenging aspect periods
            - **Currency Hedge**: If significant Pluto aspects present
            
            **Growth Opportunities (20-30%):**
            - **Momentum Plays**: Mars-Jupiter aspect beneficiaries
            - **Breakout Candidates**: Technical + astrological confluence
            - **Sector Rotation**: Following planetary emphasis shifts
            - **Emerging Themes**: Uranus aspect innovation plays
            
            **Speculative/Trading (10-20%):**
            - **High Beta Names**: For Mars-Uranus periods
            - **Volatility Plays**: Options during aspect peaks
            - **Contrarian Bets**: Against crowd during extremes
            - **Crypto Allocation**: If comfortable with high volatility
            
            **📊 Risk Management Framework:**
            
            **Position Sizing Rules:**
            - **Maximum single position**: 5% during stable periods
            - **Reduce to 3%**: During challenging aspects
            - **Increase to 7%**: During strong favorable confluences
            - **Cash levels**: 10-20% based on aspect favorability
            
            **Stop Loss Strategy:**
            - **Tight stops** (3-5%): During Mercury retrograde periods
            - **Normal stops** (5-8%): During regular market conditions
            - **Wide stops** (8-12%): During high volatility aspect periods
            - **No stops**: For long-term Jupiter-blessed positions
            
            **📅 Monthly Rebalancing Schedule:**
            
            **Week 1**: Review and adjust based on new lunar cycle
            **Week 2**: Add to momentum winners if aspects support
            **Week 3**: Prepare for Full Moon profit-taking opportunities
            **Week 4**: Position for next month's astrological themes
            
            **🔄 Sector Rotation Strategy:**
            
            **Early Month**: Follow Jupiter aspects for growth sectors
            **Mid Month**: Mars aspects may favor energy/materials
            **Late Month**: Venus aspects support financials/consumer
            **Month End**: Saturn aspects favor defensives/utilities
            
            **💡 Advanced Strategies:**
            
            **Pairs Trading**: Long favored sectors, short challenged sectors
            **Options Overlay**: Sell calls during Full Moons, buy calls during New Moons
            **Currency Hedge**: Hedge foreign exposure during Pluto aspects
            **Volatility Trading**: Long volatility before aspect peaks
            
            **📈 Performance Tracking:**
            
            **Monthly Metrics**:
            - Absolute return vs. benchmark
            - Risk-adjusted return (Sharpe ratio)
            - Maximum drawdown during challenging aspects
            - Hit rate on astrological predictions
            
            **Aspect Correlation Analysis**:
            - Track which aspects work best for {symbol}
            - Note sector rotation timing accuracy
            - Measure volatility prediction success
            - Document lunar cycle correlations
            """)
        
        # Additional insights for monthly strategy
        st.subheader('🎭 Market Psychology & Sentiment Analysis')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            #### 🧠 Psychological Drivers - {month_name}
            
            **Institutional Behavior:**
            - Month-end window dressing effects
            - Quarterly rebalancing influences  
            - Earnings season psychological impacts
            - Fed meeting anticipation/reaction
            
            **Retail Investor Patterns:**
            - Payroll cycle investment flows
            - Tax implications (if year-end)
            - Holiday season spending impacts
            - Social media sentiment amplification
            
            **Global Sentiment Factors:**
            - US-China trade relationship status
            - European economic data impacts
            - Emerging market flow dynamics
            - Cryptocurrency correlation effects
            """)
        
        with col2:
            st.markdown(f"""
            #### 📊 Sentiment Indicators to Watch
            
            **Technical Sentiment:**
            - VIX levels and term structure
            - Put/Call ratios by sector
            - High-low index readings
            - Advance-decline line trends
            
            **Fundamental Sentiment:**
            - Earnings revision trends
            - Analyst recommendation changes
            - Insider buying/selling activity
            - Share buyback announcements
            
            **Alternative Data:**
            - Google search trends
            - Social media mention analysis
            - Options flow analysis
            - Crypto correlation strength
            """)

# Footer with additional information
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <h4>🌟 Disclaimer & Important Notes</h4>
        <p><strong>Educational Purpose Only:</strong> This dashboard is for educational and research purposes. 
        Astrological analysis should be combined with fundamental and technical analysis for trading decisions.</p>
        
        <p><strong>Risk Warning:</strong> All trading involves risk. Past performance and astrological correlations 
        do not guarantee future results. Always use proper risk management and never invest more than you can afford to lose.</p>
        
        <p><strong>Data Sources:</strong> Simulated price data based on astrological aspect calculations. 
        For live trading, use real market data and professional trading platforms.</p>
        
        <p style='font-size: 12px; margin-top: 20px;'>
        🔮 <em>"The stars impel, they do not compel. Wisdom lies in using all available tools - 
        fundamental, technical, and cosmic - for informed decision making."</em>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main(), 'default_price': 65000.0, 'sector': 'Cryptocurrency'},
}

# --- STOCK DATABASE ---
stock_data = {
    'Symbol': [
        'TCS', 'ICICIBANK', 'MARUTI', 'DLF', 'NESTLEIND', 
        'RELIANCE', 'SBI', 'SUNPHARMA', 'DRREDDY', 'GOLD',
        'DOWJONES', 'SILVER', 'CRUDE', 'BTC'
    ],
    'Sector': [
        'Technology', 'Banking', 'Automotive', 'Realty', 'FMCG',
        'Energy', 'PSUs', 'Pharma', 'Pharma', 'Precious Metals',
        'US Index', 'Precious Metals', 'Energy', 'Cryptocurrency'
    ],
    'MarketCap': [
        'Large', 'Large', 'Large', 'Large', 'Large',
        'Large', 'Large', 'Large', 'Large', 'Commodity',
        'Index', 'Commodity', 'Commodity', 'Crypto'
    ]
}

STOCK_DATABASE = pd.DataFrame(stock_data)

# --- SECTOR-PLANETARY MAPPINGS ---
SECTOR_PLANETARY_INFLUENCES = {
    'Technology': ['Mercury'],
    'Banking': ['Jupiter', 'Saturn'],
    'FMCG': ['Moon'],
    'Pharma': ['Neptune'],
    'Energy': ['Mars'],
    'Automotive': ['Saturn'],
    'Realty': ['Saturn'],
    'PSUs': ['Pluto'],
    'Midcaps': ['Uranus'],
    'Smallcaps': ['Pluto'],
    'Precious Metals': ['Venus', 'Jupiter'],
    'US Index': ['Jupiter', 'Saturn'],
    'Cryptocurrency': ['Uranus', 'Pluto']
}

# --- ASPECT-SECTOR IMPACT ---
ASPECT_SECTOR_IMPACTS = {
    'Square': {
        'Technology': 'Negative', 'Banking': 'Negative', 'FMCG': 'Neutral',
        'Pharma': 'Neutral', 'Energy': 'Neutral', 'Automotive': 'Neutral',
        'Realty': 'Neutral', 'PSUs': 'Neutral', 'Midcaps': 'Neutral',
        'Smallcaps': 'Neutral', 'Precious Metals': 'Positive', 'US Index': 'Negative',
        'Cryptocurrency': 'Negative'
    },
    'Opposition': {
        'Technology': 'Neutral', 'Banking': 'Neutral', 'FMCG': 'Neutral',
        'Pharma': 'Neutral', 'Energy': 'Neutral', 'Automotive': 'Negative',
        'Realty': 'Negative', 'PSUs': 'Neutral', 'Midcaps': 'Neutral',
        'Smallcaps': 'Neutral', 'Precious Metals': 'Positive', 'US Index': 'Neutral',
        'Cryptocurrency': 'Positive'
    },
    'Trine': {
        'Technology': 'Neutral', 'Banking': 'Neutral', 'FMCG': 'Positive',
        'Pharma': 'Positive', 'Energy': 'Neutral', 'Automotive': 'Neutral',
        'Realty': 'Neutral', 'PSUs': 'Neutral', 'Midcaps': 'Neutral',
        'Smallcaps': 'Neutral', 'Precious Metals': 'Positive', 'US Index': 'Positive',
        'Cryptocurrency': 'Neutral'
    },
    'Conjunction': {
        'Technology': 'Neutral', 'Banking': 'Neutral', 'FMCG': 'Neutral',
        'Pharma': 'Neutral', 'Energy': 'Positive', 'Automotive': 'Neutral',
        'Realty': 'Neutral', 'PSUs': 'Neutral', 'Midcaps': 'Negative',
        'Smallcaps': 'Neutral', 'Precious Metals': 'Positive', 'US Index': 'Neutral',
        'Cryptocurrency': 'Positive'
    },
    'Sextile': {
        'Technology': 'Neutral', 'Banking': 'Neutral', 'FMCG': 'Neutral',
        'Pharma': 'Neutral', 'Energy': 'Neutral', 'Automotive': 'Neutral',
        'Realty': 'Neutral', 'PSUs': 'Positive', 'Midcaps': 'Neutral',
        'Smallcaps': 'Negative', 'Precious Metals': 'Positive', 'US Index': 'Positive',
        'Cryptocurrency': 'Neutral'
    }
}

# --- PLANETARY POSITION VISUALIZATION ---
def draw_planetary_wheel(ax, input_date, size=0.3):
    """Draw a simplified astrological wheel showing planetary positions"""
    base_date = datetime(2025, 8, 1)
    
    if isinstance(input_date, date_class) and not isinstance(input_date, datetime):
        date_obj = datetime.combine(input_date, datetime.min.time())
    else:
        date_obj = input_date
    
    days_diff = (date_obj.date() - base_date.date()).days
    
    base_positions = {
        'Sun': 135, 'Moon': 225, 'Mercury': 120, 'Venus': 170,
        'Mars': 85, 'Jupiter': 45, 'Saturn': 315
    }
    
    daily_movement = {
        'Sun': 1.0, 'Moon': 13.2, 'Mercury': 1.5, 'Venus': 1.2,
        'Mars': 0.5, 'Jupiter': 0.08, 'Saturn': 0.03
    }
    
    planets = {}
    for planet, base_pos in base_positions.items():
        angle = (base_pos + daily_movement[planet] * days_diff) % 360
        planets[planet] = {
            'angle': angle,
            'color': {
                'Sun': 'gold', 'Moon': 'silver', 'Mercury': 'gray',
                'Venus': 'lightgreen', 'Mars': 'red', 'Jupiter': 'orange',
                'Saturn': 'darkgoldenrod'
            }[planet],
            'size': {
                'Sun': 8, 'Moon': 6, 'Mercury': 5, 'Venus': 7,
                'Mars': 6, 'Jupiter': 10, 'Saturn': 9
            }[planet]
        }
    
    zodiac = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 
              'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
    
    for i, sign in enumerate(zodiac):
        angle = i * 30
        ax.add_patch(Wedge((0, 0), size, angle, angle+30, width=size*0.8, 
                          facecolor='lightgray', edgecolor='black', alpha=0.3))
        ax.text(0.85*size * np.cos(np.radians(angle+15)), 
                0.85*size * np.sin(np.radians(angle+15)), 
                sign[:3], ha='center', va='center', fontsize=6)
    
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
    ax.set_title(f'Planetary Positions\n{date_obj.strftime("%b %d, %Y")}', fontsize=8)

# --- GET TRADING HOURS FOR SYMBOL ---
def get_trading_hours(symbol):
    """Get trading hours for a specific symbol"""
    symbol = symbol.upper()
    if symbol in TRADING_HOURS:
        return TRADING_HOURS[symbol]
    else:
        # Default to Indian market hours for unknown symbols
        return TRADING_HOURS['NIFTY']

# --- GET SYMBOL INFO ---
def get_symbol_info(symbol):
    """Get symbol configuration info"""
    symbol = symbol.upper()
    if symbol in SYMBOL_CONFIG:
        return SYMBOL_CONFIG[symbol]
    else:
        # Default configuration
        return {
            'name': symbol,
            'currency': '₹',
            'default_price': 1000.0,
            'sector': 'Unknown'
        }

# --- GENERATE ASPECTS ---
def generate_todays_aspects():
    """Generate astrological aspects for today based on the provided table"""
    base_aspects = [
        {"planets": "Mercury-Jupiter", "aspect_type": "Square", "impact": -0.7, "type": "bearish"},
        {"planets": "Venus-Saturn", "aspect_type": "Opposition", "impact": -0.8, "type": "bearish"},
        {"planets": "Moon-Neptune", "aspect_type": "Trine", "impact": 0.6, "type": "bullish"},
        {"planets": "Mars-Uranus", "aspect_type": "Conjunction", "impact": 0.9, "type": "bullish"},
        {"planets": "Sun-Pluto", "aspect_type": "Sextile", "impact": 0.5, "type": "bullish"}
    ]
    
    aspects = []
    for aspect in base_aspects:
        aspects.append({
            "planets": aspect["planets"],
            "aspect_type": aspect["aspect_type"],
            "impact": aspect["impact"],
            "type": aspect["type"]
        })
    
    return aspects

# --- CREATE SUMMARY TABLE ---
def create_summary_table(aspects):
    """Create a summary table based on the astrological aspects"""
    summary_data = {
        'Aspect': [],
        'Nifty/Bank Nifty': [],
        'Bullish Sectors/Stocks': [],
        'Bearish Sectors/Stocks': []
    }
    
    for aspect in aspects:
        planets = aspect["planets"]
        aspect_type = aspect["aspect_type"]
        
        if planets == "Mercury-Jupiter" and aspect_type == "Square":
            summary_data['Aspect'].append("Mercury-Jupiter (Square)")
            summary_data['Nifty/Bank Nifty'].append("Volatile")
            summary_data['Bullish Sectors/Stocks'].append("IT (TCS), Gold")
            summary_data['Bearish Sectors/Stocks'].append("Banking (ICICI Bank), Crypto")
        
        elif planets == "Venus-Saturn" and aspect_type == "Opposition":
            summary_data['Aspect'].append("Venus-Saturn (Opposition)")
            summary_data['Nifty/Bank Nifty'].append("Downside")
            summary_data['Bullish Sectors/Stocks'].append("Gold, Silver, BTC")
            summary_data['Bearish Sectors/Stocks'].append("Auto (Maruti), Realty (DLF)")
        
        elif planets == "Moon-Neptune" and aspect_type == "Trine":
            summary_data['Aspect'].append("Moon-Neptune (Trine)")
            summary_data['Nifty/Bank Nifty'].append("Mild Support")
            summary_data['Bullish Sectors/Stocks'].append("FMCG (Nestlé), Pharma, Gold, Dow Jones")
            summary_data['Bearish Sectors/Stocks'].append("-")
        
        elif planets == "Mars-Uranus" and aspect_type == "Conjunction":
            summary_data['Aspect'].append("Mars-Uranus (Conjunction)")
            summary_data['Nifty/Bank Nifty'].append("Sharp Moves")
            summary_data['Bullish Sectors/Stocks'].append("Energy (Reliance, Crude), Gold, BTC")
            summary_data['Bearish Sectors/Stocks'].append("Weak Midcaps")
        
        elif planets == "Sun-Pluto" and aspect_type == "Sextile":
            summary_data['Aspect'].append("Sun-Pluto (Sextile)")
            summary_data['Nifty/Bank Nifty'].append("Structural Shift")
            summary_data['Bullish Sectors/Stocks'].append("PSUs (SBI), Gold, Dow Jones")
            summary_data['Bearish Sectors/Stocks'].append("Overvalued Smallcaps")
    
    return pd.DataFrame(summary_data)

# --- FILTER STOCKS BASED ON ASPECTS ---
def filter_stocks_by_aspects(aspects, stock_database):
    """Filter stocks based on today's astrological aspects"""
    sector_impacts = {sector: 0 for sector in stock_database['Sector'].unique()}
    
    for aspect in aspects:
        planet1, planet2 = aspect["planets"].split("-")
        
        for sector, planets in SECTOR_PLANETARY_INFLUENCES.items():
            if planet1 in planets or planet2 in planets:
                if sector not in sector_impacts:
                    sector_impacts[sector] = 0
                
                aspect_impact = ASPECT_SECTOR_IMPACTS[aspect["aspect_type"]].get(sector, "Neutral")
                
                if aspect_impact == "Positive":
                    sector_impacts[sector] += abs(aspect["impact"])
                elif aspect_impact == "Negative":
                    sector_impacts[sector] -= abs(aspect["impact"])
    
    bullish_sectors = [sector for sector, impact in sector_impacts.items() if impact > 0]
    bearish_sectors = [sector for sector, impact in sector_impacts.items() if impact < 0]
    neutral_sectors = [sector for sector, impact in sector_impacts.items() if impact == 0]
    
    bullish_stocks = stock_database[stock_database['Sector'].isin(bullish_sectors)].copy()
    bearish_stocks = stock_database[stock_database['Sector'].isin(bearish_sectors)].copy()
    neutral_stocks = stock_database[stock_database['Sector'].isin(neutral_sectors)].copy()
    
    bullish_stocks['Impact Score'] = bullish_stocks['Sector'].apply(lambda x: sector_impacts[x])
    bearish_stocks['Impact Score'] = bearish_stocks['Sector'].apply(lambda x: abs(sector_impacts[x]))
    neutral_stocks['Impact Score'] = 0
    
    bullish_stocks = bullish_stocks.sort_values('Impact Score', ascending=False)
    bearish_stocks = bearish_stocks.sort_values('Impact Score', ascending=False)
    
    return {
        'bullish': bullish_stocks,
        'bearish': bearish_stocks,
        'neutral': neutral_stocks,
        'sector_impacts': sector_impacts
    }

# --- GENERATE ASTROLOGICAL EVENTS ---
def generate_astrological_events(input_date, event_type='intraday', symbol='NIFTY'):
    """Generate astrological events for any given date and symbol"""
    
    if event_type == 'intraday':
        trading_hours = get_trading_hours(symbol)
        
        # Different event patterns based on trading hours
        if trading_hours['end_hour'] > 16:  # Extended hours (global markets)
            # More events spread across longer trading day
            base_events = [
                {"time_offset": 0, "aspect": "Pre-market: Mercury square Jupiter", "impact": -0.5, "type": "bearish"},
                {"time_offset": 120, "aspect": "Asian session: Moon trine Jupiter", "impact": 0.8, "type": "bullish"},
                {"time_offset": 240, "aspect": "London open: Mars sextile Jupiter", "impact": 0.4, "type": "neutral"},
                {"time_offset": 360, "aspect": "European session: Venus opposition Saturn", "impact": -0.6, "type": "bearish"},
                {"time_offset": 480, "aspect": "NY pre-market: Sun conjunct Mercury", "impact": 0.3, "type": "neutral"},
                {"time_offset": 600, "aspect": "US open: Mars conjunct Uranus", "impact": 1.0, "type": "bullish"},
                {"time_offset": 720, "aspect": "Mid-day: Moon square Saturn", "impact": -0.4, "type": "bearish"},
                {"time_offset": 840, "aspect": "Afternoon: Jupiter trine Neptune", "impact": 0.7, "type": "bullish"},
                {"time_offset": 960, "aspect": "US close approach", "impact": 0.2, "type": "neutral"},
                {"time_offset": 1080, "aspect": "After hours: Void Moon", "impact": -0.3, "type": "bearish"},
                {"time_offset": 1135, "aspect": "Session close", "impact": 0.1, "type": "neutral"}
            ]
        else:  # Standard Indian market hours
            base_events = [
                {"time_offset": 0, "aspect": "Opening: Mercury square Jupiter + Void Moon", "impact": -0.5, "type": "bearish"},
                {"time_offset": 45, "aspect": "Early trade: Moon trine Jupiter", "impact": 1.0, "type": "bullish"},
                {"time_offset": 135, "aspect": "Mid-morning: Mars sextile Jupiter", "impact": 0.3, "type": "neutral"},
                {"time_offset": 195, "aspect": "Pre-lunch: Sun in Leo (no aspects)", "impact": 0.0, "type": "neutral"},
                {"time_offset": 285, "aspect": "Post-lunch: Moon square Saturn", "impact": -0.8, "type": "bearish"},
                {"time_offset": 345, "aspect": "Late trade: Venus-Saturn opposition", "impact": -0.6, "type": "bearish"},
                {"time_offset": 375, "aspect": "Close", "impact": 0.1, "type": "neutral"}
            ]
        
        events = []
        if isinstance(input_date, date_class) and not isinstance(input_date, datetime):
            dt = datetime.combine(input_date, datetime.min.time())
        else:
            dt = input_date
            
        start_time = dt.replace(hour=trading_hours['start_hour'], minute=trading_hours['start_minute'])
        
        for event in base_events:
            event_time = start_time + timedelta(minutes=event["time_offset"])
            events.append({
                "time": event_time,
                "aspect": event["aspect"],
                "impact": event["impact"],
                "type": event["type"],
                "price": 0
            })
        
        return events
    
    else:  # monthly events remain the same
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
        
        if isinstance(input_date, datetime):
            year, month = input_date.year, input_date.month
        else:
            year, month = input_date.year, input_date.month
            
        days_in_month = calendar.monthrange(year, month)[1]
        
        events = []
        for event in base_events:
            day = min(event["day_offset"], days_in_month)
            event_date = datetime(year, month, day)
            events.append({
                "date": event_date,
                "aspect": event["aspect"],
                "impact": event["impact"],
                "type": event["type"],
                "price": 0
            })
        
        return events

# --- ENHANCED INTRADAY CHART ---
def generate_intraday_chart(symbol, starting_price, selected_date):
    """Generate enhanced intraday chart with dynamic layout"""
    symbol_info = get_symbol_info(symbol)
    trading_hours = get_trading_hours(symbol)
    
    if isinstance(selected_date, date_class) and not isinstance(selected_date, datetime):
        selected_date = datetime.combine(selected_date, datetime.min.time())
    
    start_time = selected_date.replace(hour=trading_hours['start_hour'], minute=trading_hours['start_minute'])
    end_time = selected_date.replace(hour=trading_hours['end_hour'], minute=trading_hours['end_minute'])
    
    # Adjust interval based on trading session length
    session_hours = (end_time - start_time).total_seconds() / 3600
    if session_hours > 12:
        interval = '30T'  # 30-minute intervals for long sessions
    else:
        interval = '15T'  # 15-minute intervals for shorter sessions
    
    times = pd.date_range(start=start_time, end=end_time, freq=interval)
    
    prices = np.zeros(len(times))
    base_price = starting_price
    
    events = generate_astrological_events(selected_date, 'intraday', symbol)
    
    # Adjust event impacts based on symbol type
    symbol_multiplier = 1.0
    if symbol in ['GOLD', 'SILVER']:
        symbol_multiplier = 0.8  # Precious metals less volatile to aspects
    elif symbol in ['BTC']:
        symbol_multiplier = 2.0  # Crypto more volatile
    elif symbol in ['CRUDE']:
        symbol_multiplier = 1.5  # Energy commodities more responsive
    
    for event in events:
        price_change = event["impact"] * base_price * 0.01 * symbol_multiplier
        event["price"] = base_price + price_change
    
    # Generate price movements
    for i, time in enumerate(times):
        closest_event = min(events, key=lambda x: abs((x["time"] - time).total_seconds()))
        distance = abs((closest_event["time"] - time).total_seconds()) / 3600
        
        # Adjust volatility based on symbol
        base_volatility = 0.15 if distance < 0.5 else 0.05
        if symbol in ['BTC']:
            base_volatility *= 3.0
        elif symbol in ['GOLD', 'SILVER']:
            base_volatility *= 0.5
        elif symbol in ['CRUDE']:
            base_volatility *= 2.0
        
        random_change = np.random.normal(0, base_volatility)
        event_influence = closest_event["impact"] * np.exp(-distance) * symbol_multiplier
        
        if i == 0:
            prices[i] = base_price
        else:
            change = (event_influence + random_change) * base_price * 0.001
            prices[i] = prices[i-1] + change
    
    df_intraday = pd.DataFrame({
        'Time': times,
        'Price': prices,
        'Aspect': [min(events, key=lambda x: abs((x["time"] - t).total_seconds()))["aspect"] for t in times]
    })
    
    # Create dynamic figure layout
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, height_ratios=[4, 1, 1, 1], width_ratios=[5, 2, 1])
    
    # Main price chart
    ax_main = fig.add_subplot(gs[0, :2])
    
    for i in range(1, len(df_intraday)):
        color = 'green' if df_intraday['Price'].iloc[i] > df_intraday['Price'].iloc[i-1] else 'red'
        ax_main.plot(df_intraday['Time'].iloc[i-1:i+1], 
                    df_intraday['Price'].iloc[i-1:i+1], 
                    color=color, linewidth=2.5)
    
    # Mark key events
    for event in events:
        color_map = {'bullish': 'green', 'bearish': 'red', 'neutral': 'blue'}
        ax_main.axvline(x=event['time'], color=color_map[event['type']], 
                       linestyle='--', alpha=0.7, linewidth=2)
        
        marker_color = color_map[event['type']]
        ax_main.scatter(event['time'], event['price'], color=marker_color, 
                       s=100, zorder=5, edgecolor='black', linewidth=1)
        
        # Dynamic annotation positioning
        y_offset = base_price * 0.02 if len(str(int(base_price))) >= 4 else base_price * 0.05
        y_pos = event['price'] + y_offset if event['price'] < base_price * 1.01 else event['price'] - y_offset
        
        ax_main.annotate(event['aspect'][:30] + '...' if len(event['aspect']) > 30 else event['aspect'], 
                        xy=(event['time'], event['price']),
                        xytext=(event['time'], y_pos),
                        arrowprops=dict(arrowstyle='->', color=marker_color, lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                        fontsize=8, ha='center')
    
    # Dynamic title and formatting
    currency_symbol = symbol_info['currency']
    ax_main.set_title(f'{symbol_info["name"]} ({symbol}) - {selected_date.strftime("%B %d, %Y")}\n'
                     f'Astrological Trading Analysis | Sector: {symbol_info["sector"]}', 
                     fontsize=16, pad=20)
    ax_main.set_xlabel(f'Time ({trading_hours["start_hour"]}:00 - {trading_hours["end_hour"]}:00)', fontsize=12)
    ax_main.set_ylabel(f'Price ({currency_symbol})', fontsize=12)
    
    # Dynamic time formatting based on session length
    if session_hours > 12:
        ax_main.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    else:
        ax_main.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    plt.setp(ax_main.get_xticklabels(), rotation=45, ha='right')
    
    # Closing price annotation
    close_price = df_intraday["Price"].iloc[-1]
    price_change = close_price - base_price
    price_change_pct = (price_change / base_price) * 100
    
    ax_main.annotate(f'Close: {currency_symbol}{close_price:.2f}\n'
                    f'Change: {price_change:+.2f} ({price_change_pct:+.2f}%)', 
                    xy=(df_intraday['Time'].iloc[-1], close_price),
                    xytext=(df_intraday['Time'].iloc[-1] - timedelta(hours=session_hours*0.2), 
                           close_price + base_price * 0.02),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                    fontsize=11, fontweight='bold',
                    bbox=dict(facecolor='yellow', alpha=0.7, edgecolor='none', pad=2))
    
    # Planetary wheel
    ax_wheel = fig.add_subplot(gs[0, 2])
    draw_planetary_wheel(ax_wheel, selected_date, size=0.4)
    
    # Volume chart (simulated with realistic patterns)
    ax_volume = fig.add_subplot(gs[1, :2])
    
    # Generate more realistic volume based on symbol type
    if symbol in ['BTC']:
        base_volume = np.random.randint(50000, 200000, size=len(times))
    elif symbol in ['GOLD', 'SILVER', 'CRUDE']:
        base_volume = np.random.randint(10000, 50000, size=len(times))
    elif symbol in ['DOWJONES']:
        base_volume = np.random.randint(100000, 500000, size=len(times))
    else:  # Indian stocks
        base_volume = np.random.randint(1000, 10000, size=len(times))
    
    colors_volume = ['green' if df_intraday['Price'].iloc[i] > df_intraday['Price'].iloc[i-1] 
                    else 'red' for i in range(1, len(df_intraday))]
    colors_volume.insert(0, 'green')
    
    ax_volume.bar(df_intraday['Time'], base_volume, color=colors_volume, alpha=0.7)
    ax_volume.set_title('Volume', fontsize=12)
    ax_volume.set_ylabel('Volume', fontsize=10)
    ax_volume.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax_volume.get_xticklabels(), rotation=45, ha='right')
    
    # Technical indicators (RSI simulation)
    ax_rsi = fig.add_subplot(gs[2, :2])
    rsi_values = 50 + np.random.normal(0, 15, len(times))  # Simulated RSI
    rsi_values = np.clip(rsi_values, 0, 100)
    
    ax_rsi.plot(df_intraday['Time'], rsi_values, color='purple', linewidth=2)
    ax_rsi.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
    ax_rsi.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
    ax_rsi.fill_between(df_intraday['Time'], 30, 70, alpha=0.1, color='gray')
    ax_rsi.set_title('RSI (14)', fontsize=12)
    ax_rsi.set_ylabel('RSI', fontsize=10)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.legend(loc='upper right', fontsize=8)
    
    # Aspect strength indicator
    ax_aspect = fig.add_subplot(gs[3, :2])
    aspect_times = [event['time'] for event in events]
    aspect_strengths = [abs(event['impact']) for event in events]
    aspect_colors = [{'bullish': 'green', 'bearish': 'red', 'neutral': 'blue'}[event['type']] for event in events]
    
    ax_aspect.scatter(aspect_times, aspect_strengths, color=aspect_colors, s=100, zorder=3)
    ax_aspect.plot(aspect_times, aspect_strengths, color='gray', alpha=0.5, linestyle='--')
    ax_aspect.set_title('Astrological Aspect Strength', fontsize=12)
    ax_aspect.set_ylabel('Strength', fontsize=10)
    ax_aspect.set_ylim(0, max(aspect_strengths) * 1.2 if aspect_strengths else 1.5)
    
    # Symbol info panel
    ax_info = fig.add_subplot(gs[1:, 2])
    ax_info.axis('off')
    
    info_text = f"""
SYMBOL INFO
-----------
Name: {symbol_info['name']}
Sector: {symbol_info['sector']}
Currency: {symbol_info['currency']}

TRADING HOURS
-------------
Start: {trading_hours['start_hour']:02d}:{trading_hours['start_minute']:02d}
End: {trading_hours['end_hour']:02d}:{trading_hours['end_minute']:02d}
Session: {session_hours:.1f} hours

PRICE DATA
----------
Open: {currency_symbol}{base_price:.2f}
Close: {currency_symbol}{close_price:.2f}
Change: {price_change:+.2f}
Change%: {price_change_pct:+.2f}%

High: {currency_symbol}{max(prices):.2f}
Low: {currency_symbol}{min(prices):.2f}
Range: {currency_symbol}{max(prices)-min(prices):.2f}
"""
    
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig

# --- ENHANCED MONTHLY CHART ---
def generate_monthly_chart(symbol, starting_price, selected_month, selected_year):
    """Generate enhanced monthly chart with dynamic layout"""
    symbol_info = get_symbol_info(symbol)
    
    start_date = datetime(selected_year, selected_month, 1)
    days_in_month = calendar.monthrange(selected_year, selected_month)[1]
    end_date = datetime(selected_year, selected_month, days_in_month)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    prices = np.zeros(len(dates))
    base_price = starting_price
    
    events = generate_astrological_events(start_date, 'monthly', symbol)
    
    # Adjust event impacts based on symbol type
    symbol_multiplier = 1.0
    if symbol in ['GOLD', 'SILVER']:
        symbol_multiplier = 0.8
    elif symbol in ['BTC']:
        symbol_multiplier = 2.5
    elif symbol in ['CRUDE']:
        symbol_multiplier = 1.8
    
    for event in events:
        price_change = event["impact"] * base_price * 0.01 * symbol_multiplier
        event["price"] = base_price + price_change
    
    # Generate price movements
    for i, date in enumerate(dates):
        closest_event = min(events, key=lambda x: abs((x["date"].date() - date.date()).days))
        distance = abs((closest_event["date"].date() - date.date()).days)
        
        base_volatility = 0.3 if distance < 2 else 0.1
        if symbol in ['BTC']:
            base_volatility *= 4.0
        elif symbol in ['GOLD', 'SILVER']:
            base_volatility *= 0.6
        elif symbol in ['CRUDE']:
            base_volatility *= 2.5
        
        random_change = np.random.normal(0, base_volatility)
        event_influence = closest_event["impact"] * np.exp(-distance/2) * symbol_multiplier
        
        if i == 0:
            prices[i] = base_price
        else:
            change = (event_influence + random_change) * base_price * 0.002
            prices[i] = prices[i-1] + change
    
    df_monthly = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Aspect': [min(events, key=lambda x: abs((x["date"].date() - d.date()).days))["aspect"] for d in dates]
    })
    
    # Create dynamic figure layout
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, height_ratios=[4, 1, 1, 1], width_ratios=[5, 2, 1])
    
    # Main price chart
    ax_main = fig.add_subplot(gs[0, :2])
    
    for i in range(1, len(df_monthly)):
        color = 'green' if df_monthly['Price'].iloc[i] > df_monthly['Price'].iloc[i-1] else 'red'
        ax_main.plot(df_monthly['Date'].iloc[i-1:i+1], 
                    df_monthly['Price'].iloc[i-1:i+1], 
                    color=color, linewidth=3)
    
    # Mark key events
    for event in events:
        color_map = {'bullish': 'green', 'bearish': 'red', 'neutral': 'blue'}
        ax_main.axvline(x=event['date'], color=color_map[event['type']], 
                       linestyle='--', alpha=0.7, linewidth=2)
        
        marker_color = color_map[event['type']]
        ax_main.scatter(event['date'], event['price'], color=marker_color, 
                       s=150, zorder=5, edgecolor='black', linewidth=1.5)
        
        y_offset = base_price * 0.03
        y_pos = event['price'] + y_offset if event['price'] < base_price * 1.02 else event['price'] - y_offset
        
        ax_main.annotate(event['aspect'][:25] + '...' if len(event['aspect']) > 25 else event['aspect'], 
                        xy=(event['date'], event['price']),
                        xytext=(event['date'], y_pos),
                        arrowprops=dict(arrowstyle='->', color=marker_color, lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                        fontsize=9, ha='center')
    
    # Dynamic formatting
    currency_symbol = symbol_info['currency']
    ax_main.set_title(f'{symbol_info["name"]} ({symbol}) - {start_date.strftime("%B %Y")}\n'
                     f'Monthly Astrological Analysis | Sector: {symbol_info["sector"]}', 
                     fontsize=16, pad=20)
    ax_main.set_xlabel('Date', fontsize=12)
    ax_main.set_ylabel(f'Price ({currency_symbol})', fontsize=12)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax_main.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, days_in_month//10)))
    plt.setp(ax_main.get_xticklabels(), rotation=45, ha='right')
    
    # Monthly close annotation
    close_price = df_monthly["Price"].iloc[-1]
    price_change = close_price - base_price
    price_change_pct = (price_change / base_price) * 100
    
    ax_main.annotate(f'Month Close: {currency_symbol}{close_price:.2f}\n'
                    f'Monthly Change: {price_change:+.2f} ({price_change_pct:+.2f}%)', 
                    xy=(df_monthly['Date'].iloc[-1], close_price),
                    xytext=(df_monthly['Date'].iloc[-1] - timedelta(days=days_in_month//4), 
                           close_price + base_price * 0.03),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                    fontsize=12, fontweight='bold',
                    bbox=dict(facecolor='yellow', alpha=0.7, edgecolor='none', pad=2))
    
    # Planetary positions for key dates
    ax_planets = fig.add_subplot(gs[0, 2])
    ax_planets.set_title('Key Planetary\nPositions', fontsize=10)
    key_dates = [
        start_date,
        start_date + timedelta(days=days_in_month//3),
        start_date + timedelta(days=2*days_in_month//3),
        end_date
    ]
    
    for i, date in enumerate(key_dates):
        ax_sub = fig.add_axes([0.70, 0.8-i*0.15, 0.12, 0.12])
        draw_planetary_wheel(ax_sub, date, size=0.4)
        ax_sub.set_title(f'{date.strftime("%b %d")}', fontsize=8)
    
    # Volume chart
    ax_volume = fig.add_subplot(gs[1, :2])
    
    if symbol in ['BTC']:
        volume = np.random.randint(500000, 2000000, size=len(dates))
    elif symbol in ['GOLD', 'SILVER', 'CRUDE']:
        volume = np.random.randint(100000, 500000, size=len(dates))
    elif symbol in ['DOWJONES']:
        volume = np.random.randint(1000000, 5000000, size=len(dates))
    else:
        volume = np.random.randint(10000, 100000, size=len(dates))
    
    colors_volume = ['green' if df_monthly['Price'].iloc[i] > df_monthly['Price'].iloc[i-1] 
                    else 'red' for i in range(1, len(df_monthly))]
    colors_volume.insert(0, 'green')
    
    ax_volume.bar(df_monthly['Date'], volume, color=colors_volume, alpha=0.7)
    ax_volume.set_title('Daily Volume', fontsize=12)
    ax_volume.set_ylabel('Volume', fontsize=10)
    ax_volume.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    # Moving averages
    ax_ma = fig.add_subplot(gs[2, :2])
    ma_5 = df_monthly['Price'].rolling(window=5, min_periods=1).mean()
    ma_20 = df_monthly['Price'].rolling(window=min(20, len(df_monthly)), min_periods=1).mean()
    
    ax_ma.plot(df_monthly['Date'], ma_5, color='blue', linewidth=2, label='MA5', alpha=0.7)
    ax_ma.plot(df_monthly['Date'], ma_20, color='red', linewidth=2, label='MA20', alpha=0.7)
    ax_ma.fill_between(df_monthly['Date'], ma_5, ma_20, alpha=0.1, 
                      color='green' if ma_5.iloc[-1] > ma_20.iloc[-1] else 'red')
    ax_ma.set_title('Moving Averages', fontsize=12)
    ax_ma.set_ylabel('Price', fontsize=10)
    ax_ma.legend(loc='upper left', fontsize=10)
    
    # Aspect calendar
    ax_calendar = fig.add_subplot(gs[3, :2])
    aspect_dates = [event['date'] for event in events]
    aspect_strengths = [abs(event['impact']) for event in events]
    aspect_colors = [{'bullish': 'green', 'bearish': 'red', 'neutral': 'blue'}[event['type']] for event in events]
    
    ax_calendar.scatter(aspect_dates, aspect_strengths, color=aspect_colors, s=200, zorder=3)
    ax_calendar.plot(aspect_dates, aspect_strengths, color='gray', alpha=0.5, linestyle='--')
    ax_calendar.set_title('Monthly Astrological Event Strength', fontsize=12)
    ax_calendar.set_ylabel('Impact Strength', fontsize=10)
    ax_calendar.set_ylim(0, max(aspect_strengths) * 1.2 if aspect_strengths else 2)
    
    # Monthly summary panel
    ax_summary = fig.add_subplot(gs[1:, 2])
    ax_summary.axis('off')
    
    monthly_high = max(prices)
    monthly_low = min(prices)
    monthly_range = monthly_high - monthly_low
    avg_price = np.mean(prices)
    
    summary_text = f"""
MONTHLY SUMMARY
--------------
Symbol: {symbol}
Sector: {symbol_info['sector']}
Month: {start_date.strftime('%B %Y')}

PRICE STATISTICS
---------------
Open: {currency_symbol}{base_price:.2f}
Close: {currency_symbol}{close_price:.2f}
Change: {price_change:+.2f}
Change%: {price_change_pct:+.2f}%

High: {currency_symbol}{monthly_high:.2f}
Low: {currency_symbol}{monthly_low:.2f}
Range: {currency_symbol}{monthly_range:.2f}
Average: {currency_symbol}{avg_price:.2f}

VOLATILITY
----------
Daily Avg: {np.std(np.diff(prices)):.2f}
Monthly Vol: {(monthly_range/avg_price)*100:.1f}%

TREND ANALYSIS
--------------
Bullish Days: {sum(1 for i in range(1, len(prices)) if prices[i] > prices[i-1])}
Bearish Days: {sum(1 for i in range(1, len(prices)) if prices[i] < prices[i-1])}
Neutral Days: {sum(1 for i in range(1, len(prices)) if prices[i] == prices[i-1])}
"""
    
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes, fontsize=8,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    return fig

# --- ANALYZE ASPECTS ---
def analyze_aspects():
    """Enhanced aspect analysis with dynamic content"""
    aspects_data = {
        'Aspect': [
            'Mercury Retrograde', 'Venus Opposition Saturn', 'Moon-Jupiter Trine', 
            'Full Moon', 'Jupiter Square Saturn', 'Mercury Direct',
            'Venus enters Libra', 'New Moon', 'Mars-Uranus Conjunction',
            'Sun-Pluto Sextile'
        ],
        'Market Impact': [
            'High Volatility', 'Bearish Pressure', 'Bullish Surge', 
            'Trend Reversal', 'Major Tension', 'Clarity Returns',
            'Financial Rally', 'Strong Bullish', 'Energy Surge',
            'Structural Change'
        ],
        'Typical Price Change': [
            '±2-3%', '-1.5-2%', '+1-2%', 
            '±1-1.5%', '-2-3%', '+0.5-1%',
            '+0.5-1%', '+1-2%', '+2-4%',
            '±1-2%'
        ],
        'Sector Focus': [
            'All Sectors', 'Banking/Realty', 'Broad Market', 
            'Technology', 'Financials', 'Technology',
            'Banking/Finance', 'Broad Market', 'Energy/Commodities',
            'Infrastructure/PSUs'
        ],
        'Best Symbols': [
            'Gold, BTC', 'Gold, Silver', 'FMCG, Pharma', 
            'Tech Stocks', 'Defensive', 'Tech, Crypto',
            'Banking', 'Growth Stocks', 'Energy, Crude',
            'PSU, Infrastructure'
        ]
    }
    
    df_aspects = pd.DataFrame(aspects_data)
    
    # Create enhanced visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Price change impact chart
    price_changes = []
    for change in df_aspects['Typical Price Change']:
        clean_change = change.replace('%', '').replace('±', '')
        if '-' in clean_change and not clean_change.startswith('-'):
            num_str = clean_change.split('-')[1]  # Take higher value for impact
        else:
            num_str = clean_change.replace('+', '')
        
        try:
            num = float(num_str)
        except:
            num = 1.0
        price_changes.append(num)
    
    colors = ['red' if 'Bearish' in impact or 'Tension' in impact or 'Volatility' in impact 
              else 'orange' if 'Reversal' in impact or 'Change' in impact
              else 'green' for impact in df_aspects['Market Impact']]
    
    bars1 = ax1.bar(range(len(df_aspects)), price_changes, color=colors, alpha=0.7)
    ax1.set_title('Astrological Aspect Impact on Price Changes', fontsize=14)
    ax1.set_ylabel('Maximum Price Change (%)', fontsize=12)
    ax1.set_xticks(range(len(df_aspects)))
    ax1.set_xticklabels(df_aspects['Aspect'], rotation=45, ha='right')
    
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.annotate(f'{height}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    # Sector distribution pie chart
    sector_counts = {}
    for sectors in df_aspects['Sector Focus']:
        for sector in sectors.split('/'):
            sector = sector.strip()
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
    
    ax2.pie(sector_counts.values(), labels=sector_counts.keys(), autopct='%1.1f%%', startangle=90)
    ax2.set_title('Most Affected Sectors by Astrological Aspects', fontsize=14)
    
    # Market impact distribution
    impact_counts = {}
    for impact in df_aspects['Market Impact']:
        impact_type = 'Bullish' if any(word in impact for word in ['Bullish', 'Rally', 'Surge', 'Returns']) else \
                     'Bearish' if any(word in impact for word in ['Bearish', 'Pressure', 'Tension']) else \
                     'Neutral'
        impact_counts[impact_type] = impact_counts.get(impact_type, 0) + 1
    
    colors_impact = ['green', 'red', 'gray']
    ax3.bar(impact_counts.keys(), impact_counts.values(), color=colors_impact, alpha=0.7)
    ax3.set_title('Distribution of Market Impact Types', fontsize=14)
    ax3.set_ylabel('Number of Aspects', fontsize=12)
    
    # Best performing symbols chart
    symbol_mentions = {}
    for symbols in df_aspects['Best Symbols']:
        for symbol in symbols.split(', '):
            symbol = symbol.strip()
            symbol_mentions[symbol] = symbol_mentions.get(symbol, 0) + 1
    
    sorted_symbols = sorted(symbol_mentions.items(), key=lambda x: x[1], reverse=True)
    symbols, counts = zip(*sorted_symbols) if sorted_symbols else ([], [])
    
    ax4.barh(symbols, counts, color='gold', alpha=0.7)
    ax4.set_title('Most Favorable Symbols Across Aspects', fontsize=14)
    ax4.set_xlabel('Favorable Mentions', fontsize=12)
    
    return fig, df_aspects

# --- STREAMLIT APP ---
def main():
    # Page configuration for better responsive design
    st.set_page_config(
        page_title="🌟 Astrological Trading Dashboard",
        page_icon="🌟",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .symbol-input {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #e8f4fd;
        border-left: 5px solid #1f77b4;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🌟 Astrological Trading Dashboard</h1>
        <p>Advanced Financial Analysis through Planetary Movements & Cosmic Patterns</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for inputs with enhanced design
    with st.sidebar:
        st.markdown("### 📊 Dashboard Controls")
        
        # Dashboard section selection with better descriptions
        dashboard_section = st.selectbox(
            '🎯 Choose Analysis Section:',
            [
                'Summary Table - Market Overview',
                'Stock Filter - Sector Analysis', 
                'Aspect Analysis - Deep Insights',
                'Intraday Chart - Live Patterns',
                'Monthly Chart - Trend Analysis'
            ]
        )
        
        # Extract the main section name
        section_name = dashboard_section.split(' - ')[0]
        
        st.markdown("---")
        
        # Symbol selection with enhanced interface
        if section_name in ['Intraday Chart', 'Monthly Chart']:
            st.markdown("### 📈 Symbol Configuration")
            
            # Popular symbols with categories
            symbol_categories = {
                'Indian Indices': ['NIFTY', 'BANKNIFTY'],
                'Indian Stocks': ['TCS', 'ICICIBANK', 'MARUTI', 'DLF', 'NESTLEIND', 'RELIANCE', 'SBI', 'SUNPHARMA', 'DRREDDY'],
                'Global Markets': ['DOWJONES'],
                'Commodities': ['GOLD', 'SILVER', 'CRUDE'],
                'Cryptocurrency': ['BTC']
            }
            
            selected_category = st.selectbox('📂 Select Category:', list(symbol_categories.keys()))
            
            if selected_category:
                symbol_options = symbol_categories[selected_category]
                selected_symbol = st.selectbox('🎯 Choose Symbol:', symbol_options)
                
                # Custom symbol input
                custom_symbol = st.text_input('✏️ Or enter custom symbol:', max_chars=10)
                symbol = custom_symbol.upper() if custom_symbol else selected_symbol
                
                # Get symbol info for dynamic defaults
                symbol_info = get_symbol_info(symbol)
                trading_hours = get_trading_hours(symbol)
                
                # Display symbol information
                st.markdown(f"""
                <div class="info-box">
                    <strong>📊 {symbol_info['name']}</strong><br>
                    <small>Sector: {symbol_info['sector']}</small><br>
                    <small>Currency: {symbol_info['currency']}</small><br>
                    <small>Trading: {trading_hours['start_hour']:02d}:{trading_hours['start_minute']:02d} - {trading_hours['end_hour']:02d}:{trading_hours['end_minute']:02d}</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Price input with dynamic default
                starting_price = st.number_input(
                    f'💰 Starting Price ({symbol_info["currency"]}):',
                    min_value=0.01,
                    value=symbol_info['default_price'],
                    step=1.0 if symbol_info['default_price'] > 100 else 0.01,
                    format="%.2f"
                )
                
                # Date/time selection based on chart type
                if section_name == 'Intraday Chart':
                    selected_date = st.date_input(
                        '📅 Select Trading Date:',
                        value=datetime(2025, 8, 5).date(),
                        min_value=datetime(2020, 1, 1).date(),
                        max_value=datetime(2030, 12, 31).date()
                    )
                elif section_name == 'Monthly Chart':
                    col1, col2 = st.columns(2)
                    with col1:
                        selected_month = st.selectbox(
                            '📅 Month:',
                            range(1, 13),
                            format_func=lambda x: calendar.month_name[x],
                            index=7  # August
                        )
                    with col2:
                        selected_year = st.selectbox(
                            '📅 Year:',
                            range(2020, 2031),
                            index=5  # 2025
                        )
        
        # Trading insights
        st.markdown("---")
        st.markdown("### 🔮 Quick Insights")
        
        # Generate today's aspects for sidebar display
        aspects = generate_todays_aspects()
        bullish_count = sum(1 for aspect in aspects if aspect['type'] == 'bullish')
        bearish_count = sum(1 for aspect in aspects if aspect['type'] == 'bearish')
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🟢 Bullish Aspects", bullish_count)
        with col2:
            st.metric("🔴 Bearish Aspects", bearish_count)
        
        # Market sentiment
        if bullish_count > bearish_count:
            sentiment = "🟢 Bullish"
            sentiment_color = "green"
        elif bearish_count > bullish_count:
            sentiment = "🔴 Bearish"
            sentiment_color = "red"
        else:
            sentiment = "🟡 Neutral"
            sentiment_color = "orange"
        
        st.markdown(f"**Overall Sentiment:** <span style='color:{sentiment_color}'>{sentiment}</span>", 
                   unsafe_allow_html=True)

    # Main content area
    aspects = generate_todays_aspects()
    
    if section_name == 'Summary Table':
        st.header('📋 Market Summary & Astrological Overview')
        
        # Create columns for better layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader('📊 Today\'s Astrological Aspects Impact')
            summary_df = create_summary_table(aspects)
            
            # Style the dataframe
            styled_df = summary_df.style.apply(
                lambda x: ['background-color: #d4edda' if 'Bullish' in str(val) or '+' in str(val) 
                          else 'background-color: #f8d7da' if 'Bearish' in str(val) or 'Downside' in str(val)
                          else '' for val in x], axis=0
            )
            
            st.dataframe(styled_df, use_container_width=True)
        
        with col2:
            st.subheader('🎯 Key Metrics')
            
            # Calculate impact scores
            total_impact = sum(abs(aspect['impact']) for aspect in aspects)
            avg_impact = total_impact / len(aspects) if aspects else 0
            
            st.metric("Total Cosmic Energy", f"{total_impact:.1f}")
            st.metric("Average Impact", f"{avg_impact:.2f}")
            st.metric("Active Aspects", len(aspects))
            
            # Risk assessment
            high_risk_aspects = sum(1 for aspect in aspects if abs(aspect['impact']) > 0.7)
            risk_level = "High" if high_risk_aspects >= 3 else "Medium" if high_risk_aspects >= 1 else "Low"
            risk_color = "red" if risk_level == "High" else "orange" if risk_level == "Medium" else "green"
            
            st.markdown(f"**Risk Level:** <span style='color:{risk_color}'>{risk_level}</span>", 
                       unsafe_allow_html=True)
        
        # Detailed insights
        st.subheader('🔮 Detailed Market Insights')
        
        # Create tabs for different insights
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Strategy", "📈 Opportunities", "⚠️ Risks", "🌟 Cosmic Events"])
        
        with tab1:
            st.markdown("""
            **🎯 Recommended Trading Strategy:**
            
            **🟢 Bullish Opportunities:**
            - **Energy Sector**: Mars-Uranus conjunction favors Reliance, Crude Oil
            - **Precious Metals**: Multiple aspects support Gold and Silver
            - **FMCG & Pharma**: Moon-Neptune trine provides defensive strength
            - **PSU Stocks**: Sun-Pluto sextile indicates structural positives
            
            **🔴 Bearish Risks:**
            - **Banking Sector**: Mercury-Jupiter square creates volatility
            - **Automotive & Realty**: Venus-Saturn opposition brings pressure
            - **Technology**: Mixed signals, trade with caution
            
            **⚡ High-Impact Trades:**
            - Consider Gold positions during Venus-Saturn opposition
            - Energy stocks may see sharp moves (Mars-Uranus)
            - BTC could be volatile but trending up on global aspects
            """)
        
        with tab2:
            st.markdown("""
            **📈 Sector-wise Opportunities:**
            
            **🥇 Top Picks:**
            1. **Gold/Silver**: Multiple supportive aspects across all planetary configurations
            2. **Energy Commodities**: Mars-Uranus conjunction + global supply dynamics
            3. **Pharmaceutical**: Moon-Neptune trine supports defensive healthcare
            4. **PSU Banking**: Sun-Pluto sextile for structural transformation
            
            **🎯 Specific Symbols:**
            - **GOLD**: $2,050+ target on safe-haven demand
            - **CRUDE**: Energy transition + Mars-Uranus = volatility opportunities
            - **BTC**: Crypto favorable on Uranus-Pluto aspects
            - **SBI**: PSU transformation theme
            """)
        
        with tab3:
            st.markdown("""
            **⚠️ Risk Management:**
            
            **🔴 High-Risk Sectors:**
            - **Private Banking**: ICICI Bank under Mercury-Jupiter square pressure
            - **Automotive**: Maruti facing Venus-Saturn headwinds
            - **Real Estate**: DLF vulnerable to credit tightening aspects
            
            **📊 Risk Mitigation:**
            - Reduce position sizes during Mercury-Jupiter square (high volatility)
            - Use stop-losses 2-3% below support for Venus-Saturn affected stocks
            - Avoid leveraged positions in Midcap segment (Mars-Uranus volatility)
            
            **⏰ Timing Risks:**
            - Morning session volatility expected (Mercury aspects)
            - Post-lunch session may see pressure (Saturn influence)
            """)
        
        with tab4:
            st.markdown("""
            **🌟 Today's Cosmic Events Schedule:**
            
            **🌅 Pre-Market (Before 9:15 AM):**
            - Mercury-Jupiter square builds tension
            - Global markets influence domestic opening
            
            **🌄 Morning Session (9:15-12:00):**
            - Initial volatility from Mercury aspects
            - Energy stocks may show strength
            
            **🌞 Afternoon Session (12:00-15:30):**
            - Venus-Saturn opposition peaks
            - Defensive sectors gain relative strength
            - Banking sector under pressure
            
            **🌆 Post-Market:**
            - Global commodity movements (Gold, Crude)
            - Crypto markets reaction to day's developments
            
            **📊 Weekly Outlook:**
            - Aspects intensify mid-week
            - Weekend planetary shifts to monitor
            """)
    
    elif section_name == 'Stock Filter':
        st.header('🔍 Advanced Stock Filtering & Sector Analysis')
        
        # Create main layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader('🌟 Today\'s Astrological Configuration')
            
            # Display aspects in a nice format
            aspects_data = []
            for aspect in aspects:
                aspects_data.append({
                    'Planets': aspect['planets'],
                    'Aspect': aspect['aspect_type'],
                    'Impact': f"{aspect['impact']:+.1f}",
                    'Sentiment': aspect['type'].title(),
                    'Strength': '🔥' * min(3, int(abs(aspect['impact']) * 3))
                })
            
            aspects_df = pd.DataFrame(aspects_data)
            
            # Color code the dataframe
            def color_sentiment(val):
                if 'Bullish' in str(val):
                    return 'background-color: #d4edda; color: #155724'
                elif 'Bearish' in str(val):
                    return 'background-color: #f8d7da; color: #721c24'
                return ''
            
            styled_aspects = aspects_df.style.applymap(color_sentiment, subset=['Sentiment'])
            st.dataframe(styled_aspects, use_container_width=True)
        
        with col2:
            st.subheader('📊 Aspect Statistics')
            
            # Create a simple pie chart for aspect types
            aspect_types = {}
            for aspect in aspects:
                aspect_types[aspect['type']] = aspect_types.get(aspect['type'], 0) + 1
            
            if aspect_types:
                fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
                colors = ['green' if k == 'bullish' else 'red' if k == 'bearish' else 'gray' 
                         for k in aspect_types.keys()]
                wedges, texts, autotexts = ax_pie.pie(aspect_types.values(), 
                                                     labels=[k.title() for k in aspect_types.keys()], 
                                                     colors=colors, autopct='%1.0f%%', startangle=90)
                ax_pie.set_title('Today\'s Aspect Distribution')
                st.pyplot(fig_pie)
        
        # Sector analysis
        st.subheader('📈 Sector Impact Analysis')
        
        filtered_stocks = filter_stocks_by_aspects(aspects, STOCK_DATABASE)
        
        # Create sector impact visualization
        sector_impacts_df = pd.DataFrame({
            'Sector': list(filtered_stocks['sector_impacts'].keys()),
            'Impact Score': list(filtered_stocks['sector_impacts'].values()),
            'Recommendation': ['Strong Buy' if x > 0.5 else 'Buy' if x > 0 else 'Hold' if x == 0 
                             else 'Sell' if x > -0.5 else 'Strong Sell' 
                             for x in filtered_stocks['sector_impacts'].values()]
        })
        sector_impacts_df = sector_impacts_df.sort_values('Impact Score', ascending=False)
        
        # Enhanced bar chart
        fig_sectors, ax_sectors = plt.subplots(figsize=(14, 8))
        colors = ['darkgreen' if x > 0.5 else 'green' if x > 0 else 'gray' if x == 0 
                 else 'red' if x > -0.5 else 'darkred' 
                 for x in sector_impacts_df['Impact Score']]
        
        bars = ax_sectors.bar(sector_impacts_df['Sector'], sector_impacts_df['Impact Score'], 
                             color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax_sectors.set_title('Sector Impact Scores - Astrological Analysis', fontsize=16, pad=20)
        ax_sectors.set_ylabel('Impact Score', fontsize=12)
        ax_sectors.set_xlabel('Sector', fontsize=12)
        ax_sectors.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax_sectors.grid(True, alpha=0.3)
        
        # Add value labels and recommendations
        for i, (bar, rec) in enumerate(zip(bars, sector_impacts_df['Recommendation'])):
            height = bar.get_height()
            ax_sectors.annotate(f'{height:.2f}\n{rec}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 5 if height >= 0 else -25),
                               textcoords="offset points",
                               ha='center', va='bottom' if height >= 0 else 'top',
                               fontsize=9, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig_sectors)
        
        # Stock recommendations in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader('🟢 Bullish Stocks')
            if not filtered_stocks['bullish'].empty:
                bullish_df = filtered_stocks['bullish'][['Symbol', 'Sector', 'Impact Score']].copy()
                bullish_df['Action'] = bullish_df['Impact Score'].apply(
                    lambda x: 'Strong Buy' if x > 0.5 else 'Buy'
                )
                
                for _, row in bullish_df.iterrows():
                    with st.container():
                        st.markdown(f"""
                        **{row['Symbol']}** ({row['Sector']})  
                        Score: {row['Impact Score']:.2f} | **{row['Action']}**
                        """)
                        st.progress(min(1.0, row['Impact Score']))
            else:
                st.info("No strong bullish signals today")
        
        with col2:
            st.subheader('🔴 Bearish Stocks')
            if not filtered_stocks['bearish'].empty:
                bearish_df = filtered_stocks['bearish'][['Symbol', 'Sector', 'Impact Score']].copy()
                bearish_df['Action'] = bearish_df['Impact Score'].apply(
                    lambda x: 'Strong Sell' if x > 0.5 else 'Sell'
                )
                
                for _, row in bearish_df.iterrows():
                    with st.container():
                        st.markdown(f"""
                        **{row['Symbol']}** ({row['Sector']})  
                        Risk Score: {row['Impact Score']:.2f} | **{row['Action']}**
                        """)
                        st.progress(min(1.0, row['Impact Score']))
            else:
                st.info("No strong bearish signals today")
        
        with col3:
            st.subheader('⚪ Neutral Stocks')
            if not filtered_stocks['neutral'].empty:
                neutral_df = filtered_stocks['neutral'][['Symbol', 'Sector']].head(5)
                
                for _, row in neutral_df.iterrows():
                    st.markdown(f"**{row['Symbol']}** ({row['Sector']}) - Hold")
            else:
                st.info("All stocks showing directional bias")
    
    elif section_name == 'Aspect Analysis':
        st.header('📋 Deep Astrological Aspect Analysis')
        
        # Generate enhanced analysis
        fig, df_aspects = analyze_aspects()
        st.pyplot(fig)
        
        # Display detailed aspect table
        st.subheader('📊 Detailed Aspect Reference Table')
        
        # Add more columns for better analysis
        df_enhanced = df_aspects.copy()
        df_enhanced['Trading Action'] = df_enhanced.apply(
            lambda row: 'Hedge/Reduce' if 'Bearish' in row['Market Impact'] or 'Tension' in row['Market Impact']
            else 'Accumulate' if 'Bullish' in row['Market Impact'] or 'Rally' in row['Market Impact']
            else 'Monitor', axis=1
        )
        
        df_enhanced['Risk Level'] = df_enhanced['Typical Price Change'].apply(
            lambda x: 'High' if any(num in x for num in ['3', '4']) 
            else 'Medium' if '2' in x else 'Low'
        )
        
        # Style the enhanced dataframe
        def highlight_risk(val):
            if val == 'High':
                return 'background-color: #ffcccc'
            elif val == 'Medium':
                return 'background-color: #ffffcc'
            elif val == 'Low':
                return 'background-color: #ccffcc'
            return ''
        
        def highlight_action(val):
            if val == 'Accumulate':
                return 'background-color: #d4edda; color: #155724'
            elif val == 'Hedge/Reduce':
                return 'background-color: #f8d7da; color: #721c24'
            elif val == 'Monitor':
                return 'background-color: #fff3cd; color: #856404'
            return ''
        
        styled_enhanced = df_enhanced.style.applymap(highlight_risk, subset=['Risk Level']).applymap(highlight_action, subset=['Trading Action'])
        st.dataframe(styled_enhanced, use_container_width=True)
        
        # Aspect interpretation guide
        st.subheader('🔭 Astrological Aspect Interpretation Guide')
        
        tab1, tab2, tab3 = st.tabs(["🌟 Aspect Types", "🪐 Planetary Influences", "📈 Trading Applications"])
        
        with tab1:
            st.markdown("""
            ### Understanding Astrological Aspects
            
            **🔄 Conjunction (0°)**: 
            - *Market Effect*: Powerful combining of energies, can create sharp moves
            - *Trading*: Expect significant price action, potential breakouts
            - *Example*: Mars-Uranus conjunction = explosive energy moves
            
            **⚔️ Square (90°)**: 
            - *Market Effect*: Tension, conflict, volatility
            - *Trading*: Increased intraday swings, good for scalping
            - *Example*: Mercury-Jupiter square = communication/policy confusion
            
            **🎯 Trine (120°)**: 
            - *Market Effect*: Harmonious, easy flow of energy
            - *Trading*: Trending moves, good for position trading
            - *Example*: Moon-Neptune trine = emotional/intuitive support
            
            **⚖️ Opposition (180°)**: 
            - *Market Effect*: Polarization, requires balance
            - *Trading*: Range-bound action, reversals possible
            - *Example*: Venus-Saturn opposition = value vs. restriction
            
            **🤝 Sextile (60°)**: 
            - *Market Effect*: Opportunity aspects, mild positive
            - *Trading*: Gentle trends, good for swing trades
            - *Example*: Sun-Pluto sextile = gradual transformation
            """)
        
        with tab2:
            st.markdown("""
            ### Planetary Market Influences
            
            **☀️ Sun**: Leadership, government policy, large-cap stocks, gold
            **🌙 Moon**: Public sentiment, emotions, consumer sectors, silver
            **☿️ Mercury**: Communication, technology, volatility, news-driven moves
            **♀️ Venus**: Finance, banking, luxury goods, relationships, copper
            **♂️ Mars**: Energy, metals, defense, aggressive moves, oil
            **♃ Jupiter**: Growth, expansion, optimism, financial sector
            **♄ Saturn**: Restriction, discipline, structure, defensive sectors
            **♅ Uranus**: Innovation, technology, sudden changes, crypto
            **♆ Neptune**: Illusion, oil, pharma, confusion, speculation
            **♇ Pluto**: Transformation, power, mining, major shifts
            
            ### Sector-Planet Correlations
            - **Technology**: Mercury, Uranus
            - **Banking**: Jupiter, Venus, Saturn  
            - **Energy**: Mars, Sun, Pluto
            - **Healthcare**: Neptune, Moon
            - **Precious Metals**: Venus, Jupiter, Sun
            - **Cryptocurrency**: Uranus, Pluto
            """)
        
        with tab3:
            st.markdown("""
            ### Practical Trading Applications
            
            **📊 Intraday Trading:**
            - Use Moon aspects for sentiment shifts (2-4 hour cycles)
            - Mercury aspects for news/volatility spikes
            - Mars aspects for energy sector breakouts
            
            **📈 Swing Trading:**
            - Venus aspects for financial sector trends (3-7 days)
            - Jupiter aspects for broad market optimism
            - Saturn aspects for defensive positioning
            
            **📉 Position Trading:**
            - Outer planet aspects (Uranus, Neptune, Pluto) for long-term themes
            - Eclipse patterns for major sector rotations
            - Retrograde periods for trend reversals
            
            **⚠️ Risk Management:**
            - Increase cash during multiple challenging aspects
            - Reduce position size during Mercury retrograde
            - Use tighter stops during Mars-Saturn squares
            
            **🎯 Sector Rotation:**
            - Follow Jupiter through zodiac signs for sector leadership
            - Track Saturn aspects for value opportunities
            - Monitor Uranus for innovation themes
            """)
    
    elif section_name == 'Intraday Chart':
        st.header(f'📈 {symbol} - Intraday Astrological Analysis')
        
        # Display symbol information prominently
        symbol_info = get_symbol_info(symbol)
        trading_hours = get_trading_hours(symbol)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Symbol", symbol)
        with col2:
            st.metric("Sector", symbol_info['sector'])
        with col3:
            st.metric("Currency", symbol_info['currency'])
        with col4:
            session_length = trading_hours['end_hour'] - trading_hours['start_hour'] + \
                           (trading_hours['end_minute'] - trading_hours['start_minute']) / 60
            st.metric("Session Hours", f"{session_length:.1f}h")
        
        # Generate and display chart
        with st.spinner(f'Generating astrological analysis for {symbol}...'):
            fig = generate_intraday_chart(symbol, starting_price, selected_date)
            st.pyplot(fig)
        
        # Trading insights based on symbol
        st.subheader(f'🎯 {symbol} Trading Insights')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Technical Outlook")
            
            # Generate insights based on symbol type
            if symbol in ['GOLD', 'SILVER']:
                st.markdown("""
                **Precious Metals Analysis:**
                - Multiple planetary aspects favor safe-haven demand
                - Venus-Saturn opposition creates financial stress → Gold strength  
                - Moon-Neptune trine supports intuitive precious metal buying
                - Best trading windows: During global uncertainty aspects
                
                **Key Levels:**
                - Watch for breakouts during Mars-Uranus conjunction
                - Support likely during Moon aspects
                - Resistance at previous highs during Saturn aspects
                """)
            
            elif symbol in ['BTC']:
                st.markdown("""
                **Cryptocurrency Analysis:**
                - Uranus aspects strongly favor crypto volatility
                - Mars-Uranus conjunction = explosive price moves
                - Traditional financial stress (Venus-Saturn) → Crypto rotation
                - High volatility expected - use proper risk management
                
                **Trading Strategy:**
                - Momentum plays during Uranus aspects
                - Contrarian plays during Saturn oppositions
                - Volume spikes likely at aspect peaks
                """)
            
            elif symbol in ['CRUDE']:
                st.markdown("""
                **Energy Commodity Analysis:**
                - Mars-Uranus conjunction directly impacts energy sector
                - Global supply disruption themes (Pluto aspects)
                - Geopolitical tensions favor energy prices
                - Weather and seasonal patterns amplified by aspects
                
                **Supply-Demand Factors:**
                - Production disruptions during Mars aspects
                - Demand surges during economic aspects
                - Storage plays during Saturn aspects
                """)
            
            elif symbol in ['DOWJONES']:
                st.markdown("""
                **US Index Analysis:**
                - Jupiter aspects favor broad market optimism
                - Saturn aspects create rotation into defensive sectors
                - Mercury aspects increase intraday volatility
                - Fed policy sensitivity during Venus-Saturn opposition
                
                **Sector Rotation:**
                - Technology during Mercury aspects
                - Energy during Mars aspects  
                - Financials during Jupiter aspects
                """)
            
            else:  # Indian stocks
                st.markdown(f"""
                **{symbol_info['sector']} Sector Analysis:**
                - Domestic market influenced by global planetary patterns
                - FII/DII flows affected by Venus-Saturn aspects
                - Sector rotation based on planetary emphasis
                - Currency impacts during outer planet aspects
                
                **Indian Market Specifics:**
                - Opening gap influenced by global overnight aspects
                - Lunch hour volatility during Moon aspects
                - Closing session strength during Jupiter aspects
                """)
        
        with col2:
            st.markdown("#### ⏰ Timing Analysis")
            
            # Generate time-specific insights based on trading hours
            if trading_hours['end_hour'] > 16:  # Extended hours
                st.markdown("""
                **Extended Session Analysis:**
                
                **🌅 Asian Session (5:00-8:00):**
                - Pre-market positioning based on overnight aspects
                - Lower volumes, higher impact from aspects
                - Key economic data releases amplified
                
                **🌍 European Session (8:00-16:00):**
                - Peak liquidity and aspect impacts
                - Central bank policy influences
                - Cross-asset correlations strongest
                
                **🌎 US Session (16:00-20:00):**
                - Maximum volatility potential
                - Aspect peaks create significant moves
                - News flow interaction with cosmic patterns
                
                **🌙 After Hours (20:00-23:55):**
                - Reduced liquidity amplifies aspect effects
                - Position adjustments for next day
                - Asian preview impact
                """)
            else:  # Indian market hours
                st.markdown("""
                **Indian Session Analysis:**
                
                **🌅 Opening (9:15-10:30):**
                - Gap opening based on global aspects
                - High volatility, aspect impacts magnified
                - Initial trend direction setting
                
                **🌞 Mid-Morning (10:30-12:00):**
                - Institutional activity peaks
                - Aspect-driven sector rotation
                - News flow integration
                
                **🍽️ Lunch Hour (12:00-13:00):**
                - Reduced activity, Moon aspects dominate
                - Range-bound unless strong aspects active
                - Position consolidation period
                
                **🌆 Closing (13:00-15:30):**
                - Final institutional positioning
                - Aspect resolution for day
                - Next-day setup formation
                """)
            
            # Risk management
            st.markdown("#### ⚠️ Risk Management")
            st.markdown(f"""
            **Position Sizing:**
            - Standard position: 1-2% of capital
            - High aspect days: Reduce to 0.5-1%
            - Strong confluence: Increase to 2-3%
            
            **Stop Loss Levels:**
            - Tight stops during Mercury aspects: 1-2%
            - Normal stops during stable aspects: 2-3%
            - Wide stops during Mars aspects: 3-5%
            
            **Profit Targets:**
            - Quick scalps: 0.5-1% (15-30 minutes)
            - Swing trades: 2-5% (2-4 hours)
            - Position trades: 5-10% (1-3 days)
            
            **Volatility Adjustments:**
            - {symbol}: Expected daily range ±{2.5 if symbol in ['BTC'] else 1.5 if symbol in ['CRUDE'] else 1.0 if symbol in ['GOLD', 'SILVER'] else 0.8}%
            - Adjust position size inversely to volatility
            - Use options for high-volatility periods
            """)
    
    elif section_name == 'Monthly Chart':
        st.header(f'📊 {symbol} - Monthly Astrological Trend Analysis')
        
        # Display symbol information
        symbol_info = get_symbol_info(symbol)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Symbol", symbol)
        with col2:
            st.metric("Analysis Period", f"{calendar.month_name[selected_month]} {selected_year}")
        with col3:
            st.metric("Sector Focus", symbol_info['sector'])
        with col4:
            st.metric("Currency", symbol_info['currency'])
        
        # Generate and display chart
        with st.spinner(f'Generating monthly analysis for {symbol}...'):
            fig = generate_monthly_chart(symbol, starting_price, selected_month, selected_year)
            st.pyplot(fig)
        
        # Monthly analysis insights
        st.subheader(f'📈 {calendar.month_name[selected_month]} {selected_year} - Strategic Analysis')
        
        # Create tabs for different analysis aspects
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Monthly Outlook", "📊 Technical Analysis", "🌙 Lunar Cycles", "💼 Portfolio Strategy"])
        
        with tab1:
            month_name = calendar.month_name[selected_month]
            
            if symbol in ['GOLD', 'SILVER']:
                st.markdown(f"""
                ### {symbol} - {month_name} {selected_year} Precious Metals Outlook
                
                **🌟 Astrological Themes:**
                - **Venus-Jupiter aspects**: Strong precious metals demand from financial uncertainty
                - **Saturn transits**: Safe-haven buying during economic restrictions
                - **Moon phases**: Emotional buying patterns aligned with lunar cycles
                - **Mercury retrograde periods**: Technical analysis less reliable, fundamentals dominate
                
                **📈 Price Drivers:**
                - Central bank policy uncertainty (Saturn aspects)
                - Currency devaluation themes (Pluto aspects)
                - Geopolitical tensions (Mars aspects)
                - Inflation hedging demand (Jupiter-Saturn aspects)
                
                **🎯 Trading Strategy:**
                - **Accumulate** during New Moon phases (stronger buying interest)
                - **Profit-take** during Full Moon phases (emotional peaks)
                - **Hold through** Mercury retrograde (avoid technical trading)
                - **Scale in** during Saturn aspects (structural support)
                
                **📊 Target Levels:**
                - **Monthly High**: Expect during Jupiter-Venus trine periods
                - **Monthly Low**: Likely during Mars-Saturn square periods
                - **Breakout Potential**: Mars-Uranus conjunction periods
                - **Support Zones**: Previous month's Jupiter aspect levels
                """)
            
            elif symbol in ['BTC']:
                st.markdown(f"""
                ### {symbol} - {month_name} {selected_year} Cryptocurrency Outlook
                
                **⚡ Astrological Themes:**
                - **Uranus-Pluto aspects**: Revolutionary technology adoption waves
                - **Mercury-Uranus aspects**: Network upgrades and technical developments
                - **Mars-Uranus conjunctions**: Explosive price movements and FOMO
                - **Saturn aspects**: Regulatory clarity or restrictions
                
                **🚀 Volatility Drivers:**
                - Institutional adoption news (Jupiter aspects)
                - Regulatory developments (Saturn aspects)
                - Technical network changes (Mercury-Uranus)
                - Market manipulation concerns (Neptune aspects)
                
                **⚠️ Risk Factors:**
                - **High volatility** during Mars-Uranus aspects (±10-20% daily swings)
                - **Regulatory risks** during Saturn-Pluto aspects
                - **Technical failures** during Mercury retrograde
                - **Market manipulation** during Neptune-Mercury aspects
                
                **💡 Strategic Approach:**
                - **DCA strategy** during volatile periods
                - **Momentum trading** during Uranus aspects
                - **Risk-off** during Saturn hard aspects
                - **HODL mentality** during Jupiter-Pluto trines
                """)
            
            elif symbol in ['CRUDE']:
                st.markdown(f"""
                ### {symbol} - {month_name} {selected_year} Energy Commodity Outlook
                
                **🛢️ Astrological Themes:**
                - **Mars-Pluto aspects**: Geopolitical tensions affecting supply
                - **Jupiter-Saturn cycles**: Economic growth vs. restriction cycles
                - **Uranus aspects**: Renewable energy transition impacts
                - **Moon phases**: Seasonal demand patterns amplified
                
                **⚡ Supply-Demand Dynamics:**
                - Production disruptions (Mars-Saturn squares)
                - Economic growth spurts (Jupiter aspects)
                - Weather pattern extremes (Uranus-Neptune aspects)
                - Strategic reserve changes (Pluto aspects)
                
                **🌍 Geopolitical Factors:**
                - **OPEC decisions** aligned with Saturn aspects
                - **Pipeline disruptions** during Mars-Uranus periods
                - **Currency impacts** during Venus-Pluto aspects
                - **Seasonal patterns** enhanced by lunar cycles
                
                **📈 Trading Levels:**
                - **Resistance**: Previous Jupiter aspect highs
                - **Support**: Saturn aspect consolidation zones
                - **Breakout zones**: Mars-Uranus conjunction levels
                - **Reversal points**: Full Moon technical confluences
                """)
            
            elif symbol in ['DOWJONES']:
                st.markdown(f"""
                ### {symbol} - {month_name} {selected_year} US Index Outlook
                
                **🇺🇸 Macro Astrological Themes:**
                - **Jupiter-Saturn cycles**: Economic expansion vs. contraction
                - **Mercury-Venus aspects**: Corporate earnings and consumer spending
                - **Mars-Jupiter aspects**: Business investment and growth
                - **Outer planet aspects**: Long-term structural changes
                
                **🏛️ Federal Reserve Alignment:**
                - **Venus-Saturn aspects**: Interest rate policy changes
                - **Mercury-Jupiter aspects**: Fed communication clarity
                - **Moon phases**: Market sentiment around FOMC meetings
                - **Eclipse periods**: Major policy shift announcements
                
                **🔄 Sector Rotation Patterns:**
                - **Technology** leadership during Mercury-Uranus aspects
                - **Energy** strength during Mars-Pluto periods
                - **Financials** favor during Venus-Jupiter trines
                - **Healthcare** defensive during Saturn aspects
                
                **📊 Technical Confluence:**
                - **Monthly resistance**: Jupiter aspect previous highs
                - **Monthly support**: Saturn aspect previous lows
                - **Breakout potential**: New Moon near technical levels
                - **Reversal zones**: Full Moon at key Fibonacci levels
                """)
            
            else:  # Indian stocks
                st.markdown(f"""
                ### {symbol} ({symbol_info['sector']}) - {month_name} {selected_year} Indian Market Outlook
                
                **🇮🇳 Domestic Astrological Influences:**
                - **Jupiter transits**: Market leadership and FII flows
                - **Saturn aspects**: Regulatory changes and policy shifts
                - **Mars-Venus aspects**: Consumer spending and investment flows
                - **Moon phases**: Retail investor sentiment cycles
                
                **💹 Sector-Specific Themes:**
                - **{symbol_info['sector']} sector** influenced by specific planetary combinations
                - **Monsoon patterns** (if applicable) aligned with water sign emphasis
                - **Festival seasons** amplified by benefic planetary aspects
                - **Budget impacts** during Saturn-Jupiter aspects
                
                **🌏 Global Correlation Factors:**
                - **US Fed policy** impacts during Venus-Saturn aspects
                - **China growth** concerns during Mars-Saturn periods  
                - **Oil prices** affecting through Mars-Pluto aspects
                - **Dollar strength** impacts during Pluto aspects
                
                **📈 Monthly Strategy:**
                - **Accumulate** during Saturn aspects (value opportunities)
                - **Momentum plays** during Mars-Jupiter periods
                - **Defensive positioning** during challenging outer planet aspects
                - **Sector rotation** based on planetary emphasis shifts
                """)
        
        with tab2:
            st.markdown(f"""
            ### Technical Analysis Integration with Astrological Cycles
            
            **📊 Moving Average Alignment:**
            - **MA5 vs MA20**: Bullish when Jupiter aspects dominate
            - **Golden Cross** potential during Venus-Jupiter trines
            - **Death Cross** risk during Saturn-Mars squares
            - **MA support/resistance** stronger during lunar phases
            
            **🎯 Support & Resistance Levels:**
            - **Primary resistance**: Previous month's Jupiter aspect highs
            - **Primary support**: Saturn aspect consolidation lows
            - **Secondary levels**: Full Moon reversal points
            - **Breakout levels**: New Moon momentum points
            
            **📈 Momentum Indicators:**
            - **RSI overbought** (>70) more reliable during Full Moons
            - **RSI oversold** (<30) stronger signal during New Moons
            - **MACD divergences** amplified during Mercury aspects
            - **Volume confirmations** critical during Mars aspects
            
            **🌙 Lunar Cycle Technical Correlation:**
            - **New Moon**: Trend initiation, breakout potential
            - **Waxing Moon**: Momentum continuation, bullish bias
            - **Full Moon**: Trend exhaustion, reversal potential
            - **Waning Moon**: Correction phases, consolidation
            
            **⚡ Volatility Patterns:**
            - **Highest volatility**: Mars-Uranus aspect periods
            - **Lowest volatility**: Venus-Jupiter trine periods
            - **Unexpected moves**: Mercury-Neptune confusion aspects
            - **Gap movements**: Eclipse and outer planet aspects
            
            **🔄 Pattern Recognition:**
            - **Triangle breakouts** during Uranus aspects
            - **Flag patterns** during Mars aspects  
            - **Head & Shoulders** during Saturn aspects
            - **Double tops/bottoms** during opposition aspects
            """)
        
        with tab3:
            st.markdown(f"""
            ### Lunar Cycles & Market Psychology for {month_name} {selected_year}
            
            **🌑 New Moon Phases (Market Initiation):**
            - **Energy**: Fresh starts, new trend beginnings
            - **Psychology**: Optimism, risk-taking increases
            - **Trading**: Look for breakout setups, trend initiations
            - **Volume**: Often lower but quality moves
            - **Best for**: Opening new positions, trend following
            
            **🌓 Waxing Moon (Building Momentum):**
            - **Energy**: Growth, expansion, building confidence  
            - **Psychology**: FOMO starts building, bullish sentiment
            - **Trading**: Momentum continuation, pyramid additions
            - **Volume**: Increasing participation
            - **Best for**: Adding to winning positions
            
            **🌕 Full Moon Phases (Emotional Peaks):**
            - **Energy**: Maximum emotion, extremes, reversals
            - **Psychology**: Euphoria or panic peaks
            - **Trading**: Reversal setups, profit-taking
            - **Volume**: Often highest of cycle
            - **Best for**: Profit booking, contrarian plays
            
            **🌗 Waning Moon (Consolidation):**
            - **Energy**: Release, correction, cooling off
            - **Psychology**: Reality check, risk assessment
            - **Trading**: Consolidation patterns, value hunting
            - **Volume**: Declining, selective moves
            - **Best for**: Position adjustments, planning
            
            **🔮 {month_name} {selected_year} Specific Lunar Events:**
            
            **Key Lunar Dates to Watch:**
            - **New Moon**: Potential trend change or continuation signal
            - **First Quarter**: Momentum confirmation or failure
            - **Full Moon**: Profit-taking opportunity or reversal signal  
            - **Last Quarter**: Consolidation phase or weakness signal
            
            **Moon Sign Influences:**
            - **Fire Signs** (Aries, Leo, Sagittarius): Aggressive moves, energy sector strength
            - **Earth Signs** (Taurus, Virgo, Capricorn): Value focus, stability preference
            - **Air Signs** (Gemini, Libra, Aquarius): Communication, technology emphasis
            - **Water Signs** (Cancer, Scorpio, Pisces): Emotional decisions, defensive moves
            """)
        
        with tab4:
            st.markdown(f"""
            ### Portfolio Strategy for {month_name} {selected_year}
            
            **🎯 Strategic Asset Allocation:**
            
            **Core Holdings (50-60%):**
            - **Large Cap Stability**: Jupiter-aspected blue chips
            - **Sector Leaders**: Dominant players in favored sectors
            - **Defensive Assets**: During challenging aspect periods
            - **Currency Hedge**: If significant Pluto aspects present
            
            **Growth Opportunities (20-30%):**
            - **Momentum Plays**: Mars-Jupiter aspect beneficiaries
            - **Breakout Candidates**: Technical + astrological confluence
            - **Sector Rotation**: Following planetary emphasis shifts
            - **Emerging Themes**: Uranus aspect innovation plays
            
            **Speculative/Trading (10-20%):**
            - **High Beta Names**: For Mars-Uranus periods
            - **Volatility Plays**: Options during aspect peaks
            - **Contrarian Bets**: Against crowd during extremes
            - **Crypto Allocation**: If comfortable with high volatility
            
            **📊 Risk Management Framework:**
            
            **Position Sizing Rules:**
            - **Maximum single position**: 5% during stable periods
            - **Reduce to 3%**: During challenging aspects
            - **Increase to 7%**: During strong favorable confluences
            - **Cash levels**: 10-20% based on aspect favorability
            
            **Stop Loss Strategy:**
            - **Tight stops** (3-5%): During Mercury retrograde periods
            - **Normal stops** (5-8%): During regular market conditions
            - **Wide stops** (8-12%): During high volatility aspect periods
            - **No stops**: For long-term Jupiter-blessed positions
            
            **📅 Monthly Rebalancing Schedule:**
            
            **Week 1**: Review and adjust based on new lunar cycle
            **Week 2**: Add to momentum winners if aspects support
            **Week 3**: Prepare for Full Moon profit-taking opportunities
            **Week 4**: Position for next month's astrological themes
            
            **🔄 Sector Rotation Strategy:**
            
            **Early Month**: Follow Jupiter aspects for growth sectors
            **Mid Month**: Mars aspects may favor energy/materials
            **Late Month**: Venus aspects support financials/consumer
            **Month End**: Saturn aspects favor defensives/utilities
            
            **💡 Advanced Strategies:**
            
            **Pairs Trading**: Long favored sectors, short challenged sectors
            **Options Overlay**: Sell calls during Full Moons, buy calls during New Moons
            **Currency Hedge**: Hedge foreign exposure during Pluto aspects
            **Volatility Trading**: Long volatility before aspect peaks
            
            **📈 Performance Tracking:**
            
            **Monthly Metrics**:
            - Absolute return vs. benchmark
            - Risk-adjusted return (Sharpe ratio)
            - Maximum drawdown during challenging aspects
            - Hit rate on astrological predictions
            
            **Aspect Correlation Analysis**:
            - Track which aspects work best for {symbol}
            - Note sector rotation timing accuracy
            - Measure volatility prediction success
            - Document lunar cycle correlations
            """)
        
        # Additional insights for monthly strategy
        st.subheader('🎭 Market Psychology & Sentiment Analysis')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            #### 🧠 Psychological Drivers - {month_name}
            
            **Institutional Behavior:**
            - Month-end window dressing effects
            - Quarterly rebalancing influences  
            - Earnings season psychological impacts
            - Fed meeting anticipation/reaction
            
            **Retail Investor Patterns:**
            - Payroll cycle investment flows
            - Tax implications (if year-end)
            - Holiday season spending impacts
            - Social media sentiment amplification
            
            **Global Sentiment Factors:**
            - US-China trade relationship status
            - European economic data impacts
            - Emerging market flow dynamics
            - Cryptocurrency correlation effects
            """)
        
        with col2:
            st.markdown(f"""
            #### 📊 Sentiment Indicators to Watch
            
            **Technical Sentiment:**
            - VIX levels and term structure
            - Put/Call ratios by sector
            - High-low index readings
            - Advance-decline line trends
            
            **Fundamental Sentiment:**
            - Earnings revision trends
            - Analyst recommendation changes
            - Insider buying/selling activity
            - Share buyback announcements
            
            **Alternative Data:**
            - Google search trends
            - Social media mention analysis
            - Options flow analysis
            - Crypto correlation strength
            """)

# Footer with additional information
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <h4>🌟 Disclaimer & Important Notes</h4>
        <p><strong>Educational Purpose Only:</strong> This dashboard is for educational and research purposes. 
        Astrological analysis should be combined with fundamental and technical analysis for trading decisions.</p>
        
        <p><strong>Risk Warning:</strong> All trading involves risk. Past performance and astrological correlations 
        do not guarantee future results. Always use proper risk management and never invest more than you can afford to lose.</p>
        
        <p><strong>Data Sources:</strong> Simulated price data based on astrological aspect calculations. 
        For live trading, use real market data and professional trading platforms.</p>
        
        <p style='font-size: 12px; margin-top: 20px;'>
        🔮 <em>"The stars impel, they do not compel. Wisdom lies in using all available tools - 
        fundamental, technical, and cosmic - for informed decision making."</em>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
