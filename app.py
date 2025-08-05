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
    'NIFTY': {'name': 'Nifty 50', 'currency': 'INR', 'default_price': 24620.0, 'sector': 'Index'},
    'BANKNIFTY': {'name': 'Bank Nifty', 'currency': 'INR', 'default_price': 52000.0, 'sector': 'Banking Index'},
    'TCS': {'name': 'Tata Consultancy Services', 'currency': 'INR', 'default_price': 4200.0, 'sector': 'Technology'},
    'ICICIBANK': {'name': 'ICICI Bank', 'currency': 'INR', 'default_price': 1200.0, 'sector': 'Banking'},
    'MARUTI': {'name': 'Maruti Suzuki', 'currency': 'INR', 'default_price': 12000.0, 'sector': 'Automotive'},
    'DLF': {'name': 'DLF Limited', 'currency': 'INR', 'default_price': 800.0, 'sector': 'Realty'},
    'NESTLEIND': {'name': 'Nestle India', 'currency': 'INR', 'default_price': 2400.0, 'sector': 'FMCG'},
    'RELIANCE': {'name': 'Reliance Industries', 'currency': 'INR', 'default_price': 3000.0, 'sector': 'Energy'},
    'SBI': {'name': 'State Bank of India', 'currency': 'INR', 'default_price': 850.0, 'sector': 'PSU Banking'},
    'SUNPHARMA': {'name': 'Sun Pharma', 'currency': 'INR', 'default_price': 1700.0, 'sector': 'Pharma'},
    'DRREDDY': {'name': 'Dr Reddy Labs', 'currency': 'INR', 'default_price': 6800.0, 'sector': 'Pharma'},
    
    'GOLD': {'name': 'Gold Futures', 'currency': 'USD', 'default_price': 2050.0, 'sector': 'Precious Metals'},
    'DOWJONES': {'name': 'Dow Jones Industrial Average', 'currency': 'USD', 'default_price': 35000.0, 'sector': 'US Index'},
    'SILVER': {'name': 'Silver Futures', 'currency': 'USD', 'default_price': 25.50, 'sector': 'Precious Metals'},
    'CRUDE': {'name': 'Crude Oil WTI', 'currency': 'USD', 'default_price': 82.50, 'sector': 'Energy'},
    'BTC': {'name': 'Bitcoin', 'currency': 'USD', 'default_price': 65000.0, 'sector': 'Cryptocurrency'},
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
        info = SYMBOL_CONFIG[symbol].copy()
        # Convert currency codes to symbols for display
        if info['currency'] == 'INR':
            info['currency'] = 'Rs.'
        elif info['currency'] == 'USD':
            info['currency'] = '$'
        return info
    else:
        # Default configuration
        return {
            'name': symbol,
            'currency': 'Rs.',
            'default_price': 1000.0,
            'sector': 'Unknown'
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
            summary_data['Bullish Sectors/Stocks'].append("FMCG (Nestle), Pharma, Gold, Dow Jones")
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
    
    if not bullish_stocks.empty:
        bullish_stocks['Impact Score'] = bullish_stocks['Sector'].apply(lambda x: sector_impacts[x])
        bullish_stocks = bullish_stocks.sort_values('Impact Score', ascending=False)
    
    if not bearish_stocks.empty:
        bearish_stocks['Impact Score'] = bearish_stocks['Sector'].apply(lambda x: abs(sector_impacts[x]))
        bearish_stocks = bearish_stocks.sort_values('Impact Score', ascending=False)
    
    if not neutral_stocks.empty:
        neutral_stocks['Impact Score'] = 0
    
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
            {"day_offset": 5, "aspect": "Moon-Jupiter trine to Moon-Saturn square", "impact": 1.2, "type": "bullish"},
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
        
        aspect_text = event['aspect'][:30] + '...' if len(event['aspect']) > 30 else event['aspect']
        ax_main.annotate(aspect_text, 
                        xy=(event['time'], event['price']),
                        xytext=(event['time'], y_pos),
                        arrowprops=dict(arrowstyle='->', color=marker_color, lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                        fontsize=8, ha='center')
    
    # Dynamic title and formatting
    currency_symbol = symbol_info['currency']
    chart_title = f"{symbol_info['name']} ({symbol}) - {selected_date.strftime('%B %d, %Y')}\nAstrological Trading Analysis | Sector: {symbol_info['sector']}"
    ax_main.set_title(chart_title, fontsize=16, pad=20)
    
    hours_text = f"Time ({trading_hours['start_hour']}:00 - {trading_hours['end_hour']}:00)"
    ax_main.set_xlabel(hours_text, fontsize=12)
    ax_main.set_ylabel(f"Price ({currency_symbol})", fontsize=12)
    
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
    
    close_text = f"Close: {currency_symbol}{close_price:.2f}\nChange: {price_change:+.2f} ({price_change_pct:+.2f}%)"
    ax_main.annotate(close_text, 
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
    max_strength = max(aspect_strengths) if aspect_strengths else 1.5
    ax_aspect.set_ylim(0, max_strength * 1.2)
    
    # Symbol info panel
    ax_info = fig.add_subplot(gs[1:, 2])
    ax_info.axis('off')
    
    high_price = max(prices)
    low_price = min(prices)
    
    info_text = f"""SYMBOL INFO
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

High: {currency_symbol}{high_price:.2f}
Low: {currency_symbol}{low_price:.2f}
Range: {currency_symbol}{high_price-low_price:.2f}
"""
    
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
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
            '2-3', '-1.5-2', '1-2', 
            '1-1.5', '-2-3', '0.5-1',
            '0.5-1', '1-2', '2-4',
            '1-2'
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
    price_changes = [float(change.split('-')[-1]) for change in df_aspects['Typical Price Change']]
    
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
        if any(word in impact for word in ['Bullish', 'Rally', 'Surge', 'Returns']):
            impact_type = 'Bullish'
        elif any(word in impact for word in ['Bearish', 'Pressure', 'Tension']):
            impact_type = 'Bearish'
        else:
            impact_type = 'Neutral'
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
        page_title="Astrological Trading Dashboard",
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
                info_html = f"""
                <div class="info-box">
                    <strong>📊 {symbol_info['name']}</strong><br>
                    <small>Sector: {symbol_info['sector']}</small><br>
                    <small>Currency: {symbol_info['currency']}</small><br>
                    <small>Trading: {trading_hours['start_hour']:02d}:{trading_hours['start_minute']:02d} - {trading_hours['end_hour']:02d}:{trading_hours['end_minute']:02d}</small>
                </div>
                """
                st.markdown(info_html, unsafe_allow_html=True)
                
                # Price input with dynamic default
                step_size = 1.0 if symbol_info['default_price'] > 100 else 0.01
                starting_price = st.number_input(
                    f"💰 Starting Price ({symbol_info['currency']}):",
                    min_value=0.01,
                    value=symbol_info['default_price'],
                    step=step_size,
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
        
        sentiment_html = f"**Overall Sentiment:** <span style='color:{sentiment_color}'>{sentiment}</span>"
        st.markdown(sentiment_html, unsafe_allow_html=True)

    # Main content area
    aspects = generate_todays_aspects()
    
    if section_name == 'Summary Table':
        st.header('📋 Market Summary & Astrological Overview')
        
        # Create columns for better layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader('📊 Today\'s Astrological Aspects Impact')
            summary_df = create_summary_table(aspects)
            st.dataframe(summary_df, use_container_width=True)
        
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
            if high_risk_aspects >= 3:
                risk_level = "High"
                risk_color = "red"
            elif high_risk_aspects >= 1:
                risk_level = "Medium"
                risk_color = "orange"
            else:
                risk_level = "Low"
                risk_color = "green"
            
            risk_html = f"**Risk Level:** <span style='color:{risk_color}'>{risk_level}</span>"
            st.markdown(risk_html, unsafe_allow_html=True)
    
    elif section_name == 'Stock Filter':
        st.header('🔍 Advanced Stock Filtering & Sector Analysis')
        
        # Display aspects in a nice format
        st.subheader('🌟 Today\'s Astrological Configuration')
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
        st.dataframe(aspects_df, use_container_width=True)
        
        # Sector analysis
        st.subheader('📈 Sector Impact Analysis')
        
        filtered_stocks = filter_stocks_by_aspects(aspects, STOCK_DATABASE)
        
        # Create sector impact visualization
        sector_impacts_df = pd.DataFrame({
            'Sector': list(filtered_stocks['sector_impacts'].keys()),
            'Impact Score': list(filtered_stocks['sector_impacts'].values())
        })
        sector_impacts_df = sector_impacts_df.sort_values('Impact Score', ascending=False)
        
        # Enhanced bar chart
        fig_sectors, ax_sectors = plt.subplots(figsize=(14, 8))
        colors = []
        for x in sector_impacts_df['Impact Score']:
            if x > 0.5:
                colors.append('darkgreen')
            elif x > 0:
                colors.append('green')
            elif x == 0:
                colors.append('gray')
            elif x > -0.5:
                colors.append('red')
            else:
                colors.append('darkred')
        
        bars = ax_sectors.bar(sector_impacts_df['Sector'], sector_impacts_df['Impact Score'], 
                             color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax_sectors.set_title('Sector Impact Scores - Astrological Analysis', fontsize=16, pad=20)
        ax_sectors.set_ylabel('Impact Score', fontsize=12)
        ax_sectors.set_xlabel('Sector', fontsize=12)
        ax_sectors.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax_sectors.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax_sectors.annotate(f'{height:.2f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 5 if height >= 0 else -15),
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
                for _, row in bullish_df.iterrows():
                    action = 'Strong Buy' if row['Impact Score'] > 0.5 else 'Buy'
                    st.markdown(f"""
                    **{row['Symbol']}** ({row['Sector']})  
                    Score: {row['Impact Score']:.2f} | **{action}**
                    """)
                    st.progress(min(1.0, row['Impact Score']))
            else:
                st.info("No strong bullish signals today")
        
        with col2:
            st.subheader('🔴 Bearish Stocks')
            if not filtered_stocks['bearish'].empty:
                bearish_df = filtered_stocks['bearish'][['Symbol', 'Sector', 'Impact Score']].copy()
                for _, row in bearish_df.iterrows():
                    action = 'Strong Sell' if row['Impact Score'] > 0.5 else 'Sell'
                    st.markdown(f"""
                    **{row['Symbol']}** ({row['Sector']})  
                    Risk Score: {row['Impact Score']:.2f} | **{action}**
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
        st.dataframe(df_aspects, use_container_width=True)
    
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
            session_length = trading_hours['end_hour'] - trading_hours['start_hour']
            session_length += (trading_hours['end_minute'] - trading_hours['start_minute']) / 60
            st.metric("Session Hours", f"{session_length:.1f}h")
        
        # Generate and display chart
        with st.spinner(f'Generating astrological analysis for {symbol}...'):
            fig = generate_intraday_chart(symbol, starting_price, selected_date)
            st.pyplot(fig)
        
        # Trading insights
        st.subheader(f'🎯 {symbol} Trading Insights')
        st.info("Analysis generated based on astrological aspects and market correlations.")

    # Footer with additional information
    st.markdown("---")
    footer_html = """
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
    """
    st.markdown(footer_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
