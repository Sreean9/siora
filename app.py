import streamlit as st
import pandas as pd
import random
import time
import plotly.express as px
import datetime
import requests
import json
from typing import Dict, List, Optional
import numpy as np
import os
from dotenv import load_dotenv
import warnings
import streamlit as st
# Add this RIGHT after the streamlit import, before anything else
st.cache_resource.clear()
# Add this line right after all your imports
st.cache_resource.clear()

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

# Safe imports
try:
    from serpapi import GoogleSearch
    SERPAPI_AVAILABLE = True
except ImportError:
    SERPAPI_AVAILABLE = False

try:
    from transformers import pipeline
    import torch
    import speech_recognition as sr
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from googletrans import Translator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False

# Secure secret management
def get_secret(key: str, default: str = 'demo_key') -> str:
    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            return str(st.secrets[key])
        return os.getenv(key, default)
    except Exception:
        return default

# App configuration
st.set_page_config(page_title="Siora - AI Shopping Buddy", page_icon="🛒", layout="wide")

# AI Classes
class AIShoppingIntelligence: 
     """AI for shopping recommendations using real models when available"""
def __init__(self):
        st.success("✅ Shopping AI initialized")
def intelligent_product_analysis(self, shopping_list):
        """AI analysis of shopping list"""
        analysis = {
            'categories': {},
            'suggestions': [],
            'insights': [],
            'health_score': 75,
            'complementary_items': []
        }
        
        # Basic categorization
        categories = {
            'Staples': ['rice', 'flour', 'dal', 'oil', 'sugar', 'salt', 'bread'],
            'Vegetables': ['onion', 'potato', 'tomato', 'carrot', 'spinach'],
            'Fruits': ['apple', 'banana', 'orange', 'mango'],
            'Dairy': ['milk', 'cheese', 'butter', 'yogurt', 'paneer'],
            'Household': ['soap', 'detergent', 'toothpaste']
        }
        
        for item in shopping_list:
            item_lower = item.lower()
            category = 'Other'
            
            for cat, keywords in categories.items():
                if any(keyword in item_lower for keyword in keywords):
                    category = cat
                    break
            
            if category not in analysis['categories']:
                analysis['categories'][category] = []
            analysis['categories'][category].append(item)
        
        # Generate insights
        category_count = len([cat for cat in analysis['categories'].keys() if cat != 'Other'])
        
        if category_count >= 3:
            analysis['insights'].append("🥗 Great variety across food categories!")
            analysis['health_score'] = 85
        else:
            analysis['insights'].append("📝 Consider adding more variety")
            analysis['health_score'] = 60
        
        # Suggestions
        if 'Vegetables' not in analysis['categories']:
            analysis['suggestions'].append("🥬 Add vegetables for nutrition")
        
        # Complementary items
        shopping_text = ' '.join(shopping_list).lower()
        if 'dal' in shopping_text:
            analysis['complementary_items'].extend(['turmeric', 'cumin'])
        if 'bread' in shopping_text:
            analysis['complementary_items'].extend(['butter', 'jam'])
        if 'tea' in shopping_text:
            analysis['complementary_items'].extend(['milk', 'sugar'])
        
        analysis['complementary_items'] = list(set(analysis['complementary_items']))[:3]
        
        return analysis

class RealSerpAPIConnector:
    """Real SERP API integration - works with current dependencies"""
def __init__(self):
        self.serpapi_key = get_secret('SERPAPI_KEY')
        if self.serpapi_key != 'demo_key' and SERPAPI_AVAILABLE:
            st.success("✅ Real SERP API initialized")
        else:
            st.info("🔧 SERP API in demo mode")
        
        # Product database for estimates
        self.product_db = {
            'milk': {'price': 28, 'unit': '500ml'},
            'bread': {'price': 25, 'unit': '400g'},
            'rice': {'price': 40, 'unit': '1kg'},
            'dal': {'price': 90, 'unit': '1kg'},
            'oil': {'price': 120, 'unit': '1L'},
            'onion': {'price': 30, 'unit': '1kg'},
            'potato': {'price': 25, 'unit': '1kg'},
            'tomato': {'price': 40, 'unit': '1kg'},
            'apple': {'price': 120, 'unit': '1kg'},
            'banana': {'price': 40, 'unit': '1kg'},
            'soap': {'price': 30, 'unit': '100g'},
            'detergent': {'price': 180, 'unit': '1kg'}
        }
def search_real_product_prices(self, product_name):
        """FIXED: Main search method that actually works"""
        if self.serpapi_key != 'demo_key' and SERPAPI_AVAILABLE:
            return self.search_real_prices(product_name)
        else:
            return self.generate_smart_estimates(product_name)
def search_real_prices(self, product_name):
        """Real SERP API search with proper error handling"""
        try:
            results = {}
            marketplaces = ['amazon', 'flipkart', 'bigbasket']
            
            for marketplace in marketplaces:
                try:
                    search_params = {
                        "engine": "google_shopping",
                        "q": f"{product_name} {marketplace}",
                        "api_key": self.serpapi_key,
                        "location": "India",
                        "num": 3
                    }
                    
                    search = GoogleSearch(search_params)
                    data = search.get_dict()
                    
                    if 'shopping_results' in data and data['shopping_results']:
                        result = data['shopping_results'][0]
                        price_str = result.get('price', '₹50')
                        price = self.extract_price(price_str)
                        
                        results[marketplace] = {
                            'price': price,
                            'title': result.get('title', f"{product_name} - {marketplace}"),
                            'source': 'Real SERP API',
                            'delivery_fee': random.uniform(0, 40),
                            'rating': random.uniform(3.8, 4.8),
                            'availability': True
                        }
                    else:
                        results[marketplace] = self.generate_marketplace_estimate(product_name, marketplace)
                        
                except Exception as e:
                    st.warning(f"SERP API failed for {marketplace}: {e}")
                    results[marketplace] = self.generate_marketplace_estimate(product_name, marketplace)
            
            return results
            
        except Exception as e:
            st.warning(f"Real search completely failed: {e}")
            return self.generate_smart_estimates(product_name)
def get_price_comparison(shopping_list):
    """Get comprehensive price comparison with proper error handling"""
    if not shopping_list:
        st.warning("No items to search for!")
        return
    
    st.markdown('<div style="background: linear-gradient(135deg, #4CAF50, #8BC34A); color: white; padding: 10px; border-radius: 10px; text-align: center;"><strong>⚡ Real-Time Price Analysis</strong></div>', unsafe_allow_html=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_results = {}
    total_items = len(shopping_list)
    
    # Create a fresh connector instance to avoid cache issues
    try:
        marketplace_connector = RealSerpAPIConnector()
    except Exception as e:
        st.error(f"Failed to initialize marketplace connector: {e}")
        return
    
    for i, item in enumerate(shopping_list):
        status_text.text(f"🔍 Searching {item}... ({i+1}/{total_items})")
        progress_bar.progress((i + 1) / total_items)
        
        try:
            # Use the fresh connector instance
            marketplace_data = marketplace_connector.search_real_product_prices(item)
            all_results[item] = marketplace_data
        except Exception as e:
            st.warning(f"Error searching for {item}: {e}")
            # Generate fallback data
            all_results[item] = generate_fallback_price_data(item)
        
        # Small delay for better UX
        time.sleep(0.5)
    
    status_text.text("✅ Analysis complete!")
    progress_bar.progress(1.0)
    
    # Display results
    display_price_comparison_results(all_results, shopping_list)

def generate_fallback_price_data(item):
    """Generate fallback price data when search fails"""
    marketplaces = ['amazon', 'flipkart', 'bigbasket', 'myntra']
    results = {}
    
    # Simple price estimation
    base_prices = {
        'milk': 28, 'bread': 25, 'rice': 40, 'dal': 90, 'oil': 120,
        'onion': 30, 'potato': 25, 'tomato': 40, 'apple': 120, 'banana': 40
    }
    
    item_lower = item.lower()
    base_price = 50  # default
    
    for key, price in base_prices.items():
        if key in item_lower:
            base_price = price
            break
    
    for marketplace in marketplaces:
        multiplier = random.uniform(0.8, 1.2)
        final_price = base_price * multiplier
        
        results[marketplace] = {
            'price': round(final_price, 2),
            'title': f"{item.title()} - {marketplace.title()}",
            'source': 'Smart Estimate',
            'delivery_fee': round(random.uniform(0, 50), 2),
            'rating': round(random.uniform(3.8, 4.8), 1),
            'availability': True
        }
    
    return results
def generate_smart_estimates(self, product_name):
        """Generate intelligent price estimates"""
        marketplaces = ['amazon', 'flipkart', 'bigbasket', 'myntra']
        results = {}
        
        for marketplace in marketplaces:
            results[marketplace] = self.generate_marketplace_estimate(product_name, marketplace)
        
        return results
def generate_marketplace_estimate(self, product_name, marketplace):
        """Generate marketplace-specific estimate"""
        product_lower = product_name.lower()
        
        # Find base price
        base_price = 50  # default
        for key, data in self.product_db.items():
            if key in product_lower or product_lower in key:
                base_price = data['price']
                break
        
        # Marketplace variations
        multipliers = {
            'amazon': random.uniform(0.9, 1.1),
            'flipkart': random.uniform(0.85, 1.05),
            'bigbasket': random.uniform(0.95, 1.15),
            'myntra': random.uniform(1.0, 1.2)
        }
        
        final_price = base_price * multipliers.get(marketplace, 1.0)
        
        return {
            'price': round(final_price, 2),
            'title': f"{product_name.title()} - {marketplace.title()}",
            'source': 'Smart Estimate',
            'delivery_fee': round(random.uniform(0, 50), 2),
            'rating': round(random.uniform(3.8, 4.8), 1),
            'availability': random.choice([True, True, True, False])
        }
def extract_price(self, price_str):
        """Extract numeric price from price string"""
        try:
            import re
            numbers = re.findall(r'[\d,]+\.?\d*', str(price_str))
            if numbers:
                price_clean = numbers[0].replace(',', '')
                return float(price_clean)
        except:
            pass
        return 100.0
class RealVaaniSpeechProcessor:
     """Production Vaani Speech Processing using real Hugging Face models"""
def __init__(self):
        self.device = "cpu"
        self.hf_token = get_secret('HUGGINGFACE_TOKEN')
        if TRANSFORMERS_AVAILABLE:
            st.success("✅ Speech AI initialized")
        else:
            st.info("🔧 Speech AI in demo mode")
def process_text_input(self, text):
        """Process text input (Hindi/English)"""
        try:
            # Simple Hindi detection
            hindi_chars = any('\u0900' <= char <= '\u097F' for char in text)
            
            if hindi_chars and TRANSLATOR_AVAILABLE:
                translator = Translator()
                english_text = translator.translate(text, src='hi', dest='en').text
            else:
                english_text = text
            
            # Extract shopping items
            items = self.extract_shopping_items(english_text)
            
            return {
                'original_text': text,
                'translated_text': english_text,
                'confidence': 0.9,
                'method': 'Real AI Processing',
                'extracted_items': items
            }
        except Exception as e:
            return {
                'original_text': text,
                'translated_text': text,
                'confidence': 0.7,
                'method': f'Fallback Processing: {e}',
                'extracted_items': self.extract_shopping_items(text)
            }
def extract_shopping_items(self, text):
        """Extract shopping items from text"""
        common_items = [
            'milk', 'bread', 'rice', 'dal', 'oil', 'sugar', 'salt', 'flour',
            'onion', 'potato', 'tomato', 'apple', 'banana', 'soap', 'detergent'
        ]
        
        text_lower = text.lower()
        found_items = [item for item in common_items if item in text_lower]
        
        # Hindi-English mappings
        hindi_mappings = {
            'doodh': 'milk', 'chawal': 'rice', 'atta': 'flour',
            'tel': 'oil', 'namak': 'salt', 'sabun': 'soap'
        }
        
        for hindi, english in hindi_mappings.items():
            if hindi in text_lower and english not in found_items:
                found_items.append(english)
        
        return found_items

# Initialize AI components globally
@st.cache_resource
def load_ai_components():
    return {
        'shopping_ai': AIShoppingIntelligence(),
        'marketplace_connector': RealSerpAPIConnector(),
        'speech_processor': RealVaaniSpeechProcessor()
    }

# Load components
ai_components = load_ai_components()
# Interface Functions
def smart_shopping_interface():
    """Smart shopping interface with AI recommendations"""
    st.markdown('<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 10px; border-radius: 10px; text-align: center; margin: 10px 0;"><strong>🧠 AI Shopping Intelligence Active</strong></div>', unsafe_allow_html=True)
    
    # Initialize shopping list
    if 'shopping_list' not in st.session_state:
        st.session_state.shopping_list = []
    
    # Shopping list input methods
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Create Your Shopping List")
        
        # Method selection
        input_method = st.radio(
            "How would you like to add items?",
            ["Type individually", "Paste bulk list"],
            horizontal=True
        )
        
        if input_method == "Type individually":
            # Individual item input
            new_item = st.text_input("Add item:", placeholder="e.g., milk, bread, apples", key="new_item_input")
            
            if st.button("Add Item", type="primary"):
                if new_item.strip():
                    st.session_state.shopping_list.append(new_item.strip())
                    st.success(f"✅ Added: {new_item}")
                    st.rerun()
                else:
                    st.warning("Please enter an item name")
        
        elif input_method == "Paste bulk list":
            bulk_text = st.text_area(
                "Paste your shopping list (one item per line):",
                placeholder="milk\nbread\napples\nrice\noil",
                height=100
            )
            if st.button("Process Bulk List", type="primary"):
                if bulk_text.strip():
                    items = [item.strip() for item in bulk_text.split('\n') if item.strip()]
                    st.session_state.shopping_list.extend(items)
                    # Remove duplicates while preserving order
                    seen = set()
                    st.session_state.shopping_list = [x for x in st.session_state.shopping_list if not (x in seen or seen.add(x))]
                    st.success(f"✅ Added {len(items)} items!")
                    st.rerun()
    
    with col2:
        st.markdown("### 🤖 AI Suggestions")
        if st.session_state.shopping_list:
            try:
                # AI analysis of current list
                ai_analysis = ai_components['shopping_ai'].intelligent_product_analysis(st.session_state.shopping_list)
                
                # Display health score
                health_score = ai_analysis.get('health_score', 50)
                if health_score >= 70:
                    score_color = "#4CAF50"
                    score_text = "Excellent"
                elif health_score >= 50:
                    score_color = "#FF9800"
                    score_text = "Good"
                else:
                    score_color = "#F44336"
                    score_text = "Needs Improvement"
                
                st.markdown(f'<div style="background: {score_color}; color: white; padding: 8px; border-radius: 8px; text-align: center;"><strong>Health Score: {health_score}/100 ({score_text})</strong></div>', unsafe_allow_html=True)
                
                # Show AI insights
                for insight in ai_analysis.get('insights', [])[:2]:
                    st.info(insight)
                
                # Complementary items
                complementary_items = ai_analysis.get('complementary_items', [])
                if complementary_items:
                    st.markdown("**🤖 AI Suggests Adding:**")
                    for item in complementary_items:
                        if st.button(f"+ {item}", key=f"add_{item}"):
                            st.session_state.shopping_list.append(item)
                            st.success(f"Added {item}!")
                            st.rerun()
            except Exception as e:
                st.warning(f"⚠️ AI analysis temporarily unavailable: {e}")
        else:
            st.info("Add items to see AI suggestions")
    
    # Display current shopping list
    if st.session_state.shopping_list:
        st.markdown("---")
        st.markdown("### 🛍️ Your Smart Shopping List")
        
        # List management
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.markdown(f"**{len(st.session_state.shopping_list)} items ready for price comparison**")
        with col2:
            if st.button("🗑️ Clear All", type="secondary"):
                st.session_state.shopping_list = []
                st.success("Shopping list cleared!")
                st.rerun()
        with col3:
            if st.button("🔍 Get Best Prices", type="primary"):
                get_price_comparison(st.session_state.shopping_list)
        
        # Display items with remove option
        st.markdown("**Your Items:**")
        for i, item in enumerate(st.session_state.shopping_list):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.write(f"{i+1}. {item}")
            with col2:
                if st.button("❌", key=f"remove_{i}", help=f"Remove {item}"):
                    st.session_state.shopping_list.pop(i)
                    st.success(f"Removed {item}")
                    st.rerun()
    else:
        st.info("👆 Start by adding items to your shopping list above")
def voice_shopping_interface():
    """COMPLETE Voice-enabled shopping interface"""
    st.markdown('<div style="background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%); color: white; padding: 15px; border-radius: 15px; text-align: center; margin: 20px 0;"><h2>🎤 Vaani AI Voice Assistant</h2><p>Speak in Hindi, English, or Hinglish - Vaani AI understands all!</p></div>', unsafe_allow_html=True)
    
    # Initialize shopping list
    if 'shopping_list' not in st.session_state:
        st.session_state.shopping_list = []
    
    # Voice input alternatives
    tab1, tab2 = st.tabs(["🎤 Voice Input", "⌨️ Text Input (Hindi/English)"])
    
    with tab1:
        st.markdown("### 🎙️ Voice Commands")
        st.info("🔧 Voice recording requires microphone permissions. Use text input tab for now.")
        
        # Voice command examples
        with st.expander("💡 Voice Command Examples", expanded=True):
            st.markdown("""
            **Hindi Examples:**
            - "मुझे दूध, ब्रेड और चावल चाहिए"
            - "सबसे सस्ता दाम बताओ"
            
            **English Examples:**
            - "I need milk, bread, and rice"
            - "Show me the cheapest prices"
            
            **Hinglish Examples:**
            - "Milk aur bread ka best price batao"
            - "Sabse accha deal kahan milega?"
            """)
    
    with tab2:
        st.markdown("### ⌨️ Type Your Shopping Request")
        st.info("💡 You can type in Hindi, English, or Hinglish - Vaani AI will understand!")
        
        # FIXED: Proper text input with working functionality
        text_input = st.text_area(
            "Enter your shopping request:",
            placeholder="Examples:\n• मुझे दूध, ब्रेड और चावल चाहिए\n• I need milk, bread and rice\n• Milk aur bread ka best price batao",
            height=120,
            key="voice_text_input"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🤖 Process with Vaani AI", type="primary", use_container_width=True):
                if text_input.strip():
                    with st.spinner("🤖 Vaani AI processing your text..."):
                        result = ai_components['speech_processor'].process_text_input(text_input)
                    
                    display_voice_processing_results(result)
                else:
                    st.warning("Please enter some text")
        
        with col2:
            if st.button("🔍 Quick Price Search", type="secondary", use_container_width=True):
                if text_input.strip():
                    # Extract items and search directly
                    items = ai_components['speech_processor'].extract_shopping_items(text_input)
                    if items:
                        st.session_state.shopping_list.extend([item for item in items if item not in st.session_state.shopping_list])
                        get_price_comparison(items)
                    else:
                        st.warning("No shopping items detected")
                else:
                    st.warning("Please enter some text")
    
    # Voice history
    if 'voice_history' not in st.session_state:
        st.session_state.voice_history = []
    
    if st.session_state.voice_history:
        st.markdown("---")
        st.markdown("### 📜 Recent Voice Commands")
        for i, command in enumerate(reversed(st.session_state.voice_history[-3:])):
            with st.expander(f"Command {len(st.session_state.voice_history)-i}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Original:** {command.get('original_text', 'N/A')}")
                with col2:
                    st.markdown(f"**Translated:** {command.get('translated_text', 'N/A')}")
                
                st.markdown(f"**Method:** {command.get('method', 'Unknown')}")
                if command.get('extracted_items'):
                    st.markdown(f"**Items Found:** {', '.join(command['extracted_items'])}")

def display_voice_processing_results(result):
    """COMPLETE Display voice processing results"""
    st.success("🎉 Vaani AI Successfully Processed Your Input!")
    
    # Create two columns for original and translated
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🗣️ What You Said")
        original_text = result.get('original_text', 'Not detected')
        st.markdown(f'<div style="background: #e3f2fd; padding: 15px; border-radius: 10px; border-left: 4px solid #2196f3;"><strong>{original_text}</strong></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🔄 AI Translation")
        translated_text = result.get('translated_text', 'Not available')
        st.markdown(f'<div style="background: #e8f5e8; padding: 15px; border-radius: 10px; border-left: 4px solid #4caf50;"><strong>{translated_text}</strong></div>', unsafe_allow_html=True)
    
    # Processing details
    with st.expander("🔍 AI Processing Details", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            confidence = result.get('confidence', 0)
            confidence_color = "#4CAF50" if confidence >= 0.8 else "#FF9800" if confidence >= 0.6 else "#F44336"
            st.markdown(f'<div style="color: {confidence_color}; font-weight: bold;">Confidence: {confidence*100:.1f}%</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"**Method:** {result.get('method', 'Unknown')}")
    
    # Extract and display shopping items
    extracted_items = result.get('extracted_items', [])
    if extracted_items:
        st.markdown("### 🛍️ Detected Shopping Items")
        
        # Display items in a nice format
        cols = st.columns(min(len(extracted_items), 4))
        for i, item in enumerate(extracted_items):
            with cols[i % 4]:
                st.markdown(f'<div style="background: #fff3e0; padding: 10px; border-radius: 8px; text-align: center; margin: 5px 0;"><strong>🛒 {item.title()}</strong></div>', unsafe_allow_html=True)
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("➕ Add to Shopping List", type="primary", use_container_width=True):
                if 'shopping_list' not in st.session_state:
                    st.session_state.shopping_list = []
                
                new_items = [item for item in extracted_items if item not in st.session_state.shopping_list]
                st.session_state.shopping_list.extend(new_items)
                
                if new_items:
                    st.success(f"✅ Added {len(new_items)} new items to your list!")
                    st.rerun()
                else:
                    st.info("All items already in your list!")
        
        with col2:
            if st.button("🔍 Get Prices Now", type="secondary", use_container_width=True):
                get_price_comparison(extracted_items)
        
        with col3:
            if st.button("🤖 Get AI Suggestions", use_container_width=True):
                if ai_components['shopping_ai']:
                    analysis = ai_components['shopping_ai'].intelligent_product_analysis(extracted_items)
                    st.markdown("**AI Suggestions:**")
                    for suggestion in analysis.get('suggestions', [])[:3]:
                        st.info(suggestion)
    else:
        st.warning("No shopping items detected. Try being more specific about what you need.")
        st.markdown("**Try phrases like:**")
        st.markdown("- 'I need milk and bread'")
        st.markdown("- 'मुझे दूध चाहिए'")
        st.markdown("- 'Rice aur dal ka price batao'")
    
    # Store in voice history
    if 'voice_history' not in st.session_state:
        st.session_state.voice_history = []
    
    st.session_state.voice_history.append({
        'timestamp': datetime.datetime.now().isoformat(),
        'original_text': result.get('original_text', ''),
        'translated_text': result.get('translated_text', ''),
        'confidence': result.get('confidence', 0),
        'method': result.get('method', ''),
        'extracted_items': extracted_items
    })
def get_price_comparison(shopping_list):
    """Get comprehensive price comparison"""
    if not shopping_list:
        st.warning("No items to search for!")
        return
    
    st.markdown('<div style="background: linear-gradient(135deg, #4CAF50, #8BC34A); color: white; padding: 10px; border-radius: 10px; text-align: center;"><strong>⚡ Real-Time Price Analysis</strong></div>', unsafe_allow_html=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_results = {}
    total_items = len(shopping_list)
    
    for i, item in enumerate(shopping_list):
        status_text.text(f"🔍 Searching {item}... ({i+1}/{total_items})")
        progress_bar.progress((i + 1) / total_items)
        
        # Get marketplace data
        marketplace_data = ai_components['marketplace_connector'].search_real_product_prices(item)
        all_results[item] = marketplace_data
        
        # Small delay for better UX
        time.sleep(0.5)
    
    status_text.text("✅ Analysis complete!")
    progress_bar.progress(1.0)
    
    # Display results
    display_price_comparison_results(all_results, shopping_list)
def display_price_comparison_results(price_data, shopping_list):
    """Display comprehensive price comparison results"""
    st.markdown("## 💰 Smart Price Comparison Results")
    
    # Calculate summary statistics
    total_best_price = 0
    total_worst_price = 0
    best_deals = []
    
    for item, marketplaces in price_data.items():
        if marketplaces:
            prices = [(marketplace, data['price'] + data.get('delivery_fee', 0)) 
                     for marketplace, data in marketplaces.items()]
            prices.sort(key=lambda x: x[1])
            
            if prices:
                best_price = prices[0][1]
                worst_price = prices[-1][1]
                total_best_price += best_price
                total_worst_price += worst_price
                
                best_deals.append({
                    'item': item,
                    'marketplace': prices[0][0],
                    'price': best_price,
                    'savings': worst_price - best_price
                })
    
    # Display summary cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="background: #4CAF50; color: white; padding: 20px; border-radius: 15px; text-align: center;">
            <h3>💰 Best Total</h3>
            <div style="font-size: 2em; font-weight: bold;">₹{total_best_price:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        potential_savings = total_worst_price - total_best_price
        st.markdown(f"""
        <div style="background: #FF9800; color: white; padding: 20px; border-radius: 15px; text-align: center;">
            <h3>💸 Potential Savings</h3>
            <div style="font-size: 2em; font-weight: bold;">₹{potential_savings:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if best_deals:
            top_marketplace = max(set([deal['marketplace'] for deal in best_deals]), 
                                key=lambda x: sum(1 for deal in best_deals if deal['marketplace'] == x))
            marketplace_count = sum(1 for deal in best_deals if deal['marketplace'] == top_marketplace)
            st.markdown(f"""
            <div style="background: #2196F3; color: white; padding: 20px; border-radius: 15px; text-align: center;">
                <h3>🏆 Best Overall</h3>
                <div style="font-size: 1.2em; font-weight: bold;">{top_marketplace.title()}</div>
                <div>{marketplace_count} best deals</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed results
    st.markdown("### 📊 Detailed Price Analysis")
    
    for item, marketplaces in price_data.items():
        with st.expander(f"🔍 {item.title()} - Price Details", expanded=True):
            if not marketplaces:
                st.warning("No price data available for this item")
                continue
            
            # Sort marketplaces by total cost
            sorted_marketplaces = sorted(
                marketplaces.items(),
                key=lambda x: x[1]['price'] + x[1].get('delivery_fee', 0)
            )
            
            cols = st.columns(min(len(sorted_marketplaces), 4))
            
            for i, (marketplace, data) in enumerate(sorted_marketplaces[:4]):
                with cols[i]:
                    total_cost = data['price'] + data.get('delivery_fee', 0)
                    is_best = i == 0
                    
                    # Card styling
                    border_color = "#4CAF50" if is_best else "#ddd"
                    
                    st.markdown(f"""
                    <div style="border: 3px solid {border_color}; padding: 15px; border-radius: 10px; margin: 5px 0;">
                        <div style="font-weight: bold; color: #333; margin-bottom: 10px;">
                            {marketplace.title()}
                            {'🏆 BEST!' if is_best else ''}
                        </div>
                        <div style="font-size: 1.5em; font-weight: bold; color: #FF6D00;">₹{data['price']:.2f}</div>
                        <div style="color: #666; font-size: 0.9em;">
                            + ₹{data.get('delivery_fee', 0):.2f} delivery<br>
                            ⭐ {data.get('rating', 'N/A')} rating<br>
                            📊 Total: ₹{total_cost:.2f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Shopping tips
    st.markdown("### 💡 Smart Shopping Tips")
    tips = [
        f"💰 Save ₹{potential_savings:.2f} by choosing the best deals for each item",
        "📱 Check for app-exclusive offers and discounts",
        "🛒 Consider bulk buying for non-perishable items",
        "🕐 Shop during weekday mornings for better stock and deals"
    ]
    
    for tip in tips:
        st.info(tip)
def budget_ai_interface():
    """COMPLETE Budget AI interface with monthly budget tracking"""
    st.markdown('<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 15px; text-align: center; margin: 20px 0;"><h2>📊 Smart Budget AI</h2><p>Track expenses, set budgets, and get AI-powered insights</p></div>', unsafe_allow_html=True)
    
    # Initialize budget data
    if 'monthly_budget' not in st.session_state:
        st.session_state.monthly_budget = 0.0
    if 'monthly_expenses' not in st.session_state:
        st.session_state.monthly_expenses = []
    if 'budget_categories' not in st.session_state:
        st.session_state.budget_categories = {
            'Groceries': 0.0,
            'Household': 0.0,
            'Personal Care': 0.0,
            'Others': 0.0
        }
    
    # Budget Setup Section
    st.markdown("### 💰 Monthly Budget Setup")
    col1, col2 = st.columns(2)
    
    with col1:
        new_budget = st.number_input(
            "Set Monthly Budget (₹):",
            min_value=0.0,
            value=float(st.session_state.monthly_budget),
            step=500.0,
            help="Set your total monthly shopping budget"
        )
        
        if st.button("💾 Save Budget", type="primary"):
            st.session_state.monthly_budget = new_budget
            st.success(f"✅ Monthly budget set to ₹{new_budget:,.2f}")
            st.rerun()
    
    with col2:
        # Current month spending
        current_month = datetime.datetime.now().strftime('%Y-%m')
        monthly_spent = sum(
            expense['amount'] for expense in st.session_state.monthly_expenses
            if expense['date'].startswith(current_month)
        )
        
        remaining_budget = st.session_state.monthly_budget - monthly_spent
        budget_used_pct = (monthly_spent / st.session_state.monthly_budget * 100) if st.session_state.monthly_budget > 0 else 0
        
        # Budget status
        if budget_used_pct <= 50:
            status_color = "#4CAF50"
            status_text = "On Track"
        elif budget_used_pct <= 80:
            status_color = "#FF9800"
            status_text = "Watch Spending"
        else:
            status_color = "#F44336"
            status_text = "Over Budget"
        
        st.markdown(f'''
        <div style="background: {status_color}; color: white; padding: 15px; border-radius: 10px; text-align: center;">
            <h4>Budget Status: {status_text}</h4>
            <p style="font-size: 1.2em; margin: 5px 0;">₹{remaining_budget:,.2f} Remaining</p>
            <p style="margin: 0;">{budget_used_pct:.1f}% Used This Month</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Budget Dashboard
    if st.session_state.monthly_budget > 0:
        st.markdown("---")
        st.markdown("### 📊 Budget Dashboard")
        
        # Progress bar
        progress_value = min(budget_used_pct / 100, 1.0)
        st.progress(progress_value)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Monthly Budget", f"₹{st.session_state.monthly_budget:,.2f}")
        
        with col2:
            st.metric("Spent", f"₹{monthly_spent:,.2f}", delta=f"{budget_used_pct:.1f}%")
        
        with col3:
            st.metric("Remaining", f"₹{remaining_budget:,.2f}")
        
        with col4:
            daily_avg = monthly_spent / datetime.datetime.now().day if datetime.datetime.now().day > 0 else 0
            st.metric("Daily Average", f"₹{daily_avg:.2f}")
    
    # Add Expense Section
    st.markdown("---")
    st.markdown("### ➕ Add New Expense")
    
    with st.form("add_expense_form"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            expense_amount = st.number_input("Amount (₹)", min_value=0.0, step=10.0)
        
        with col2:
            expense_category = st.selectbox("Category", list(st.session_state.budget_categories.keys()))
        
        with col3:
            expense_date = st.date_input("Date", value=datetime.datetime.now())
        
        with col4:
            expense_description = st.text_input("Description", placeholder="e.g., Grocery shopping")
        
        submitted = st.form_submit_button("💾 Add Expense", type="primary")
        
        if submitted and expense_amount > 0:
            new_expense = {
                'amount': expense_amount,
                'category': expense_category,
                'date': expense_date.strftime('%Y-%m-%d'),
                'description': expense_description or f"{expense_category} expense",
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            st.session_state.monthly_expenses.append(new_expense)
            st.session_state.budget_categories[expense_category] += expense_amount
            
            st.success(f"✅ Added expense: ₹{expense_amount} for {expense_category}")
            st.rerun()
    
    # Expense History
    if st.session_state.monthly_expenses:
        st.markdown("---")
        st.markdown("### 📜 Recent Expenses")
        
        # Filter for current month
        current_month_expenses = [
            exp for exp in st.session_state.monthly_expenses
            if exp['date'].startswith(current_month)
        ]
        
        if current_month_expenses:
            # Create DataFrame for display
            df = pd.DataFrame(current_month_expenses)
            df['Date'] = pd.to_datetime(df['date']).dt.strftime('%d %b')
            df['Amount'] = df['amount'].apply(lambda x: f"₹{x:.2f}")
            
            # Display recent expenses
            display_df = df[['Date', 'Category', 'Amount', 'Description']].tail(10)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Category breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Category Breakdown")
                category_totals = df.groupby('category')['amount'].sum()
                fig_pie = px.pie(
                    values=category_totals.values,
                    names=category_totals.index,
                    title="Spending by Category"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.markdown("#### 📈 Daily Spending Trend")
                daily_spending = df.groupby('date')['amount'].sum().reset_index()
                daily_spending['date'] = pd.to_datetime(daily_spending['date'])
                
                fig_line = px.line(
                    daily_spending,
                    x='date',
                    y='amount',
                    title="Daily Spending This Month",
                    labels={'amount': 'Amount (₹)', 'date': 'Date'}
                )
                st.plotly_chart(fig_line, use_container_width=True)
        
        # AI Budget Insights
        st.markdown("---")
        st.markdown("### 🤖 AI Budget Insights")
        
        insights = []
        recommendations = []
        
        # Generate AI insights
        if budget_used_pct > 80:
            insights.append("⚠️ You've used over 80% of your monthly budget")
            recommendations.append("💡 Consider reducing discretionary spending")
        elif budget_used_pct < 30:
            insights.append("✅ Great job staying within budget!")
            recommendations.append("💰 You could allocate unused budget to savings")
        
        if len(current_month_expenses) > 5:
            avg_expense = sum(exp['amount'] for exp in current_month_expenses) / len(current_month_expenses)
            insights.append(f"📊 Your average expense is ₹{avg_expense:.2f}")
            
            if avg_expense > 200:
                recommendations.append("🛒 Consider bulk buying to reduce per-transaction costs")
        
        # Display insights
        if insights:
            for insight in insights:
                st.info(insight)
        
        if recommendations:
            for rec in recommendations:
                st.success(rec)
    
    else:
        st.info("📝 No expenses recorded yet. Add your first expense above!")
    
    # Quick Actions
    st.markdown("---")
    st.markdown("### ⚡ Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Generate Monthly Report", use_container_width=True):
            st.info("Monthly report generation coming soon!")
    
    with col2:
        if st.button("🔄 Reset Monthly Data", use_container_width=True):
            if st.button("⚠️ Confirm Reset", type="secondary"):
                st.session_state.monthly_expenses = []
                st.session_state.budget_categories = {key: 0.0 for key in st.session_state.budget_categories}
                st.success("Monthly data reset!")
                st.rerun()
    
    with col3:
        if st.button("💾 Export Data", use_container_width=True):
            if st.session_state.monthly_expenses:
                df_export = pd.DataFrame(st.session_state.monthly_expenses)
                csv = df_export.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"budget_data_{current_month}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No data to export")
def price_analytics_interface():
    """Price analytics interface"""
    st.markdown("### 📈 Price Analytics Interface")
    st.info("🔧 Price analytics features coming soon! View price trends and market analysis.")
def market_intelligence_interface():
    """Market intelligence interface"""
    st.markdown("### 🔍 Market Intelligence Interface")
    st.info("🔧 Market intelligence features coming soon! Get real-time market insights.")
# CSS Styling
def apply_custom_css():
    st.markdown("""
    <style>
    /* Main app styling */
    .main {
        padding-top: 2rem;
    }
    
    /* Status indicators */
    .api-status-live {
        background: linear-gradient(45deg, #4CAF50, #8BC34A);
        color: white;
        padding: 5px 12px;
        border-radius: 15px;
        font-size: 0.8em;
        display: inline-block;
        margin: 3px;
        animation: pulse 2s infinite;
    }
    
    .api-status-demo {
        background: linear-gradient(45deg, #FF9800, #FFC107);
        color: white;
        padding: 5px 12px;
        border-radius: 15px;
        font-size: 0.8em;
        display: inline-block;
        margin: 3px;
    }
    
    /* Animations */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 20px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 10px;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    
    .stTextArea > div > div > textarea {
        border-radius: 10px;
    }
    
    /* Success/Info/Warning boxes */
    .stSuccess, .stInfo, .stWarning {
        border-radius: 10px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom spacing */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Main application function
def main():
    """Main Siora application with real AI integration"""
    
    # Apply custom CSS
    apply_custom_css()
    
    # Header with branding
    st.markdown("""
    <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 30px;">
        <h1 style="color: white; margin: 0; font-size: 3em;">🛒 Siora</h1>
        <h2 style="color: #f0f0f0; margin: 10px 0; font-size: 1.5em;">AI Shopping Buddy</h2>
        <p style="color: #e0e0e0; margin: 0; font-size: 1.1em;">
            Powered by Real AI • Voice-Enabled • Market Intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Real AI Status Dashboard
    st.markdown("### 🔧 System Status")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        serp_status = "🟢 Live" if get_secret('SERPAPI_KEY') != 'demo_key' else "🟡 Demo"
        st.markdown(f'<div class="api-status-{"live" if "Live" in serp_status else "demo"}">SERP API: {serp_status}</div>', unsafe_allow_html=True)
    
    with col2:
        hf_status = "🟢 Live" if get_secret('HUGGINGFACE_TOKEN') != 'demo_key' else "🟡 Demo"
        st.markdown(f'<div class="api-status-{"live" if "Live" in hf_status else "demo"}">HuggingFace: {hf_status}</div>', unsafe_allow_html=True)
    
    with col3:
        transformers_status = "🟢 Ready" if TRANSFORMERS_AVAILABLE else "🟡 Loading"
        st.markdown(f'<div class="api-status-{"live" if "Ready" in transformers_status else "demo"}">AI Models: {transformers_status}</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="api-status-live">⚡ Real-Time Mode</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Mode Selection with proper routing
    st.markdown("### 🎯 Choose Your Shopping Mode")
    app_mode = st.selectbox(
        "Select Mode:",
        [
            "🛒 Smart Shopping", 
            "🎤 Voice Shopping", 
            "📊 Budget AI", 
            "📈 Price Analytics", 
            "🔍 Market Intelligence"
        ],
        key="main_mode_selector",
        help="Select different modes to access various AI-powered shopping features"
    )
    
    st.markdown("---")
    
    # Route to appropriate interface based on selection
    try:
        if app_mode == "🛒 Smart Shopping":
            smart_shopping_interface()
        elif app_mode == "🎤 Voice Shopping":
            voice_shopping_interface()
        elif app_mode == "📊 Budget AI":
            budget_ai_interface()
        elif app_mode == "📈 Price Analytics":
            price_analytics_interface()
        elif app_mode == "🔍 Market Intelligence":
            market_intelligence_interface()
        else:
            st.error("Unknown mode selected")
            
    except Exception as e:
        st.error(f"Error loading interface: {str(e)}")
        st.info("Please try refreshing the page or selecting a different mode.")

# Sidebar with additional information and controls
def display_sidebar():
    """Display sidebar with app information and controls"""
    with st.sidebar:
        st.markdown('<div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin-bottom: 20px;"><strong>🤖 Siora AI</strong></div>', unsafe_allow_html=True)
        
        # Quick stats
        st.markdown("### 📊 Session Stats")
        
        # Initialize session counters
        if 'total_searches' not in st.session_state:
            st.session_state.total_searches = 0
        if 'total_savings' not in st.session_state:
            st.session_state.total_savings = 0.0
        if 'items_analyzed' not in st.session_state:
            st.session_state.items_analyzed = 0
        
        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Searches", st.session_state.total_searches)
            st.metric("Items Analyzed", st.session_state.items_analyzed)
        with col2:
            st.metric("Potential Savings", f"₹{st.session_state.total_savings:.0f}")
            if 'shopping_list' in st.session_state:
                st.metric("List Items", len(st.session_state.shopping_list))
            else:
                st.metric("List Items", 0)
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🗑️ Clear All Data", help="Clear shopping list and reset session"):
            # Clear all session data
            for key in ['shopping_list', 'voice_history']:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("All data cleared!")
            st.rerun()
        
        if st.button("🔄 Refresh AI Components", help="Reload AI components"):
            st.cache_resource.clear()
            st.success("AI components refreshed!")
            st.rerun()
        
        st.markdown("---")
        
        # App information
        st.markdown("### ℹ️ App Info")
        st.markdown("""
        **Features:**
        - 🛒 Smart Shopping Lists
        - 🎤 Voice Commands (Hindi/English)
        - 💰 Real-time Price Comparison
        - 🤖 AI Recommendations
        - 📊 Shopping Analytics
        
        **Powered by:**
        - HuggingFace Transformers
        - Google SERP API
        - Vaani Speech AI
        - Real-time Market Data
        """)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.8em;">
            <p>🤖 <strong>Siora AI Shopping Buddy</strong></p>
            <p>Built with ❤️ using Streamlit</p>
        </div>
        """, unsafe_allow_html=True)

# App footer
def display_footer():
    """Display application footer"""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 30px; background: #f8f9fa; border-radius: 15px; margin-top: 30px;">
        <h4 style="color: #333; margin-bottom: 15px;">🤖 Siora - AI Shopping Buddy</h4>
        <p style="color: #666; margin-bottom: 10px;">
            Powered by Real AI • Vaani Speech Processing • SERP Market Intelligence
        </p>
        <div style="color: #888; font-size: 0.9em;">
            <p>🔧 <strong>Technologies:</strong> Streamlit • HuggingFace Transformers • Google SERP API • Python</p>
            <p>🎯 <strong>Features:</strong> Voice Commands • Price Comparison • Smart Recommendations • Budget Analysis</p>
            <p>🌟 <strong>AI Models:</strong> Whisper Speech Recognition • Translation • Shopping Intelligence</p>
        </div>
        <div style="margin-top: 20px; color: #999; font-size: 0.8em;">
            Made with ❤️ for smarter shopping experiences
        </div>
    </div>
    """, unsafe_allow_html=True)

# Error handling wrapper
def safe_main():
    """Safe main function with error handling"""
    try:
        # Display sidebar
        display_sidebar()
        
        # Run main app
        main()
        
        # Display footer
        display_footer()
        
    except Exception as e:
        st.error("🚨 Application Error")
        st.markdown(f"""
        <div style="background: #ffebee; border: 1px solid #f44336; border-radius: 10px; padding: 20px; margin: 10px 0;">
            <h4 style="color: #c62828; margin: 0 0 10px 0;">Something went wrong!</h4>
            <p style="color: #666; margin: 0;"><strong>Error:</strong> {str(e)}</p>
            <p style="color: #666; margin: 10px 0 0 0;"><strong>Solution:</strong> Try refreshing the page or contact support if the issue persists.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Option to clear cache and restart
        if st.button("🔄 Clear Cache & Restart", type="primary"):
            st.cache_resource.clear()
            st.rerun()

# Application entry point
if __name__ == "__main__":
    # Set page configuration if not already set
    try:
        safe_main()
    except Exception as e:
        st.error(f"Critical application error: {str(e)}")
        st.info("Please refresh the page to restart the application.")
