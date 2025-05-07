import streamlit as st
import pandas as pd
import random
import time
import plotly.express as px
import datetime
import base64
from io import BytesIO
from PIL import Image, ImageDraw

# App configuration
st.set_page_config(page_title="Siora - Your Shopping Buddy", page_icon="🛒", layout="wide")

# Function to simulate Vaani Hindi to English translation
def simulate_vaani_translation():
    """
    Simulate Hindi to English translation using Vaani API.
    For the hackathon demo, this simulates the process.
    """
    # Hindi phrases and their English translations
    hindi_english_samples = [
        {
            "hindi": "दूध, रोटी, और अंडे",
            "english": "milk, bread, and eggs"
        },
        {
            "hindi": "चावल, दाल, आटा, और तेल",
            "english": "rice, lentils, flour, and oil"
        },
        {
            "hindi": "सब्जियां, फल, और मसाले",
            "english": "vegetables, fruits, and spices"
        },
        {
            "hindi": "साबुन, डिटर्जेंट, और टूथपेस्ट",
            "english": "soap, detergent, and toothpaste"
        }
    ]
    
    # Randomly select a sample for the demo
    sample = random.choice(hindi_english_samples)
    
    return sample

# Function to display an advertisement
def display_ad():
    ad_types = [
        {
            "title": "Save 20% on Groceries",
            "description": "Use code SIORA20 for 20% off your first order at BigBasket",
            "color": "#e3f2fd",
            "text_color": "#1565c0",
            "company": "BigBasket"
        },
        {
            "title": "Free Delivery on Swiggy",
            "description": "Order above ₹199 and get free delivery on your first 3 orders",
            "color": "#fff3e0",
            "text_color": "#e65100",
            "company": "Swiggy"
        },
        {
            "title": "Flash Sale on Electronics",
            "description": "50% off on selected electronics this weekend only",
            "color": "#e8f5e9",
            "text_color": "#2e7d32",
            "company": "Flipkart"
        }
    ]
    
    ad = random.choice(ad_types)
    
    st.markdown(f"""
    <div style="background-color: {ad['color']}; border-radius: 8px; padding: 15px; margin: 10px 0; border: 1px solid #E4E6E8; border-left: 5px solid {ad['text_color']}; text-align: center;">
        <div style="font-size: 10px; text-transform: uppercase; color: #666; margin-bottom: 5px;">Advertisement</div>
        <div style="display: flex; align-items: center; justify-content: center;">
            <div>
                <h4 style="color: {ad['text_color']}; margin: 0;">{ad['title']}</h4>
                <p style="margin: 5px 0">{ad['description']}</p>
                <p style="font-size: 11px; margin: 0;">Sponsored by {ad['company']}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Custom CSS for colorful design
def apply_custom_css():
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary: #2962FF;
        --primary-light: #768fff;
        --primary-dark: #0039cb;
        --secondary: #FF6D00;
        --secondary-light: #ff9e40;
        --secondary-dark: #c43c00;
        --background: #F5F7FF;
        --surface: #FFFFFF;
        --text: #333333;
    }
    
    /* Overall page styling */
    .main {
        background-color: var(--background);
        color: var(--text);
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: var(--primary-dark);
    }
    
    /* Custom containers */
    .card {
        background-color: var(--surface);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    
    .highlight-card {
        background: linear-gradient(135deg, var(--primary), var(--primary-light));
        color: white;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }
    
    /* Success message */
    .success-message {
        background-color: #00C853;
        color: white;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Transaction history table */
    .transaction-table {
        width: 100%;
        border-collapse: collapse;
    }
    
    .transaction-table th {
        background-color: var(--primary);
        color: white;
        padding: 12px;
        text-align: left;
    }
    
    .transaction-table td {
        padding: 12px;
        border-bottom: 1px solid #ddd;
    }
    
    .transaction-table tr:nth-child(even) {
        background-color: #f2f2f2;
    }
    
    .transaction-table tr:hover {
        background-color: #e1f5fe;
    }
    
    /* Voice button styling */
    .voice-button {
        background-color: #FF6D00;
        color: white;
        border: none;
        border-radius: 50%;
        width: 36px;
        height: 36px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin: 0;
        padding: 0;
        cursor: pointer;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    .voice-button:hover {
        background-color: #FF9E40;
    }
    
    /* Custom tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #f0f7ff;
        border-bottom: 3px solid #2962FF;
    }
    </style>
    """, unsafe_allow_html=True)

# Apply custom CSS
apply_custom_css()

# Initialize session state
if "shopping_list" not in st.session_state:
    st.session_state.shopping_list = []
if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = None
if "monthly_spending" not in st.session_state:
    st.session_state.monthly_spending = {"Groceries": 0}  # Only Groceries category now
if "order_placed" not in st.session_state:
    st.session_state.order_placed = False
if "order_details" not in st.session_state:
    st.session_state.order_details = {}
if "transaction_history" not in st.session_state:
    st.session_state.transaction_history = []
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Shop"
if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""

# Function to create grocery spending pie chart
def create_grocery_spending_chart():
    # Get current month for the title
    current_month = datetime.datetime.now().strftime("%B")
    
    # Get actual budget and spending values for groceries
    grocery_budget = st.session_state.get("grocery_budget", 5000)
    grocery_spent = st.session_state.monthly_spending.get("Groceries", 0)
    
    # Calculate remaining
    grocery_remaining = max(0, grocery_budget - grocery_spent)
    
    # Create data for the pie chart
    spend_data = {
        'Status': ['Spent', 'Remaining'],
        'Amount': [grocery_spent, grocery_remaining]
    }
    
    df = pd.DataFrame(spend_data)
    
    # Create a pie chart with colorful theme
    fig = px.pie(
        df, 
        values='Amount', 
        names='Status', 
        title=f'Grocery Budget for {current_month}',
        color_discrete_sequence=['#FF6D00', '#2962FF'],
        hole=0.4
    )
    
    # Update to show actual percentages
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+value',
        marker=dict(line=dict(color='#FFFFFF', width=2)),
        texttemplate='%{percent:.1f}%<br>₹%{value:.0f}'
    )
    
    fig.update_layout(
        font_family="Arial",
        title_font_size=20,
        title_font_color="#333333",
        legend_title_font_color="#333333"
    )
    
    return fig

# Display improved header
st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 30px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); border-radius: 12px; overflow: hidden;">
    <div style="background: linear-gradient(135deg, #2962FF, #1565C0); padding: 25px; color: white; text-align: center; width: 150px; height: 150px; display: flex; align-items: center; justify-content: center;">
        <h1 style="font-size: 2rem; margin: 0; font-weight: bold;">SIORA</h1>
    </div>
    <div style="padding: 20px 30px; flex: 1;">
        <h1 style="color: #2962FF; margin: 0 0 5px 0; font-size: 2.4rem;">Shopping Optimization Agent</h1>
        <p style="color: #555; margin: 0; font-size: 1.1rem;">Compare prices, track budgets, shop smarter</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Main navigation tabs
tab1, tab2, tab3 = st.tabs(["🛒 Shop", "📊 Budget", "📜 Transaction History"])
# Display order confirmation message if an order was just placed
if st.session_state.order_placed:
    order_details = st.session_state.order_details
    
    with tab1:
        # Create a highlighted order confirmation message
        st.markdown("""
        <div style="background-color: #00C853; color: white; padding: 20px; border-radius: 8px; text-align: center; margin-bottom: 20px; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 24px; margin-right: 10px;">✓</span>
            <span style="font-size: 20px; font-weight: bold;">Your order has been placed successfully!</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Order details in a nice container
        st.markdown("""
        <div class="highlight-card">
            <h3 style="margin-top: 0;">Order Summary</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="card">
                <p><strong>Marketplace:</strong> {order_details.get('marketplace', '').capitalize()}</p>
                <p><strong>Order Total:</strong> ₹{order_details.get('total', 0):.2f}</p>
                <p><strong>Order Date:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="card">
                <p><strong>Items:</strong> {', '.join(order_details.get('items', []))}</p>
                <p><strong>Delivery Estimate:</strong> {order_details.get('delivery_time', '30-60 minutes')}</p>
                <p><strong>Transaction ID:</strong> TXN-{datetime.datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Add transaction to history
        transaction = {
            "date": datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
            "items": order_details.get('items', []),
            "marketplace": order_details.get('marketplace', '').capitalize(),
            "amount": order_details.get('total', 0),
            "transaction_id": f"TXN-{datetime.datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}",
            "delivery_time": order_details.get('delivery_time', '30-60 minutes')
        }
        st.session_state.transaction_history.append(transaction)
        
        # Updated grocery spending chart
        st.subheader("Updated Budget Status")
        spending_fig = create_grocery_spending_chart()
        st.plotly_chart(spending_fig, use_container_width=True, key="order_confirmation_chart")
        
        # Transaction history
        st.markdown("""
        <div class="highlight-card">
            <h3 style="margin-top: 0;">Latest Transaction</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a table for the latest transaction
        st.markdown("""
        <table class="transaction-table">
            <tr>
                <th>Date</th>
                <th>Items</th>
                <th>Marketplace</th>
                <th>Amount</th>
                <th>Transaction ID</th>
            </tr>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <tr>
                <td>{transaction['date']}</td>
                <td>{', '.join(transaction['items'])}</td>
                <td>{transaction['marketplace']}</td>
                <td>₹{transaction['amount']:.2f}</td>
                <td>{transaction['transaction_id']}</td>
            </tr>
        </table>
        """, unsafe_allow_html=True)
        
        # Navigate to transaction history
        st.markdown("""
        <div class="card">
            <p>Your transaction has been added to your history. View all your past purchases in the Transaction History tab.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Button to continue shopping
        if st.button("Continue Shopping", key="continue_shopping"):
            st.session_state.order_placed = False
            st.rerun()

# Tab 1: Shopping Interface
with tab1:
    if not st.session_state.order_placed:
        # Shopping list input
        st.markdown("""
        <div class="highlight-card">
            <h2 style="margin-top: 0;">Create Shopping List</h2>
            <p>Enter the items you want to buy, separated by commas</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display ad at the top of the shopping interface
        display_ad()
        
        # Text input with voice button
        col1, col2, col3 = st.columns([4, 0.5, 1])
        
        with col1:
            new_item = st.text_input("", value=st.session_state.voice_text, 
                                     placeholder="e.g., milk, bread, eggs", 
                                     label_visibility="collapsed")
        
        with col2:
            # Voice input button with better styling
            if st.button("🎤", help="Click for Hindi voice input that will be translated to English"):
                with st.spinner("Listening to Hindi speech..."):
                    # Simulate processing time
                    time.sleep(1.5)
                    
                    # Get simulated translation
                    translation = simulate_vaani_translation()
                    
                    # Show the "detected" Hindi and translation
                    st.info(f"Detected Hindi: **{translation['hindi']}**")
                    time.sleep(0.8)  # Brief pause to simulate translation time
                    
                    # Show the English translation
                    st.success(f"Translated to English: **{translation['english']}**")
                    
                    # Update the text input field with the English translation
                    st.session_state.voice_text = translation['english']
                    
                    # Show integration message
                    st.markdown("""
                    <div style="background-color: #EDE7F6; padding: 10px; border-radius: 5px; border-left: 5px solid #673AB7; margin-top: 10px; font-size: 0.9em;">
                        <b>Integration:</b> Using Vaani API for Hindi-to-English speech translation
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Rerun to update the UI
                    st.rerun()
        
        with col3:
            if st.button("Compare Prices", type="primary"):
                if new_item:
                    # Parse items from input
                    items = [item.strip() for item in new_item.split(",") if item.strip()]
                    st.session_state.shopping_list = items
                    st.session_state.voice_text = ""  # Clear voice text after submission
                    
                    # Show loading indicator
                    with st.spinner("Finding the best deals for you..."):
                        time.sleep(2)  # Simulate processing time
                        
                        # Generate simulated comparison results
                        marketplaces = ["zepto", "swiggy", "blinkit", "bigbasket"]
                        results = {}
                        
                        for item in items:
                            # Set base price based on item
                            if "milk" in item.lower():
                                base_price = 50
                            elif "bread" in item.lower():
                                base_price = 35
                            elif "eggs" in item.lower():
                                base_price = 80
                            elif "rice" in item.lower():
                                base_price = 60
                            else:
                                base_price = random.randint(30, 100)
                            
                            # Generate prices for each marketplace
                            item_results = {}
                            for marketplace in marketplaces:
                                # Add some price variation
                                modifier = {
                                    "zepto": random.uniform(0.8, 1.1),
                                    "swiggy": random.uniform(0.9, 1.2),
                                    "blinkit": random.uniform(0.85, 1.05),
                                    "bigbasket": random.uniform(0.95, 1.15)
                                }
                                
                                price = round(base_price * modifier[marketplace], 2)
                                delivery_fee = round(random.uniform(20, 40), 2)
                                
                                item_results[marketplace] = {
                                    "price": price,
                                    "delivery_fee": delivery_fee,
                                    "delivery_time": f"{random.randint(15, 60)} mins"
                                }
                            
                            results[item] = item_results
                        
                        st.session_state.comparison_results = results

        # Display comparison results
        if st.session_state.comparison_results:
            st.markdown("""
            <div class="highlight-card">
                <h2 style="margin-top: 0;">Price Comparison Results</h2>
                <p>Compare prices across marketplaces and find the best deals</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display another ad
            display_ad()
            
            results = st.session_state.comparison_results
            marketplaces = ["zepto", "swiggy", "blinkit", "bigbasket"]
            
            # Create tabs for different views
            item_tab, marketplace_tab = st.tabs(["By Item", "By Marketplace"])
            
            with item_tab:
                # Show each item comparison
                for item, marketplaces_data in results.items():
                    with st.expander(f"{item.capitalize()}", expanded=True):
                        # Create a dataframe for this item
                        data = []
                        for marketplace, details in marketplaces_data.items():
                            data.append({
                                "Marketplace": marketplace.capitalize(),
                                "Price (₹)": details["price"],
                                "Delivery (₹)": details["delivery_fee"],
                                "Delivery Time": details["delivery_time"],
                                "Total (₹)": details["price"] + details["delivery_fee"]
                            })
                        
                        df = pd.DataFrame(data)
                        best_price_idx = df["Price (₹)"].idxmin()
                        
                        st.dataframe(df, use_container_width=True)
                        st.markdown(f"""
                        <div style="background-color: #E8F5E9; border-radius: 5px; padding: 10px; border-left: 5px solid #4CAF50;">
                            <strong>Best price:</strong> {df.iloc[best_price_idx]['Marketplace']} - ₹{df.iloc[best_price_idx]['Price (₹)']}
                        </div>
                        """, unsafe_allow_html=True)
            
            with marketplace_tab:
                # Calculate totals for each marketplace
                marketplace_totals = {}
                for marketplace in marketplaces:
                    # Calculate item totals for this marketplace
                    item_total = sum(results[item][marketplace]["price"] for item in results)
                    # Highest delivery fee (assuming one delivery)
                    delivery = max(results[item][marketplace]["delivery_fee"] for item in results)
                    grand_total = item_total + delivery
                    delivery_time = results[list(results.keys())[0]][marketplace]["delivery_time"]  # Just use first item's delivery time
                    
                    marketplace_totals[marketplace] = {
                        "item_total": item_total,
                        "delivery_fee": delivery,
                        "grand_total": grand_total,
                        "delivery_time": delivery_time
                    }
                
                # Find best marketplace
                best_marketplace = min(marketplace_totals.items(), key=lambda x: x[1]["grand_total"])[0]
                
                # Show marketplace comparison
                for marketplace, totals in marketplace_totals.items():
                    is_best = marketplace == best_marketplace
                    with st.expander(
                        f"{marketplace.capitalize()}{' (BEST DEAL)' if is_best else ''}",
                        expanded=is_best
                    ):
                        cols = st.columns([1, 1, 1])
                        with cols[0]:
                            st.markdown(f"""
                            <div class="card" style="height: 100%;">
                                <h4 style="margin-top: 0;">Items</h4>
                                <p><strong>Total:</strong> ₹{totals['item_total']:.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with cols[1]:
                            st.markdown(f"""
                            <div class="card" style="height: 100%;">
                                <h4 style="margin-top: 0;">Delivery</h4>
                                <p><strong>Fee:</strong> ₹{totals['delivery_fee']:.2f}</p>
                                <p><strong>Time:</strong> {totals['delivery_time']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        with cols[2]:
                            st.markdown(f"""
                            <div class="card" style="height: 100%;">
                                <h4 style="margin-top: 0;">Summary</h4>
                                <p><strong>Grand Total:</strong> ₹{totals['grand_total']:.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Buy button for each marketplace
                        if st.button(f"Buy from {marketplace.capitalize()}", key=f"buy_{marketplace}", 
                                    type="primary" if is_best else "secondary"):
                            # Update monthly spending - ALL GOES TO GROCERIES NOW
                            total_spent = totals["grand_total"]
                            # CHANGE: All spending goes to Groceries category
                            st.session_state.monthly_spending["Groceries"] += total_spent
                            
                            # Store order details to display on confirmation page
                            st.session_state.order_details = {
                                "marketplace": marketplace,
                                "items": st.session_state.shopping_list,
                                "total": total_spent,
                                "delivery_time": totals["delivery_time"]
                            }
                            
                            # Set order placed flag
                            st.session_state.order_placed = True
                            
                            # Reset shopping list and comparison results
                            st.session_state.comparison_results = None
                            st.session_state.shopping_list = []
                            
                            # Rerun to show confirmation screen
                            st.rerun()
                
                # Display another ad before purchase button
                display_ad()
                
                # Buy button for best marketplace (outside the expanders)
                st.markdown("""
                <div class="highlight-card">
                    <h3 style="margin-top: 0;">Ready to Purchase?</h3>
                    <p>Choose the best deal to maximize your savings</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Buy from {best_marketplace.capitalize()} (BEST DEAL)", type="primary"):
                    # Update monthly spending - ALL GOES TO GROCERIES NOW
                    total_spent = marketplace_totals[best_marketplace]["grand_total"]
                    # CHANGE: All spending goes to Groceries category
                    st.session_state.monthly_spending["Groceries"] += total_spent
                    
                    # Store order details to display on confirmation page
                    st.session_state.order_details = {
                        "marketplace": best_marketplace,
                        "items": st.session_state.shopping_list,
                        "total": total_spent,
                        "delivery_time": marketplace_totals[best_marketplace]["delivery_time"]
                    }
                    
                    # Set order placed flag
                    st.session_state.order_placed = True
                    
                    # Reset shopping list and comparison results
                    st.session_state.comparison_results = None
                    st.session_state.shopping_list = []
                    
                    # Rerun to show confirmation screen
                    st.rerun()

        # Display welcome message for first-time users
        if not st.session_state.shopping_list and not st.session_state.comparison_results and not st.session_state.order_placed:
            st.markdown("""
            <div class="card" style="background-color: #E3F2FD; border-left: 5px solid #2196F3;">
                <h3 style="margin-top: 0;">👋 Welcome to Siora!</h3>
                <p>Enter your shopping list above to compare prices across marketplaces.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show demo items
            st.markdown("""
            <div class="card">
                <h3 style="margin-top: 0;">Try these example items:</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Milk, Bread, Eggs", key="example1"):
                    st.session_state.shopping_list = ["Milk", "Bread", "Eggs"]
                    st.rerun()
            with col2:
                if st.button("Rice, Flour, Soap", key="example2"):
                    st.session_state.shopping_list = ["Rice", "Flour", "Soap"]
                    st.rerun()
            with col3:
                if st.button("Vegetables, Fruits", key="example3"):
                    st.session_state.shopping_list = ["Vegetables", "Fruits"]
                    st.rerun()
            
            # Display ad at the bottom
            display_ad()

# Tab 2: Budget Management
with tab2:
    st.markdown("""
    <div class="highlight-card">
        <h2 style="margin-top: 0;">Budget Management</h2>
        <p>Track your spending and manage your monthly budget</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display ad in budget tab
    display_ad()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3 style="margin-top: 0;">Set Monthly Budget</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Store these values in session state so they persist
        st.session_state.grocery_budget = st.number_input("Grocery Budget (₹)", value=st.session_state.get("grocery_budget", 5000), min_value=0)
        
        # Add a reset budget button
        if st.button("Reset Monthly Spending"):
            # Reset only the spending, not the budgets
            st.session_state.monthly_spending = {"Groceries": 0}
            st.success("Monthly spending has been reset!")
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3 style="margin-top: 0;">Current Budget Status</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Show grocery spending pie chart
        spending_fig = create_grocery_spending_chart()
        st.plotly_chart(spending_fig, use_container_width=True, key="budget_spending_chart")
    
    # Monthly spending summary
    st.markdown("""
    <div class="card">
        <h3 style="margin-top: 0;">Spending Summary</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # CHANGE: Only show grocery spending
    grocery_spent = st.session_state.monthly_spending.get("Groceries", 0)
    
    st.markdown(f"""
    <div class="card" style="text-align: center; background-color: #E3F2FD; height: 100%;">
        <h3 style="margin-top: 0; color: #2962FF;">Grocery Spending</h3>
        <p style="font-size: 2.5rem; font-weight: bold; color: #2962FF;">₹{grocery_spent:.2f}</p>
        <p>of ₹{st.session_state.grocery_budget:.2f} budget</p>
    </div>
    """, unsafe_allow_html=True)

# Tab 3: Transaction History
with tab3:
    st.markdown("""
    <div class="highlight-card">
        <h2 style="margin-top: 0;">Transaction History</h2>
        <p>View all your past purchases and track your spending</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display ad in transaction history tab
    display_ad()
    
    if st.session_state.transaction_history:
        # Display all transactions in a table
        st.markdown("""
        <table class="transaction-table">
            <tr>
                <th>Date</th>
                <th>Items</th>
                <th>Marketplace</th>
                <th>Amount</th>
                <th>Transaction ID</th>
                <th>Delivery Estimate</th>
            </tr>
        """, unsafe_allow_html=True)
        
        for transaction in reversed(st.session_state.transaction_history):
            st.markdown(f"""
            <tr>
                <td>{transaction['date']}</td>
                <td>{', '.join(transaction['items'])}</td>
                <td>{transaction['marketplace']}</td>
                <td>₹{transaction['amount']:.2f}</td>
                <td>{transaction['transaction_id']}</td>
                <td>{transaction['delivery_time']}</td>
            </tr>
            """, unsafe_allow_html=True)
        
        st.markdown("</table>", unsafe_allow_html=True)
        
        # Add transaction summary
        st.markdown("""
        <div class="card">
            <h3 style="margin-top: 0;">Transaction Summary</h3>
        </div>
        """, unsafe_allow_html=True)
        
        total_spent = sum(transaction['amount'] for transaction in st.session_state.transaction_history)
        avg_transaction = total_spent / len(st.session_state.transaction_history) if st.session_state.transaction_history else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="card" style="text-align: center; background-color: #E3F2FD;">
                <h3 style="margin-top: 0; color: #2962FF;">Total Transactions</h3>
                <p style="font-size: 2.5rem; font-weight: bold; color: #2962FF;">{len(st.session_state.transaction_history)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="card" style="text-align: center; background-color: #FFF3E0;">
                <h3 style="margin-top: 0; color: #FF6D00;">Total Spent</h3>
                <p style="font-size: 2.5rem; font-weight: bold; color: #FF6D00;">₹{total_spent:.2f}</p>
                <p>Average: ₹{avg_transaction:.2f} per transaction</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Display ad at the bottom of transaction history
        display_ad()
    else:
        st.markdown("""
        <div class="card" style="text-align: center; padding: 40px;">
            <h3 style="margin-top: 0;">No transactions yet</h3>
            <p>Your transaction history will appear here after your first purchase</p>
            <p>Go to the Shop tab to start shopping!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display ad when no transactions
        display_ad()
