import streamlit as st
import os
from datetime import datetime
import pandas as pd
import json
import time
import plotly.express as px  # This was the missing import
from dotenv import load_dotenv

# Import project modules
from config import OPENAI_API_KEY, STRIPE_API_KEY, MARKETPLACES, DB_CONFIG
from database.db_manager import DatabaseManager
from utils.price_scraper import PriceComparer
from utils.payment_handler import PaymentProcessor
from agents.shopping_agent import ShoppingAgent


# Load environment variables
load_dotenv()

# App configuration
st.set_page_config(
    page_title="Siora - Shopping Optimization Agent",
    page_icon="🛒",
    layout="wide",
)

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_user" not in st.session_state:
    st.session_state.current_user = "user123"  # Default user ID for demo

if "shopping_list" not in st.session_state:
    st.session_state.shopping_list = []

if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = None

if "payment_authorized" not in st.session_state:
    st.session_state.payment_authorized = False

if "show_budget_form" not in st.session_state:
    st.session_state.show_budget_form = False

# Initialize components
@st.cache_resource
def initialize_components():
    db_manager = DatabaseManager(DB_CONFIG.get("path", "siora_database.db"))
    price_comparer = PriceComparer(MARKETPLACES)
    payment_processor = PaymentProcessor(STRIPE_API_KEY)
    
    # Initialize agent
    agent = ShoppingAgent(
        api_key=OPENAI_API_KEY,
        db_manager=db_manager,
        price_comparer=price_comparer,
        payment_processor=payment_processor,
        model_name="gpt-4o",  # Use the best model for the hackathon
        temperature=0.2,
        verbose=True
    )
    
    return db_manager, price_comparer, payment_processor, agent

db_manager, price_comparer, payment_processor, agent = initialize_components()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #4A90E2;
        margin-bottom: 10px;
        text-align: center;
    }
    .subheader {
        font-size: 20px;
        color: #555555;
        margin-bottom: 30px;
        text-align: center;
    }
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        display: flex;
        flex-direction: row;
    }
    .chat-message.user {
        background-color: #F0F2F6;
        border-left: 5px solid #4A90E2;
    }
    .chat-message.assistant {
        background-color: #EEF9FF;
        border-left: 5px solid #25C7B7;
    }
    .avatar {
        min-width: 40px;
        margin-right: 10px;
    }
    .avatar img {
        max-width: 40px;
        max-height: 40px;
        border-radius: 50%;
    }
    .message {
        flex-grow: 1;
    }
    .comparison-card {
        border: 1px solid #DDD;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: white;
    }
    .comparison-header {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #333;
    }
    .marketplace-card {
        border: 1px solid #EEE;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        background-color: #FAFAFA;
    }
    .marketplace-name {
        font-size: 16px;
        font-weight: bold;
        color: #4A90E2;
    }
    .best-deal {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
    }
    .action-button {
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='main-header'>🛒 Siora</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Your Intelligent Shopping Assistant</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Menu")
    
    # User settings
    st.subheader("User Settings")
    user_id = st.text_input("User ID", value=st.session_state.current_user)
    if user_id != st.session_state.current_user:
        st.session_state.current_user = user_id
    
    # Budget management
    st.subheader("Budget Management")
    if st.button("Set Monthly Budget"):
        st.session_state.show_budget_form = True
    
    # Display current budget if available
    try:
        budget_items = db_manager.get_budget(
            user_id, 
            month=datetime.now().strftime("%B"), 
            year=datetime.now().year
        )
        
        if budget_items:
            st.write("Current Budget:")
            budget_df = pd.DataFrame(budget_items, columns=["Category", "Amount"])
            st.dataframe(budget_df, hide_index=True, use_container_width=True)
            
            # Budget visualization
            if not budget_df.empty:
                fig = px.pie(budget_df, values='Amount', names='Category', title='Budget Allocation')
                st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading budget: {str(e)}")
    
    # Shopping history
    st.subheader("Shopping History")
    try:
        history = db_manager.get_shopping_history(user_id)
        if history:
            for i, transaction in enumerate(history[:5]):  # Show last 5 transactions
                with st.expander(f"Purchase on {transaction['date'][:10]}"):
                    st.write(f"**Marketplace:** {transaction['marketplace']}")
                    st.write(f"**Items:** {', '.join(transaction['items'])}")
                    st.write(f"**Total:** ₹{transaction['total_amount']:.2f}")
    except Exception as e:
        st.error(f"Error loading history: {str(e)}")
    
    # Reset button
    if st.button("Reset Chat"):
        st.session_state.messages = []
        st.session_state.shopping_list = []
        st.session_state.comparison_results = None
        st.session_state.payment_authorized = False

# Budget form
if st.session_state.show_budget_form:
    with st.expander("Set Monthly Budget", expanded=True):
        with st.form("budget_form"):
            st.write(f"Setting budget for {datetime.now().strftime('%B %Y')}")
            
            # Dynamic category inputs
            categories = ["Groceries", "Household", "Personal Care", "Electronics", "Other"]
            category_amounts = {}
            
            for category in categories:
                amount = st.number_input(f"{category} Budget (₹)", min_value=0.0, step=100.0)
                category_amounts[category] = amount
            
            # Add custom category
            custom_category = st.text_input("Add Custom Category")
            if custom_category:
                custom_amount = st.number_input(f"{custom_category} Budget (₹)", min_value=0.0, step=100.0)
                if custom_amount > 0:
                    category_amounts[custom_category] = custom_amount
            
            submitted = st.form_submit_button("Save Budget")
            if submitted:
                try:
                    # Save each category to the database
                    for category, amount in category_amounts.items():
                        if amount > 0:  # Only save categories with non-zero budgets
                            db_manager.save_budget(
                                st.session_state.current_user, 
                                category, 
                                amount
                            )
                    
                    st.success("Budget saved successfully!")
                    st.session_state.show_budget_form = False
                    st.rerun()  # Refresh the page to show updated budget
                except Exception as e:
                    st.error(f"Error saving budget: {str(e)}")

# Main chat interface
st.subheader("Chat with Siora")

# Display chat messages
for message in st.session_state.messages:
    avatar_img = "👤" if message["role"] == "user" else "🤖"
    
    with st.container():
        st.markdown(f"""
        <div class="chat-message {message['role']}">
            <div class="avatar">
                {avatar_img}
            </div>
            <div class="message">
                {message['content']}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Process comparison results if available
if st.session_state.comparison_results:
    with st.container():
        st.markdown("""
        <div class='comparison-card'>
            <div class='comparison-header'>Shopping Optimization Results</div>
        """, unsafe_allow_html=True)
        
        results = st.session_state.comparison_results
        
        # Display marketplace summaries
        if "marketplace_summary" in results:
            col1, col2 = st.columns(2)
            
            # Find the best marketplace (lowest total)
            best_marketplace = min(
                results["marketplace_summary"].items(),
                key=lambda x: x[1]["total"]
            )[0] if results["marketplace_summary"] else None
            
            with col1:
                st.subheader("By Marketplace")
                for marketplace, summary in results["marketplace_summary"].items():
                    css_class = "marketplace-card best-deal" if marketplace == best_marketplace else "marketplace-card"
                    st.markdown(f"""
                    <div class='{css_class}'>
                        <div class='marketplace-name'>{marketplace.capitalize()}</div>
                        <p>Items: {", ".join(summary["items"])}</p>
                        <p>Item Total: ₹{summary["item_total"]:.2f}</p>
                        <p>Delivery Fee: ₹{summary["delivery_fee"]:.2f}</p>
                        <p><strong>Total: ₹{summary["total"]:.2f}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("Item Details")
                
                # Create a dataframe for item details
                item_data = []
                for item_name, item_details in results["item_details"].items():
                    item_data.append({
                        "Item": item_name,
                        "Best Price (₹)": item_details["price"],
                        "Marketplace": item_details["marketplace"].capitalize(),
                        "Delivery Fee (₹)": item_details.get("delivery_fee", 0),
                        "Delivery Time": item_details.get("delivery_time", "N/A")
                    })
                
                if item_data:
                    item_df = pd.DataFrame(item_data)
                    st.dataframe(item_df, hide_index=True, use_container_width=True)
            
            # Summary of savings
            st.subheader("Savings Summary")
            st.write(f"**Optimized Total Cost:** ₹{results.get('optimized_total', 0):.2f}")
            
            if results.get("cheapest_single_marketplace"):
                marketplace, details = results["cheapest_single_marketplace"]
                st.write(f"**Cheapest Single Marketplace:** {marketplace.capitalize()} (₹{details['total']:.2f})")
                
                savings = results.get("potential_savings", 0)
                if savings > 0:
                    st.success(f"**Potential Savings:** ₹{savings:.2f} by shopping optimally across marketplaces!")
                else:
                    st.info("It's most economical to get everything from a single marketplace in this case.")
            
            # Payment authorization
            st.subheader("Proceed with Purchase")
            col3, col4 = st.columns(2)
            
            with col3:
                if st.button("Buy All From Best Marketplace", key="buy_best"):
                    st.session_state.payment_authorized = True
                    marketplace = best_marketplace
                    items = st.session_state.shopping_list
                    
                    # Process payment
                    try:
                        order = payment_processor.process_order(
                            st.session_state.current_user,
                            items,
                            results["marketplace_summary"][marketplace]["total"]
                        )
                        
                        # Save to database
                        db_manager.save_shopping(
                            st.session_state.current_user,
                            items,
                            results["marketplace_summary"][marketplace]["total"],
                            marketplace,
                            order["transaction_id"]
                        )
                        
                        # Add a message to chat
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"✅ Payment processed successfully! Your items from {marketplace.capitalize()} will be delivered soon. Transaction ID: {order['transaction_id']}"
                        })
                        
                        # Clear comparison results
                        st.session_state.comparison_results = None
                        st.session_state.shopping_list = []
                        st.rerun()
                    except Exception as e:
                        st.error(f"Payment error: {str(e)}")
            
            with col4:
                if st.button("Shop Optimally Across Marketplaces", key="shop_optimal"):
                    st.session_state.payment_authorized = True
                    
                    # For each marketplace, process a separate order
                    for marketplace, summary in results["marketplace_summary"].items():
                        try:
                            order = payment_processor.process_order(
                                st.session_state.current_user,
                                summary["items"],
                                summary["total"]
                            )
                            
                            # Save to database
                            db_manager.save_shopping(
                                st.session_state.current_user,
                                summary["items"],
                                summary["total"],
                                marketplace,
                                order["transaction_id"]
                            )
                        except Exception as e:
                            st.error(f"Error processing {marketplace} order: {str(e)}")
                    
                    # Add a message to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"✅ Orders placed optimally across {len(results['marketplace_summary'])} marketplaces! You saved approximately ₹{results.get('potential_savings', 0):.2f}."
                    })
                    
                    # Clear comparison results
                    st.session_state.comparison_results = None
                    st.session_state.shopping_list = []
                    st.rerun()
        
        # Close the card div
        st.markdown("</div>", unsafe_allow_html=True)

# Input for new message
user_input = st.chat_input("Message Siora...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Check for shopping list
    if "shopping list" in user_input.lower():
        # Extract items from input using simple parsing
        # In real app, we'd use NLP
        potential_items = [item.strip() for item in user_input.lower().split("list")[1].split(",")]
        shopping_list = [item for item in potential_items if len(item) > 0]
        
        if shopping_list:
            st.session_state.shopping_list = shopping_list
            
            # Start price comparison
            with st.spinner("Comparing prices across marketplaces..."):
                comparison_results = price_comparer.compare_prices(shopping_list)
                st.session_state.comparison_results = comparison_results
                
                # Process with agent
                prompt = f"I want to buy these items: {', '.join(shopping_list)}. Compare prices and suggest the best options."
                response = agent.run(prompt)
                
                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                st.rerun()
        else:
            # If no clear list is found, just process with agent
            response = agent.run(user_input)
            
            # Add assistant message to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    else:
        # Process input with agent
        response = agent.run(user_input)
        
        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

# Display welcome message if this is the first interaction
if not st.session_state.messages:
    with st.container():
        st.markdown("""
        <div class="chat-message assistant">
            <div class="avatar">
                🤖
            </div>
            <div class="message">
                <p>👋 Hello! I'm Siora, your Shopping Optimization Agent.</p>
                <p>I can help you:</p>
                <ul>
                    <li>Compare prices across different marketplaces</li>
                    <li>Optimize your shopping to save money</li>
                    <li>Track your monthly budget</li>
                    <li>Process payments securely</li>
                </ul>
                <p>Try saying:</p>
                <ul>
                    <li>"I want to buy milk, bread, eggs, and rice"</li>
                    <li>"Set my monthly grocery budget to ₹5000"</li>
                    <li>"Show me my shopping history"</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
