import streamlit as st
import pandas as pd
import random
import time
import plotly.express as px

# App configuration
st.set_page_config(page_title="Siora - Shopping Assistant", page_icon="🛒", layout="wide")

# Initialize session state
if "shopping_list" not in st.session_state:
    st.session_state.shopping_list = []
if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = None

# App header
st.title("🛒 Siora - Your Shopping Buddy")
st.write("Compare prices across marketplaces to save money!")

# Sidebar for user info
with st.sidebar:
    st.header("User Information")
    user_name = st.text_input("Your Name", "Demo User")
    
    st.header("Monthly Budget")
    grocery_budget = st.number_input("Grocery Budget (₹)", value=5000)
    household_budget = st.number_input("Household Budget (₹)", value=2000)
    
    # Budget visualization
    if grocery_budget or household_budget:
        data = {'Category': ['Groceries', 'Household'], 
                'Amount': [grocery_budget, household_budget]}
        df = pd.DataFrame(data)
        fig = px.pie(df, values='Amount', names='Category', title='Budget Allocation')
        st.plotly_chart(fig)

# Shopping list input
st.header("Your Shopping List")
new_item = st.text_input("Add items (separated by commas)")

if st.button("Compare Prices", type="primary"):
    if new_item:
        # Parse items from input
        items = [item.strip() for item in new_item.split(",") if item.strip()]
        st.session_state.shopping_list = items
        
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
    st.header("Price Comparison Results")
    
    results = st.session_state.comparison_results
    marketplaces = ["zepto", "swiggy", "blinkit", "bigbasket"]
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["By Item", "By Marketplace"])
    
    with tab1:
        # Show each item comparison
        for item, marketplaces_data in results.items():
            st.subheader(f"{item}")
            
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
            st.success(f"Best price: {df.iloc[best_price_idx]['Marketplace']} - ₹{df.iloc[best_price_idx]['Price (₹)']}") 
    
    with tab2:
        # Calculate totals for each marketplace
        marketplace_totals = {}
        for marketplace in marketplaces:
            # Calculate item totals for this marketplace
            item_total = sum(results[item][marketplace]["price"] for item in results)
            # Highest delivery fee (assuming one delivery)
            delivery = max(results[item][marketplace]["delivery_fee"] for item in results)
            grand_total = item_total + delivery
            
            marketplace_totals[marketplace] = {
                "item_total": item_total,
                "delivery_fee": delivery,
                "grand_total": grand_total
            }
        
        # Find best marketplace
        best_marketplace = min(marketplace_totals.items(), key=lambda x: x[1]["grand_total"])[0]
        
        # Show marketplace comparison
        for marketplace, totals in marketplace_totals.items():
            with st.expander(
                f"{marketplace.capitalize()}{' (BEST DEAL)' if marketplace == best_marketplace else ''}",
                expanded=(marketplace == best_marketplace)
            ):
                st.write(f"**Items Total:** ₹{totals['item_total']:.2f}")
                st.write(f"**Delivery Fee:** ₹{totals['delivery_fee']:.2f}")
                st.write(f"**Grand Total:** ₹{totals['grand_total']:.2f}")
        
        # Buy button
        st.subheader("Ready to Purchase?")
        if st.button(f"Buy from {best_marketplace.capitalize()}", type="primary"):
            st.balloons()
            st.success(f"Order placed with {best_marketplace.capitalize()}! Your items will be delivered soon.")
