import streamlit as st
import pandas as pd
import random
import time
import plotly.express as px
import datetime

# App configuration
st.set_page_config(page_title="Siora - Shopping Assistant", page_icon="🛒", layout="wide")

# Initialize session state
if "shopping_list" not in st.session_state:
    st.session_state.shopping_list = []
if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = None
if "monthly_spending" not in st.session_state:
    st.session_state.monthly_spending = {"Groceries": 0, "Household": 0}

# Function to update the budget chart
def update_budget_chart():
    # Get current month for the title
    current_month = datetime.datetime.now().strftime("%B")
    
    # Create data for the chart
    budget_data = {
        'Category': ['Groceries', 'Household'],
        'Budget': [st.session_state.get("grocery_budget", 5000), 
                  st.session_state.get("household_budget", 2000)],
        'Spent': [st.session_state.monthly_spending["Groceries"], 
                 st.session_state.monthly_spending["Household"]]
    }
    
    df = pd.DataFrame(budget_data)
    
    # Calculate remaining budget
    df['Remaining'] = df['Budget'] - df['Spent']
    
    # Create a stacked bar chart
    fig = px.bar(df, x='Category', y=['Spent', 'Remaining'], 
                title=f'Budget Tracking for {current_month}',
                labels={'value': 'Amount (₹)', 'variable': 'Status'},
                color_discrete_map={'Spent': '#FF6B6B', 'Remaining': '#4ECDC4'})
    
    # Calculate percentage spent
    for i, row in df.iterrows():
        percentage = (row['Spent'] / row['Budget'] * 100) if row['Budget'] > 0 else 0
        fig.add_annotation(
            x=row['Category'],
            y=row['Spent'] / 2,
            text=f"{percentage:.1f}%",
            showarrow=False,
            font=dict(color="white" if percentage > 30 else "black")
        )
    
    return fig, df

# App header
st.title("🛒 Siora - Shopping Optimization Agent")
st.write("Compare prices across marketplaces to save money!")

# Sidebar for user info
with st.sidebar:
    st.header("User Information")
    user_name = st.text_input("Your Name", "Demo User")
    
    st.header("Monthly Budget")
    # Store these values in session state so they persist
    st.session_state.grocery_budget = st.number_input("Grocery Budget (₹)", value=st.session_state.get("grocery_budget", 5000))
    st.session_state.household_budget = st.number_input("Household Budget (₹)", value=st.session_state.get("household_budget", 2000))
    
    # Show budget tracking chart if we have budgets set
    if st.session_state.grocery_budget > 0 or st.session_state.household_budget > 0:
        st.subheader("Budget Tracking")
        budget_fig, _ = update_budget_chart()
        st.plotly_chart(budget_fig, use_container_width=True)
        
        # Add a reset budget button
        if st.button("Reset Monthly Spending"):
            st.session_state.monthly_spending = {"Groceries": 0, "Household": 0}
            st.success("Monthly spending has been reset!")
            st.rerun()

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
            # Process the purchase
            st.balloons()
            st.success(f"Order placed with {best_marketplace.capitalize()}! Your items will be delivered soon.")
            
            # Update monthly spending
            # Determine if this is primarily groceries or household items
            # (a simple approach - you can make this more sophisticated)
            grocery_items = ["milk", "bread", "eggs", "rice", "flour", "vegetables", "fruits"]
            household_items = ["soap", "detergent", "tissues", "cleaner"]
            
            # Calculate how much of the total is groceries vs household items
            total_spent = marketplace_totals[best_marketplace]["grand_total"]
            grocery_count = sum(1 for item in st.session_state.shopping_list if any(g in item.lower() for g in grocery_items))
            household_count = sum(1 for item in st.session_state.shopping_list if any(h in item.lower() for h in household_items))
            other_count = len(st.session_state.shopping_list) - grocery_count - household_count
            
            # Distribute cost proportionally
            if grocery_count + household_count > 0:
                grocery_ratio = grocery_count / (grocery_count + household_count + other_count) if (grocery_count + household_count + other_count) > 0 else 0
                household_ratio = household_count / (grocery_count + household_count + other_count) if (grocery_count + household_count + other_count) > 0 else 0
                
                # Default distribution if we can't categorize
                if grocery_ratio + household_ratio == 0:
                    grocery_ratio = 0.7  # Assume 70% groceries by default
                    household_ratio = 0.3  # and 30% household items
                
                # Update spending in session state
                st.session_state.monthly_spending["Groceries"] += total_spent * grocery_ratio
                st.session_state.monthly_spending["Household"] += total_spent * household_ratio
            else:
                # If no categorization, put all in groceries by default
                st.session_state.monthly_spending["Groceries"] += total_spent
            
            # Show budget tracking after purchase
            st.subheader("Updated Budget Status")
            budget_fig, budget_df = update_budget_chart()
            st.plotly_chart(budget_fig, use_container_width=True)
            
            # Show warnings if over budget
            for idx, row in budget_df.iterrows():
                if row['Spent'] > row['Budget']:
                    st.warning(f"⚠️ You have exceeded your {row['Category']} budget by ₹{row['Spent'] - row['Budget']:.2f}!")
                elif row['Spent'] > row['Budget'] * 0.8:
                    st.info(f"ℹ️ You have used {row['Spent'] / row['Budget'] * 100:.1f}% of your {row['Category']} budget.")
            
            # Reset for next comparison
            st.session_state.comparison_results = None
            st.session_state.shopping_list = []

# Display welcome message for first-time users
if not st.session_state.shopping_list and not st.session_state.comparison_results:
    st.info("👋 Welcome to Siora! Enter a shopping list above to compare prices across marketplaces.")
    
    # Show demo items
    st.write("Try these example items:")
    if st.button("Milk, Bread, Eggs"):
        st.session_state.shopping_list = ["Milk", "Bread", "Eggs"]
        st.rerun()
    if st.button("Rice, Flour, Soap"):
        st.session_state.shopping_list = ["Rice", "Flour", "Soap"]
        st.rerun()
