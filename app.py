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
if "order_placed" not in st.session_state:
    st.session_state.order_placed = False
if "order_details" not in st.session_state:
    st.session_state.order_details = {}

# Function to create budget allocation pie chart
def create_budget_allocation_chart():
    # Create data for the pie chart
    budget_data = {
        'Category': ['Groceries', 'Household'],
        'Budget': [st.session_state.get("grocery_budget", 5000), 
                  st.session_state.get("household_budget", 2000)]
    }
    
    df = pd.DataFrame(budget_data)
    
    # Create a pie chart
    fig = px.pie(
        df, 
        values='Budget', 
        names='Category', 
        title='Budget Allocation',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        marker=dict(line=dict(color='#FFFFFF', width=2))
    )
    
    return fig

# Function to update the budget remaining chart
def update_budget_remaining_chart():
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
    fig = px.bar(
        df, 
        x='Category', 
        y=['Spent', 'Remaining'], 
        title=f'Remaining Budget for {current_month}',
        labels={'value': 'Amount (₹)', 'variable': 'Status'},
        color_discrete_map={'Spent': '#FF6B6B', 'Remaining': '#4ECDC4'}
    )
    
    # Calculate percentage spent
    for i, row in df.iterrows():
        percentage = (row['Spent'] / row['Budget'] * 100) if row['Budget'] > 0 else 0
        remaining_percent = 100 - percentage
        
        # Add spent percentage annotation
        fig.add_annotation(
            x=row['Category'],
            y=row['Spent'] / 2,
            text=f"{percentage:.1f}% Spent",
            showarrow=False,
            font=dict(color="white" if percentage > 30 else "black", size=12)
        )
        
        # Add remaining percentage annotation
        if remaining_percent > 15:  # Only add if there's enough space
            fig.add_annotation(
                x=row['Category'],
                y=row['Spent'] + (row['Remaining'] / 2),
                text=f"{remaining_percent:.1f}% Left",
                showarrow=False,
                font=dict(color="black", size=12)
            )
    
    return fig, df

# App header
st.title("🛒 Siora - Shopping Optimization Agent")
st.write("Compare prices across marketplaces to save money!")

# Display order confirmation message if an order was just placed
if st.session_state.order_placed:
    order_details = st.session_state.order_details
    
    # Create a highlighted order confirmation message
    st.success(f"✅ Your order has been placed successfully!")
    
    # Order details in a nice container
    with st.container():
        st.markdown("### Order Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Marketplace:** {order_details.get('marketplace', '').capitalize()}")
            st.markdown(f"**Order Total:** ₹{order_details.get('total', 0):.2f}")
            st.markdown(f"**Order Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        with col2:
            st.markdown(f"**Items:** {', '.join(order_details.get('items', []))}")
            st.markdown(f"**Delivery Estimate:** {order_details.get('delivery_time', '30-60 minutes')}")
    
    # Budget update after purchase
    st.markdown("### Updated Budget")
    budget_fig, budget_df = update_budget_remaining_chart()
    st.plotly_chart(budget_fig, use_container_width=True)
    
    # Show warnings if over budget
    for idx, row in budget_df.iterrows():
        if row['Spent'] > row['Budget']:
            st.warning(f"⚠️ You have exceeded your {row['Category']} budget by ₹{row['Spent'] - row['Budget']:.2f}!")
        elif row['Spent'] > row['Budget'] * 0.8:
            st.info(f"ℹ️ You have used {row['Spent'] / row['Budget'] * 100:.1f}% of your {row['Category']} budget for this month.")
    
    # Button to continue shopping
    if st.button("Continue Shopping"):
        st.session_state.order_placed = False
        st.rerun()

# Only show regular UI if no order was just placed
if not st.session_state.order_placed:
    # Sidebar for user info
    with st.sidebar:
        st.header("User Information")
        user_name = st.text_input("Your Name", "Demo User")
        
        st.header("Monthly Budget")
        # Store these values in session state so they persist
        st.session_state.grocery_budget = st.number_input("Grocery Budget (₹)", value=st.session_state.get("grocery_budget", 5000))
        st.session_state.household_budget = st.number_input("Household Budget (₹)", value=st.session_state.get("household_budget", 2000))
        
        # Show budget allocation pie chart
        if st.session_state.grocery_budget > 0 or st.session_state.household_budget > 0:
            st.subheader("Budget Allocation")
            budget_allocation_fig = create_budget_allocation_chart()
            st.plotly_chart(budget_allocation_fig, use_container_width=True)
            
            # Show budget tracking chart
            st.subheader("Budget Status")
            budget_remaining_fig, _ = update_budget_remaining_chart()
            st.plotly_chart(budget_remaining_fig, use_container_width=True)
            
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
                with st.expander(
                    f"{marketplace.capitalize()}{' (BEST DEAL)' if marketplace == best_marketplace else ''}",
                    expanded=(marketplace == best_marketplace)
                ):
                    st.write(f"**Items Total:** ₹{totals['item_total']:.2f}")
                    st.write(f"**Delivery Fee:** ₹{totals['delivery_fee']:.2f}")
                    st.write(f"**Delivery Time:** {totals['delivery_time']}")
                    st.write(f"**Grand Total:** ₹{totals['grand_total']:.2f}")
                    
                    # Buy button for each marketplace
                    if st.button(f"Buy from {marketplace.capitalize()}", key=f"buy_{marketplace}", type="primary" if marketplace == best_marketplace else "secondary"):
                        # Process the purchase
                        
                        # Update monthly spending
                        # Determine if this is primarily groceries or household items
                        grocery_items = ["milk", "bread", "eggs", "rice", "flour", "vegetables", "fruits"]
                        household_items = ["soap", "detergent", "tissues", "cleaner"]
                        
                        # Calculate how much of the total is groceries vs household items
                        total_spent = totals["grand_total"]
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
                        
                        # Store order details to display on confirmation page
                        st.session_state.order_details = {
                            "marketplace": marketplace,
                            "items": st.session_state.shopping_list,
                            "total": total_spent,
                            "delivery_time": totals["delivery_time"]
                        }
                        
                        # Set order placed flag
                        st.session_state.order_placed = True
                        
                        # Show success notification
                        st.balloons()
                        
                        # Reset shopping list and comparison results
                        st.session_state.comparison_results = None
                        st.session_state.shopping_list = []
                        
                        # Rerun to show confirmation screen
                        st.rerun()
            
            # Buy button for best marketplace (outside the expanders)
            st.subheader("Ready to Purchase?")
            if st.button(f"Buy from {best_marketplace.capitalize()} (BEST DEAL)", type="primary"):
                # Process the purchase
                
                # Update monthly spending
                # Determine if this is primarily groceries or household items
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
                
                # Store order details to display on confirmation page
                st.session_state.order_details = {
                    "marketplace": best_marketplace,
                    "items": st.session_state.shopping_list,
                    "total": total_spent,
                    "delivery_time": marketplace_totals[best_marketplace]["delivery_time"]
                }
                
                # Set order placed flag
                st.session_state.order_placed = True
                
                # Show success notification
                st.balloons()
                
                # Reset shopping list and comparison results
                st.session_state.comparison_results = None
                st.session_state.shopping_list = []
                
                # Rerun to show confirmation screen
                st.rerun()

    # Display welcome message for first-time users
    if not st.session_state.shopping_list and not st.session_state.comparison_results:
        st.info("👋 Welcome to Siora! Enter a shopping list above to compare prices across marketplaces.")
        
        # Show demo items
        st.write("Try these example items:")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Milk, Bread, Eggs"):
                st.session_state.shopping_list = ["Milk", "Bread", "Eggs"]
                st.rerun()
        with col2:
            if st.button("Rice, Flour, Soap"):
                st.session_state.shopping_list = ["Rice", "Flour", "Soap"]
                st.rerun()
        with col3:
            if st.button("Vegetables, Fruits"):
                st.session_state.shopping_list = ["Vegetables", "Fruits"]
                st.rerun()
