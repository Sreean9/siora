from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from typing import List, Dict, Any, Optional
import json

from .base_agent import BaseAgent
from utils.price_scraper import PriceComparer
from utils.payment_handler import PaymentProcessor
from database.db_manager import DatabaseManager


class GetBudgetInput(BaseModel):
    user_id: str = Field(..., description="User ID to retrieve budget for")
    month: Optional[str] = Field(None, description="Month to retrieve budget for, defaults to current month")
    year: Optional[int] = Field(None, description="Year to retrieve budget for, defaults to current year")


class ComparePricesInput(BaseModel):
    items: List[str] = Field(..., description="List of items to compare prices for")


class ProcessPaymentInput(BaseModel):
    user_id: str = Field(..., description="User ID making the purchase")
    items: List[str] = Field(..., description="List of items being purchased")
    marketplace: str = Field(..., description="Marketplace to purchase from")
    payment_method: Optional[str] = Field(None, description="Payment method ID")


class ShoppingAgent(BaseAgent):
    """Agent specialized for shopping optimization."""
    
    def __init__(self, api_key, db_manager, price_comparer, payment_processor, **kwargs):
        """
        Initialize the shopping agent.
        
        Args:
            api_key (str): OpenAI API key
            db_manager (DatabaseManager): Database manager instance
            price_comparer (PriceComparer): Price comparison utility
            payment_processor (PaymentProcessor): Payment processing utility
        """
        self.db_manager = db_manager
        self.price_comparer = price_comparer
        self.payment_processor = payment_processor
        
        # Define agent-specific tools
        tools = self._create_tools()
        
        # Initialize base agent with tools
        super().__init__(api_key=api_key, tools=tools, **kwargs)
        
        # Customize system prompt for shopping agent
        self._customize_prompt()
    
    def _create_tools(self):
        """Create specialized tools for the shopping agent."""
        
        # Tool for getting user budget
        class GetBudgetTool(BaseTool):
            name = "get_budget"
            description = "Get the user's budget for various categories"
            args_schema = GetBudgetInput
            
            def _run(self, user_id, month=None, year=None):
                budget_items = self.db_manager.get_budget(user_id, month, year)
                return json.dumps([{"category": item[0], "amount": item[1]} for item in budget_items])
            
            def __init__(self, db_manager):
                self.db_manager = db_manager
                super().__init__()
        
        # Tool for comparing prices
        class ComparePricesTool(BaseTool):
            name = "compare_prices"
            description = "Compare prices of items across different marketplaces"
            args_schema = ComparePricesInput
            
            def _run(self, items):
                results = self.price_comparer.compare_prices(items)
                return json.dumps(results)
            
            def __init__(self, price_comparer):
                self.price_comparer = price_comparer
                super().__init__()
        
        # Tool for processing payments
        class ProcessPaymentTool(BaseTool):
            name = "process_payment"
            description = "Process payment for items after user authorization"
            args_schema = ProcessPaymentInput
            
            def _run(self, user_id, items, marketplace, payment_method=None):
                # Calculate total from items
                # In a real app, we'd get precise prices from the database or API
                # For hackathon simplicity, we'll use the price comparer
                item_results = self.price_comparer.compare_prices(items)
                marketplace_items = [
                    item for item_name, item in item_results["item_details"].items()
                    if item["marketplace"] == marketplace
                ]
                
                if not marketplace_items:
                    return json.dumps({
                        "error": f"No items found for marketplace: {marketplace}"
                    })
                
                # Calculate total
                item_total = sum(item["price"] for item in marketplace_items)
                delivery_fee = max(item.get("delivery_fee", 0) for item in marketplace_items)
                total_amount = item_total + delivery_fee
                
                # Process payment
                order_result = self.payment_processor.process_order(
                    user_id, items, total_amount, payment_method
                )
                
                # Save to shopping history
                self.db_manager.save_shopping(
                    user_id, items, total_amount, marketplace, order_result["transaction_id"]
                )
                
                return json.dumps(order_result)
            
            def __init__(self, db_manager, price_comparer, payment_processor):
                self.db_manager = db_manager
                self.price_comparer = price_comparer
                self.payment_processor = payment_processor
                super().__init__()
        
        # Create tool instances
        get_budget_tool = GetBudgetTool(self.db_manager)
        compare_prices_tool = ComparePricesTool(self.price_comparer)
        process_payment_tool = ProcessPaymentTool(
            self.db_manager, self.price_comparer, self.payment_processor
        )
        
        return [get_budget_tool, compare_prices_tool, process_payment_tool]
    
    def _customize_prompt(self):
        """Customize the agent prompt for shopping optimization."""
        system_message = """You are Siora, an intelligent shopping assistant designed to help users optimize their shopping experiences.

Core capabilities:
1. Budget tracking - Track monthly budgets for different shopping categories
2. Price comparison - Compare prices of items across marketplaces like Zepto, Swiggy, and others
3. Shopping optimization - Find the most cost-effective way to purchase items
4. Transaction processing - Handle payments securely after explicit user authorization

Interaction guidelines:
- When a user provides a shopping list, always compare prices and provide a summary of the best options
- When recommending purchases, clearly show price breakdowns including delivery fees
- IMPORTANT: ALWAYS request explicit user confirmation before proceeding with any payment
- Provide helpful suggestions for budget management
- Be conversational and friendly while remaining professional

Your primary goal is to help users save money while making their shopping experience seamless.
"""
        
        # Update the agent's system prompt
        self.agent_executor.agent.prompt.messages[0].content = system_message
