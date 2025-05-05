import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import logging
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PriceComparer:
    def __init__(self, marketplaces=None):
        """
        Initialize the price comparer.
        
        Args:
            marketplaces (dict): Dictionary containing marketplace configurations
        """
        self.marketplaces = marketplaces or {}
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def search_item(self, item_name, marketplace_name):
        """
        Search for item in a specific marketplace.
        This would usually connect to each marketplace's API, but for the hackathon
        we're using a simulated version.
        
        Args:
            item_name (str): Name of the item to search
            marketplace_name (str): Name of the marketplace
        
        Returns:
            list: List of items found
        """
        try:
            # This would be replaced with actual API calls in production
            # Simulated data for hackathon purposes
            return self._simulate_search_results(item_name, marketplace_name)
        except Exception as e:
            logger.error(f"Error searching for {item_name} in {marketplace_name}: {str(e)}")
            return []
    
    def _simulate_search_results(self, item_name, marketplace):
        """Simulate search results for hackathon demo."""
        # Add random delay to simulate network latency
        time.sleep(random.uniform(0.1, 0.5))
        
        # Generate simulated results
        base_price = 0
        if "milk" in item_name.lower():
            base_price = 50.0
        elif "bread" in item_name.lower():
            base_price = 35.0
        elif "rice" in item_name.lower():
            base_price = 60.0 
        elif "soap" in item_name.lower():
            base_price = 45.0
        else:
            base_price = random.uniform(20.0, 100.0)
            
        # Different marketplaces have different price ranges
        price_modifiers = {
            "zepto": random.uniform(0.8, 1.1),
            "swiggy": random.uniform(0.9, 1.2),
            "blinkit": random.uniform(0.85, 1.05),
            "bigbasket": random.uniform(0.95, 1.15)
        }
        
        modifier = price_modifiers.get(marketplace, 1.0)
        price = round(base_price * modifier, 2)
        
        return [{
            "name": item_name,
            "price": price,
            "marketplace": marketplace,
            "available": True,
            "delivery_fee": round(random.uniform(10, 30), 2),
            "delivery_time": f"{random.randint(15, 60)} mins"
        }]
    
    def compare_prices(self, shopping_list):
        """
        Compare prices across all marketplaces for each item in the shopping list.
        
        Args:
            shopping_list (list): List of item names
        
        Returns:
            dict: Dictionary containing best options for each item and overall summary
        """
        results = {}
        all_options = {}
        
        # Search each item across all marketplaces in parallel
        for item in shopping_list:
            item_options = []
            # Use ThreadPoolExecutor for parallel API calls
            with ThreadPoolExecutor(max_workers=len(self.marketplaces)) as executor:
                future_to_marketplace = {
                    executor.submit(self.search_item, item, marketplace): marketplace
                    for marketplace in self.marketplaces
                }
                
                for future in future_to_marketplace:
                    marketplace = future_to_marketplace[future]
                    try:
                        search_results = future.result()
                        item_options.extend(search_results)
                    except Exception as e:
                        logger.error(f"Error processing {marketplace} results: {str(e)}")
            
            # Find the cheapest option for this item
            if item_options:
                best_option = min(item_options, key=lambda x: x["price"] + x.get("delivery_fee", 0))
                results[item] = best_option
                all_options[item] = item_options
        
        # Generate summary
        total_by_marketplace = {}
        for marketplace in self.marketplaces:
            marketplace_items = [results[item] for item in results if results[item]["marketplace"] == marketplace]
            if marketplace_items:
                item_total = sum(item["price"] for item in marketplace_items)
                delivery_fee = max(item.get("delivery_fee", 0) for item in marketplace_items)
                total_by_marketplace[marketplace] = {
                    "items": [item["name"] for item in marketplace_items],
                    "item_total": item_total,
                    "delivery_fee": delivery_fee,
                    "total": item_total + delivery_fee
                }
        
        # Calculate potential savings
        all_in_one_marketplace = {}
        for marketplace in self.marketplaces:
            marketplace_total = 0
            delivery_fee = 0
            
            for item in shopping_list:
                item_options = all_options.get(item, [])
                marketplace_items = [option for option in item_options if option["marketplace"] == marketplace]
                
                if marketplace_items:
                    best_option = min(marketplace_items, key=lambda x: x["price"])
                    marketplace_total += best_option["price"]
                    delivery_fee = max(delivery_fee, best_option.get("delivery_fee", 0))
            
            if marketplace_total > 0:
                all_in_one_marketplace[marketplace] = {
                    "item_total": marketplace_total,
                    "delivery_fee": delivery_fee,
                    "total": marketplace_total + delivery_fee
                }
        
        # Find the cheapest marketplace for buying everything
        cheapest_marketplace = min(all_in_one_marketplace.items(), key=lambda x: x[1]["total"]) if all_in_one_marketplace else None
        
        # Compare with optimized approach (buying from different marketplaces)
        optimized_total = sum(results[item]["price"] for item in results) + sum(total_by_marketplace[marketplace]["delivery_fee"] for marketplace in total_by_marketplace)
        
        summary = {
            "item_details": results,
            "marketplace_summary": total_by_marketplace,
            "optimized_total": optimized_total,
            "cheapest_single_marketplace": cheapest_marketplace,
            "potential_savings": cheapest_marketplace[1]["total"] - optimized_total if cheapest_marketplace else 0
        }
        
        return summary
