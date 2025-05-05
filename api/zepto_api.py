import requests
import json
import time
import random
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ZeptoAPI:
    """Simulated Zepto API client for the hackathon."""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """
        Initialize the Zepto API client.
        
        Args:
            api_key (str): API key for authentication
            base_url (str): Base URL for API endpoints
        """
        self.api_key = api_key
        self.base_url = base_url or "https://www.zepto.com/api/v1"
        
        # Headers for API requests
        self.headers = {
            "Authorization": f"Bearer {api_key}" if api_key else "",
            "Content-Type": "application/json",
            "User-Agent": "Siora-Shopping-Assistant/1.0"
        }
    
    def search_products(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for products.
        
        Args:
            query (str): Search query
            limit (int): Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of products
        """
        try:
            # This would normally be an API call
            # For the hackathon, let's simulate a response
            return self._simulate_search_results(query, limit)
        except Exception as e:
            logger.error(f"Error searching products: {str(e)}")
            return []
    
    def _simulate_search_results(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Simulate search results for demo purposes."""
        # Add delay to simulate network latency
        time.sleep(random.uniform(0.2, 0.6))
        
        # Common grocery items and their price ranges
        grocery_items = {
            "milk": {"min": 45, "max": 65, "unit": "liter"},
            "bread": {"min": 30, "max": 45, "unit": "loaf"},
            "eggs": {"min": 70, "max": 90, "unit": "dozen"},
            "rice": {"min": 55, "max": 80, "unit": "kg"},
            "flour": {"min": 40, "max": 60, "unit": "kg"},
            "oil": {"min": 100, "max": 150, "unit": "liter"},
            "sugar": {"min": 40, "max": 55, "unit": "kg"},
            "salt": {"min": 20, "max": 30, "unit": "kg"},
            "butter": {"min": 50, "max": 70, "unit": "500g"},
            "cheese": {"min": 80, "max": 120, "unit": "200g"}
        }
        
        results = []
        query_lower = query.lower()
        
        # Check if query matches any known items
        matched_items = [item for item in grocery_items.keys() if item in query_lower]
        
        if matched_items:
            # Generate results for matched items
            for item in matched_items[:limit]:
                item_info = grocery_items[item]
                results.append({
                    "id": f"zepto_{item}_" + str(random.randint(1000, 9999)),
                    "name":
