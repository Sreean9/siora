import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy-key-for-demo")
STRIPE_API_KEY = os.getenv("STRIPE_API_KEY", "dummy-key-for-demo")

# Marketplace Configurations
MARKETPLACES = {
    "zepto": {
        "base_url": "https://www.zepto.com/api/v1",
        "api_key": os.getenv("ZEPTO_API_KEY", "dummy-key"),
    },
    "swiggy": {
        "base_url": "https://www.swiggy.com/api/v1",
        "api_key": os.getenv("SWIGGY_API_KEY", "dummy-key"),
    },
    "blinkit": {
        "base_url": "https://www.blinkit.com/api/v1",
        "api_key": os.getenv("BLINKIT_API_KEY", "dummy-key"),
    },
    "bigbasket": {
        "base_url": "https://www.bigbasket.com/api/v1",
        "api_key": os.getenv("BIGBASKET_API_KEY", "dummy-key"),
    }
}

# Database Configuration
DB_CONFIG = {
    "type": "sqlite",  # for simplicity in the hackathon
    "path": "siora_database.db"
}

# Agent Configuration
AGENT_CONFIG = {
    "model": "gpt-4o",
    "temperature": 0.2,
    "max_tokens": 2000
}
