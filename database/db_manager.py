import sqlite3
import json
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path="siora_database.db"):
        """Initialize database connection."""
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_tables()
        
    def _create_tables(self):
        """Create necessary tables if they don't exist."""
        # Table for user budgets
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS budgets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            category TEXT NOT NULL,
            amount REAL NOT NULL,
            month TEXT NOT NULL,
            year INTEGER NOT NULL
        )
        ''')
        
        # Table for shopping history
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS shopping_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            date TEXT NOT NULL,
            items TEXT NOT NULL,
            total_amount REAL NOT NULL,
            marketplace TEXT NOT NULL,
            transaction_id TEXT
        )
        ''')
        
        self.conn.commit()
    
    def save_budget(self, user_id, category, amount, month=None, year=None):
        """Save user budget for a specific category."""
        if month is None:
            month = datetime.now().strftime("%B")
        if year is None:
            year = datetime.now().year
            
        self.cursor.execute(
            "INSERT INTO budgets (user_id, category, amount, month, year) VALUES (?, ?, ?, ?, ?)",
            (user_id, category, amount, month, year)
        )
        self.conn.commit()
        
    def get_budget(self, user_id, month=None, year=None):
        """Get user budget for all categories in a month."""
        if month is None:
            month = datetime.now().strftime("%B")
        if year is None:
            year = datetime.now().year
            
        self.cursor.execute(
            "SELECT category, amount FROM budgets WHERE user_id = ? AND month = ? AND year = ?",
            (user_id, month, year)
        )
        return self.cursor.fetchall()
    
    def save_shopping(self, user_id, items, total_amount, marketplace, transaction_id=None):
        """Save shopping transaction to history."""
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        items_json = json.dumps(items)
        
        self.cursor.execute(
            "INSERT INTO shopping_history (user_id, date, items, total_amount, marketplace, transaction_id) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, date, items_json, total_amount, marketplace, transaction_id)
        )
        self.conn.commit()
        
    def get_shopping_history(self, user_id, limit=10):
        """Get recent shopping history for user."""
        self.cursor.execute(
            "SELECT date, items, total_amount, marketplace, transaction_id FROM shopping_history WHERE user_id = ? ORDER BY date DESC LIMIT ?",
            (user_id, limit)
        )
        
        result = []
        for row in self.cursor.fetchall():
            date, items_json, total_amount, marketplace, transaction_id = row
            items = json.loads(items_json)
            result.append({
                "date": date,
                "items": items,
                "total_amount": total_amount,
                "marketplace": marketplace,
                "transaction_id": transaction_id
            })
        
        return result
    
    def close(self):
        """Close the database connection."""
        self.conn.close()
