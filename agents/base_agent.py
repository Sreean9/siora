import logging
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool
from typing import List, Optional, Type

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseAgent:
    """Base agent class for Siora platform."""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.2,
        tools: List[BaseTool] = None,
        verbose: bool = False
    ):
        """
        Initialize the base agent.
        
        Args:
            api_key (str): OpenAI API key
            model_name (str): Name of the OpenAI model to use
            temperature (float): Temperature parameter for sampling
            tools (List[BaseTool]): List of LangChain tools
            verbose (bool): Whether to enable verbose output
        """
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.tools = tools or []
        self.verbose = verbose
        
        # Initialize the language model
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            api_key=api_key,
            streaming=True
        )
        
        # Set up the agent with default system prompt
        self._setup_agent()
    
    def _setup_agent(self):
        """Set up the agent with tools and prompt."""
        # Default system prompt
        system_message = """You are Siora, an AI shopping assistant designed to help users optimize their shopping experiences. 
Your main capabilities are:
1. Budget tracking and management
2. Price comparison across different marketplaces
3. Finding the best deals and discounts
4. Handling payment processing after user authorization
5. Providing personalized shopping recommendations

Always be helpful, accurate, and friendly in your interactions.
"""
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create React agent
        agent = create_react_agent(self.llm, self.tools, prompt)
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=self.verbose,
            handle_parsing_errors=True
        )
    
    def run(self, query: str, chat_history=None):
        """
        Run the agent on a query.
        
        Args:
            query (str): User query
            chat_history (list): Optional chat history
            
        Returns:
            str: Agent response
        """
        try:
            # Prepare inputs
            inputs = {
                "input": query,
                "chat_history": chat_history or []
            }
            
            # Run agent
            response = self.agent_executor.invoke(inputs)
            return response["output"]
        except Exception as e:
            logger.error(f"Error running agent: {str(e)}")
            return f"I encountered an error while processing your request: {str(e)}"
