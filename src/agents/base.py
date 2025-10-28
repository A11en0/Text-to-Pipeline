"""
Base Agent Definition
"""

class Agent:
    """
    Base class for agents, defining basic interfaces and common methods
    """
    
    def __init__(self, config):
        """
        Initialize the agent
        
        Args:
            config: Configuration for the agent
        """
        self.config = config
        self.debug_mode = config.get("debug_mode", False)
    
    def run(self, *args, **kwargs):
        """
        Execute the main functionality of the agent. 
        Subclasses must implement this method.
        
        Returns:
            Returns the result of execution depending on the agent
        """
        raise NotImplementedError("The 'run' method must be implemented in the subclass")
    
    def _debug_print(self, message):
        """
        Print debug information when in debug mode
        
        Args:
            message: Message to be printed
        """
        if self.debug_mode:
            print(f"[{self.__class__.__name__}] {message}")
