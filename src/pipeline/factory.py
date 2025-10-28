"""
Operation factory class, responsible for creating various operation instances
"""
from typing import Dict, Optional, Type, Union, List
import pandas as pd
from src.pipeline.base import Operation

class OperationFactory:
    """Operation factory class"""
    
    _registry = {}
    
    @classmethod
    def register(cls, op_name: str, operation_class: Type[Operation]):
        """
        Register an operation class
        
        Args:
            op_name: Operation name
            operation_class: Operation class
        """
        cls._registry[op_name] = operation_class
    
    @classmethod
    def create(cls, op_name: str, params: Dict = None, table: Union[pd.DataFrame, List[pd.DataFrame]] = None, table_info: Union[Dict, List[Dict]] = None, validate: bool = True) -> Optional[Operation]:
        """
        Create an operation instance
        
        Args:
            op_name: Operation name
            params: Operation parameters
            validate: Whether to validate parameters
            
        Returns:
            Operation: Operation instance, returns None if the operation does not exist
        """
        if op_name not in cls._registry:
            return None
        
        operation_class = cls._registry[op_name]
        processed_table = []
        if table is not None:
            if isinstance(table, pd.DataFrame):
                if not table.empty:
                    processed_table = table
            elif isinstance(table, list):
                if table:  # Non-empty list
                    processed_table = table
            else:
                raise TypeError("table must be a pandas DataFrame or a list of DataFrames")
        return operation_class(
            params or {},
            table=processed_table,
            table_info=table_info if table_info else [],
            validate=validate
        )
    
    @classmethod
    def get_available_operations(cls) -> Dict[str, Type[Operation]]:
        """
        Get all available operations
        
        Returns:
            Dict: Mapping of operation names to operation classes
        """
        return cls._registry.copy()