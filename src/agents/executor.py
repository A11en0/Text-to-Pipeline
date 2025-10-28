"""
Executor agent that executes code and validates results.
"""
import pandas as pd
import numpy as np
import time
from src.agents.base import Agent
from src.pipeline import Pipeline, OperationFactory
from typing import List, Dict, Tuple, Optional, Union
from src.utils.logger import get_logger
from pandas.api.types import is_numeric_dtype


class ExecutorAgent(Agent):
    """
    Executor agent that executes code and validates results.
    """
    
    def __init__(self, config):
        """
        Initialize the executor agent.
        
        Args:
            config: Agent configuration.
        """
        super().__init__(config)
        self.timeout = config.get("timeout", 10)  # Code execution timeout (seconds)
        self.logger = get_logger("executor")
    
    def run(self, input_table: Union[pd.DataFrame, List[pd.DataFrame]], transform_ops: List[Dict] = None, code: str = None) -> Tuple[pd.DataFrame, bool, str]:
        """
        Execute transformation operations.
        
        Args:
            input_table: Input table.
            transform_ops: Transformation operation chain.
            code: pandas code.
            
        Returns:
            Tuple[pd.DataFrame, bool, str]: (Transformed table, success flag, error message)
        """
        # Check parameters
        if transform_ops is None and code is None:
            return input_table, False, "No transformation operations or code provided."
        
        result = None
        error = None
        
        try:
            # Prioritize using the transformation operation chain, execute using the pipeline
            if transform_ops:
                pipeline, result = Pipeline.from_transform_chain(transform_ops, input_table)  # Use the newly added method
                if isinstance(result, List):
                    result = result[0]
                # result = pipeline.transform(input_table.copy())
                return result, True, None
        
        except Exception as e:
            self.logger.error(f"Failed to execute transformation chain: {str(e)}")
            error = str(e)
        
        # Return the original table on failure
        return input_table, False, error or "Execution failed, no detailed information."
    
    @staticmethod
    def validate(T_generated: pd.DataFrame, T_target: pd.DataFrame, tolerance: float = 1e-6):
        """
        Validate whether the generated table matches the target table, allowing a certain tolerance for numeric columns.

        Args:
            T_generated (pd.DataFrame): Generated table.
            T_target (pd.DataFrame): Target table.
            tolerance (float): Absolute error tolerance for numeric column comparison, default is 1e-6.

        Returns:
            tuple: (Validation result: bool, Difference information: str or None)
        """
        # 1. Type check
        if not isinstance(T_generated, pd.DataFrame) or not isinstance(T_target, pd.DataFrame):
            return False, "Generated table or target table is not a DataFrame."
        T_generated.columns = T_generated.columns.astype(str)  # Convert column names to strings
        T_target.columns = T_target.columns.astype(str)  # Convert column names to strings
        # 2. Column name check (ignore order)
        cols_gen = set(T_generated.columns.astype(str))
        cols_tgt = set(T_target.columns.astype(str))
        if cols_gen != cols_tgt:
            missing = cols_tgt - cols_gen
            extra = cols_gen - cols_tgt
            msg = []
            if missing:
                msg.append(f"Missing columns: {missing}")
            if extra:
                msg.append(f"Extra columns: {extra}")
            return False, "；".join(msg)

        # 3. Align columns, sort by all columns, and reset index
        cols = sorted(cols_gen)
        df1 = T_generated[cols].copy()
        df2 = T_target[cols].copy()
        df1 = df1.sort_values(
            by=cols,
            key=lambda col_series: col_series.astype(str),
            na_position="first"
        ).reset_index(drop=True)

        df2 = df2.sort_values(
            by=cols,
            key=lambda col_series: col_series.astype(str),
            na_position="first"
        ).reset_index(drop=True)

        # 4. Row count check
        if len(df1) != len(df2):
            return False, f"Row count mismatch: Generated table has {len(df1)} rows, target table has {len(df2)} rows."

        # 5. Compare columns one by one
        for col in cols:
            s1, s2 = df1[col], df2[col]
            # Numeric column comparison
            if is_numeric_dtype(s1.dtype) and is_numeric_dtype(s2.dtype):
                arr1 = s1.to_numpy()
                arr2 = s2.to_numpy()
                mask = np.isclose(arr1, arr2, atol=tolerance, equal_nan=True)

            # One is numeric, the other is object type, try converting and comparing
            elif is_numeric_dtype(s1.dtype) != is_numeric_dtype(s2.dtype):
                # s1 is numeric, s2 is object
                if is_numeric_dtype(s1.dtype) and not is_numeric_dtype(s2.dtype):
                    s2_conv = pd.to_numeric(s2, errors='coerce')
                    # If all non-null values can be successfully converted, compare as numeric
                    if s2.notna().eq(s2_conv.notna()).all():
                        arr1 = s1.to_numpy()
                        arr2 = s2_conv.to_numpy()
                        mask = np.isclose(arr1, arr2, atol=tolerance, equal_nan=True)
                    else:
                        mask = s1.fillna("__NA__").eq(s2.fillna("__NA__"))

                # s1 is object, s2 is numeric
                else:
                    s1_conv = pd.to_numeric(s1, errors='coerce')
                    if s1.notna().eq(s1_conv.notna()).all():
                        arr1 = s1_conv.to_numpy()
                        arr2 = s2.to_numpy()
                        mask = np.isclose(arr1, arr2, atol=tolerance, equal_nan=True)
                    else:
                        mask = s1.fillna("__NA__").eq(s2.fillna("__NA__"))

            # Other cases compare as object/string
            else:
                mask = s1.fillna("__NA__").astype(str).eq(s2.fillna("__NA__").astype(str))

            if not mask.all():
                bad_idx = np.nonzero(~mask)[0][:5].tolist()
                return False, (
                    f"Column '{col}' mismatches at rows {bad_idx}.\n"
                    f"Generated table samples: {df1.loc[bad_idx, col].tolist()}\n"
                    f"Target table samples: {df2.loc[bad_idx, col].tolist()}"
                )

        return True, None