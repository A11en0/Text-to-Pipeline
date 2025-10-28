"""
File utility module, providing file and data processing functions
"""
import os
import json
import pandas as pd
from typing import List, Dict, Optional
import copy


def load_tasks(task_file: str, logger) -> List[dict]:
    """
    Load task data
    
    Args:
        task_file: Path to the task file
        logger: Logger
    
    Returns:
        List[dict]: List of tasks
    """
    if not os.path.exists(task_file):
        logger.error(f"Task file {task_file} does not exist, please check the path")
        return []
    
    try:
        with open(task_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading task file: {e}")
        return []


def load_input_tables(base_dir: str, input_files: List[str], logger) -> List[pd.DataFrame]:
    """
    Load input table data
    
    Args:
        base_dir: Base directory
        input_files: List of input files
        logger: Logger
        
    Returns:
        List[pd.DataFrame]: List of data tables
    """
    input_tables = []
    for idx, filename in enumerate(input_files, start=1):
        file_path = os.path.join(base_dir, filename)
        if not os.path.exists(file_path):
            if logger:
                logger.warning(f"Input table file {file_path} does not exist, skipping")
            continue
        
        try:
            df = pd.read_csv(file_path)
            input_tables.append(df)
        except Exception as e:
            if logger:
                logger.error(f"Error reading file {file_path}: {e}")
    
    return input_tables


def format_input_tables(input_tables: List[pd.DataFrame], transform_chain=None) -> str:
    """
    Format input tables into a string
    
    Args:
        input_tables: List of input tables
        
    Returns:
        str: Formatted table string
    """
    input_tables_filtered = []
    if transform_chain:
        # Correctly generate table_1 to table_n
        input_table_names = [f"table_{i}" for i in range(1, len(input_tables) + 1)]
        # Collect all table names that appear in the operations
        op_tables = []
        for step in transform_chain:
            op_tables.extend(step.get("op_tables", []))
        # Traverse in order, append the corresponding DataFrame if matched
        for idx, table_name in enumerate(input_table_names):
            if table_name in op_tables:
                input_tables_filtered.append(input_tables[idx])
    else:
        input_tables_filtered = input_tables

    # 2. Generate display string based on the filtered tables
    input_table_str = []
    for idx, df in enumerate(input_tables_filtered, start=1):
        header = f"table_{idx}:\n"
        content = df.head(10).to_string(index=False)
        input_table_str.append(header + content)

    # 3. Return the final concatenated string
    return "\n\n".join(input_table_str)


def save_task_results(result: List[dict], result_dir: str) -> None:
    """
    Save task result files
    
    Args:
        result: List of results
        result_dir: Result directory
    """
    # Create result directory
    os.makedirs(result_dir, exist_ok=True)
    
    # Process results to ensure error information is retained
    processed_result = []
    for item in result:
        # Create a deep copy to avoid modifying the original data
        processed_item = copy.deepcopy(item)
        
        # Ensure execution_result exists and contains error information
        if "execution_result" not in processed_item:
            processed_item["execution_result"] = {
                "status": "unknown",
                "result": None,
                "error": processed_item.get("error", None),
                "error_type": processed_item.get("error_type", "unknown"),
                "execution_method": "unknown"
            }
        elif isinstance(processed_item["execution_result"], dict):
            # Ensure error information is retained
            if "error" not in processed_item["execution_result"] and "error" in processed_item:
                processed_item["execution_result"]["error"] = processed_item["error"]
            
            # Ensure error type field exists
            if "error_type" not in processed_item["execution_result"]:
                # Try to infer error type from error message
                error_msg = processed_item["execution_result"].get("error", "")
                if error_msg:
                    error_type = infer_error_type(error_msg)
                else:
                    error_type = processed_item.get("error_type", "unknown")
                processed_item["execution_result"]["error_type"] = error_type
                
        # Process result field to avoid serializing DataFrame
        if isinstance(processed_item.get("execution_result", {}).get("result"), pd.DataFrame):
            try:
                # Save DataFrame as CSV file
                task_id = processed_item.get("task_id", "unknown")
                result_filename = f"task_{task_id}_result.csv"
                # result_path = os.path.join(result_dir, result_filename)
                
                # Save DataFrame
                # processed_item["execution_result"]["result"].to_csv(result_path, index=False)
                
                # Record file name reference in JSON
                processed_item["execution_result"]["result"] = result_filename
                # print(f"Result table saved to: {result_path}")
            except Exception as e:
                # print(f"Failed to save result table: {e}")
                # Use placeholder if saving fails
                processed_item["execution_result"]["result"] = "<DataFrame save failed>"
                # Set save error
                if "error" not in processed_item["execution_result"] or not processed_item["execution_result"]["error"]:
                    processed_item["execution_result"]["error"] = f"Failed to save DataFrame: {str(e)}"
                    processed_item["execution_result"]["error_type"] = "io_error"
        
        processed_result.append(processed_item)
    
    # Save original results
    result_file = os.path.join(result_dir, "result.json")
    try:
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(processed_result, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {result_file}")
    except Exception as e:
        print(f"Error saving results to {result_file}: {e}")


def infer_error_type(error_msg: str) -> str:
    """
    Infer error type based on the error message, providing finer-grained error classification
    
    Args:
        error_msg: Error message
        
    Returns:
        str: Error type
    """
    if not error_msg:
        return "unknown_error"
        
    error_msg = error_msg.lower()
    
    # ==================== Parsing Errors ====================
    if any(kw in error_msg for kw in ["parsing failed", "content parsing", "parse", "parsing"]):
        if any(kw in error_msg for kw in ["format", "structure"]):
            return "parsing_format_error"  # Format parsing error
        elif any(kw in error_msg for kw in ["syntax", "field"]):
            return "parsing_syntax_error"  # Syntax parsing error
        elif any(kw in error_msg for kw in ["empty content", "empty"]):
            return "parsing_empty_error"  # Empty content error
        else:
            return "parsing_general_error"  # General parsing error
    
    # ==================== Execution Errors ====================
    elif any(kw in error_msg for kw in ["execution failed", "pipeline execution", "execution", "running"]):
        if any(kw in error_msg for kw in ["parameter", "param"]):
            return "execution_param_error"  # Parameter error
        elif any(kw in error_msg for kw in ["type", "conversion"]):
            return "execution_type_error"  # Type error
        elif any(kw in error_msg for kw in ["table", "data"]):
            return "execution_table_error"  # Table processing error
        elif any(kw in error_msg for kw in ["operator", "operation", "op"]):
            return "execution_op_error"  # Operation execution error
        elif any(kw in error_msg for kw in ["invalid execution table", "no valid table"]):
            return "execution_invalid_result_error"  # Invalid execution result error
        else:
            return "execution_general_error"  # General execution error
    
    # ==================== Validation Errors ====================
    elif any(kw in error_msg for kw in ["validation failed", "mismatch", "validation"]):
        if any(kw in error_msg for kw in ["schema", "column", "field"]):
            return "validation_schema_error"  # Schema mismatch
        elif any(kw in error_msg for kw in ["content", "value"]):
            return "validation_content_error"  # Content mismatch
        elif any(kw in error_msg for kw in ["size", "row", "column count"]):
            return "validation_size_error"  # Size mismatch
        else:
            return "validation_general_error"  # General validation error
    
    # ==================== Input Errors ====================
    elif any(kw in error_msg for kw in ["input table", "input", "failed to read target"]):
        if any(kw in error_msg for kw in ["missing", "not found"]):
            return "input_missing_error"  # Input missing
        elif any(kw in error_msg for kw in ["format", "invalid input"]):
            return "input_format_error"  # Input format error
        elif "user intent not found" in error_msg:
            return "input_intent_missing_error"  # User intent missing
        else:
            return "input_general_error"  # General input error
    
    # ==================== Model Errors ====================
    elif any(kw in error_msg for kw in ["model call failed", "model", "llm"]):
        if any(kw in error_msg for kw in ["timeout"]):
            return "model_timeout_error"  # Model timeout
        elif any(kw in error_msg for kw in ["empty response", "empty"]):
            return "model_empty_response_error"  # Empty response
        elif any(kw in error_msg for kw in ["format"]):
            return "model_format_error"  # Response format error
        elif any(kw in error_msg for kw in ["connection", "network"]):
            return "model_connection_error"  # Connection error
        else:
            return "model_general_error"  # General model error
    
    # ==================== Operation Chain Errors ====================
    elif any(kw in error_msg for kw in ["operation chain is empty", "operation transformation failed", "transform", "chain"]):
        if "operation chain is empty" in error_msg:
            return "chain_empty_error"  # Empty operation chain
        elif any(kw in error_msg for kw in ["length mismatch", "length"]):
            return "chain_length_error"  # Operation chain length mismatch
        elif any(kw in error_msg for kw in ["parameter", "param"]):
            return "chain_param_error"  # Operation chain parameter error
        elif any(kw in error_msg for kw in ["order", "sequence"]):
            return "chain_order_error"  # Operation chain order error
        elif any(kw in error_msg for kw in ["mismatch", "operation mismatch"]):
            return "chain_op_mismatch_error"  # Operation mismatch error
        else:
            return "chain_general_error"  # General operation chain error
    
    # ==================== IO Errors ====================
    elif any(kw in error_msg for kw in ["read file", "file", "io", "save"]):
        if any(kw in error_msg for kw in ["read", "load"]):
            return "io_read_error"  # Read error
        elif any(kw in error_msg for kw in ["write", "save"]):
            return "io_write_error"  # Write error
        elif any(kw in error_msg for kw in ["not found", "not exist"]):
            return "io_not_found_error"  # File not found
        elif any(kw in error_msg for kw in ["permission", "access"]):
            return "io_permission_error"  # Permission error
        else:
            return "io_general_error"  # General IO error
    
    # ==================== Unknown Errors ====================
    return "unknown_error"  # Unknown error type


def save_evaluation_results(metrics: Dict, result_dir: str) -> None:
    """
    Save evaluation result files
    
    Args:
        metrics: Evaluation metrics
        result_dir: Result directory
    """
    # Create result directory
    os.makedirs(result_dir, exist_ok=True)
    
    # Save evaluation results
    accuracy_file = os.path.join(result_dir, "accuracy.json")
    try:
        with open(accuracy_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Accuracy results saved to {accuracy_file}")
    except Exception as e:
        print(f"Error saving accuracy results to {accuracy_file}: {e}")


def save_results(result: List[dict], result_dir: str, metrics: Optional[Dict] = None) -> None:
    """
    Save result files and evaluation metrics
    
    Args:
        result: List of results
        result_dir: Result directory
        metrics: Evaluation metrics
    """
    # Save task results
    save_task_results(result, result_dir)
    
    # Save evaluation results
    if metrics:
        save_evaluation_results(metrics, result_dir)


def save_table_result(df: pd.DataFrame, base_dir: str, task_idx: int, logger) -> None:
    """
    Save table results
    
    Args:
        df: Result data table
        base_dir: Base directory
        task_idx: Task index
        logger: Logger
    """
    try:
        result_df_path = os.path.join(base_dir, f"result_{task_idx}.csv")
        df.to_csv(result_df_path, index=False)
        logger.info(f"Result table saved to {result_df_path}")
    except Exception as e:
        logger.warning(f"Failed to save result table: {e}")