"""
Storage agent responsible for saving generated tasks and results.
"""
import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from src.agents.base import Agent
from src.utils.logger import get_logger
from typing import Dict

class StorageAgent(Agent):
    """
    Storage agent for saving generated tasks and benchmark datasets.
    """
    
    def __init__(self, storage_config):
        """
        Initialize the storage agent.
        
        Args:
            storage_config: Storage configuration.
        """
        super().__init__(storage_config)
        
        # Get the base output directory
        base_output_dir = storage_config.get("output_path", "output/benchmark")
        
        # Add a timestamp to the output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(base_output_dir, f"benchmark_{timestamp}")
        self.benchmark_file = storage_config.get("benchmark_file", "benchmark.json")
        
        # Create the output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize the task list
        self.tasks = []
        
        # Logger
        self.logger = get_logger("storage")
        self.logger.info(f"Storage agent initialized, output directory: {self.output_dir}")
    
    def run(self, task_data):
        """
        Save task data.
        
        Args:
            task_data: Task data.
            
        Returns:
            str: Task ID.
        """
        task_id = task_data["id"]
        
        # Copy data to avoid modifying the original object
        task_copy = task_data.copy()
        
        # Convert DataFrame to HTML table string
        if "input_table" in task_copy and isinstance(task_copy["input_table"], pd.DataFrame):
            # Save DataFrame to CSV
            input_path = os.path.join(self.output_dir, f"{task_id}_input.csv")
            task_copy["input_table"].to_csv(input_path, index=False)
            # Generate HTML preview
            task_copy["input_preview"] = task_copy["input_table"].head().to_html(classes="table table-striped table-sm")
            # Record file path
            task_copy["input_table_path"] = input_path
            # Delete DataFrame object
            del task_copy["input_table"]
        
        if "target_table" in task_copy and isinstance(task_copy["target_table"], pd.DataFrame):
            # Save DataFrame to CSV
            target_path = os.path.join(self.output_dir, f"{task_id}_target.csv")
            task_copy["target_table"].to_csv(target_path, index=False)
            # Generate HTML preview
            task_copy["target_preview"] = task_copy["target_table"].head().to_html(classes="table table-striped table-sm")
            # Record file path
            task_copy["target_table_path"] = target_path
            # Delete DataFrame object
            del task_copy["target_table"]
        
        # Process DataFrame in execution results
        if "execution_result" in task_copy and isinstance(task_copy["execution_result"], dict):
            if "result" in task_copy["execution_result"] and isinstance(task_copy["execution_result"]["result"], pd.DataFrame):
                # Save result DataFrame to CSV
                result_path = os.path.join(self.output_dir, f"{task_id}_result.csv")
                task_copy["execution_result"]["result"].to_csv(result_path, index=False)
                # Record file path
                task_copy["execution_result"]["result_path"] = result_path
                # Delete DataFrame object
                del task_copy["execution_result"]["result"]
        
        # Add timestamp
        task_copy["timestamp"] = datetime.now().isoformat()
        
        # Add the task to the list
        self.tasks.append(task_copy)
        
        self.logger.info(f"Task saved: {task_id}")
        return task_id
    
    def default_serializer(self, obj):
        """
        Default serializer for unsupported data types.
        """
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.ndarray, list, tuple)):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, pd.Timedelta):
            return str(obj)
        else:
            return str(obj)  # Safest, convert to string if all else fails
        
    def finalize(self):
        """
        Finalize benchmark dataset generation and save the entire benchmark dataset.
        
        Returns:
            str: Path to the saved file.
        """
        try:
            # Try to serialize each task individually, skip those that cannot be serialized
            serializable_tasks = []
            for task in self.tasks:
                try:
                    # Try to convert the task to a JSON string individually
                    json.dumps(task, ensure_ascii=False, default=self.default_serializer)
                    serializable_tasks.append(task)
                except Exception as task_error:
                    self.logger.warning(f"Skipped task that cannot be serialized: {task}, reason: {str(task_error)}")

            # Create benchmark dataset data
            benchmark_data = {
                "metadata": {
                    "creation_time": datetime.now().isoformat(),
                    "task_count": len(serializable_tasks)
                },
                "tasks": serializable_tasks
            }
            # Save as JSON file
            benchmark_path = os.path.join(self.output_dir, "benchmark.json")
            with open(benchmark_path, 'w', encoding='utf-8') as f:
                json.dump(benchmark_data, f, ensure_ascii=False, indent=2, default=self.default_serializer)

            self.logger.info(f"Benchmark dataset saved to: {benchmark_path}, valid task count: {len(serializable_tasks)}")
            return benchmark_path

        except Exception as e:
            self.logger.error(f"Failed to save benchmark dataset: {str(e)}")
            return None
    
    def get_benchmark_stats(self):
        """
        Get simple statistics of the benchmark dataset.
        
        Returns:
            dict: Statistics.
        """
        # Operation type statistics
        task_types = {}
        for task in self.tasks:
            for transform in task.get("transform_chain", []):
                if isinstance(transform, dict):
                    op = transform.get("op", "unknown")
                else:
                    op = str(transform)
                    
                if op in task_types:
                    task_types[op] += 1
                else:
                    task_types[op] = 1
        
        # Simple complexity statistics (based on transform chain length)
        complexity = {"low": 0, "medium": 0, "high": 0}
        
        for task in self.tasks:
            chain_length = len(task.get("transform_chain", []))
            
            if chain_length == 1:
                complexity["low"] += 1
            elif chain_length == 2:
                complexity["medium"] += 1
            else:
                complexity["high"] += 1
        
        return {
            "task_count": len(self.tasks),
            "task_types": task_types,
            "complexity": complexity
        }

    def save_task(self, task_id: str, task_data: Dict) -> bool:
        """
        Save task data.
        
        Args:
            task_id: Task ID.
            task_data: Task data.
            
        Returns:
            bool: Whether the save was successful.
        """
        try:
            # Separate table data and save
            task_copy = task_data.copy()
            
            # Save input tables
            if "input_table" in task_copy:
                path = []
                for idx, input_table in enumerate(task_copy["input_table"]):
                    input_path = os.path.join(self.output_dir, f"task_{task_id}_input_{idx}.csv")
                    input_table.to_csv(input_path, index=False)
                    path.append(os.path.basename(input_path))
                task_copy["input_table"] = path
            
            # Save target table
            if "target_table" in task_copy:
                target_path = os.path.join(self.output_dir, f"task_{task_id}_target.csv")
                task_copy["target_table"].to_csv(target_path, index=False)
                task_copy["target_table"] = os.path.basename(target_path)
            
            # Save execution result table
            if "execution_result" in task_copy and "result" in task_copy["execution_result"]:
                if isinstance(task_copy["execution_result"]["result"], pd.DataFrame):
                    result_path = os.path.join(self.output_dir, f"task_{task_id}_result.csv")
                    task_copy["execution_result"]["result"].to_csv(result_path, index=False)
                    task_copy["execution_result"]["result"] = os.path.basename(result_path)
            
            # Add task data to task list (for final benchmark.json generation)
            self.tasks.append(task_copy)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save task {task_id}: {str(e)}")
            return False
    
    def load_task(self, task_id):
        """
        Load task data.
        
        Args:
            task_id: Task ID.
            
        Returns:
            dict: Task data, or None if loading failed.
        """
        try:
            # Load path
            task_path = os.path.join(self.output_dir, f"{task_id}.json")
            
            # Check if file exists
            if not os.path.exists(task_path):
                self.logger.warning(f"Task {task_id} does not exist")
                return None
            
            # Load from JSON
            with open(task_path, 'r', encoding='utf-8') as f:
                task_data = json.load(f)
            
            # Convert back to DataFrame
            task_data = self._convert_to_dataframes(task_data)
            
            self.logger.info(f"Task {task_id} loaded successfully")
            return task_data
            
        except Exception as e:
            self.logger.error(f"Failed to load task {task_id}: {str(e)}")
            return None
    
    def save_benchmark_report(self, report_data):
        """
        Save benchmark report.
        
        Args:
            report_data: Report data.
            
        Returns:
            str: Report ID, or None if saving failed.
        """
        try:
            # Generate report ID
            report_id = f"report_{int(time.time())}"
            
            # Prepare data for saving
            save_data = self._prepare_for_save(report_data)
            
            # Save path
            report_path = os.path.join(self.output_dir, f"{report_id}.json")
            
            # Save as JSON
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Report {report_id} saved successfully: {report_path}")
            return report_id
            
        except Exception as e:
            self.logger.error(f"Failed to save report: {str(e)}")
            return None
    
    def _prepare_for_save(self, data):
        """
        Prepare data for saving.
        Mainly handles objects like DataFrame that cannot be directly serialized.
        
        Args:
            data: Data to be saved.
            
        Returns:
            dict: Processed data.
        """
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                result[key] = self._prepare_for_save(value)
            return result
        elif isinstance(data, list):
            return [self._prepare_for_save(item) for item in data]
        elif isinstance(data, pd.DataFrame):
            # Convert DataFrame to a serializable form
            return {
                "_type": "dataframe",
                "data": data.to_dict(orient="records"),
                "index": data.index.tolist(),
                "columns": data.columns.tolist()
            }
        elif isinstance(data, pd.Series):
            # Convert Series to a serializable form
            return {
                "_type": "series",
                "data": data.tolist(),
                "index": data.index.tolist(),
                "name": data.name
            }
        else:
            # Try to return other types directly
            return data
    
    def _convert_to_dataframes(self, data):
        """
        Convert saved data back to original types.
        
        Args:
            data: Saved data.
            
        Returns:
            Converted data.
        """
        if isinstance(data, dict):
            if "_type" in data:
                if data["_type"] == "dataframe":
                    # Convert data back to DataFrame
                    df = pd.DataFrame(data["data"])
                    if len(data["columns"]) > 0:
                        df = df[data["columns"]]  # Restore column order
                    return df
                elif data["_type"] == "series":
                    # Convert data back to Series
                    return pd.Series(data["data"], index=data["index"], name=data["name"])
            
            # Recursively process other items in the dictionary
            result = {}
            for key, value in data.items():
                result[key] = self._convert_to_dataframes(value)
            return result
        elif isinstance(data, list):
            # Recursively process items in the list
            return [self._convert_to_dataframes(item) for item in data]
        else:
            # Return other types directly
            return data