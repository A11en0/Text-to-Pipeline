"""
IMACS system orchestrator, responsible for managing agents and workflows
"""
import multiprocessing
import queue
import random
import time
from typing import List
import uuid
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from src.agents.connector import WebTablesConnector
from src.agents.sampler import SamplerAgent
from src.agents.intent import IntentAgent
from src.agents.coder import CoderAgent
from src.agents.executor import ExecutorAgent
from src.agents.storage import StorageAgent
from src.llm.client import LLMClient
from src.utils.logger import get_logger
from src.config import SAMPLER_CONFIG, DEFAULT_CONFIG
from src.pipeline.base import Pipeline

class IMACSOrchestrator:
    """
    IMACS system orchestrator, manages agents and executes workflows
    """
    
    def __init__(self, llm_config, webtables_config, agent_config, storage_config, log_config=None):
        """
        Initialize the orchestrator
        
        Args:
            llm_config: LLM configuration
            webtables_config: WebTables data connector configuration
            agent_config: Agent configuration
            storage_config: Storage configuration
            log_config: Log configuration
        """
        # Set up logging
        self.logger = get_logger("orchestrator")
        
        # Read configuration
        self.debug_mode = agent_config.get("debug_mode", False)
        self.max_attempts = agent_config.get("max_attempts", 3)
        
        # Merge workflow configuration from DEFAULT_CONFIG
        workflow_config = DEFAULT_CONFIG.get("workflow", {})
        if "workflow" not in agent_config:
            agent_config["workflow"] = workflow_config
        else:
            agent_config["workflow"] = {**workflow_config, **agent_config["workflow"]}
        
        # Initialize LLM client
        # Supports new configuration format, uses the specified model if model_name is provided
        model_name = llm_config.get("model_name") or llm_config.get("default_model")
        self.llm_client = LLMClient(llm_config, model_name)
        
        # Update sampler configuration
        sampler_config = {**agent_config, **SAMPLER_CONFIG}
        
        # Initialize agents
        self.connector = WebTablesConnector(webtables_config)
        self.sampler_agent = SamplerAgent(sampler_config, self.connector)
        self.intent_agent = IntentAgent(agent_config, self.llm_client)
        # Retain coder_agent for compatibility, but not needed in a pure pipeline system
        self.coder_agent = CoderAgent(agent_config, self.llm_client)
        self.executor_agent = ExecutorAgent(agent_config)
        self.storage_agent = StorageAgent(storage_config)
        
        # Save configuration for later access
        self.agent_config = agent_config
        
        self.logger.info("IMACS orchestrator initialized")
    
    def generate_benchmark(self, n_tasks=10, max_complexity=None, auto_save=True, auto_report=False, timeout=20):
        """
        Generate benchmark dataset
        
        Args:
            n_tasks: Number of tasks to generate
            max_complexity: Maximum complexity of tasks
            auto_save: Whether to automatically save to storage
            auto_report: Whether to automatically generate a report

        Returns:
            dict: Statistics of the generated tasks
        """
        start_time = time.time()
        successful_tasks = 0
        failed_tasks = 0
        tasks = []
        timeout = timeout  # Set timeout to 20 seconds

        # Allocate proportions based on task count and complexity
        low_part = int(n_tasks * 0.15)   
        mid_part = int(n_tasks * 0.55)   
        high_part = n_tasks - low_part - mid_part
        complexities = []
        complexities += [1] * int(low_part * 0.6)  # Increase the number of tasks with complexity 1
        complexities += [random.randint(2, 3) for _ in range(low_part - int(low_part * 0.6))]  # Randomly generate 2 or 3 for the remaining part
        complexities += [random.randint(4, 6) for _ in range(mid_part)]   # 4-6 level
        complexities += [random.randint(7, max_complexity) for _ in range(high_part)]  # 7-max level

        self.logger.info(f"Starting to generate {n_tasks} tasks with hierarchical complexity (max={max_complexity})...")
        # Create a task iterator with a progress bar
        task_iter = tqdm(enumerate(complexities), total=n_tasks, desc="Generating tasks", unit="tasks")

        # Iterate through task generation and execution
        for i, complexity in task_iter:
            task_start = time.time()
            task_id = str(uuid.uuid4())[:8]  # Generate unique task ID

            # Update progress bar description
            task_iter.set_description(f"Task {i+1}/{n_tasks} [ID:{task_id} C:{complexity}]")
            self.logger.info(f"Generating task {i+1}/{n_tasks} [ID:{task_id} Complexity:{complexity}]...")

            # Use multiprocessing.Process to start a subprocess to execute the task
            result_queue = multiprocessing.Queue()
            p = multiprocessing.Process(
                target=self.task_worker,
                args=(
                    complexity,
                    self.sampler_agent,
                    self.intent_agent,
                    self._execute_with_pipeline,
                    self.logger.name,  # Pass the logger's name, reconstruct the logger inside the subprocess
                    result_queue
                )
            )
            p.start()
            try:
                # Wait for result_queue to return results, timeout after 30 seconds
                result = result_queue.get(timeout=timeout)
                task_time = time.time() - task_start
                if result.get("status") == "success":
                    successful_tasks += 1
                    result["success"] = True
                else:
                    failed_tasks += 1
                    result["success"] = False
            except queue.Empty:
                # Queue timeout -> indicates the task took too long or got stuck
                p.terminate()
                p.join()
                self.logger.warning(f"Task {task_id} timed out (exceeded {timeout} seconds), skipping")
                result = {
                    "status": "timeout",
                    "success": False,
                    "error": "Task timed out"
                }
                task_time = time.time() - task_start
                failed_tasks += 1
            except Exception as e:
                p.terminate()
                p.join()
                self.logger.error(f"Task {task_id} execution error: {str(e)}")
                result = {
                    "status": "error",
                    "success": False,
                    "error": str(e)
                }
                task_time = time.time() - task_start
                failed_tasks += 1

            result["task_id"] = task_id
            result["exec_time"] = task_time
            tasks.append(result)

            # Automatically save task results
            if auto_save:
                save_result = self.storage_agent.save_task(task_id, result)
                if not save_result:
                    self.logger.warning(f"Task {task_id} save failed")

        # Summarize overall information
        total_time = time.time() - start_time
        success_rate = successful_tasks / n_tasks if n_tasks > 0 else 0

        benchmark_result = {
            "total_tasks": n_tasks,
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": success_rate,
            "total_time": total_time,
            "avg_time": total_time / n_tasks if n_tasks > 0 else 0,
            "tasks": tasks,
            "complexity": complexity,
            "start_time": start_time,
            "end_time": time.time()
        }
        
        self.logger.info(f"Benchmark completed. Success rate: {success_rate:.2%}, Total time: {total_time:.2f} seconds")

        if auto_save:
            benchmark_path = self.storage_agent.finalize()
            if benchmark_path:
                benchmark_result["benchmark_path"] = benchmark_path
            else:
                self.logger.warning("Failed to save benchmark dataset")

        if auto_report:
            report_id = self.storage_agent.save_benchmark_report(benchmark_result)
            if report_id:
                benchmark_result["report_path"] = report_id
                self.logger.info(f"Benchmark report saved, ID: {report_id}")
            else:
                self.logger.warning("Failed to generate benchmark report")

        return benchmark_result
        
    def task_worker(self, complexity, sampler_agent, intent_agent, execute_with_pipeline, logger_name, result_queue):
        """
        Execute task workflow in a subprocess and return results via a queue.
        """
        # Reinitialize logger in the subprocess, do not directly pass self.logger
        
        logger = get_logger(logger_name)
        try:
            logger.info(f"Starting task workflow execution, complexity: {complexity}")
            task_start = time.time()
            if 1 <= complexity <= 3:
                difficulty = "easy"
            elif 4 <= complexity <= 6:
                difficulty = "medium"
            else: 
                difficulty = "hard"
            # Generate transformation task
            input_table, target_table, transform_chain, src = sampler_agent.run(complexity)
            
            # Generate natural language instructions
            instruction, initial_intent, is_valid, rewritten_intent = intent_agent.run(input_table, target_table, transform_chain)
            
            # instruction, initial_intent, is_valid, rewritten_intent = 1, 2, 3, 4
            logger.info("Executing transformation chain using pipeline system")
            execution_result = execute_with_pipeline(input_table, transform_chain, target_table)

            transform_code = Pipeline.transform_chain_to_code(transform_chain, input_table)            
            task_time = time.time() - task_start
            
            task_result = {
                "status": execution_result.get("status", "error"),
                "difficulty": difficulty,
                "data_source": src,
                "input_table": input_table,
                "target_table": target_table,
                "transform_chain": transform_chain,
                "transform_code": transform_code,                
                "instruction": instruction,
                "initial_intent": initial_intent,
                "intent": rewritten_intent,
                "rewrite": not is_valid,
                "execution_result": execution_result,
                "execution_time": task_time
            }
            
            if task_result["status"] == "success":
                logger.info("✅ Task executed successfully!")
            else:
                logger.warning(f"❌ Task execution failed, status: {task_result['status']}")
            # Return results via the queue
            result_queue.put(task_result)
        except Exception as e:
            logger.exception(f"❌ Task execution exception: {str(e)}")
            result_queue.put({"status": "error", "error": str(e)})
    
    def _execute_with_pipeline(self, input_table, transform_chain, target_table):
        """
        Execute transformation chain using the pipeline system
        
        Args:
            input_table: Input table
            transform_chain: Transformation chain
            target_table: Target table
            
        Returns:
            dict: Execution result
        """
        # Execute transformation chain
        T_generated, success, error = self.executor_agent.run(input_table, transform_ops=transform_chain)
        if isinstance(T_generated, List):
            T_generated = T_generated[0]
        if success:
            # Validate results
            is_valid, difference = self.executor_agent.validate(T_generated, target_table)
            if is_valid:
                return {
                    "status": "success",
                    "result": T_generated,
                    "error": None,
                    "execution_method": "pipeline"
                }
            else:
                return {
                    "status": "partial",
                    "result": T_generated,
                    "error": difference,
                    "execution_method": "pipeline"
                }
        else:
            return {
                "status": "error",
                "result": None,
                "error": error,
                "execution_method": "pipeline"
            }