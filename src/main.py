"""
IMACS System Main Entry
"""
import os
import argparse
from pathlib import Path
from src.config import load_config_from_file
from src.orchestrator import IMACSOrchestrator
from src.utils.logger import get_logger, init_logging

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='IMACS System - Intelligent Data Transformation Benchmark Generator')
    
    parser.add_argument('--config', type=str, default='config/default_config.yaml', help='Path to the configuration file')    
    parser.add_argument('--n_tasks', type=int, default=10, help='Number of tasks to generate')    
    parser.add_argument('--complexity', type=int, choices=[7,8], default=None, help='Task complexity (1-8)')    
    parser.add_argument('--auto_report', action='store_true', help='Automatically generate a report')
    parser.add_argument('--timeout', type=int, default=25, help='Timeout (default is 25 seconds)')    
    return parser.parse_args()
    
def main():
    """Main function"""
    try:
        # Parse command-line arguments
        args = parse_args()
        
        # Load configuration
        config = load_config_from_file(args.config)
        
        # Get logging configuration from the config
        log_config = config.get('log_config', {})
        log_level = log_config.get('log_level', 'debug')  # Default to debug level for troubleshooting
        log_dir = log_config.get('log_dir', 'logs')
        
        # Initialize logging system
        logger = init_logging(log_level=log_level, log_dir=log_dir)
        
        logger.info("🚀 IMACS System Starting")
        
        # Ensure the configuration contains all necessary sections
        required_configs = ['llm_config', 'webtables_config', 'agent_config', 'storage_config']
        for cfg in required_configs:
            if cfg not in config:
                raise ValueError(f"⚠️ Missing required configuration item in the config file: {cfg}")
        
        # Initialize orchestrator
        logger.info("🔧 Initializing system orchestrator...")
        orchestrator = IMACSOrchestrator(
            llm_config=config['llm_config'],
            webtables_config=config['webtables_config'],
            agent_config=config['agent_config'],
            storage_config=config['storage_config']
        )
        
        # Generate benchmark dataset
        logger.info(f"📊 Starting to generate {args.n_tasks} tasks, complexity: {args.complexity or 'default'}...")
        result = orchestrator.generate_benchmark(
            n_tasks=args.n_tasks,
            max_complexity=args.complexity,
            auto_save=True,
            auto_report=args.auto_report,
            timeout=args.timeout
        )
        
        # Output result statistics
        total = result['total_tasks']
        
        logger.info("=" * 50)
        logger.info(f"✨ Benchmark dataset generation completed!")
        logger.info(f"📝 Total tasks: {total}")
        
        # If a report was generated, display the report path
        if args.auto_report and 'report_path' in result:
            logger.info(f"📋 Report generated: {result['report_path']}")
        
        # Display total time taken
        if 'start_time' in result and 'end_time' in result:
            total_time = result['end_time'] - result['start_time']
            logger.info(f"⏱️ Total time taken: {total_time:.2f} seconds")
            if total > 0:
                logger.info(f"⏱️ Average time per task: {total_time/total:.2f} seconds")
        
        logger.info("=" * 50)
        logger.info("🏁 System execution completed")
        
        return 0
    
    except Exception as e:
        logger.exception(f"❌ An error occurred during execution: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())