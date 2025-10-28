"""
Logging utility module
"""
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Log level mapping
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

def setup_logger(name="imacs", log_level="info", log_dir="logs"):
    """
    Configure and get a logger
    
    Args:
        name: Logger name
        log_level: Log level (debug, info, warning, error, critical)
        log_dir: Log file directory
    
    Returns:
        logger: Configured logger
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True, parents=True)
    
    # Create log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{name}_{timestamp}.log"
    
    # Get logger
    logger = logging.getLogger(name)
    
    # Set log level
    level = LOG_LEVELS.get(log_level.lower(), logging.INFO)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicate additions
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    
    # Set log format
    log_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler.setFormatter(log_format)
    file_handler.setFormatter(log_format)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Set the logging system to propagate, ensuring that logs from child loggers propagate to the root logger
    # Here, propagate is set to True to allow log propagation
    logger.propagate = True
    
    logger.info(f"Logging initialized, log file: {log_file}")
    
    # Add this logger to the configured loggers dictionary
    _loggers[name] = logger
    
    return logger

# Global logger dictionary
_loggers = {}

# Add an initialization flag to track whether the default logger has been initialized
_default_logger_initialized = False

def get_logger(name=None):
    """
    Get the logger with the specified name, or return the default logger if not specified.
    If the logger is not initialized, basic configuration will be performed.
    
    Args:
        name: Logger name
    
    Returns:
        logger: Logger
    """
    global _loggers, _default_logger_initialized
    
    if name is None:
        name = "imacs"
    
    # Create the full logger name
    full_name = f"imacs.{name}" if name != "imacs" else "imacs"
    
    # If a logger with this name already exists, return it directly
    if full_name in _loggers:
        return _loggers[full_name]
    
    # If the main logger is requested but not yet initialized, initialize it
    if name == "imacs" and not _default_logger_initialized:
        return setup_logger(name="imacs")
    
    # Get or create the logger
    logger = logging.getLogger(full_name)
    
    # Ensure the logger has the correct level
    # If the parent logger is configured, use the same level
    parent_logger = logging.getLogger("imacs")
    if parent_logger.level != logging.NOTSET:
        logger.setLevel(parent_logger.level)
    else:
        # If the parent logger is not configured, set a default level
        logger.setLevel(logging.INFO)
        
        # If this is the first request for a logger and the parent logger is not configured, add a simple console handler
        if not logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
    
    # Ensure propagation is set to True
    logger.propagate = True
    
    # Store the logger reference
    _loggers[full_name] = logger
    
    return logger

# Add an initialization function that can be called at application startup
def init_logging(log_level="info", log_dir="logs"):
    """
    Initialize the logging system
    
    Args:
        log_level: Log level
        log_dir: Log directory
    
    Returns:
        logger: Main logger
    """
    global _default_logger_initialized
    logger = setup_logger(name="imacs", log_level=log_level, log_dir=log_dir)
    _default_logger_initialized = True
    return logger