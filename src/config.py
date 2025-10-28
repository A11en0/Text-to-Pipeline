"""
IMACS system configuration file
"""
import os
import yaml
import json
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Get project root directory
ROOT_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = ROOT_DIR / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default_config.yaml"

def load_yaml_config(config_path):
    """Load configuration from a YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# Load configuration from the default configuration file
if DEFAULT_CONFIG_PATH.exists():
    DEFAULT_CONFIG = load_yaml_config(DEFAULT_CONFIG_PATH)
else:
    print(f"Warning: Default configuration file does not exist: {DEFAULT_CONFIG_PATH}")
    DEFAULT_CONFIG = {}

# Retain these constants for backward compatibility
LLM_CONFIG = DEFAULT_CONFIG.get("llm_config", {})
WEBTABLES_CONFIG = DEFAULT_CONFIG.get("webtables_config", {})
AGENT_CONFIG = DEFAULT_CONFIG.get("agent_config", {})
STORAGE_CONFIG = DEFAULT_CONFIG.get("storage_config", {})
LOG_CONFIG = DEFAULT_CONFIG.get("log_config", {})
SAMPLER_CONFIG = DEFAULT_CONFIG.get("sampler_config", {})

# Apply environment variable overrides (for backward compatibility only)
# Note: New code should directly use configurations from models
default_model_name = LLM_CONFIG.get("default_model", "gpt-4o-mini")
if "models" in LLM_CONFIG and default_model_name in LLM_CONFIG["models"]:
    default_model_config = LLM_CONFIG["models"][default_model_name]
    # For backward compatibility, copy the default model's configuration to the top level of LLM_CONFIG
    for key, value in default_model_config.items():
        if key not in LLM_CONFIG:
            LLM_CONFIG[key] = value

# Compatible environment variable overrides
AGENT_CONFIG["debug_mode"] = os.getenv("DEBUG_MODE", str(AGENT_CONFIG.get("debug_mode", "True"))).lower() == "true"
LOG_CONFIG["log_level"] = os.getenv("LOG_LEVEL", LOG_CONFIG.get("log_level", "info"))

def load_config_from_file(config_path):
    """
    Load configuration from a YAML or JSON file
    
    Args:
        config_path: Path to the configuration file
    
    Returns:
        dict: Loaded configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    
    if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    elif config_path.suffix.lower() == '.json':
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    # Merge configurations
    result = {}
    if 'llm_config' in config:
        result['llm_config'] = {**LLM_CONFIG, **config['llm_config']}
    else:
        result['llm_config'] = LLM_CONFIG
        
    if 'webtables_config' in config:
        result['webtables_config'] = {**WEBTABLES_CONFIG, **config['webtables_config']}
    else:
        result['webtables_config'] = WEBTABLES_CONFIG
        
    if 'agent_config' in config:
        result['agent_config'] = {**AGENT_CONFIG, **config['agent_config']}
    else:
        result['agent_config'] = AGENT_CONFIG
        
    if 'storage_config' in config:
        result['storage_config'] = {**STORAGE_CONFIG, **config['storage_config']}
    else:
        result['storage_config'] = STORAGE_CONFIG
    
    return result