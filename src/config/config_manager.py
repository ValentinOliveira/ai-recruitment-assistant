#!/usr/bin/env python3
"""
Configuration Management System
===============================

Flexible configuration system for the AI Recruitment Assistant.
Supports YAML configs, environment variable overrides, and runtime modifications.

Features:
- YAML-based configuration files
- Environment variable overrides
- Configuration validation
- Multiple config profiles
- Runtime parameter modification
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: Optional[str] = None, profile: str = "base"):
        self.config_path = config_path
        self.profile = profile
        self.config = {}
        self.config_dir = Path(__file__).parent.parent.parent / "configs"
        
        # Load configuration
        self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if self.config_path:
            config_file = Path(self.config_path)
        else:
            # Use profile-based config
            config_file = self.config_dir / f"{self.profile}_config.yaml"
            
            if not config_file.exists():
                # Fall back to base config
                config_file = self.config_dir / "base_config.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        logger.info(f"Loading configuration from: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        # Validate configuration
        self._validate_config()
        
        return self.config
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides to configuration."""
        # Model settings
        if os.getenv("MODEL_NAME"):
            self.config["model"]["name"] = os.getenv("MODEL_NAME")
        
        # Training settings
        if os.getenv("BATCH_SIZE"):
            self.config["training"]["batch_size"] = int(os.getenv("BATCH_SIZE"))
        
        if os.getenv("LEARNING_RATE"):
            self.config["training"]["learning_rate"] = float(os.getenv("LEARNING_RATE"))
        
        if os.getenv("NUM_EPOCHS"):
            self.config["training"]["num_epochs"] = int(os.getenv("NUM_EPOCHS"))
        
        # Data settings
        if os.getenv("DATASET_PATH"):
            self.config["data"]["dataset_path"] = os.getenv("DATASET_PATH")
        
        # W&B settings
        if os.getenv("WANDB_PROJECT"):
            self.config["wandb"]["project"] = os.getenv("WANDB_PROJECT")
        
        if os.getenv("WANDB_ENTITY"):
            self.config["wandb"]["entity"] = os.getenv("WANDB_ENTITY")
        
        if os.getenv("WANDB_DISABLED"):
            self.config["wandb"]["enabled"] = os.getenv("WANDB_DISABLED").lower() != "true"
        
        # Hardware settings
        if os.getenv("CUDA_VISIBLE_DEVICES"):
            self.config["hardware"]["device"] = "cuda"
    
    def _validate_config(self):
        """Validate configuration values."""
        required_sections = ["model", "lora", "training", "data", "paths"]
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate training parameters
        training = self.config["training"]
        if training["batch_size"] < 1:
            raise ValueError("batch_size must be >= 1")
        
        if training["learning_rate"] <= 0:
            raise ValueError("learning_rate must be > 0")
        
        if training["num_epochs"] < 1:
            raise ValueError("num_epochs must be >= 1")
        
        # Validate LoRA parameters
        lora = self.config["lora"]
        if lora["r"] < 1:
            raise ValueError("LoRA rank (r) must be >= 1")
        
        if lora["alpha"] < 1:
            raise ValueError("LoRA alpha must be >= 1")
        
        # Create necessary directories
        paths = self.config["paths"]
        for path_key, path_value in paths.items():
            Path(path_value).mkdir(parents=True, exist_ok=True)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the value
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple configuration values."""
        for key_path, value in updates.items():
            self.set(key_path, value)
    
    def save(self, output_path: Optional[str] = None) -> None:
        """Save current configuration to file."""
        if output_path is None:
            output_path = self.config_dir / f"runtime_{self.profile}_config.yaml"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to: {output_path}")
    
    def get_training_args(self) -> Dict[str, Any]:
        """Get Hugging Face TrainingArguments compatible parameters."""
        training_config = self.config["training"]
        paths_config = self.config["paths"]
        logging_config = self.config["logging"]
        wandb_config = self.config["wandb"]
        
        args = {
            "output_dir": paths_config["output_dir"],
            "num_train_epochs": training_config["num_epochs"],
            "per_device_train_batch_size": training_config["batch_size"],
            "gradient_accumulation_steps": training_config["gradient_accumulation_steps"],
            "learning_rate": training_config["learning_rate"],
            "weight_decay": training_config["weight_decay"],
            "warmup_ratio": training_config["warmup_ratio"],
            "max_grad_norm": training_config["max_grad_norm"],
            "optim": training_config["optim"],
            "lr_scheduler_type": training_config["lr_scheduler_type"],
            "fp16": training_config["fp16"],
            "bf16": training_config["bf16"],
            "gradient_checkpointing": training_config["gradient_checkpointing"],
            "dataloader_drop_last": training_config["dataloader_drop_last"],
            "group_by_length": training_config["group_by_length"],
            "logging_steps": logging_config["log_steps"],
            "save_steps": logging_config["save_steps"],
            "save_total_limit": logging_config["save_total_limit"],
            "report_to": "wandb" if wandb_config.get("enabled", True) else None,
            "run_name": wandb_config.get("run_name", "recruitment-assistant"),
        }
        
        # Add evaluation settings if present
        if "evaluation" in self.config:
            eval_config = self.config["evaluation"]
            args.update({
                "evaluation_strategy": eval_config.get("eval_strategy", "no"),
                "eval_steps": eval_config.get("eval_steps", 500),
                "metric_for_best_model": eval_config.get("metric_for_best_model", "eval_loss"),
                "greater_is_better": eval_config.get("greater_is_better", False),
                "load_best_model_at_end": eval_config.get("load_best_model_at_end", False),
            })
        
        # Add dataloader settings if present
        if "dataloader_pin_memory" in training_config:
            args["dataloader_pin_memory"] = training_config["dataloader_pin_memory"]
        
        if "dataloader_num_workers" in training_config:
            args["dataloader_num_workers"] = training_config["dataloader_num_workers"]
        
        return args
    
    def get_lora_config(self) -> Dict[str, Any]:
        """Get LoRA configuration parameters."""
        lora_config = self.config["lora"].copy()
        
        # Convert task_type string to enum if needed
        return lora_config
    
    def get_quantization_config(self) -> Dict[str, Any]:
        """Get quantization configuration parameters."""
        return self.config["quantization"].copy()
    
    def get_generation_config(self) -> Dict[str, Any]:
        """Get text generation configuration parameters."""
        return self.config.get("generation", {}).copy()
    
    def display(self) -> None:
        """Display current configuration in a readable format."""
        print("🔧 Current Configuration:")
        print("=" * 50)
        
        def print_section(data: Dict[str, Any], indent: int = 0):
            for key, value in data.items():
                prefix = "  " * indent
                if isinstance(value, dict):
                    print(f"{prefix}{key}:")
                    print_section(value, indent + 1)
                elif isinstance(value, list):
                    print(f"{prefix}{key}: [{', '.join(map(str, value))}]")
                else:
                    print(f"{prefix}{key}: {value}")
        
        print_section(self.config)
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware-specific configuration."""
        import torch
        
        info = {
            "device": self.get("hardware.device", "auto"),
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        return info

def load_config(profile: str = "base", config_path: Optional[str] = None) -> ConfigManager:
    """Convenience function to load configuration."""
    return ConfigManager(config_path=config_path, profile=profile)

def get_available_profiles() -> list:
    """Get list of available configuration profiles."""
    config_dir = Path(__file__).parent.parent.parent / "configs"
    profiles = []
    
    for config_file in config_dir.glob("*_config.yaml"):
        profile_name = config_file.stem.replace("_config", "")
        profiles.append(profile_name)
    
    return profiles

# Quick configuration presets
def get_rtx4060_config() -> ConfigManager:
    """Get RTX 4060 optimized configuration."""
    return ConfigManager(profile="rtx4060")

def get_base_config() -> ConfigManager:
    """Get base configuration."""
    return ConfigManager(profile="base")

if __name__ == "__main__":
    # Example usage
    print("Available profiles:", get_available_profiles())
    
    # Load RTX 4060 config
    config = get_rtx4060_config()
    config.display()
    
    print("\n" + "=" * 50)
    print("Hardware Info:", config.get_hardware_info())
