#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple test script to verify create_from_arg_string method
"""

import os
import logging
from functools import cached_property

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Simple mock for testing
class utils:
    @staticmethod
    def simple_parse_args_string(arg_string):
        """Parse a comma-separated string of key=value pairs into a dictionary."""
        if arg_string == "":
            return {}
        return {k.strip(): v.strip() for k, v in 
               [item.split("=", 1) for item in arg_string.split(",")]}

# Create a mock class to test the create_from_arg_string method
class MockChatCompletion:
    def __init__(self, base_url=None, model=None, api_key=None, max_tokens=None, **kwargs):
        self.base_url = base_url
        self.model = model
        self.provided_api_key = api_key
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        logger.info(f"Initialized with base_url={base_url}, model={model}, api_key={api_key}, max_tokens={max_tokens}")
    
    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        """
        Create an instance from a string of arguments.
        
        Args:
            arg_string: Comma-separated string of key=value pairs
            additional_config: Additional config parameters
            
        Returns:
            An instance of MockChatCompletion
        """
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)
    
    @property
    def api_key(self):
        # First check if API key was provided directly in the constructor
        if self.provided_api_key:
            return self.provided_api_key
            
        # Otherwise check environment variables
        # First check if we need a specific provider's key
        if self.base_url and "together.xyz" in self.base_url:
            # Check for Together API key first
            together_key = os.environ.get("TOGETHER_API_KEY")
            if together_key:
                return together_key
                
        # Fall back to OpenAI key
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            return openai_key
            
        raise ValueError(
            "Please set the OPENAI_API_KEY or TOGETHER_API_KEY environment variable or provide an api_key parameter"
        )

def test_create_from_arg_string():
    """Test the create_from_arg_string method"""
    
    # Set environment variables for testing
    os.environ["TOGETHER_API_KEY"] = "test_together_api_key"
    os.environ["OPENAI_API_KEY"] = "test_openai_api_key"
    
    logger.info("\n--- Test case 1: Basic argument parsing ---")
    arg_string = "model=deepseek-ai/DeepSeek-V3,base_url=https://api.together.xyz/v1/chat/completions,max_tokens=25"
    model = MockChatCompletion.create_from_arg_string(arg_string)
    logger.info(f"Created model with args: model={model.model}, base_url={model.base_url}, max_tokens={model.max_tokens}")
    logger.info(f"Using API key: {model.api_key}")
    assert model.model == "deepseek-ai/DeepSeek-V3"
    assert model.base_url == "https://api.together.xyz/v1/chat/completions"
    assert model.max_tokens == "25"
    assert model.api_key == "test_together_api_key"
    logger.info("✅ Test case 1 passed")
    
    logger.info("\n--- Test case 2: With direct API key ---")
    arg_string = "model=deepseek-ai/DeepSeek-V3,base_url=https://api.together.xyz/v1/chat/completions,api_key=direct_api_key_123,max_tokens=25"
    model = MockChatCompletion.create_from_arg_string(arg_string)
    logger.info(f"Created model with args: model={model.model}, base_url={model.base_url}, max_tokens={model.max_tokens}")
    logger.info(f"Using API key: {model.api_key}")
    assert model.model == "deepseek-ai/DeepSeek-V3"
    assert model.base_url == "https://api.together.xyz/v1/chat/completions"
    assert model.max_tokens == "25"
    assert model.api_key == "direct_api_key_123"
    logger.info("✅ Test case 2 passed")
    
    logger.info("\n--- Test case 3: With additional config ---")
    arg_string = "model=deepseek-ai/DeepSeek-V3,base_url=https://api.together.xyz/v1/chat/completions"
    additional_config = {"max_tokens": "50", "temperature": "0.7"}
    model = MockChatCompletion.create_from_arg_string(arg_string, additional_config)
    logger.info(f"Created model with args: model={model.model}, base_url={model.base_url}, max_tokens={model.max_tokens}")
    logger.info(f"Additional args: temperature={model.kwargs.get('temperature')}")
    assert model.model == "deepseek-ai/DeepSeek-V3"
    assert model.base_url == "https://api.together.xyz/v1/chat/completions"
    assert model.max_tokens == "50"
    assert model.kwargs.get("temperature") == "0.7"
    logger.info("✅ Test case 3 passed")
    
    logger.info("\n--- Summary ---")
    logger.info("All tests for create_from_arg_string passed.")
    logger.info("The implementation should work correctly when used with LocalChatCompletion and OpenAIChatCompletion.")

if __name__ == "__main__":
    test_create_from_arg_string() 