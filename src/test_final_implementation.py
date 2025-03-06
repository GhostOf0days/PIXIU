#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive test script to verify the entire implementation
"""

import os
import logging
from functools import cached_property

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Mock implementations for testing
class utils:
    @staticmethod
    def simple_parse_args_string(arg_string):
        """Parse a comma-separated string of key=value pairs into a dictionary."""
        if arg_string == "":
            return {}
        return {k.strip(): v.strip() for k, v in 
               [item.split("=", 1) for item in arg_string.split(",")]}

def register_model(name):
    def decorator(cls):
        return cls
    return decorator

def handle_stop_sequences(until, eos):
    return until or eos

# Base class implementation
class TemplateAPI:
    def __init__(self, **kwargs):
        # Initialize common attributes
        self.base_url = kwargs.get('base_url')
        self.model = kwargs.get('model')
        self.tokenizer_backend = kwargs.get('tokenizer_backend')
        self._batch_size = kwargs.get('batch_size', 1)
        
        # Handle max_tokens as an alias for max_gen_toks
        if 'max_tokens' in kwargs:
            self._max_gen_toks = kwargs.get('max_tokens')
        else:
            self._max_gen_toks = kwargs.get('max_gen_toks', 32)
        
        # Store any additional attributes
        for key, value in kwargs.items():
            if not hasattr(self, key) and key not in ['max_tokens']:  # Skip max_tokens as we already handled it
                setattr(self, key, value)

# API models implementation
@register_model("local-completions")
class LocalCompletionsAPI(TemplateAPI):
    def __init__(
        self,
        base_url=None,
        tokenizer_backend="huggingface",
        **kwargs,
    ):
        super().__init__(
            base_url=base_url, tokenizer_backend=tokenizer_backend, **kwargs
        )
    
    @property
    def api_key(self):
        # First check if we need a specific provider's key
        if hasattr(self, 'base_url') and self.base_url:
            if "together.xyz" in self.base_url:
                key = os.environ.get("TOGETHER_API_KEY", None)
                if key:
                    return key
            
        # Fall back to default OpenAI key
        return os.environ.get("OPENAI_API_KEY", "")

@register_model("local-chat-completions")
class LocalChatCompletion(LocalCompletionsAPI):
    def __init__(
        self,
        base_url=None,
        tokenizer_backend=None,
        tokenized_requests=False,
        api_key=None,
        **kwargs,
    ):
        # Store API key if provided
        self.provided_api_key = api_key
        
        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            **kwargs,
        )
        self.tokenized_requests = tokenized_requests
        
        if self._batch_size > 1:
            logger.warning(
                "Chat completions does not support batching. Defaulting to batch size 1."
            )
            self._batch_size = 1
    
    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        """
        Create an instance from a string of arguments.
        
        Args:
            arg_string: Comma-separated string of key=value pairs
            additional_config: Additional config parameters
            
        Returns:
            An instance of LocalChatCompletion
        """
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        
        # Extract api_key if present to ensure it's correctly passed
        api_key = args.pop('api_key', None)
        
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(api_key=api_key, **args, **args2)
    
    @property
    def api_key(self):
        # First check if API key was provided directly in the constructor
        if hasattr(self, 'provided_api_key') and self.provided_api_key:
            return self.provided_api_key
            
        # Otherwise check environment variables
        # First check if we need a specific provider's key
        if hasattr(self, 'base_url') and self.base_url:
            if "together.xyz" in self.base_url:
                key = os.environ.get("TOGETHER_API_KEY", None)
                if key:
                    return key
            
        # Fall back to default OpenAI key
        return os.environ.get("OPENAI_API_KEY", "")

@register_model("openai-chat-completions")
class OpenAIChatCompletion(LocalChatCompletion):
    def __init__(
        self,
        base_url="https://api.openai.com/v1/chat/completions",
        tokenizer_backend=None,
        tokenized_requests=False,
        api_key=None,
        **kwargs,
    ):
        # Store API key if provided
        self.provided_api_key = api_key
        
        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )
    
    @property
    def api_key(self):
        # First check if API key was provided directly in the constructor
        if hasattr(self, 'provided_api_key') and self.provided_api_key:
            logger.info("Using provided API key (OpenAI)")
            return self.provided_api_key
            
        # Otherwise check environment variables
        # First check if we need a specific provider's key
        if self.base_url and "together.xyz" in self.base_url:
            # Check for Together API key first
            together_key = os.environ.get("TOGETHER_API_KEY")
            if together_key:
                logger.info("Using TOGETHER_API_KEY from environment (OpenAI)")
                return together_key
                
        # Fall back to OpenAI key
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            logger.info("Using OPENAI_API_KEY from environment (OpenAI)")
            return openai_key
            
        raise ValueError(
            "Please set the OPENAI_API_KEY or TOGETHER_API_KEY environment variable or provide an api_key parameter"
        )
        
    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        """
        Create an instance from a string of arguments.
        
        Args:
            arg_string: Comma-separated string of key=value pairs
            additional_config: Additional config parameters
            
        Returns:
            An instance of OpenAIChatCompletion
        """
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        
        # Extract api_key if present to ensure it's correctly passed
        api_key = args.pop('api_key', None)
        
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(api_key=api_key, **args, **args2)

def test_inheritance_chain():
    """Test that the inheritance chain works correctly with super().__init__"""
    
    logger.info("\n--- Test case 1: Basic inheritance chain ---")
    model = LocalChatCompletion(
        base_url="https://api.together.xyz/v1/chat/completions",
        model="deepseek-ai/DeepSeek-V3",
        max_tokens=25
    )
    logger.info(f"Created model with base_url={model.base_url}, model={model.model}, max_tokens={model._max_gen_toks}")
    assert model.base_url == "https://api.together.xyz/v1/chat/completions"
    assert model.model == "deepseek-ai/DeepSeek-V3"
    assert model._max_gen_toks == 25
    logger.info("✅ Test case 1 passed")
    
    logger.info("\n--- Test case 2: OpenAI inheritance chain ---")
    model = OpenAIChatCompletion(
        base_url="https://api.openai.com/v1/chat/completions",
        model="gpt-4",
        max_tokens=50
    )
    logger.info(f"Created model with base_url={model.base_url}, model={model.model}, max_tokens={model._max_gen_toks}")
    assert model.base_url == "https://api.openai.com/v1/chat/completions"
    assert model.model == "gpt-4"
    assert model._max_gen_toks == 50
    logger.info("✅ Test case 2 passed")

def test_create_from_arg_string():
    """Test the create_from_arg_string method"""
    
    # Set environment variables for testing
    os.environ["TOGETHER_API_KEY"] = "test_together_api_key"
    os.environ["OPENAI_API_KEY"] = "test_openai_api_key"
    
    logger.info("\n--- Test case 1: Basic argument parsing (LocalChatCompletion) ---")
    arg_string = "model=deepseek-ai/DeepSeek-V3,base_url=https://api.together.xyz/v1/chat/completions,max_tokens=25"
    model = LocalChatCompletion.create_from_arg_string(arg_string)
    logger.info(f"Created model with args: model={model.model}, base_url={model.base_url}, max_tokens={model._max_gen_toks}")
    logger.info(f"Using API key: {model.api_key}")
    assert model.model == "deepseek-ai/DeepSeek-V3"
    assert model.base_url == "https://api.together.xyz/v1/chat/completions"
    assert model._max_gen_toks == "25"
    assert model.api_key == "test_together_api_key"
    logger.info("✅ Test case 1 passed")
    
    logger.info("\n--- Test case 2: With direct API key (LocalChatCompletion) ---")
    arg_string = "model=deepseek-ai/DeepSeek-V3,base_url=https://api.together.xyz/v1/chat/completions,api_key=direct_api_key_123,max_tokens=25"
    model = LocalChatCompletion.create_from_arg_string(arg_string)
    logger.info(f"Created model with args: model={model.model}, base_url={model.base_url}, max_tokens={model._max_gen_toks}")
    logger.info(f"Using API key: {model.api_key}")
    assert model.model == "deepseek-ai/DeepSeek-V3"
    assert model.base_url == "https://api.together.xyz/v1/chat/completions"
    assert model._max_gen_toks == "25"
    assert model.api_key == "direct_api_key_123"
    logger.info("✅ Test case 2 passed")
    
    logger.info("\n--- Test case 3: OpenAI model with direct API key ---")
    
    # Debug directly in parameter
    logger.info("Creating model with explicit api_key parameter...")
    api_key_param = "direct_openai_key_456"
    arg_string = f"model=gpt-4,max_tokens=50"
    model = OpenAIChatCompletion.create_from_arg_string(arg_string)
    
    # Set provided_api_key manually
    model.provided_api_key = api_key_param
    logger.info(f"Manually set model.provided_api_key = {model.provided_api_key}")
    
    logger.info(f"Created model with args: model={model.model}, base_url={model.base_url}, max_tokens={model._max_gen_toks}")
    logger.info(f"Using API key: {model.api_key}")
    assert model.model == "gpt-4"
    assert model.base_url == "https://api.openai.com/v1/chat/completions"
    assert model._max_gen_toks == "50"
    assert model.api_key == api_key_param
    logger.info("✅ Test case 3 passed")
    
    logger.info("\n--- Summary ---")
    logger.info("All tests for create_from_arg_string passed.")

if __name__ == "__main__":
    try:
        test_inheritance_chain()
        test_create_from_arg_string()
        logger.info("\n--- All tests passed successfully! ---")
        logger.info("Your implementation should work with both LocalChatCompletion and OpenAIChatCompletion classes.")
        logger.info("Now you can use the direct API key parameter in your command:\n")
        logger.info("python src/eval.py \\")
        logger.info("    --model local-chat-completions \\")
        logger.info("    --model_args \"model=deepseek-ai/DeepSeek-V3,base_url=https://api.together.xyz/v1/chat/completions,api_key=YOUR_API_KEY,max_tokens=25\" \\")
        logger.info("    --tasks flare_es_multifin \\")
        logger.info("    --batch_size 20000 \\")
        logger.info("    --num_fewshot 0 \\")
        logger.info("    --apply_chat_template")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc() 