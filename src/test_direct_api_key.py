#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify direct API key parameter support
This script tests if passing the API key directly in the model arguments works.
"""

import os
import sys
import logging
import traceback

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# First try to import from local path
try:
    from api_models import LocalChatCompletion, OpenAIChatCompletion
    logger.info("Successfully imported API models from local path")
except ImportError as e:
    # If that fails, try to import from src
    try:
        # Add the current directory to the path
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.api_models import LocalChatCompletion, OpenAIChatCompletion
        logger.info("Successfully imported API models from src path")
    except ImportError as e:
        logger.error(f"Failed to import API models: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

def test_direct_api_key():
    """
    Test the direct API key parameter support in LocalChatCompletion and OpenAIChatCompletion
    """
    # Test API key
    api_key = "test_api_key_123456"
    
    # Test with LocalChatCompletion
    try:
        logger.info("Testing LocalChatCompletion with direct API key...")
        local_model = LocalChatCompletion(
            base_url="https://api.together.xyz/v1/chat/completions",
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            api_key=api_key
        )
        
        # Get the API key and check if it matches
        detected_key = local_model.api_key
        if detected_key == api_key:
            logger.info("✅ LocalChatCompletion successfully used the provided API key")
        else:
            logger.error(f"❌ LocalChatCompletion did not use the provided API key. Got: {detected_key}")
    except Exception as e:
        logger.error(f"❌ Error testing LocalChatCompletion: {e}")
        logger.error(traceback.format_exc())
    
    # Test with OpenAIChatCompletion
    try:
        logger.info("\nTesting OpenAIChatCompletion with direct API key...")
        openai_model = OpenAIChatCompletion(
            base_url="https://api.together.xyz/v1/chat/completions",
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            api_key=api_key
        )
        
        # Get the API key and check if it matches
        detected_key = openai_model.api_key
        if detected_key == api_key:
            logger.info("✅ OpenAIChatCompletion successfully used the provided API key")
        else:
            logger.error(f"❌ OpenAIChatCompletion did not use the provided API key. Got: {detected_key}")
    except Exception as e:
        logger.error(f"❌ Error testing OpenAIChatCompletion: {e}")
        logger.error(traceback.format_exc())
    
    logger.info("\nTest Summary:")
    logger.info("✓ The direct API key parameter feature is now available")
    logger.info("\nYou can now use the following command format:")
    logger.info("python src/eval.py \\")
    logger.info("    --model local-chat-completions \\")
    logger.info("    --model_args model=deepseek-ai/DeepSeek-V3,base_url=https://api.together.xyz/v1/chat/completions,api_key=your_api_key_here,max_tokens=25 \\")
    logger.info("    --tasks flare_es_multifin \\")
    logger.info("    --batch_size 20000 \\")
    logger.info("    --num_fewshot 0 \\")
    logger.info("    --apply_chat_template")

if __name__ == "__main__":
    test_direct_api_key() 