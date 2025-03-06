#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple test script to verify Together AI API key environment variables
"""

import os
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_api_key_logic():
    """
    Test the Together AI API key detection logic manually
    """
    # Check environment variables
    together_key = os.environ.get("TOGETHER_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    
    logger.info(f"TOGETHER_API_KEY present: {together_key is not None}")
    logger.info(f"OPENAI_API_KEY present: {openai_key is not None}")
    
    # Simulate our API key logic
    def get_api_key(base_url):
        # This mimics our implementation in the API models
        if "together.xyz" in base_url:
            if together_key:
                logger.info("Together AI URL detected, using TOGETHER_API_KEY")
                return together_key
            elif openai_key:
                logger.info("Together AI URL detected, but TOGETHER_API_KEY not found, falling back to OPENAI_API_KEY")
                return openai_key
            else:
                logger.error("No API keys found in environment variables")
                return None
        else:
            if openai_key:
                logger.info("Standard OpenAI URL detected, using OPENAI_API_KEY")
                return openai_key
            elif together_key:
                logger.info("Standard OpenAI URL detected, but OPENAI_API_KEY not found, trying TOGETHER_API_KEY")
                return together_key
            else:
                logger.error("No API keys found in environment variables")
                return None
    
    # Test with different base URLs
    logger.info("\n--- Testing Together AI URL ---")
    together_url = "https://api.together.xyz/v1/chat/completions"
    key_for_together = get_api_key(together_url)
    if key_for_together:
        # Don't print the actual key, just confirm detection
        logger.info("✅ API key detected for Together AI")
        
        # Check if the right key was selected
        if together_key and key_for_together == together_key:
            logger.info("✅ Using TOGETHER_API_KEY as expected")
        elif openai_key and key_for_together == openai_key:
            logger.info("⚠️ Using OPENAI_API_KEY as fallback")
        
    logger.info("\n--- Testing OpenAI URL ---")
    openai_url = "https://api.openai.com/v1/chat/completions"
    key_for_openai = get_api_key(openai_url)
    if key_for_openai:
        # Don't print the actual key, just confirm detection
        logger.info("✅ API key detected for OpenAI")
        
        # Check if the right key was selected
        if openai_key and key_for_openai == openai_key:
            logger.info("✅ Using OPENAI_API_KEY as expected")
        elif together_key and key_for_openai == together_key:
            logger.info("⚠️ Using TOGETHER_API_KEY as fallback")
    
    logger.info("\n--- Test Summary ---")
    if together_key:
        logger.info("✓ TOGETHER_API_KEY is properly set")
    else:
        logger.info("✗ TOGETHER_API_KEY is not set")
        
    if openai_key:
        logger.info("✓ OPENAI_API_KEY is properly set")
    else:
        logger.info("✗ OPENAI_API_KEY is not set")
        
    logger.info("\nTest completed!")
    logger.info("\nTo run an actual evaluation with Together AI:")
    logger.info("1. Set the TOGETHER_API_KEY environment variable: export TOGETHER_API_KEY=your_key_here")
    logger.info("2. Run the evaluation script:")
    logger.info("   python src/eval.py \\")
    logger.info("     --model local-chat-completions \\")
    logger.info("     --model_args model=meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo,base_url=https://api.together.xyz/v1/chat/completions \\")
    logger.info("     --tasks finqa \\")
    logger.info("     --apply_chat_template")

if __name__ == "__main__":
    check_api_key_logic() 