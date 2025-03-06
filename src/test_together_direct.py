#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Direct test of Together AI's API using the OpenAI client
This script performs a simple API call without depending on the evaluation framework.
"""

import os
import json
import time
import logging
from openai import OpenAI

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_together_api():
    """
    Test the Together AI API directly using the OpenAI client
    """
    # Get API key from environment
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        logger.error("TOGETHER_API_KEY environment variable not set")
        return
    
    logger.info("TOGETHER_API_KEY found in environment")
    
    # Initialize the OpenAI client with Together AI's base URL
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.together.xyz/v1"
    )
    
    # Create a simple prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    logger.info("Sending request to Together AI API...")
    try:
        start_time = time.time()
        # Make the API call
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=messages,
            max_tokens=50,
            temperature=0.0
        )
        end_time = time.time()
        
        # Process the response
        response_text = response.choices[0].message.content
        logger.info(f"API call completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Response: {response_text}")
        
        logger.info("\n✅ Together AI API test successful!")
        logger.info("\nThis confirms that:")
        logger.info("1. Your Together API key is valid")
        logger.info("2. The OpenAI client correctly connects to Together's API")
        logger.info("3. The integration approach we implemented should work correctly")
        
    except Exception as e:
        logger.error(f"❌ API call failed: {str(e)}")
        logger.error("Please check your API key and internet connection")

if __name__ == "__main__":
    test_together_api() 