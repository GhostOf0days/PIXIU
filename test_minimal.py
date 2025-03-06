import os
import logging
import requests
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock request class
class MockRequest:
    def __init__(self, context="", until=None, doc=None):
        self.context = context
        self.until = until
        self.doc = doc

    def __iter__(self):
        return iter((self.context, self.until))

def greedy_until(requests, api_key, base_url="https://api.together.xyz/v1/chat/completions"):
    """Simple implementation of greedy_until logic to test handling empty requests with docs"""
    results = []
    
    for request in requests:
        context, until = request
        
        # For any task: Handle empty context by checking if there's a document with query
        if not context and hasattr(request, 'doc'):
            doc = request.doc
            if isinstance(doc, dict) and 'query' in doc:
                context = doc['query']
                logger.info(f"Using document query from request.doc")
        
        # Skip empty requests that can't be recovered
        if not context:
            logger.warning("Received empty request with no recoverable context, returning empty string")
            results.append("")
            continue
        
        # Convert the context to a message format for the API
        messages = [{"role": "user", "content": context}]
        
        # Create payload for API request
        payload = {
            "model": "deepseek-ai/DeepSeek-V3",
            "messages": messages,
            "max_tokens": 50,
            "temperature": 0.0
        }
        
        try:
            # Make API request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            response = requests.post(
                base_url,
                headers=headers,
                json=payload,
                timeout=120
            )
            
            # Check response
            if response.status_code != 200:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                results.append("")
                continue
                
            # Parse response
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0].get("message", {}).get("content", "")
                results.append(content)
            else:
                logger.warning("No content in response")
                results.append("")
                
        except Exception as e:
            logger.error(f"Error in API request: {e}")
            results.append("")
    
    return results

def main():
    # Get API key (Using a dummy key for testing if not provided)
    api_key = os.environ.get("TOGETHER_API_KEY", "your_api_key_here")
    
    # Check if we need to prompt for API key
    if api_key == "your_api_key_here":
        logger.warning("Using dummy API key. Replace with your actual Together API key.")
        logger.warning("For testing purposes only, the script will continue but will likely fail API calls.")
    
    # Create test cases
    test_cases = [
        # Case 1: Empty request with doc containing query
        MockRequest(doc={
            "query": "What is the capital of France?",
            "text": "Some text about France"
        }),
        # Case 2: Normal request with context
        MockRequest(context="Who was the first person to walk on the moon?")
    ]
    
    # Run the test
    logger.info("Testing greedy_until with empty requests that have documents")
    logger.info(f"Test case 1 context: {test_cases[0].context}")
    logger.info(f"Test case 1 doc query: {test_cases[0].doc.get('query')}")
    logger.info(f"Test case 2 context: {test_cases[1].context}")
    
    # Only make API calls if we have a real API key
    if api_key != "your_api_key_here":
        results = greedy_until(test_cases, api_key)
        
        # Print results
        for i, result in enumerate(results):
            logger.info(f"Result {i+1}: {result[:50]}...")
    else:
        # Just test the document extraction logic without making API calls
        for request in test_cases:
            context, until = request
            
            # For any task: Handle empty context by checking if there's a document with query
            if not context and hasattr(request, 'doc'):
                doc = request.doc
                if isinstance(doc, dict) and 'query' in doc:
                    context = doc['query']
                    logger.info(f"Successfully extracted query from document: {context}")

if __name__ == "__main__":
    main() 