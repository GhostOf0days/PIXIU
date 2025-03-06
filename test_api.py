import os
import sys
import logging
from importlib.util import spec_from_file_location, module_from_spec

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import the LocalChatCompletion class
try:
    # Add the src directory to the path
    sys.path.insert(0, os.path.abspath("src"))
    
    # Import the LocalChatCompletion class
    from api_models import LocalChatCompletion
    logger.info("Successfully imported LocalChatCompletion")
except ImportError as e:
    logger.error(f"Failed to import LocalChatCompletion: {e}")
    sys.exit(1)

# Create a mock request class
class MockRequest:
    def __init__(self, context="", until=None, doc=None):
        self.context = context
        self.until = until
        self.doc = doc

    def __iter__(self):
        return iter((self.context, self.until))

def main():
    logger.info("Testing the greedy_until method for handling document queries")
    
    # Check if API key is set
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        logger.error("TOGETHER_API_KEY environment variable not set")
        return
    
    # Create a mock document
    doc = {
        "query": "What is the capital of France?",
        "text": "This is some text about France"
    }
    
    # Create a mock request with empty context but with doc attached
    request = MockRequest(doc=doc)
    
    try:
        # Create a LocalChatCompletion instance
        model = LocalChatCompletion(
            base_url="https://api.together.xyz/v1/chat/completions",
            api_key=api_key,
            max_tokens=50
        )
        
        # Test greedy_until with an empty request that has a doc
        logger.info("Testing greedy_until with document")
        result = model.greedy_until([request])
        
        # Print the result
        logger.info(f"Result: {result[0][:50]}...")
        
        logger.info("Test completed successfully")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
