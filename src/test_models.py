#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify that all models are properly registered.
Run this script to check if both the original models and our new API models are available.
"""

import os
import sys
import importlib.util

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# First try to import our API models
try:
    from src.api_models import LocalCompletionsAPI, LocalChatCompletion, OpenAICompletionsAPI, OpenAIChatCompletion
    print("✅ Successfully imported API models from src.api_models")
except ImportError as e:
    print(f"❌ Failed to import API models: {e}")

# Now check if the models are registered
try:
    # Import the model registry
    # Use importlib to import from a path with hyphens
    fin_eval_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "financial-evaluation")
    sys.path.append(fin_eval_path)
    
    try:
        from lm_eval.models import MODEL_REGISTRY, get_model
    except ImportError:
        print("❌ Could not import directly, using importlib")
        
        # Try using importlib
        spec = importlib.util.spec_from_file_location(
            "models",
            os.path.join(fin_eval_path, "lm_eval", "models", "__init__.py")
        )
        models_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(models_module)
        
        MODEL_REGISTRY = models_module.MODEL_REGISTRY
        get_model = models_module.get_model
    
    # Print all registered models
    print("\n📋 Registered models:")
    for idx, (model_name, model_class) in enumerate(MODEL_REGISTRY.items(), 1):
        print(f"{idx}. {model_name}: {model_class.__module__}.{model_class.__name__}")
    
    # Test accessing specific models
    test_models = [
        "hf-causal-vllm", 
        "hf-chatglm", 
        "hf-causal-llama",
        "local-chat-completions",
        "openai-chat-completions"
    ]
    
    print("\n🧪 Testing model access:")
    for model_name in test_models:
        try:
            model_class = get_model(model_name)
            print(f"✅ Successfully accessed model: {model_name} → {model_class.__module__}.{model_class.__name__}")
        except KeyError:
            print(f"❌ Failed to access model: {model_name} (not registered)")
        except Exception as e:
            print(f"❌ Error accessing model: {model_name} - {str(e)}")
            
except Exception as e:
    print(f"❌ Error testing model registry: {e}")

print("\n✨ Test completed") 