# Together AI Integration for PIXIU

This document describes the integration of Together AI with the PIXIU financial evaluation framework. Together AI allows you to use various AI models through their API, which is compatible with the OpenAI API format.

## Changes Made

We have enhanced the PIXIU framework to better support Together AI integration:

1. **Fixed Import Issues**:

   - Added handling for missing `AutoMLM` and `AutoPrefixLM` attributes in Hugging Face models
   - Added graceful error handling for missing imports

2. **Enhanced API Key Handling**:

   - Added the ability to provide the API key directly in the model arguments
   - Implemented logic to check for provided API keys before falling back to environment variables

3. **Added Missing Functionality**:

   - Implemented the `create_from_arg_string` method for both `LocalChatCompletion` and `OpenAIChatCompletion` classes
   - Added proper parsing of argument strings to create model instances

4. **Fixed Initialization Issues**:
   - Ensured proper initialization of the model classes through the inheritance chain
   - Added support for `max_tokens` as an alias for `max_gen_toks`

## How to Use

You can now run evaluations with Together AI models in two ways:

### Option 1: Using Environment Variables

```bash
# Set your Together AI API key as an environment variable
export TOGETHER_API_KEY=your_api_key_here

# Run the evaluation
python src/eval.py \
    --model local-chat-completions \
    --model_args "model=deepseek-ai/DeepSeek-V3,base_url=https://api.together.xyz/v1/chat/completions,max_tokens=25" \
    --tasks flare_es_multifin \
    --batch_size 20000 \
    --num_fewshot 0 \
    --apply_chat_template
```

### Option 2: Passing the API Key Directly

```bash
python src/eval.py \
    --model local-chat-completions \
    --model_args "model=deepseek-ai/DeepSeek-V3,base_url=https://api.together.xyz/v1/chat/completions,api_key=your_api_key_here,max_tokens=25" \
    --tasks flare_es_multifin \
    --batch_size 20000 \
    --num_fewshot 0 \
    --apply_chat_template
```

## Important Notes

1. The `base_url` must include `/chat/completions` at the end
2. The `api_key` parameter can be passed directly in the model_args
3. Don't forget to include the `--apply_chat_template` flag when using chat models
4. You can find available models at the [Together AI Model Library](https://docs.together.ai/docs/inference-models)

## Troubleshooting

If you encounter errors related to the environment:

1. **NumPy Compatibility Issues**: If you see "numpy.dtype size changed, may indicate binary incompatibility", try creating a fresh virtual environment and installing the required packages.

2. **TensorFlow Warnings**: To disable TensorFlow custom operations warnings, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`

3. **Import Errors**: The code now has improved error handling for missing modules, but if you encounter additional import issues, make sure all dependencies are installed correctly.

4. **Authentication Failures**: Double-check your API key and ensure you're using the correct base URL for the Together AI API.

## Testing

A comprehensive test suite has been implemented to verify the functionality of the Together AI integration. You can run the tests with:

```bash
python src/test_final_implementation.py
```
