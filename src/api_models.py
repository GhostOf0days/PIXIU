import logging
import os
from functools import cached_property
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple, Union
import traceback
import requests
import json
import sys

# Add financial-evaluation to Python path
src_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(src_dir)
financial_eval_path = os.path.join(parent_dir, "financial-evaluation")
if os.path.exists(financial_eval_path) and financial_eval_path not in sys.path:
    sys.path.insert(0, financial_eval_path)

# Try importing from lm_eval with explicit error handling
try:
    from lm_eval.api.registry import register_model
    from lm_eval import utils
    from lm_eval.models.api_models import TemplateAPI
    from lm_eval.models.utils import handle_stop_sequences
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current sys.path: {sys.path}")
    
    # Create fallback implementations
    def register_model(name):
        def decorator(cls):
            print(f"Registering model {name} using fallback implementation")
            return cls
        return decorator
    
    # Import local utils or create simple version
    try:
        # Try to import utils from the current directory
        from utils import simple_parse_args_string
        
        class utils:
            @staticmethod
            def simple_parse_args_string(arg_string):
                return simple_parse_args_string(arg_string)
    except ImportError:
        # Fallback to a simple implementation
        class utils:
            @staticmethod
            def simple_parse_args_string(arg_string):
                """Parse a comma-separated string of key=value pairs into a dictionary."""
                if arg_string == "":
                    return {}
                return {k.strip(): v.strip() for k, v in 
                       [item.split("=", 1) for item in arg_string.split(",")]}
    
    # Create a simple TemplateAPI class
    class TemplateAPI:
        def __init__(self, **kwargs):
            # Initialize cache_hook attribute
            self.cache_hook = None
            
            # Store base_url
            self._base_url = kwargs.get('base_url')
            
            # Store model name
            self.model = kwargs.get('model')
            
            # Store tokenizer backend
            self.tokenizer_backend = kwargs.get('tokenizer_backend')
            
            # Handle batch_size
            batch_size = kwargs.get('batch_size', 1)
            if isinstance(batch_size, str):
                try:
                    batch_size = int(batch_size)
                except ValueError:
                    batch_size = 1
            self._batch_size = batch_size
            
            # Handle max_tokens/max_gen_toks
            max_gen_toks = 32
            if 'max_tokens' in kwargs:
                max_tokens = kwargs.get('max_tokens')
                if isinstance(max_tokens, str):
                    try:
                        max_tokens = int(max_tokens)
                    except ValueError:
                        max_tokens = 32
                max_gen_toks = max_tokens
            elif 'max_gen_toks' in kwargs:
                max_gen_toks_val = kwargs.get('max_gen_toks')
                if isinstance(max_gen_toks_val, str):
                    try:
                        max_gen_toks = int(max_gen_toks_val)
                    except ValueError:
                        max_gen_toks = 32
            
            self._max_gen_toks = max_gen_toks
            
            # Store additional attributes
            for key, value in kwargs.items():
                if not hasattr(self, key) and key not in ['max_tokens', 'batch_size', 'max_gen_toks']:
                    setattr(self, key, value)

        def set_cache_hook(self, cache_hook):
            """Set the cache hook for caching API results."""
            self.cache_hook = cache_hook
    
    # Simple implementation of handle_stop_sequences
    def handle_stop_sequences(until, eos):
        """Handle stop sequences for generation."""
        if until is None:
            return eos
        return until


# Configure logging
eval_logger = logging.getLogger(__name__)


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
        # Ensure base_url is accessible
        if base_url:
            self._base_url = base_url
    
    @property
    def base_url(self):
        """Get the base URL for API requests."""
        return getattr(self, '_base_url', None)
        
    @base_url.setter
    def base_url(self, value):
        """Set the base URL for API requests."""
        self._base_url = value

    def _create_payload(self, messages, generate=True, gen_kwargs=None, logit_bias=None):
        """
        Create the payload for the API request.
        
        Args:
            messages: List of message objects
            generate: Whether to generate text or calculate logprobs
            gen_kwargs: Generation parameters
            logit_bias: Logit bias parameters
            
        Returns:
            Dict containing the API request payload
        """
        # Enable debug mode
        DEBUG_NER = os.environ.get("DEBUG_NER", "0") == "1"
        gen_kwargs = gen_kwargs or {}
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,  # Default to 0.0 for deterministic outputs
        }
        
        # Add generation parameters if provided
        if generate:
            # Add max_tokens parameter
            if "max_tokens" in gen_kwargs:
                payload["max_tokens"] = gen_kwargs["max_tokens"]
            elif hasattr(self, "_max_gen_toks"):
                payload["max_tokens"] = self._max_gen_toks
            
            # Add stop parameter if provided
            if "until" in gen_kwargs and gen_kwargs["until"]:
                # Convert to list if not already
                until = gen_kwargs["until"]
                if not isinstance(until, list):
                    until = [until]
                
                # Filter out None values and empty strings
                until = [u for u in until if u]
                
                # Add to payload if not empty
                if until:
                    payload["stop"] = until
            
            # Add other parameters
            if "temperature" in gen_kwargs:
                payload["temperature"] = gen_kwargs["temperature"]
                
            if "top_p" in gen_kwargs:
                payload["top_p"] = gen_kwargs["top_p"]
                
            if "top_k" in gen_kwargs:
                payload["top_k"] = gen_kwargs["top_k"]
        
        # Add logit bias if provided
        if logit_bias:
            payload["logit_bias"] = logit_bias
        
        if DEBUG_NER:
            logging.info(f"Created payload for API request: {payload}")
        
        return payload

    @staticmethod
    def parse_logprobs(
        outputs: Union[Dict, List[Dict]],
        tokens: List[List[int]] = None,
        ctxlens: List[int] = None,
        **kwargs,
    ) -> List[Tuple[float, bool]]:
        """
        Parse logprobs from the API response.
        
        Args:
            outputs: API response
            tokens: List of token lists
            ctxlens: List of context lengths
            
        Returns:
            List of (logprob, is_greedy) tuples
        """
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            for choice, ctxlen in zip(
                sorted(out["choices"], key=itemgetter("index")), ctxlens
            ):
                assert ctxlen > 0, "Context length must be greater than 0"
                logprobs = sum(choice["logprobs"]["token_logprobs"][ctxlen:-1])
                tokens_logprobs = choice["logprobs"]["token_logprobs"][ctxlen:-1]
                top_logprobs = choice["logprobs"]["top_logprobs"][ctxlen:-1]
                is_greedy = True
                for tok, top in zip(tokens_logprobs, top_logprobs):
                    if tok != max(top.values()):
                        is_greedy = False
                        break
                res.append((logprobs, is_greedy))
        return res

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        """
        Parse the generations from the API response.
        
        Args:
            outputs: API response
            
        Returns:
            List of generated text strings
        """
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
            
        for out in outputs:
            # Check for error field in response
            if "error" in out and out["error"]:
                logging.error(f"API error: {out['error']}")
                res.append("")
                continue
                    
            # Check if there are any choices
            if "choices" not in out or not out["choices"]:
                logging.error("No choices in API response")
                res.append("")
                continue
            
            # Create a placeholder array sized to match the number of choices
            tmp = [None] * len(out["choices"])
            
            # Extract the content from each choice based on the response format
            for choice in out["choices"]:
                # Get the index to place this result
                idx = choice.get("index", 0)
                
                # Extract the content based on whether this is a chat or completion API
                if "message" in choice and isinstance(choice["message"], dict) and "content" in choice["message"]:
                    # This is the chat completions API format
                    tmp[idx] = choice["message"]["content"]
                elif "text" in choice:
                    # This is the completions API format
                    tmp[idx] = choice["text"]
                else:
                    # Couldn't determine the format, log a warning and continue
                    logging.warning(f"Could not extract content from choice: {choice}")
                    tmp[idx] = ""
            
            # Add the results to the output list
            res.extend([t if t is not None else "" for t in tmp])
            
        return res

    @property
    def api_key(self):
        """Get the API key for the API request."""
        # First check if we need a specific provider's key
        if hasattr(self, 'base_url') and self.base_url:
            if "together.xyz" in self.base_url:
                key = os.environ.get("TOGETHER_API_KEY", None)
                if key:
                    return key
            
        # Fall back to default OpenAI key
        return os.environ.get("OPENAI_API_KEY", "")

    def _make_request(self, payload):
        try:
            # Get base URL and API key
            base_url = self.base_url
            api_key = self.api_key
            
            # Debug output
            print(f"DEBUG: Using API key: {api_key[:5]}...{api_key[-5:]}")
            print(f"DEBUG: Making request to: {base_url}")
            
            # Add API key to headers
            headers = {
                "Content-Type": "application/json", 
                "Authorization": f"Bearer {api_key}"
            }
            
            # Make the request to the API
            response = requests.post(
                base_url,
                headers=headers,
                json=payload,
                timeout=180  # Increased timeout for larger requests
            )
            
            # Check if the request was successful
            if response.status_code != 200:
                error_message = f"API request failed with status code {response.status_code}: {response.text}"
                logging.error(error_message)
                # Return a structured error format that can be handled by parse methods
                return {
                    "error": error_message,
                    "choices": []
                }
            
            # Parse the response
            result = response.json()
            
            if DEBUG_NER:
                try:
                    logging.info(f"API response received: {json.dumps(result, indent=2)}")
                except:
                    logging.info(f"API response received (not serializable to JSON)")
            
            # Handle Together AI specific format adjustments
            if hasattr(self, 'base_url') and self.base_url and "together.xyz" in self.base_url:
                # Make sure the response has the expected format
                if "choices" in result:
                    # Ensure each choice has an index
                    for i, choice in enumerate(result["choices"]):
                        if "index" not in choice:
                            choice["index"] = i
                            
                    # Print the content of the response if debugging
                    if DEBUG_NER and result["choices"]:
                        try:
                            choice = result["choices"][0]
                            if "message" in choice and "content" in choice["message"]:
                                content = choice["message"]["content"]
                                logging.info(f"First response content: {content}")
                            elif "text" in choice:
                                content = choice["text"]
                                logging.info(f"First response content: {content}")
                        except Exception as e:
                            logging.error(f"Error extracting content from response: {e}")
            
            return result
            
        except Exception as e:
            logging.error(f"Error in API request: {e}")
            if DEBUG_NER:
                traceback.print_exc()
            
            # Return a structured error response
            return {
                "error": str(e),
                "choices": []
            }


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
        
        # Store base_url directly to ensure it's accessible
        self._base_url = base_url
        
        # Warning about chat template flag
        eval_logger.warning(
            "chat-completions endpoint requires the `--apply_chat_template` flag."
        )
        
        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            **kwargs,
        )
        self.tokenized_requests = tokenized_requests
        self.apply_chat_template = kwargs.get('apply_chat_template', True)
        
        if self._batch_size > 1:
            eval_logger.warning(
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

    def _create_payload(
        self,
        messages: List[Dict],
        generate=False,
        gen_kwargs: dict = None,
        seed=1234,
        eos=None,
        **kwargs,
    ) -> dict:
        """
        Create the payload for the chat completions API request.
        
        Args:
            messages: List of message dictionaries
            generate: Whether to generate text
            gen_kwargs: Generation parameters
            seed: Random seed
            eos: End of sequence token
            
        Returns:
            Dict containing the API request payload
        """
        assert type(messages) is not str, (
            "chat-completions require the --apply_chat_template flag."
        )
        gen_kwargs = gen_kwargs or {}
        gen_kwargs.pop("do_sample", False)
        
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
            
        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
        
        if not isinstance(stop, (list, tuple)):
            stop = [stop]
            
        # Filter out None and empty strings
        stop_list = [s for s in stop if s]
        
        payload = {
            "messages": messages,
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # Only add stop if there are valid stop sequences
        if stop_list:
            payload["stop"] = stop_list[:4]  # API limit is 4 stop sequences
            
        # Add seed for reproducibility
        payload["seed"] = seed
        
        # Add any remaining kwargs
        for k, v in gen_kwargs.items():
            payload[k] = v
            
        return payload

    def tok_encode(
        self,
        string: Union[str, Any],
        left_truncate_len=None,
        add_special_tokens=None,
        **kwargs,
    ) -> Union[List[str], List[int], Any]:
        """Pass through the string as is for chat completions."""
        return string

    def loglikelihood(self, requests, **kwargs):
        """
        Return a default loglikelihood response to prevent errors with CachingLM.
        
        Args:
            requests: The requests for which to calculate loglikelihoods
            
        Returns:
            A list of tuples containing (logprob, is_greedy) values
        """
        eval_logger.warning(
            "Loglikelihood calculation is not fully supported for chat completions. "
            "Returning default values (-1.0, False) for each request."
        )
        
        # Return a default response for each request
        return [(-1.0, False) for _ in requests]

    @property
    def api_key(self):
        """Get the API key for the API request."""
        if hasattr(self, 'provided_api_key') and self.provided_api_key:
            print(f"DEBUG: Using provided_api_key: {self.provided_api_key[:5]}...{self.provided_api_key[-5:]}")
            return self.provided_api_key
            
        # Check Together API key from environment
        together_key = os.environ.get("TOGETHER_API_KEY")
        if together_key:
            print(f"DEBUG: Using TOGETHER_API_KEY from env: {together_key[:5]}...{together_key[-5:]}")
            return together_key
            
        # Try OpenAI key as fallback
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            print(f"DEBUG: Using OPENAI_API_KEY: {openai_key[:5]}...{openai_key[-5:]}")
            return openai_key
            
        raise ValueError(
            "Please set the TOGETHER_API_KEY environment variable or provide an api_key parameter"
        )

    def greedy_until(self, requests):
        """
        Generate text greedily until a stopping sequence is reached.
        
        Args:
            requests: List of Request objects from the lm-evaluation-harness
                
        Returns:
            List of generated continuations
        """
        results = []
        
        # Enable debug mode
        DEBUG_NER = os.environ.get("DEBUG_NER", "0") == "1"
        
        if DEBUG_NER:
            logging.info(f"Processing {len(requests)} requests")
        
        for request in requests:
            context = None
            until = None
            
            # 1. Try to extract context and until from the request
            try:
                # Try to unpack as a tuple (context, until_args)
                context, until_args = request
                if isinstance(until_args, dict):
                    until = until_args.get("until", None)
                else:
                    until = until_args
                if DEBUG_NER:
                    logging.info(f"Extracted context as tuple: {context[:50] if context else None}...")
            except (ValueError, TypeError, AttributeError):
                # Not a tuple, try other methods
                if DEBUG_NER:
                    logging.info(f"Request type: {type(request)}")
                    if hasattr(request, '__dict__'):
                        logging.info(f"Request attributes: {request.__dict__}")
                    else:
                        logging.info(f"Request does not have __dict__")
                
                # 2. Try to access request.args (for RequestFactory objects)
                if hasattr(request, 'args') and request.args:
                    if DEBUG_NER:
                        logging.info(f"Request has args: {request.args}")
                    context = request.args[0] if len(request.args) > 0 else None
                    if len(request.args) > 1:
                        if isinstance(request.args[1], dict):
                            until = request.args[1].get("until", None)
                        else:
                            until = request.args[1]
                
                # 3. Try direct attribute access
                elif hasattr(request, 'context'):
                    context = request.context
                    until = getattr(request, 'until', None)
                    if DEBUG_NER:
                        logging.info(f"Used direct attribute access, context: {context[:50] if context else None}")
            
            # 4. Special handling for tasks like NER: check for document with query
            if hasattr(request, 'doc'):
                doc = request.doc
                if DEBUG_NER:
                    logging.info(f"Request has doc attribute: {type(doc)}")
                    if isinstance(doc, dict):
                        logging.info(f"Doc keys: {list(doc.keys())}")
                
                # For NER, the query is in the doc
                if isinstance(doc, dict) and 'query' in doc:
                    # For all tasks with 'query' in doc, use it as context
                    context = doc['query']
                    if DEBUG_NER:
                        logging.info(f"Using query from doc: {context[:100] if context else None}...")
            
            # Skip empty requests
            if not context:
                logging.warning("Received empty request with no recoverable context, returning empty string")
                results.append("")
                continue
            
            if DEBUG_NER:
                logging.info(f"Final context: {context[:100] if context else None}...")
            
            # Convert the context to a message format for the API
            messages = [{"role": "user", "content": context}]
            
            # Use a reasonable max_tokens value for generation tasks
            max_tokens = min(1024, self._max_gen_toks)
            
            try:
                # Create the payload
                payload = self._create_payload(
                    messages=messages,
                    generate=True,
                    gen_kwargs={"until": until, "max_tokens": max_tokens},
                    temperature=0.0
                )
                
                if DEBUG_NER:
                    logging.info(f"API payload: {json.dumps(payload, indent=2)}")
                
                # Make the API request
                response = self._make_request(payload)
                
                if DEBUG_NER:
                    logging.info(f"API response received")
                
                # Parse the generated text
                generations = self.parse_generations(response)
                
                if DEBUG_NER:
                    logging.info(f"Parsed generations: {generations}")
                
                # Handle empty generations
                if not generations or len(generations) == 0:
                    logging.warning("Received empty generations from API")
                    results.append("")
                    continue
                
                # Get the first generation
                generation = generations[0]
                
                # Process until sequences (stop sequences)
                if until:
                    for term in until:
                        if term and term in generation:
                            generation = generation.split(term)[0]
                
                # Add the result
                results.append(generation)
                
            except Exception as e:
                logging.error(f"Error in greedy_until: {e}")
                if DEBUG_NER:
                    traceback.print_exc()
                results.append("")
        
        return results


@register_model("openai-completions")
class OpenAICompletionsAPI(LocalCompletionsAPI):
    def __init__(
        self,
        base_url="https://api.openai.com/v1/completions",
        tokenizer_backend="tiktoken",
        **kwargs,
    ):
        super().__init__(
            base_url=base_url, tokenizer_backend=tokenizer_backend, **kwargs
        )

    @cached_property
    def api_key(self):
        """Override this property to return the API key for the API request."""
        key = os.environ.get("OPENAI_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the `OPENAI_API_KEY` environment variable."
            )
        return key

    def loglikelihood(self, requests, **kwargs):
        assert self.model in [
            "babbage-002",
            "davinci-002",
        ], (
            f"Prompt loglikelihoods are only supported by OpenAI's API for {['babbage-002', 'davinci-002']}."
        )
        return super().loglikelihood(requests, **kwargs)

    def chat_template(self, chat_template: Union[bool, str] = False) -> Optional[str]:
        return ""


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
        
        # Warning about response_format for O1 models
        if "o1" in kwargs.get("model", ""):
            eval_logger.warning(
                "For O1 models, consider using 'response_format={'type': 'text'}' for better quality responses."
            )
            
        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )
    
    @cached_property
    def api_key(self):
        # First check if API key was provided directly in the constructor
        if hasattr(self, 'provided_api_key') and self.provided_api_key:
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
        
    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        """
        Create an instance from a string of arguments.
        """
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        
        # Extract api_key if present to ensure it's correctly passed
        api_key = args.pop('api_key', None)
        
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(api_key=api_key, **args, **args2)

    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError(
            "Loglikelihood (and therefore `multiple_choice`-type tasks) is not supported for chat completions as OpenAI does not provide prompt logprobs."
        )

    def _create_payload(
        self,
        messages: List[Dict],
        generate=False,
        gen_kwargs: dict = None,
        seed=1234,
        eos="<|endoftext|>",
        **kwargs,
    ) -> dict:
        """Create payload for the OpenAI chat completions API."""
        assert type(messages) is not str, (
            "chat-completions require the --apply_chat_template flag."
        )
        gen_kwargs = gen_kwargs or {}
        gen_kwargs.pop("do_sample", False)
        
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
            
        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", ["<|endoftext|>"]), eos)
        
        if not isinstance(stop, (list, tuple)):
            stop = [stop]
            
        # Filter out None and empty strings
        stop_list = [s for s in stop if s]
        
        payload = {
            "messages": messages,
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # Only add stop if there are valid stop sequences
        if stop_list:
            payload["stop"] = stop_list[:4]  # API limit is 4 stop sequences
            
        # Add any remaining kwargs
        for k, v in gen_kwargs.items():
            payload[k] = v
            
        # Special case handling for different OpenAI model versions
        if "o1" in self.model:
            if "stop" in payload:
                del payload["stop"]
            payload["temperature"] = 1
        elif "o3" in self.model:
            if "temperature" in payload:
                del payload["temperature"]
                
        return payload