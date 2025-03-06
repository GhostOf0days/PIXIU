import logging
import os
from functools import cached_property
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple, Union
import traceback

# Try to import the register_model decorator, but don't fail if it's not available
try:
    from lm_eval.api.registry import register_model
    from lm_eval import utils
except ImportError:
    # Create a dummy decorator if register_model is not available
    def register_model(name):
        def decorator(cls):
            return cls
        return decorator
    
    # Create a simple dictionary parser if utils is not available
    class utils:
        @staticmethod
        def simple_parse_args_string(arg_string):
            """Parse a comma-separated string of key=value pairs into a dictionary."""
            if arg_string == "":
                return {}
            return {k.strip(): v.strip() for k, v in 
                   [item.split("=", 1) for item in arg_string.split(",")]}

try:
    from lm_eval.models.api_models import TemplateAPI
except ImportError:
    # Create a dummy base class if TemplateAPI is not available
    class TemplateAPI:
        def __init__(self, **kwargs):
            # Initialize cache_hook attribute
            self.cache_hook = None
            
            # Initialize common attributes
            self.base_url = kwargs.get('base_url')
            self.model = kwargs.get('model')
            self.tokenizer_backend = kwargs.get('tokenizer_backend')
            
            # Convert batch_size to int if it's a string
            batch_size = kwargs.get('batch_size', 1)
            if isinstance(batch_size, str):
                try:
                    batch_size = int(batch_size)
                except ValueError:
                    batch_size = 1
            self._batch_size = batch_size
            
            # Handle max_tokens as an alias for max_gen_toks
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
            
            # Store any additional attributes
            for key, value in kwargs.items():
                if not hasattr(self, key) and key not in ['max_tokens', 'batch_size', 'max_gen_toks']:
                    # Try to convert numeric strings to appropriate types
                    if isinstance(value, str):
                        # Try to convert to int first
                        try:
                            if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                                value = int(value)
                            # Then try float
                            elif value.replace('.', '', 1).isdigit() or (value.startswith('-') and value[1:].replace('.', '', 1).isdigit()):
                                value = float(value)
                        except (ValueError, AttributeError):
                            pass  # Keep as string if conversion fails
                    
                    setattr(self, key, value)

        def set_cache_hook(self, cache_hook):
            """Set the cache hook for caching API results.
            
            Args:
                cache_hook: The cache hook to use.
            """
            self.cache_hook = cache_hook

try:
    from lm_eval.models.utils import handle_stop_sequences
except ImportError:
    # Create a dummy function if handle_stop_sequences is not available
    def handle_stop_sequences(until, eos):
        return until or eos


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

    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        generate=False,
        gen_kwargs: Optional[dict] = None,
        seed: int = 1234,
        eos=None,
        **kwargs,
    ) -> dict:
        if generate:
            gen_kwargs.pop("do_sample", False)
            if "max_tokens" in gen_kwargs:
                max_tokens = gen_kwargs.pop("max_tokens")
            else:
                max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
            temperature = gen_kwargs.pop("temperature", 0)
            stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
            return {
                "prompt": messages,
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop,
                "seed": seed,
                **gen_kwargs,
            }
        else:
            return {
                "model": self.model,
                "prompt": messages,
                "temperature": 0,
                "max_tokens": 1,
                "logprobs": 1,
                "seed": seed,
                "echo": True,
            }

    @staticmethod
    def parse_logprobs(
        outputs: Union[Dict, List[Dict]],
        tokens: List[List[int]] = None,
        ctxlens: List[int] = None,
        **kwargs,
    ) -> List[Tuple[float, bool]]:
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
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            tmp = [None] * len(out["choices"])
            for choices in out["choices"]:
                tmp[choices["index"]] = choices["text"]
            res = res + tmp
        return res

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
        assert type(messages) is not str, (
            "chat-completions require the --apply_chat_template flag."
        )
        gen_kwargs.pop("do_sample", False)
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
        if not isinstance(stop, (list, tuple)):
            stop = [stop]
        return {
            "messages": messages,
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop[:4],
            "seed": seed,
            **gen_kwargs,
        }

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            tmp = [None] * len(out["choices"])
            for choices in out["choices"]:
                tmp[choices["index"]] = choices["message"]["content"]
            res = res + tmp
        return res

    def tok_encode(
        self,
        string: Union[str, Any],
        left_truncate_len=None,
        add_special_tokens=None,
        **kwargs,
    ) -> Union[List[str], List[int], Any]:
        return string

    def loglikelihood(self, requests, **kwargs):
        """
        Return a default loglikelihood response to prevent errors with CachingLM.
        
        This is a simplistic implementation that allows the CachingLM wrapper to function
        without raising errors, but doesn't provide real loglikelihood calculation.
        
        Args:
            requests: The requests for which to calculate loglikelihoods
            **kwargs: Additional keyword arguments
            
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

    def greedy_until(self, requests):
        """
        Generate text greedily until a stopping sequence is reached.
        
        Args:
            requests: List of tuples (context, until)
                context: String context for generation
                until: List of string sequences to generate until
                
        Returns:
            List of generated continuations
        """
        results = []
        import logging
        
        # Enable debug mode
        DEBUG_NER = os.environ.get("DEBUG_NER", "0") == "1"
        
        # Super verbose debug
        if DEBUG_NER:
            print(f"PIXIU DEBUG: Total requests: {len(requests)}")
            for i, req in enumerate(requests):
                print(f"PIXIU DEBUG: Request {i} type: {type(req)}")
                if hasattr(req, 'doc'):
                    print(f"PIXIU DEBUG: Request {i} has doc attribute")
                    if isinstance(req.doc, dict) and 'query' in req.doc:
                        print(f"PIXIU DEBUG: Request {i} doc has query key")
        
        for request in requests:
            context, until = request
            
            # Add explicit print statement to ensure visibility
            if DEBUG_NER:
                print(f"PIXIU DEBUG: Initial context: '{context[:20]}...' (len: {len(context) if context else 0})")
                print(f"PIXIU DEBUG: Request type: {type(request)}")
                print(f"PIXIU DEBUG: Has doc attr: {hasattr(request, 'doc')}")
            
            # For any task: Handle empty context by checking if there's a document with query
            if not context and hasattr(request, 'doc'):
                doc = request.doc
                if DEBUG_NER:
                    print(f"PIXIU DEBUG: Doc type: {type(doc)}")
                if isinstance(doc, dict) and 'query' in doc:
                    context = doc['query']
                    if DEBUG_NER:
                        print(f"PIXIU DEBUG: Extracted query from document! New context length: {len(context)}")
                    logging.info(f"Using document query from request.doc")
            
            # Additional check: if context is still empty, check if request has a query field directly
            if not context and hasattr(request, 'query'):
                context = request.query
                if DEBUG_NER:
                    print(f"PIXIU DEBUG: Extracted query directly from request.query! New context length: {len(context)}")
                logging.info(f"Using query directly from request")
            
            # Skip empty requests that can't be recovered
            if not context:
                if DEBUG_NER:
                    print("PIXIU DEBUG: Empty context with no document query available")
                logging.warning("Received empty request with no recoverable context, returning empty string")
                results.append("")
                continue
            
            # Convert the context to a message format for the API
            messages = [{"role": "user", "content": context}]
            
            # Use a reasonable max_tokens value that works for all tasks
            max_tokens = min(200, self._max_gen_toks) 
            
            try:
                # Create the payload
                payload = self._create_payload(
                    messages=messages,
                    generate=True,
                    gen_kwargs={"until": until, "max_tokens": max_tokens},
                    temperature=0.0  # Use temperature 0 for greedy generation
                )
                
                # Make the API request
                response = self._make_request(payload)
                
                # Parse the generated text
                generations = self.parse_generations(response)
                
                # Handle the case where generations might be empty
                if not generations or not generations[0]:
                    logging.warning("Received empty generation, returning empty string")
                    results.append("")
                else:
                    # Add to cache if available
                    result = generations[0]
                    if self.cache_hook is not None:
                        self.cache_hook.add_partial('greedy_until', (context, until), result)
                    results.append(result)
                
            except Exception as e:
                logging.error(f"Error in greedy_until: {e}")
                import traceback
                logging.error(traceback.format_exc())
                results.append("")
            
        return results

    def _make_request(self, payload):
        """
        Make an API request with the given payload.
        
        Args:
            payload: Dict containing the request payload
            
        Returns:
            Dict containing the API response
        """
        import requests
        import json
        import logging
        import traceback
        
        try:
            logging.debug(f"Making API request to: {self._base_url}")
            
            # Add API key to headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Make the request to the API
            response = requests.post(
                self._base_url,
                headers=headers,
                json=payload,
                timeout=120  # Increase timeout for larger requests
            )
            
            # Check if the request was successful
            if response.status_code != 200:
                error_message = f"API request failed with status code {response.status_code}: {response.text}"
                logging.error(error_message)
                # Return a structured error format that can be handled by parse methods
                return {
                    "error": error_message,
                    "choices": [{"message": {"content": ""}}]
                }
            
            # Parse the response
            result = response.json()
            logging.debug(f"API response received successfully")
            
            return result
            
        except Exception as e:
            logging.error(f"Error making API request: {str(e)}")
            logging.error(traceback.format_exc())
            # Return an error response that can be handled by parse methods
            return {
                "error": str(e),
                "choices": [{"message": {"content": ""}}]
            }


@register_model(
    "openai-completions",
)
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

    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError(
            "Loglikelihood (and therefore `multiple_choice`-type tasks) is not supported for chat completions as OpenAI does not provide prompt logprobs. See https://github.com/EleutherAI/lm-evaluation-harness/issues/942#issuecomment-1777836312 or https://github.com/EleutherAI/lm-evaluation-harness/issues/1196 for more background on this limitation."
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
        assert type(messages) is not str, (
            "chat-completions require the --apply_chat_template flag."
        )
        gen_kwargs.pop("do_sample", False)
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", ["<|endoftext|>"]), eos)
        if not isinstance(stop, (list, tuple)):
            stop = [stop]
        output = {
            "messages": messages,
            "model": self.model,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop[:4],
            "seed": seed,
            **gen_kwargs,
        }
        if "o1" in self.model:
            output.pop("stop")
            output["temperature"] = 1
        elif "o3" in self.model:
            output.pop("temperature")
        return output 