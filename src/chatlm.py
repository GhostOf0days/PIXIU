import os
import asyncio
import numpy as np
import transformers
from lm_eval.base import BaseLM
from lm_eval import utils
from tqdm import tqdm
import time
import logging
import json

BACKOFF_TIME = 0.1

async def single_chat(client, **kwargs):
    global BACKOFF_TIME
    backoff_time = BACKOFF_TIME
    while True:
        try:
            r = await client.post(**kwargs, timeout=20)
            json_response = r.json()
            s = json_response['choices'][0]["message"]['content']
            time.sleep(backoff_time)
            return s
        except Exception:
            import traceback

            traceback.print_exc()
            time.sleep(backoff_time * 30)
            BACKOFF_TIME *= 1.05


async def oa_completion(**kwargs):
    """Query OpenAI API for completion.

    Retry with back-off until they respond
    """
    import httpx

    async with httpx.AsyncClient() as client:
        tasks = [single_chat(
            client=client,
            url=kwargs["url"], headers=kwargs["headers"],
            json={
                "temperature": kwargs["temperature"], "max_tokens": kwargs["max_tokens"],
                "model": kwargs["model"], "messages": [message,],
            }
        ) for message in kwargs["messages"]]
        results = await asyncio.gather(*tasks)
        return results


class ChatLM(BaseLM):
    REQ_CHUNK_SIZE = 20

    def __init__(self, model, truncate=False):
        """

        :param model: str
        :param truncate: bool
            Truncate input if too long (if False and input is too long, throw error)
        """
        super().__init__()

        import openai

        self.model = model
        self.truncate = truncate
        # Read from environment variable OPENAI_API_SECRET_KEY
        api_key = os.environ["OPENAI_API_SECRET_KEY"]
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        # Note: the OpenAI API supports up to 2049 tokens, with the first token being the first input token
        return 4096

    @property
    def max_gen_toks(self):
        return 10

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        raise NotImplementedError()

    def greedy_until(self, requests):
        if not requests:
            return []
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return len(toks), x[0]

        re_ord = utils.Reorderer(requests, _collate)

        def sameuntil_chunks(xs, size):
            ret = []
            lastuntil = "</s>"
            for x in xs:
                if len(ret) >= size:
                    yield ret, lastuntil
                    ret = []
                    lastuntil = "</s>"
                ret.append(x)

            if ret:
                yield ret, lastuntil

        # todo: more intelligent batching for heterogeneous `until`
        for chunk, until in tqdm(
            list(sameuntil_chunks(re_ord.get_reordered(), self.REQ_CHUNK_SIZE))
        ):
            inps = []
            for context in chunk:
                inps.append(context[0])

            responses = asyncio.run(oa_completion(
                url="https://api.openai.com/v1/chat/completions",
                headers=self.headers,
                model=self.model,
                messages=[{"role": "user", "content": inp} for inp in inps],
                max_tokens=self.max_gen_toks,
                temperature=0.0,
                # stop=until,
            ))

            for resp, context in zip(responses, chunk):
                s = resp

                # partial caching
                self.cache_hook.add_partial("greedy_until", (context, "</s>"), s)

                res.append(s)

        return re_ord.get_original(res)

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override greedy_until
        raise NotImplementedError()


class OpenAIChatCompletionLM(BaseLM):
    """
    Class for OpenAI's ChatCompletion API (ChatGPT, GPT-4, etc.)
    Requires the OPENAI_API_KEY environment variable to be set
    """
    REQ_CHUNK_SIZE = 20

    def __init__(
        self,
        model,
        base_url="https://api.openai.com/v1/chat/completions",
        max_tokens=256,
        temperature=0.0,
        top_p=1.0,
        num_concurrent=1,
        max_retries=5,
        batch_size=1,
        apply_chat_template=True,
        **kwargs
    ):
        """
        :param model: str
            The specific OpenAI model to use (e.g., "gpt-4-turbo", "gpt-3.5-turbo", etc.)
        :param base_url: str
            API endpoint URL
        :param max_tokens: int
            Maximum number of tokens to generate
        :param temperature: float
            Sampling temperature
        :param top_p: float
            Nucleus sampling parameter
        :param num_concurrent: int
            Number of concurrent requests to make
        :param max_retries: int
            Maximum number of retries on rate limit or error
        :param batch_size: int
            Batch size for requests (will be set to 1 as chat completions doesn't support batching)
        :param apply_chat_template: bool
            Whether to format prompts using chat template
        """
        super().__init__()
        
        try:
            import openai
        except ImportError:
            raise ImportError("OpenAI package not found. Please install it using 'pip install openai'.")
        
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.num_concurrent = num_concurrent
        self.max_retries = max_retries
        self._apply_chat_template = apply_chat_template
        
        # Chat completions doesn't support batching
        if batch_size > 1:
            logging.warning("Chat completions does not support batching. Defaulting to batch size 1.")
            batch_size = 1
        self._batch_size = batch_size
        
        # Get API key from environment variable
        self.api_key = os.environ.get("OPENAI_API_KEY", None)
        if self.api_key is None:
            raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
        
        # Set up the client
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)
        
        # Set up the tokenizer for encoding inputs
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
    
    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id
    
    @property
    def max_length(self):
        # Model dependent, but for most modern GPT models this is a reasonable default
        return 8192
    
    @property
    def max_gen_toks(self):
        return self.max_tokens
    
    @property
    def batch_size(self):
        return self._batch_size
    
    @property
    def device(self):
        # Not applicable for API models
        return "cpu"
    
    def tok_encode(self, string: str):
        """Tokenize a string"""
        return self.tokenizer.encode(string, add_special_tokens=False)
    
    def tok_decode(self, tokens):
        """Decode tokens to a string"""
        return self.tokenizer.decode(tokens)
    
    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        """
        Not supported for chat completions as OpenAI doesn't provide prompt logprobs.
        """
        raise NotImplementedError(
            "Loglikelihood (and therefore multiple_choice-type tasks) is not supported "
            "for chat completions as OpenAI does not provide prompt logprobs."
        )
    
    def greedy_until(self, requests):
        """
        Generates text from each prompt in requests until a stopping criterion is met.
        
        :param requests: List of tuples (prompt, list_of_stops)
        :return: List of generated texts
        """
        if not requests:
            return []
        
        res = []
        
        # Sort requests by length for better batching
        def _collate(x):
            toks = self.tok_encode(x[0])
            return len(toks), x[0]
        
        re_ord = utils.Reorderer(requests, _collate)
        
        def format_message(text):
            """Format prompt text into a ChatGPT message"""
            if self._apply_chat_template:
                # Simple chat template format for OpenAI models
                return [{"role": "user", "content": text}]
            else:
                # Return raw text for models that don't need chat template
                return text
        
        def handle_stop_sequences(stop_sequences, eos=None):
            """Format stop sequences for API call"""
            if stop_sequences is None:
                if eos is not None:
                    return [eos]
                return []
            
            if isinstance(stop_sequences, str):
                return [stop_sequences]
            
            return list(stop_sequences)
        
        # Process requests in batches (though batch size will be 1 for chat completions)
        for request_batch in tqdm(
            [re_ord.get_reordered()[i:i+self.REQ_CHUNK_SIZE] 
             for i in range(0, len(re_ord.get_reordered()), self.REQ_CHUNK_SIZE)],
            disable=disable_tqdm
        ):
            batch_responses = []
            
            for request in request_batch:
                prompt, stop_sequences = request
                
                # Format the prompt as a message
                messages = format_message(prompt)
                
                # Handle stop sequences
                stops = handle_stop_sequences(stop_sequences, eos="<|endoftext|>")
                
                # Prepare the API call parameters
                params = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_gen_toks,
                    "temperature": self.temperature,
                    "top_p": self.top_p
                }
                
                # Add stop sequences if provided
                if stops:
                    params["stop"] = stops[:4]  # OpenAI API accepts at most 4 stop sequences
                
                # Call the API with retries
                response = None
                retry_count = 0
                
                while response is None and retry_count < self.max_retries:
                    try:
                        response = self.client.chat.completions.create(**params)
                    except Exception as e:
                        retry_count += 1
                        if retry_count >= self.max_retries:
                            logging.error(f"Failed to get response after {self.max_retries} retries. Error: {e}")
                            batch_responses.append("")
                            break
                        
                        wait_time = 2 ** retry_count  # Exponential backoff
                        logging.warning(f"Error in API call: {e}. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                
                if response is not None:
                    # Extract the generated text
                    generated_text = response.choices[0].message.content
                    batch_responses.append(generated_text)
                    
                    # Add to cache
                    self.cache_hook.add_partial("greedy_until", (prompt, stop_sequences), generated_text)
            
            res.extend(batch_responses)
        
        return re_ord.get_original(res)
    
    def _model_call(self, inps):
        # Not used because we override _loglikelihood_tokens
        raise NotImplementedError()
    
    def _model_generate(self, context, max_length, eos_token_id):
        # Not used because we override greedy_until
        raise NotImplementedError()
