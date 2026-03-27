import asyncio
import json
from openai import AsyncOpenAI
import os
import random
import numpy as np
import google.auth
import google.auth.transport.requests

from .base import LLMBase


class OpenAICredentialsRefresher:
    def __init__(self, **kwargs):
        self.client = AsyncOpenAI(**kwargs, api_key='PLACEHOLDER')
        self.creds, _ = google.auth.default(
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        self.request = google.auth.transport.requests.Request()

    def __getattr__(self, name):
        if not self.creds.valid or not self.creds.token:
            try:
                self.creds.refresh(self.request)
            except Exception as exc:
                raise RuntimeError(
                    "Unable to refresh auth: credential refresh call failed. "
                    "Check your Google Cloud credentials configuration and network connectivity."
                ) from exc

            if not self.creds.valid or not self.creds.token:
                raise RuntimeError(
                    "Unable to refresh auth: credentials remain invalid after refresh "
                    f"(valid={self.creds.valid}, token_present={bool(self.creds.token)}). "
                    "Verify your Google Cloud credentials and permissions."
                )

        self.client.api_key = self.creds.token
        return getattr(self.client, name)

# Set openai_api_key if there's secrets.json file

try:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, '..', 'secrets.json')) as f:
        data = json.load(f)
        aclient = AsyncOpenAI(api_key=data['ai_studio_key'], base_url='https://generativelanguage.googleapis.com/v1beta/openai/')
        client_provider = 'ai_studio'
except Exception as e:
    try:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, '..', 'secrets.json')) as f:
            data = json.load(f)
            aclient = AsyncOpenAI(api_key=data['openai_api_key'])
            client_provider = 'openai'
    except Exception as e:
        try:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            with open(os.path.join(dir_path, '..', 'secrets.json')) as f:
                data = json.load(f)
                aclient = AsyncOpenAI(api_key=data['openrouter_api_key'], base_url='https://openrouter.ai/api/v1')
                client_provider = 'openrouter'
        except:
            try:
                dir_path = os.path.dirname(os.path.realpath(__file__))
                with open(os.path.join(dir_path, '..', 'secrets.json')) as f:
                    data = json.load(f)
                    aclient = OpenAICredentialsRefresher(
                        base_url=f"https://aiplatform.googleapis.com/v1/projects/{data['vertex_project_id']}/locations/{data['vertex_location']}/endpoints/openapi"
                    )
                    client_provider = 'vertex'
            except:
                aclient = AsyncOpenAI()


def choose_provider(provider):
    global aclient
    global client_provider
    if provider == 'ai_studio':
        try:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            with open(os.path.join(dir_path, '..', 'secrets.json')) as f:
                data = json.load(f)
                aclient = AsyncOpenAI(api_key=data['ai_studio_key'], base_url='https://generativelanguage.googleapis.com/v1beta/openai/')
                client_provider = 'ai_studio'
        except:
            aclient = AsyncOpenAI()
    elif provider == 'vertex':
        try:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            with open(os.path.join(dir_path, '..', 'secrets.json')) as f:
                data = json.load(f)
                aclient = OpenAICredentialsRefresher(
                    base_url=f"https://aiplatform.googleapis.com/v1/projects/{data['vertex_project_id']}/locations/{data['vertex_location']}/endpoints/openapi"
                )
                client_provider = 'vertex'
        except:
            aclient = AsyncOpenAI()
    elif provider == 'openrouter':
        try:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            with open(os.path.join(dir_path, '..', 'secrets.json')) as f:
                data = json.load(f)
                aclient = AsyncOpenAI(api_key=data['openrouter_api_key'], base_url='https://openrouter.ai/api/v1')
                client_provider = 'openrouter'
        except:
            aclient = AsyncOpenAI(base_url='https://openrouter.ai/api/v1')
    else:
        try:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            with open(os.path.join(dir_path, '..', 'secrets.json')) as f:
                data = json.load(f)
                aclient = AsyncOpenAI(api_key=data['openai_api_key'])
                client_provider = 'openai'
        except:
            aclient = AsyncOpenAI()


def _extract_retry_after_seconds(exc):
    response = getattr(exc, 'response', None)
    headers = getattr(response, 'headers', None)
    if headers is None:
        return None

    retry_after = headers.get('retry-after')
    if retry_after is None:
        return None

    try:
        return max(0.0, float(retry_after))
    except Exception:
        return None


def _is_rate_limited_error(exc):
    status_code = getattr(exc, 'status_code', None)
    if status_code == 429:
        return True

    response = getattr(exc, 'response', None)
    if getattr(response, 'status_code', None) == 429:
        return True

    text = str(exc).lower()
    return '429' in text or 'too many requests' in text or 'rate limit' in text


def _retry_wait_seconds(exc, retry_count):
    retry_after = _extract_retry_after_seconds(exc)
    if retry_after is not None:
        return min(120.0, retry_after)

    wait_seconds = min(60.0, (2 ** min(retry_count, 6)) + random.uniform(0.0, 1.0))
    if _is_rate_limited_error(exc):
        wait_seconds = min(120.0, wait_seconds * 1.5)
    return wait_seconds


async def prompt_openai_single(model, prompt, n, **kwargs):
    global client_provider
    ct = 0
    n_retries = 30
    while ct <= n_retries:
        try:
            if client_provider == 'openrouter' or client_provider == 'ai_studio' or client_provider == 'vertex':
                responses = await asyncio.gather(*[aclient.completions.create(model=model, prompt=prompt, **kwargs) for _ in range(n)])
                return [x.text for response in responses for x in response.choices]
            else:
                response = await aclient.completions.create(model=model, prompt=prompt, n=n, **kwargs)
                return [x.text for x in response.choices]
        except Exception as e:
            ct += 1
            if ct > n_retries:
                raise
            wait_seconds = _retry_wait_seconds(e, ct)
            print(f'Exception occured: {e}')
            print(f'Retrying ({ct}/{n_retries}) after {wait_seconds:.2f} seconds')
            await asyncio.sleep(wait_seconds)


async def prompt_openai_chat_single(model, messages, n, **kwargs):
    global client_provider
    ct = 0
    n_retries = 10
    while ct <= n_retries:
        try:
            if client_provider == 'openrouter' or client_provider == 'ai_studio' or client_provider == 'vertex':
                responses = await asyncio.gather(*[aclient.chat.completions.create(model=model, messages=messages, **kwargs) for _ in range(n)])
                return [x.message.content for response in responses for x in response.choices]
            else:
                response = await aclient.chat.completions.create(model=model, messages=messages, n=n, **kwargs)
                return [x.message.content for x in response.choices]
        except Exception as e: 
            ct += 1
            if ct > n_retries:
                raise
            wait_seconds = _retry_wait_seconds(e, ct)
            print(f'Exception occured: {e}')
            print(f'Retrying ({ct}/{n_retries}) after {wait_seconds:.2f} seconds')
            await asyncio.sleep(wait_seconds)


class OpenAI_LLM(LLMBase):
    def __init__(self, model, prompt_single_func, formatter):
        self.full_model = model
        self.model = model.split('/')[-1]
        self.prompt_single_func = prompt_single_func
        self.info = {
            'input_tokens': 0,
            'output_tokens': 0,
            'calls': 0,
            'actual_input_tokens': 0,
            'actual_output_tokens': 0,
            'actual_calls': 0,
        }
        self.rng = np.random.default_rng(0)
        super().__init__(self.model, formatter)

    def handle_kwargs(self, kwargs):
        if 'temperature' not in kwargs:
            kwargs['temperature'] = 0
        # if 'max_tokens' not in kwargs:
        #     if not self.model.startswith('gpt-4'):
        #         kwargs['max_tokens'] = 1000
        if 'timeout' not in kwargs:
            kwargs['timeout'] = 180 if self.model.startswith('gpt-4') else 30
        # if 'request_timeout' not in kwargs:
        #     kwargs['request_timeout'] = 180 if self.model.startswith('gpt-4') else 30

        if (client_provider == 'ai_studio' or client_provider == 'vertex') and 'seed' in kwargs:
            del kwargs['seed']

        kwargs = {**kwargs, **self.default_kwargs}

        return kwargs

    def prompt(self, prompts, **kwargs):
        kwargs = self.handle_kwargs(kwargs)

        prompts = [self.formatter.format_prompt(prompt) for prompt in prompts]
        outputs = asyncio.run(self._prompt_batcher(prompts, **kwargs))
        outputs, input_tokens, calls, output_tokens = list(zip(*outputs)) 
        # Note that this is quite risky: https://stackoverflow.com/questions/61647815/do-coroutines-require-locks-when-reading-writing-a-shared-resource
        # Without lock, we need to ensure that operations on self.info are always atomic
        self.info['input_tokens'] += self.formatter.tiklen_formatted_prompts(prompts)
        self.info['calls'] += len(prompts)
        self.info['output_tokens'] += self.formatter.tiklen_outputs(outputs)
        self.info['actual_input_tokens'] += sum(input_tokens)
        self.info['actual_calls'] += sum(calls)
        self.info['actual_output_tokens'] += sum(output_tokens)

        return [self.formatter.format_output(output) for output in outputs]

    async def aprompt(self, prompts, **kwargs):
        kwargs = self.handle_kwargs(kwargs)

        prompts = [self.formatter.format_prompt(prompt) for prompt in prompts]
        outputs = await self._prompt_batcher(prompts, **kwargs)
        outputs, input_tokens, calls, output_tokens = list(zip(*outputs)) 
        # Note that this is quite risky: https://stackoverflow.com/questions/61647815/do-coroutines-require-locks-when-reading-writing-a-shared-resource
        # Without lock, we need to ensure that operations on self.info are always atomic
        self.info['input_tokens'] += self.formatter.tiklen_formatted_prompts(prompts)
        self.info['calls'] += len(prompts)
        self.info['output_tokens'] += self.formatter.tiklen_outputs(outputs)
        self.info['actual_input_tokens'] += sum(input_tokens)
        self.info['actual_calls'] += sum(calls)
        self.info['actual_output_tokens'] += sum(output_tokens)

        return [self.formatter.format_output(output) for output in outputs]

    def override_formatter(self, formatter):
        self.formatter = formatter

    async def _prompt_batcher(self, prompts, **kwargs):
        all_res = []
        max_concurrency = int(kwargs.pop('max_concurrency', 16))
        max_concurrency = max(1, max_concurrency)
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _bounded_get_prompt_res(prompt):
            async with semaphore:
                return await self._get_prompt_res(prompt, **kwargs)

        for ind in range(0, len(prompts), 1000): # Batch 1000 requests
            res = await asyncio.gather(*[_bounded_get_prompt_res(prompt) for prompt in prompts[ind:ind+1000]])
            all_res += res
        return all_res

    async def _get_prompt_res(self, prompt, **kwargs):
        if 'n' in kwargs:
            n = kwargs['n']
        else:
            n = 1
        cache_res = self.lookup_cache(prompt, **kwargs)
        if cache_res is not None and cache_res[0] is not None and len(cache_res) >= n and len(cache_res[0]) > 0:
            if 'n' in kwargs:
                if len(cache_res) == n:
                    return cache_res, 0, 0, 0
                else:
                    return self.rng.choice(cache_res, n), 0, 0, 0
            else:
                return cache_res[0], 0, 0, 0
        
        if 'n' in kwargs:
            if cache_res is not None and cache_res[0] is not None and len(cache_res[0]) > 0:
                n_existing = len(cache_res)
            else:
                n_existing = 0
        else:
            n_existing = 0

        n_to_prompt = n - n_existing
        
        new_kwargs = kwargs.copy()
        if 'n' in kwargs:
            del new_kwargs['n']
        res = await self.prompt_single_func(self.full_model, prompt, n_to_prompt, **new_kwargs)
        self.update_cache(prompt, n_existing, res, **new_kwargs)
        
        if n_existing > 0:
            res = cache_res + res
        
        return (res if 'n' in kwargs else res[0]), self.formatter.tiklen_formatted_prompts([prompt]), 1, self.formatter.tiklen_outputs(res)

    def get_info(self, cost_per_token=None):
        cost_per_token_dict = {
            'gpt-4-1106-preview': (0.010, 0.030),
            'gpt-4-turbo': (0.010, 0.030),
            'gpt-3.5-turbo': (0.003, 0.006),
            'gpt-4': (0.030, 0.060),
            'gpt-4o': (0.005, 0.015),
            'gpt-4o-2024-08-06': (0.0025, 0.010),
            'gpt-4o-2024-05-13': (0.005, 0.015),
            'gpt-4o-mini': (0.00015, 0.0006),
            'gpt-4o-mini-2024-07-18': (0.00015, 0.0006),
            'o1-preview': (0.015, 0.060),
            'o1-preview-2024-09-12': (0.015, 0.060),
            'o1-mini': (0.003, 0.012),
            'o1-mini-2024-09-12': (0.003, 0.012),
            'gemini-1.5-flash': (0.000075, 0.0003),
            'gemini-2.5-flash': (0.0003, 0.0025),
            'gpt-5': (0.00125, 0.010),
            'gemini-2.5-flash-lite': (0.0001, 0.0004),
            'gemini-2.0-flash-lite': (0.000075, 0.0003),
        }
        if cost_per_token is not None:
            self.info['cost_per_token'] = cost_per_token
        else:
            if self.model in cost_per_token_dict:
                self.info['cost_per_token'] = cost_per_token_dict[self.model]
            else:
                self.info['cost_per_token'] = (0, 0)
        self.info['cost'] = self.info['cost_per_token'][0] / 1000 * self.info['input_tokens'] + self.info['cost_per_token'][1] / 1000 * self.info['output_tokens']
        self.info['actual_cost'] = self.info['cost_per_token'][0] / 1000 * self.info['actual_input_tokens'] + self.info['cost_per_token'][1] / 1000 * self.info['actual_output_tokens']
        return self.info