import os
import openai
import backoff
import threading

completion_tokens = prompt_tokens = 0

api_key = os.getenv("OPENAI_API_KEY", "")
if api_key != "":
    openai.api_key = api_key
else:
    print("Warning: OPENAI_API_KEY is not set")
    
api_base = os.getenv("OPENAI_API_BASE", "")
if api_base != "":
    print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
    openai.api_base = api_base

# Lock to protect global token counters
a_global_usage_lock = threading.Lock()

@backoff.on_exception(backoff.expo, openai.error.OpenAIError, max_time=60)
def completions_with_backoff(**kwargs):
    """Wrapper around openai.ChatCompletion.create that respects DeepSeek's
    limitation of n=1 per request by sending multiple requests sequentially
    and aggregating the results when the caller specifies n>1.
    The behaviour for all other providers remains unchanged.
    """
    model_name = kwargs.get("model", "")
    n = kwargs.get("n", 1)

    # DeepSeek only supports n==1. If a larger n is requested, perform n
    # individual calls and merge the responses so that downstream code sees
    # the same structure as from a single OpenAI response.
    if model_name.startswith("deepseek") and n > 1:
        # ensure we don't forward an invalid n to DeepSeek
        kwargs_single = dict(kwargs)
        kwargs_single["n"] = 1
        responses = []
        for _ in range(n):
            responses.append(openai.ChatCompletion.create(**kwargs_single))

        # aggregate choices and usage into the first response object
        merged = responses[0]
        merged.choices = [choice for r in responses for choice in r.choices]
        merged.usage.prompt_tokens = sum(r.usage.prompt_tokens for r in responses)
        merged.usage.completion_tokens = sum(r.usage.completion_tokens for r in responses)
        return merged

    # Non-DeepSeek or n==1 → normal path
    return openai.ChatCompletion.create(**kwargs)

def gpt(prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None, return_logprobs: bool=False, top_logprobs: int=5) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop, return_logprobs=return_logprobs, top_logprobs=top_logprobs)
    
def chatgpt(messages, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None, return_logprobs: bool=False, top_logprobs: int=5) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        extra_kwargs = {}
        if return_logprobs:
            extra_kwargs = {"logprobs": True, "top_logprobs": top_logprobs}
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop, **extra_kwargs)
        if return_logprobs:
            outputs.extend([choice for choice in res.choices])  # return raw choices with logprobs
        else:
            outputs.extend([choice.message.content for choice in res.choices])
        # log completion tokens atomically
        with a_global_usage_lock:
            completion_tokens += res.usage.completion_tokens
            prompt_tokens += res.usage.prompt_tokens
    return outputs
    
def gpt_usage(backend="gpt-4"):
    global completion_tokens, prompt_tokens
    with a_global_usage_lock:
        comp = completion_tokens
        prom = prompt_tokens
    if backend == "gpt-4":
        cost = comp / 1000 * 0.06 + prom / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = comp / 1000 * 0.002 + prom / 1000 * 0.0015
    elif backend == "gpt-4o":
        cost = comp / 1000 * 0.00250 + prom / 1000 * 0.01
    elif backend == "deepseek-chat":
        # DeepSeek-Chat (DeepSeek-V3) standard price (cache miss, UTC 00:30-16:30)
        # Prompt: $0.00027 /1K, Completion: $0.00110 /1K
        cost = comp / 1000 * 0.00110 + prom / 1000 * 0.00027
    elif backend == "deepseek-reasoner":
        # DeepSeek-Reasoner (R1) standard price (cache miss, UTC 00:30-16:30)
        # Prompt: $0.00055 /1K, Completion: $0.00219 /1K
        cost = comp / 1000 * 0.00219 + prom / 1000 * 0.00055
    return {"completion_tokens": comp, "prompt_tokens": prom, "cost": cost}
    