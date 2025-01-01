
import os
import random
import httpx
import argparse
from openai import OpenAI

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from config import PROXY, ATTEMPT_COUNTER, WAIT_TIME_MIN, WAIT_TIME_MAX, VLLM_URL
from utils import token_count


def get_api_key(platform, model_name=None):
    if platform=="OpenAI":
        return os.environ["OpenAI_API_KEY"]
    elif platform=="DeepInfra":
        return os.environ["DeepInfra_API_KEY"]
    elif platform=="vllm":
        return os.environ["vllm_KEY"]
    elif platform=="SiliconFlow":
        return os.environ["SiliconFlow_API_KEY"]


class LLMAPI:
    def __init__(self, model_name, platform=None):
        self.model_name = model_name
        
        self.platform_list = ["SiliconFlow", "OpenAI", "DeepInfra", 'vllm']
        self.model_platforms = {
                    "SiliconFlow":  [
                        'llama3-8b', 'llama3-70b', 'gemma2-9b', 'gemma2-27b', 'mistral7bv2', 'qwen2-1.5b', 'qwen2-7b', 'qwen2-14b', 'qwen2-72b', 'glm4-9b', 'glm3-6b', 'deepseekv2', 'llama3.1-8b', 'llama3.1-70b', 'llama3.1-405b'] + [
                        'llama3-8b-pro', 'gemma2-9b-pro', 'mistral7bv2-pro', 'qwen2-1.5b-pro', 'qwen2-7b-pro', 'glm4-9b-pro', 'glm3-6b-pro'
                    ],
                    "OpenAI":       ['gpt35turbo', 'gpt4turbo', 'gpt4o', 'gpt4omini'],
                    "DeepInfra":    ['llama3-8b', 'llama3-70b', 'gemma2-9b', 'gemma2-27b', 'mistral7bv2', 'qwen2-7b', 'qwen2-72b', 'llama3.1-8b', 'llama3.1-70b', 'mistral7bv3', 'llama3.1-405b'],
                    "vllm": ['llama3-8B-local', 'gemma2-2b-local', 'chatglm3-citygpt', 'chatglm3-6B-local']
                }
        
        self.model_mapper = {
            'gpt35turbo': 'gpt-3.5-turbo-0125',
            'gpt4turbo': 'gpt-4-turbo-2024-04-09',
            'gpt4o': 'gpt-4o-2024-05-13',
            'gpt4omini': 'gpt-4o-mini-2024-07-18',
            'llama3-8b': 'meta-llama/Meta-Llama-3-8B-Instruct',
            'llama3.1-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
            'llama3-8b-pro': 'Pro/meta-llama/Meta-Llama-3-8B-Instruct',
            'llama3-70b': 'meta-llama/Meta-Llama-3-70B-Instruct',
            'llama3.1-70b': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
            'llama3.1-405b': 'meta-llama/Meta-Llama-3.1-405B-Instruct',
            'llama2-7b': 'meta-llama/Llama-2-7b-chat-hf',
            'llama2-13b': 'meta-llama/Llama-2-13b-chat-hf',
            'llama2-70b': 'meta-llama/Llama-2-70b-chat-hf',
            'gemma2-9b': 'google/gemma-2-9b-it',
            'gemma2-9b-pro': 'Pro/google/gemma-2-9b-it',
            'gemma2-27b': 'google/gemma-2-27b-it',
            'mistral7bv2': 'mistralai/Mistral-7B-Instruct-v0.2',
            'mistral7bv3': 'mistralai/Mistral-7B-Instruct-v0.3',
            'mistral7bv2-pro': 'Pro/mistralai/Mistral-7B-Instruct-v0.2',
            'qwen2-1.5b': 'Qwen/Qwen2-1.5B-Instruct',
            'qwen2-1.5b-pro': 'Pro/Qwen/Qwen2-1.5B-Instruct',
            'qwen2-7b': 'Qwen/Qwen2-7B-Instruct',
            'qwen2-7b-pro': "Pro/Qwen/Qwen2-7B-Instruct",
            'qwen2-14b': 'Qwen/Qwen2-57B-A14B-Instruct',
            'qwen2-72b': 'Qwen/Qwen2-72B-Instruct',
            'glm4-9b': 'THUDM/glm-4-9b-chat',
            'glm4-9b-pro': 'Pro/THUDM/glm-4-9b-chat',
            'glm3-6b': 'THUDM/chatglm3-6b',
            'glm3-6b-pro': 'Pro/THUDM/chatglm3-6b',
            'deepseekv2': 'deepseek-ai/DeepSeek-V2-Chat',
            'llama3-8B-local':'llama3-8B-local',
            'gemma2-2b-local': 'gemma2-2b-local',
            'chatglm3-citygpt': 'chatglm3-citygpt',
            'chatglm3-6B-local': 'chatglm3-6B-local'
        }

        support_models = ";".join([";".join(self.model_platforms[k]) for k in self.model_platforms])
        if self.model_name not in support_models:
            raise ValueError('Invalid model name! Please use one of the following: {}'.format(support_models))
        
        if platform is not None and platform in self.platform_list:
            self.platform = platform
        else:
            for platform in self.platform_list:
                if self.model_name in self.model_platforms[platform]:
                    self.platform = platform
                    break

        if self.platform is None:
            raise ValueError("'Invalid API platform:{} with model:{}".format(self.platform, self.model_name))

        if self.model_name not in self.model_platforms[self.platform]:
            raise ValueError('Invalid model name! Please use one of the following: {} in API platform:{}'.format(support_models, self.platform))
        

        if self.platform == "OpenAI":
            self.client = OpenAI(
                api_key=get_api_key(platform),
                http_client=httpx.Client(proxies=PROXY),
            )
        elif self.platform == "DeepInfra":
            self.client = OpenAI(
                base_url="https://api.deepinfra.com/v1/openai",
                api_key=get_api_key(platform),
                http_client=httpx.Client(proxies=PROXY),
            )
        elif self.platform == "SiliconFlow":
            self.client = OpenAI(
                base_url="https://api.siliconflow.cn/v1",
                api_key=get_api_key(platform, model_name)
            )
        elif self.platform == 'vllm':
            self.client = OpenAI(
                base_url=VLLM_URL,
                api_key=get_api_key(platform)
            )
    
    def get_client(self):
        return self.client
    
    def get_model_name(self):
        return self.model_mapper[self.model_name]
    
    def get_platform_name(self):
        return self.platform

    def get_supported_models(self):
        return self.model_platforms


class LLMWrapper:
    def __init__(self, model_name, platform=None):
        self.model_name = model_name
        self.hyperparams = {
            'temperature': 0.,  # make the LLM basically deterministic
            'max_new_tokens': 100, # not used in OpenAI API
            'max_tokens': 1000,    # The maximum number of [tokens](/tokenizer) that can be generated in the completion.
            'max_input_tokens': 2000 # The maximum number of input tokens
        }
        
        self.llm_api = LLMAPI(self.model_name, platform=platform)
        self.client = self.llm_api.get_client()
        self.api_model_name = self.llm_api.get_model_name()

    @retry(wait=wait_random_exponential(min=WAIT_TIME_MIN, max=WAIT_TIME_MAX), stop=stop_after_attempt(ATTEMPT_COUNTER))
    def get_response(self, prompt_text):
        if "gpt" in self.model_name:
            system_messages = [{"role": "system", "content": "You are a helpful assistant who predicts user next location."}]
        else:
            system_messages = []
        

        if token_count(prompt_text)>self.hyperparams['max_input_tokens']:
            prompt_text = prompt_text[-min(self.hyperparams['max_input_tokens']*3, len(prompt_text)):]
        
        response = self.client.chat.completions.create(
            model=self.api_model_name,
            messages=system_messages + [{"role": "user", "content": prompt_text}],
            max_tokens=self.hyperparams["max_tokens"],
            temperature=self.hyperparams["temperature"]
        )
        full_text = response.choices[0].message.content
        return full_text


if __name__ == "__main__":
    prompt_text = "Who are you?"

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="llama3-8b")
    parser.add_argument("--platform", type=str, default="SiliconFlow", choices=["SiliconFlow", "OpenAI", "DeepInfra"])
    args = parser.parse_args()

    llm = LLMWrapper(model_name=args.model_name, platform=args.platform)
    print(llm.get_response(prompt_text))
