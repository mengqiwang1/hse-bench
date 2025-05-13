import os
import re

import requests
import json
import openai
from openai import OpenAI
# from google import genai
import anthropic


with open("config.json", 'r') as file:
    config =  json.load(file)

# Dummy key usage for examples
API_KEYS = {
    "openai": config.get("api_keys").get("openai"),
    "google": config.get("api_keys").get("google"),
    "anthropic": config.get("api_keys").get("anthropic"),
    "nebius": config.get("api_keys").get("nebius"),
    "deepseek": config.get("api_keys").get("deepseek"),
}

# Sample text prompt
PROMPT = "Explain the concept of Occupational Health and Safety regulations."

# Model registry (text + vision)
MODELS = {
    "OpenAI GPT-4o-mini": {"type": "text", "api": "openai", "model": "gpt-4o-mini"},
    "Claude Sonnet 3.7": {"type": "text", "api": "anthropic", "model": "claude-3-7-sonnet-20250219"},
    "Gemini 2.0 Flash": {"type": "text", "api": "google", "model": "gemini-2.0-flash"},
    # "Gemini 2.5 Pro": {"type": "text", "api": "google", "model": "gemini-2.5-pro-exp-03-25"},
    "Gemini 2.5 Pro Preview": {"type": "text", "api": "google", "model": "gemini-2.5-pro-preview-03-25"},
    "DeepSeek V3": {"type": "text", "api": "deepseek", "model": "deepseek-chat"},
    "DeepSeek R1": {"type": "text", "api": "deepseek", "model": "deepseek-reasoner"},
    "QwQ": {"type": "text", "api": "nebius", "model": "Qwen/QwQ-32B"},
    "Qwen": {"type": "text", "api": "nebius", "model": "Qwen/Qwen2.5-72B-Instruct"},
    "LLaMA 3.1": {"type": "text", "api": "nebius", "model": "meta-llama/Meta-Llama-3.1-8B-Instruct"},
    "Gemma-3": {"type": "vision", "api": "nebius", "model": "google/gemma-3-27b-it"},
    "LLaVA-1.5": {"type": "vision", "api": "nebius", "model": "llava-hf/llava-1.5-7b-hf"}, # vision
    "Qwen2.5-VL": {"type": "vision", "api": "nebius", "model": "Qwen/Qwen2.5-VL-72B-Instruct"}, # vision
}

openai_client = OpenAI(
    api_key=API_KEYS["openai"],  # This is the default and can be omitted
)

ds_client = OpenAI(api_key=API_KEYS["deepseek"], base_url="https://api.deepseek.com")

nb_client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=API_KEYS["nebius"]
)
# gg_client = genai.Client(api_key=API_KEYS["google"])
ant_client = anthropic.Anthropic(
    api_key=API_KEYS["anthropic"]
)


def call_openai(model_name, prompt):
    response = openai_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500,

    )
    # print(response)
    return response.choices[0].message.content

def call_deepseek(model_name, prompt, temp=0):
    response = ds_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
        max_tokens=500,
    )
    print(response)
    return response.choices[0].message.content

def call_anthropic(model_name, prompt):
    response = ant_client.messages.create(
        model=model_name,
        max_tokens=500,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    print(response)
    return response.content[0].text

def call_google(model_name, prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": API_KEYS["google"]}
    data = {
        "generationConfig": {"temperature": 0},
        "contents": [{"parts": [{"text": prompt}]}]
    }
    response = requests.post(url, headers=headers, params=params, json=data)
    print(response)
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]



def call_nebius(model_name, prompt, metrics):
    if model_name == "Qwen/QwQ-32B" and metrics == "accuracy":
        prompt += "In the last sentence, reply in: The correct option is A/B/B/D."
    response = nb_client.chat.completions.create(
        model=model_name,
        temperature=0,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    print(response)
    if model_name == "Qwen/QwQ-32B" and metrics == "accuracy":
        match = re.search(r'the correct option is\s*([ABCD])\b', response.choices[0].message.content, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        else:
            return "?"
    return response.choices[0].message.content


