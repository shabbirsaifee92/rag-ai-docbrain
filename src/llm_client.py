# llm_client.py
import requests
from config import settings

class LLMClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.completion_endpoint = f"{base_url}/v1/completions"

    def generate_completion(self, prompt: str) -> str:
        payload = {
            "model": settings.LLM_MODEL_NAME,
            "prompt": prompt,
            "temperature": float(settings.TEMPERATURE),
            "top_p": float(settings.TOP_P),
            "stream": False,
            "max_tokens": int(settings.MAX_TOKENS)
        }

        try:
            response = requests.post(self.completion_endpoint, json=payload)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['text']
        except Exception as e:
            raise Exception(f"Error generating completion: {str(e)}")
