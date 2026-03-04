from hashlib import md5
import base64
import mimetypes
import os
import time
import re
import string
import logging

import httpx
import numpy as np
import openai
from openai import OpenAI


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + md5(content.encode()).hexdigest()


class LLM_Model:
    def __init__(self, llm_model):
        http_client = httpx.Client(timeout=60.0, trust_env=False)
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            http_client=http_client,
        )
        self.llm_config = {
            "model": llm_model,
            "max_tokens": 2000,
            "temperature": 0,
        }

    @staticmethod
    def _image_path_to_data_url(image_path):
        if not image_path or not os.path.exists(image_path):
            return None
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/png"
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def build_qa_messages(self, system_prompt, prompt_user_text, image_paths=None):
        image_paths = image_paths or []
        content = [{"type": "text", "text": prompt_user_text}]
        for image_path in image_paths:
            data_url = self._image_path_to_data_url(image_path)
            if data_url:
                content.append({"type": "image_url", "image_url": {"url": data_url}})

        if len(content) == 1:
            user_content = prompt_user_text
        else:
            user_content = content

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def _messages_to_text_only(messages):
        simplified = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                content = "\n".join([x for x in text_parts if x]).strip()
            simplified.append({"role": msg.get("role"), "content": content})
        return simplified

    def infer(self, messages):
        max_retries = 10
        retry_delay = 5

        for i in range(max_retries):
            try:
                response = self.openai_client.chat.completions.create(
                    **self.llm_config,
                    messages=messages,
                )
                return response.choices[0].message.content

            except openai.RateLimitError:
                wait_time = retry_delay * (i + 1)
                logging.warning("Rate limited, retry %d/%d in %ds", i + 1, max_retries, wait_time)
                time.sleep(wait_time)

            except Exception as e:
                # If endpoint doesn't support multimodal message schema, fallback once to text-only.
                if any(isinstance(m.get("content"), list) for m in messages):
                    try:
                        response = self.openai_client.chat.completions.create(
                            **self.llm_config,
                            messages=self._messages_to_text_only(messages),
                        )
                        return response.choices[0].message.content
                    except Exception:
                        pass
                logging.error("LLM API error: %s", e)
                time.sleep(2)
                if i == max_retries - 1:
                    raise e

        return "Error: Reached max retries due to RateLimit."


def normalize_answer(s):
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def setup_logging(log_file):
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler()]
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers,
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def min_max_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    if range_val == 0:
        return np.ones_like(x)
    return (x - min_val) / range_val
