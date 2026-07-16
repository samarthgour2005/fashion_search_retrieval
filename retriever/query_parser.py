"""
Parses a free-text query into explicit {garment: color} pairs using a small
free instruction-tuned Hugging Face model.

CLIP embeds a whole query as one vector, which is where compositional queries
get conflated ("red shirt, blue pants" vs "blue shirt, red pants"). This
parser extracts the structure needed to check attribute binding explicitly
at re-rank time (see retriever/rerank_search.py).

Runs once per query, not once per image, so its cost stays flat as the
indexed corpus grows.
"""

import re
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config

_KNOWN_GARMENTS = set(config.CATEGORY_TO_GARMENT.values())

_FEW_SHOT_PROMPT = """Extract garment-color pairs from the query as JSON. Only include garments explicitly mentioned with a color. Do not invent garments or colors not stated. Respond with JSON only, no other text.

Query: "white shirt with red tie"
Output: {{"shirt": "white", "tie": "red"}}

Query: "a bright yellow raincoat"
Output: {{"coat": "yellow"}}

Query: "professional business attire inside a modern office"
Output: {{}}

Query: "someone wearing a blue shirt sitting on a park bench"
Output: {{"shirt": "blue"}}

Query: "{query}"
Output:"""


class QueryParser:
    """Wraps the parser model + tokenizer so they're loaded once and reused."""

    def __init__(self, model_name=config.PARSER_MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if config.DEVICE == "cuda" else torch.float32,
            device_map="auto" if config.DEVICE == "cuda" else None,
        )
        if config.DEVICE != "cuda":
            self.model = self.model.to(config.DEVICE)

    def parse(self, query: str) -> dict:
        prompt = _FEW_SHOT_PROMPT.format(query=query)
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=config.PARSER_MAX_NEW_TOKENS, do_sample=False
            )

        decoded = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        match = re.search(r"\{.*?\}", decoded, re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            return {}

        # validate against known vocab -> a hallucinated garment/color is dropped,
        # not silently passed through to the scoring function
        return {
            g.lower(): c.lower()
            for g, c in parsed.items()
            if g.lower() in _KNOWN_GARMENTS and c.lower() in config.KNOWN_COLORS
        }
