"""
Parses a free-text query into explicit {garment: color} pairs using a lightweight
rule-based parser or an instruction-tuned Hugging Face LLM model.

CLIP embeds a whole query as one vector, which is where compositional queries
get conflated ("red shirt, blue pants" vs "blue shirt, red pants"). This
parser extracts the structure needed to check attribute binding explicitly
at re-rank time (see retriever/rerank_search.py).
"""

import re
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config

_KNOWN_GARMENTS = set(config.CATEGORY_TO_GARMENT.values())

COLOR_SYNONYMS = {
    "grey": "gray",
    "navy": "blue",
    "beige": "brown",
    "tan": "brown",
    "maroon": "red",
    "crimson": "red",
    "violet": "purple",
}

GARMENT_SYNONYMS = {
    "shirt": "shirt",
    "t-shirt": "shirt",
    "tshirt": "shirt",
    "tee": "shirt",
    "top": "shirt",
    "blouse": "shirt",
    "polo": "shirt",
    "pants": "pants",
    "jeans": "pants",
    "trousers": "pants",
    "slacks": "pants",
    "shorts": "pants",
    "dress": "dress",
    "gown": "dress",
    "coat": "coat",
    "raincoat": "coat",
    "overcoat": "coat",
    "parka": "coat",
    "jacket": "jacket",
    "blazer": "jacket",
    "skirt": "skirt",
    "sweater": "sweater",
    "cardigan": "sweater",
    "hoodie": "sweater",
    "pullover": "sweater",
    "suit": "suit",
    "tuxedo": "suit",
    "cape": "cape",
    "cloak": "cape",
    "tie": "tie",
    "necktie": "tie",
    "bowtie": "tie",
    "scarf": "scarf",
    "hat": "hat",
    "cap": "hat",
    "beanie": "hat",
    "beret": "hat",
    "bag": "bag",
    "handbag": "bag",
    "purse": "bag",
    "backpack": "bag",
    "wallet": "bag",
    "belt": "belt",
    "shoe": "shoe",
    "shoes": "shoe",
    "sneaker": "shoe",
    "sneakers": "shoe",
    "boot": "shoe",
    "boots": "shoe",
    "heel": "shoe",
    "heels": "shoe",
    "loafers": "shoe",
}

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


class RuleBasedQueryParser:
    """Fast, zero-RAM rule-based query parser that extracts {garment: color} pairs."""

    @staticmethod
    def parse(query: str) -> dict:
        if not query:
            return {}
        q = query.lower()
        tokens = re.findall(r"[a-z\-]+", q)
        result = {}

        # Pattern 1: <color> [modifier] <garment> (e.g., 'red tie', 'bright yellow raincoat')
        for i, tok in enumerate(tokens):
            raw_color = None
            if tok in config.KNOWN_COLORS:
                raw_color = tok
            elif tok in COLOR_SYNONYMS:
                raw_color = COLOR_SYNONYMS[tok]

            if raw_color:
                for j in range(i + 1, min(i + 4, len(tokens))):
                    g_tok = tokens[j]
                    if g_tok in GARMENT_SYNONYMS:
                        canon_garment = GARMENT_SYNONYMS[g_tok]
                        if canon_garment in _KNOWN_GARMENTS and raw_color in config.KNOWN_COLORS:
                            result[canon_garment] = raw_color
                        break

        # Pattern 2: <garment> (in|of|colored|with) <color> (e.g., 'shirt in blue')
        for i, tok in enumerate(tokens):
            if tok in GARMENT_SYNONYMS:
                canon_garment = GARMENT_SYNONYMS[tok]
                if canon_garment not in result and i + 2 < len(tokens):
                    if tokens[i + 1] in ("in", "of", "colored", "with"):
                        c_tok = tokens[i + 2]
                        raw_color = c_tok if c_tok in config.KNOWN_COLORS else COLOR_SYNONYMS.get(c_tok)
                        if raw_color and canon_garment in _KNOWN_GARMENTS and raw_color in config.KNOWN_COLORS:
                            result[canon_garment] = raw_color

        return result


class QueryParser:
    """QueryParser supports both fast rule-based parsing and instruction-tuned LLMs."""

    def __init__(self, model_name=config.PARSER_MODEL_NAME, use_llm=None):
        if use_llm is None:
            use_llm = getattr(config, "USE_LLM_QUERY_PARSER", False)
        self.use_llm = use_llm
        self.tokenizer = None
        self.model = None

        if self.use_llm:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if config.DEVICE == "cuda" else torch.bfloat16,
                    low_cpu_mem_usage=True,
                    device_map="auto" if config.DEVICE == "cuda" else None,
                )
                if config.DEVICE != "cuda":
                    self.model = self.model.to(config.DEVICE)
            except Exception as e:
                print(f"[QueryParser] LLM load failed ({e}). Falling back to RuleBasedQueryParser.")
                self.use_llm = False
                self.model = None
                self.tokenizer = None

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extracts JSON object from text, handling markdown code blocks and nested dicts."""
        if not text:
            return {}
        code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        target = code_block.group(1) if code_block else text

        start = target.find("{")
        end = target.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(target[start:end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        return {}

    def parse(self, query: str) -> dict:
        if not self.use_llm or self.model is None or self.tokenizer is None:
            return RuleBasedQueryParser.parse(query)

        try:
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

            parsed = self._extract_json(decoded)
            return {
                g.lower(): c.lower()
                for g, c in parsed.items()
                if isinstance(g, str) and isinstance(c, str)
                and g.lower() in _KNOWN_GARMENTS
                and c.lower() in config.KNOWN_COLORS
            }
        except Exception as e:
            print(f"[QueryParser] LLM inference failed ({e}). Using rule-based fallback.")
            return RuleBasedQueryParser.parse(query)
