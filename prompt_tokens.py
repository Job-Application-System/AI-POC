"""Count local-model input tokens without loading model weights."""

from functools import lru_cache

from transformers import AutoTokenizer


@lru_cache(maxsize=1)
def load_tokenizer():
    return AutoTokenizer.from_pretrained("tencent/Hy3")


def log_prompt_tokens(tokenizer, messages, task):
    # Include role markers and the generation prefix, without truncation.
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        reasoning_effort="high",
    )
    print(f"{task} prompt tokens (before generation): {len(token_ids)}", flush=True)
