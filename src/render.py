from transformers import AutoTokenizer

from .models import ModelConfig


def load_tokenizer(cfg: ModelConfig):
    return AutoTokenizer.from_pretrained(cfg.hf_repo, revision=cfg.revision)


def render(tokenizer, messages: list[dict], tools: list[dict], cfg: ModelConfig) -> str:
    """Full prompt text from the model's own HF chat template. Every kwarg is explicit."""
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
        **cfg.template_kwargs,
    )
