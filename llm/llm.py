import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import yaml

load_dotenv()

# ── Load config ───────────────────────────────────────────
def _load_config():
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)

_cfg = _load_config()

MODEL = _cfg["llm"]["model"]
# Used for short, low-stakes calls (query rewrite, guardrail, evaluator) where
# a smaller/faster model cuts latency a lot and the "reasoning" required is
# minimal (yes/no, short rewritten query). Falls back to MODEL if unset.
FAST_MODEL = _cfg["llm"].get("fast_model", MODEL)
DEFAULT_TEMPERATURE = _cfg["llm"]["temperature"]

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


DEFAULT_SYSTEM_MESSAGE = (
    "You ground every claim in the provided context and never invent facts "
    "beyond it, but you explain things in your own natural words rather "
    "than copying the context's phrasing or structure."
)


def generate_answer(
    prompt: str,
    temperature: float = None,
    max_tokens: int = None,
    system_message: str = None,
    model: str = None,
) -> str:
    """
    Call the Groq LLM with a given prompt.
    Temperature and max_tokens default to the values in config.yaml.
    system_message lets a specific mode (e.g. revise/practice) swap in its
    own framing instead of the default one.
    model lets a caller use FAST_MODEL for short utility calls instead of
    the main (slower, higher-quality) model.
    """
    temp = temperature if temperature is not None else DEFAULT_TEMPERATURE
    tokens = max_tokens if max_tokens is not None else _cfg["llm"]["max_tokens"]
    system = system_message if system_message is not None else DEFAULT_SYSTEM_MESSAGE
    use_model = model if model is not None else MODEL

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model=use_model,
        temperature=temp,
        max_tokens=tokens,
    )

    return response.choices[0].message.content