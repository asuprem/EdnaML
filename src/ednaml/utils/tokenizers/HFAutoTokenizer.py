import warnings

try:
    from transformers import AutoTokenizer
except:
    AutoTokenizer = object
    warnings.warn(
        "`transformers` not installed. If using AutoTokenizer, there may be issues."
    )


class HFAutoTokenizer:
    def __init__(self, **kwargs):
        raise RuntimeError(
            "`__init__` called for HFAutoTokenizer. Use `__new__` instead"
        )

    def __new__(cls, from_pretrained: str, **kwargs):
        return AutoTokenizer.from_pretrained(from_pretrained, **kwargs)
