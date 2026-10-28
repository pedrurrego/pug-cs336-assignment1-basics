import importlib.metadata
from cs336_basics.p03_train_bpe import pug_bpe

__all__ = [
    "pug_bpe"
]

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass
