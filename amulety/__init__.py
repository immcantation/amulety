from importlib.metadata import PackageNotFoundError, version

from .amulety import embed_airr, translate_airr

try:
    __version__ = version("amulety")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "embed_airr",
    "translate_airr",
]
