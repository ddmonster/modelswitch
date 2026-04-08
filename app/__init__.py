"""ModelSwitch - LLM Gateway Proxy with multi-provider fallback."""

try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("modelswitch")
    except PackageNotFoundError:
        __version__ = "0.0.0-dev"
except ImportError:
    __version__ = "0.0.0-dev"
