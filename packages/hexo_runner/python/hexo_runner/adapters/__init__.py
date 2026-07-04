"""Optional runner adapters for external engines and bots."""

from .sealbot import (
    SealBotConfig,
    SealBotPlayer,
    SealBotUnavailableError,
    discover_sealbot_adapters,
)

__all__ = [
    "SealBotConfig",
    "SealBotPlayer",
    "SealBotUnavailableError",
    "discover_sealbot_adapters",
]
