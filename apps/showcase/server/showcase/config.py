"""Showcase runtime settings, read from `SHOWCASE_*` env vars with sane defaults.

One frozen `Settings` object is built at startup (`Settings.from_env()`) and
threaded explicitly through `create_app` / `BotPool` / `ShowcaseDB` — no module
lives read env after import, so tests construct `Settings` directly.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, "").strip() or default)


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, "").strip() or default)


def _env_opt_bool(name: str) -> bool | None:
    """Tri-state boolean env: unset/empty -> None (caller-defined default)."""
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return None
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean (0/1/true/false), got {raw!r}")


@dataclass(frozen=True, slots=True)
class Settings:
    """Complete showcase configuration. All knobs, one place."""

    db_path: Path
    bots_toml: Path
    search_config: Path
    static_dir: Path
    workers: int
    max_active_games: int
    max_games_per_ip: int
    moves_per_minute: int
    analysis_per_minute: int
    games_per_hour: int
    idle_timeout_s: float
    bot_timeout_s: float
    finished_ttl_s: float
    sweep_interval_s: float
    analysis_search_visit_cap: int
    policy_floor: float
    torch_threads: int
    ip_salt: str
    # Inference device request (auto | cpu | xpu | cuda) — resolved per worker
    # process at init by showcase.device.resolve_device; `auto` prefers xpu,
    # then cuda, then cpu. Defaults keep existing constructors (tests) on the
    # auto->cpu path.
    device: str = "auto"
    # Startup CPU-vs-device parity self-check (None -> on when the resolved
    # device is not cpu). See showcase.device.verify_device.
    device_selfcheck: bool | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from `SHOWCASE_*` env vars.

        Path defaults assume the process runs from the repo root (the compose
        entrypoint and the README invocation both do). `SHOWCASE_IP_SALT`
        defaults to a random per-boot salt: per-IP caps still work, but client
        hashes are orphaned on restart unless a stable salt is configured.
        """
        return cls(
            db_path=Path(os.environ.get("SHOWCASE_DB_PATH", "showcase.db")),
            bots_toml=Path(
                os.environ.get("SHOWCASE_BOTS_TOML", "apps/showcase/bots.example.toml")
            ),
            search_config=Path(
                os.environ.get("SHOWCASE_SEARCH_CONFIG", "configs/hexfield_main_7.toml")
            ),
            static_dir=Path(os.environ.get("SHOWCASE_STATIC_DIR", "apps/showcase/web")),
            workers=_env_int("SHOWCASE_WORKERS", 2),
            # Capacity defaults (each env-overridable). global=16 assumes the
            # deploy has GPU/CPU headroom for that many concurrent searches;
            # an XPU-backed deploy raises real capacity well past this, but 16
            # stays the safe code default — bump via env in the compose file.
            max_active_games=_env_int("SHOWCASE_MAX_ACTIVE_GAMES", 16),
            max_games_per_ip=_env_int("SHOWCASE_MAX_GAMES_PER_IP", 4),
            moves_per_minute=_env_int("SHOWCASE_MOVES_PER_MINUTE", 120),
            analysis_per_minute=_env_int("SHOWCASE_ANALYSIS_PER_MINUTE", 40),
            games_per_hour=_env_int("SHOWCASE_GAMES_PER_HOUR", 60),
            idle_timeout_s=_env_float("SHOWCASE_IDLE_TIMEOUT_S", 600.0),
            bot_timeout_s=_env_float("SHOWCASE_BOT_TIMEOUT_S", 120.0),
            finished_ttl_s=_env_float("SHOWCASE_FINISHED_TTL_S", 6 * 3600.0),
            sweep_interval_s=_env_float("SHOWCASE_SWEEP_INTERVAL_S", 15.0),
            analysis_search_visit_cap=_env_int("SHOWCASE_ANALYSIS_VISIT_CAP", 64),
            policy_floor=_env_float("SHOWCASE_POLICY_FLOOR", 1e-4),
            torch_threads=_env_int("SHOWCASE_TORCH_THREADS", 0),
            ip_salt=os.environ.get("SHOWCASE_IP_SALT", "") or secrets.token_hex(16),
            device=os.environ.get("SHOWCASE_DEVICE", "").strip().lower() or "auto",
            device_selfcheck=_env_opt_bool("SHOWCASE_DEVICE_SELFCHECK"),
        )
