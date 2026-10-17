from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from mikirin.atomic import atomic_write_json


ModelRole = Literal["organizer", "chat", "embedding"]
ApiStyle = Literal["chat-completions", "responses"]
EmbeddingApiStyle = Literal["openai-embeddings"]


@dataclass(frozen=True, slots=True)
class ModelProfile:
    provider: str
    base_url: str
    model: str
    api_style: ApiStyle
    key_env: str

    def key(self) -> str:
        value = os.environ.get(self.key_env, "").strip()
        if not value:
            raise RuntimeError(
                f"API key is missing; put {self.key_env}=... in the project .env file"
            )
        return value

    def public_dict(self) -> dict[str, object]:
        return {**asdict(self), "key_available": bool(os.environ.get(self.key_env, "").strip())}


@dataclass(frozen=True, slots=True)
class EmbeddingProfile:
    provider: str
    base_url: str
    model: str
    api_style: EmbeddingApiStyle
    key_env: str
    dimensions: int
    distance: str = "cosine"
    query_prefix: str = ""
    document_prefix: str = ""
    normalize: bool = True
    input_normalization_version: str = "unicode-nfc-lf-v1"

    def key(self) -> str:
        value = os.environ.get(self.key_env, "").strip()
        if not value:
            raise RuntimeError(
                f"API key is missing; put {self.key_env}=... in the project .env file"
            )
        return value

    def public_dict(self) -> dict[str, object]:
        return {**asdict(self), "key_available": bool(os.environ.get(self.key_env, "").strip())}


PROVIDER_DEFAULTS: dict[str, ModelProfile] = {
    "deepseek": ModelProfile(
        provider="deepseek",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-pro",
        api_style="chat-completions",
        key_env="DEEPSEEK_API_KEY",
    ),
    "openai": ModelProfile(
        provider="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-5.6-terra",
        api_style="responses",
        key_env="OPENAI_API_KEY",
    ),
}


class ModelConfigStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[ModelRole, ModelProfile | EmbeddingProfile]:
        defaults: dict[ModelRole, ModelProfile | EmbeddingProfile] = {
            "organizer": PROVIDER_DEFAULTS["deepseek"],
            "chat": PROVIDER_DEFAULTS["deepseek"],
        }
        if not self.path.exists():
            return defaults
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        roles = payload.get("roles", {})
        for role in ("organizer", "chat"):
            raw = roles.get(role)
            if isinstance(raw, dict):
                defaults[role] = self._validate(ModelProfile(**raw))
        raw_embedding = roles.get("embedding")
        if isinstance(raw_embedding, dict):
            defaults["embedding"] = self._validate_embedding(
                EmbeddingProfile(**self._embedding_values(raw_embedding))
            )
        return defaults

    def save(self, profiles: dict[ModelRole, ModelProfile | EmbeddingProfile]) -> None:
        roles = {role: asdict(profile) for role, profile in profiles.items()}
        # A caller using the legacy chat/organizer-only API must not erase a
        # configured embedding role from an existing runtime manifest.
        if "embedding" not in roles and self.path.exists():
            existing = json.loads(self.path.read_text(encoding="utf-8"))
            existing_embedding = existing.get("roles", {}).get("embedding")
            if isinstance(existing_embedding, dict):
                roles["embedding"] = existing_embedding
        atomic_write_json(
            self.path,
            {"schema_version": 1, "roles": roles},
        )

    def set_provider(
        self,
        role: ModelRole,
        provider: str,
        *,
        model: str | None = None,
    ) -> ModelProfile:
        if role == "embedding":
            raise ValueError("embedding uses set_custom with an EmbeddingProfile")
        profiles = self.load()
        if provider not in PROVIDER_DEFAULTS:
            raise ValueError(f"unknown provider {provider!r}; use deepseek or openai")
        base = PROVIDER_DEFAULTS[provider]
        profile = ModelProfile(
            provider=base.provider,
            base_url=base.base_url,
            model=(model or base.model).strip(),
            api_style=base.api_style,
            key_env=base.key_env,
        )
        profiles[role] = self._validate(profile)
        self.save(profiles)
        return profile

    def set_custom(
        self, role: ModelRole, profile: ModelProfile | EmbeddingProfile
    ) -> ModelProfile | EmbeddingProfile:
        profiles = self.load()
        if role == "embedding":
            if not isinstance(profile, EmbeddingProfile):
                raise TypeError("embedding role requires an EmbeddingProfile")
            validated = self._validate_embedding(profile)
        else:
            if not isinstance(profile, ModelProfile):
                raise TypeError("chat and organizer roles require a ModelProfile")
            validated = self._validate(profile)
        profiles[role] = validated
        self.save(profiles)
        return validated

    @staticmethod
    def _embedding_values(raw: dict[str, object]) -> dict[str, object]:
        values = dict(raw)
        # Older manifests identify this role with api_style; retain that exact
        # on-disk compatibility while supplying a stable fingerprint default.
        values.setdefault("input_normalization_version", "unicode-nfc-lf-v1")
        return values

    @staticmethod
    def _validate(profile: ModelProfile) -> ModelProfile:
        if profile.api_style not in {"responses", "chat-completions"}:
            raise ValueError("api_style must be responses or chat-completions")
        parsed_url = urlsplit(profile.base_url)
        if parsed_url.scheme != "https" or not parsed_url.netloc:
            raise ValueError("base_url must use HTTPS")
        if not profile.model.strip() or not profile.key_env.strip():
            raise ValueError("model and key_env cannot be empty")
        return profile

    @staticmethod
    def _validate_embedding(profile: EmbeddingProfile) -> EmbeddingProfile:
        if profile.api_style != "openai-embeddings":
            raise ValueError("embedding api_style must be openai-embeddings")
        parsed_url = urlsplit(profile.base_url)
        if parsed_url.scheme != "https" or not parsed_url.netloc:
            raise ValueError("base_url must use HTTPS")
        if not profile.provider.strip() or not profile.model.strip() or not profile.key_env.strip():
            raise ValueError("provider, model, and key_env cannot be empty")
        if not isinstance(profile.dimensions, int) or isinstance(profile.dimensions, bool):
            raise ValueError("embedding dimensions must be an integer")
        if profile.dimensions < 1:
            raise ValueError("embedding dimensions must be positive")
        if profile.distance != "cosine":
            raise ValueError("only cosine embedding distance is supported")
        if not isinstance(profile.query_prefix, str) or not isinstance(profile.document_prefix, str):
            raise ValueError("embedding prefixes must be strings")
        if not isinstance(profile.normalize, bool):
            raise ValueError("embedding normalize must be boolean")
        if not isinstance(profile.input_normalization_version, str) or not profile.input_normalization_version.strip():
            raise ValueError("input_normalization_version must be a non-empty string")
        return profile
