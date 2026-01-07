"""JSON-LD Schema definitions for Magic Protocol metadata.

This module provides structured definitions for magic dependencies
that can be serialized to JSON-LD format for interoperability.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ProvisionSpec(BaseModel):
    """Specification of what a magic protocol provides."""

    model_config = ConfigDict(populate_by_name=True)

    name: str
    type_annotation: str = Field(..., alias="type")
    availability: str = Field(
        default="always", description="When this provision is available"
    )
    nullable: bool = False
    description: str = ""


class RequirementSpec(BaseModel):
    """Specification of what a magic protocol requires."""

    model_config = ConfigDict(populate_by_name=True)

    headers: list[str] = Field(default_factory=list)
    env_vars: list[str] = Field(default_factory=list, alias="envVars")
    dependencies: list[str] = Field(default_factory=list)
    config_keys: list[str] = Field(default_factory=list, alias="configKeys")


class SideEffectSpec(BaseModel):
    """Specification of a side effect caused by magic."""

    model_config = ConfigDict(populate_by_name=True)

    effect_type: str = Field(..., alias="type")
    condition: str = ""
    description: str = ""
    raises: str | None = None  # Exception that may be raised
    writes_to: list[str] = Field(default_factory=list)
    calls: list[str] = Field(default_factory=list)


class InjectionMechanismSpec(BaseModel):
    """How the magic is injected into the runtime."""

    framework: str
    pattern: str  # dependency_injection, middleware, decorator, etc.
    scope: str = "request"  # request, session, singleton, transient


class CodexRuleSpec(BaseModel):
    """A rule in the dependency codex (law book)."""

    rule_id: str
    description: str
    enforcement_level: str = "strict"  # strict, warn, info
    auto_fixable: bool = False
    fix_suggestion: str = ""


class ContractSpec(BaseModel):
    """Full contract specification for a magic protocol."""

    model_config = ConfigDict(populate_by_name=True)

    provides: dict[str, ProvisionSpec] = Field(default_factory=dict)
    requires: RequirementSpec = Field(default_factory=RequirementSpec)
    side_effects: list[SideEffectSpec] = Field(default_factory=list, alias="sideEffects")


class MagicProtocolSchema(BaseModel):
    """Complete JSON-LD schema for a magic protocol.

    This is the primary exchange format for magic dependency definitions.
    """

    context: str = Field(
        default="https://vibecoding.dev/schema/magic-dependency/v1",
        alias="@context",
    )
    type_: str = Field(default="MagicProtocol", alias="@type")
    id_: str = Field(..., alias="@id")
    name: str
    protocol_type: str = Field(..., alias="protocolType")
    injection_mechanism: InjectionMechanismSpec = Field(..., alias="injectionMechanism")
    contract: ContractSpec
    influenced_nodes: list[str] = Field(default_factory=list, alias="influencedNodes")
    codex: CodexRuleSpec

    class Config:
        populate_by_name = True

    def to_jsonld(self) -> dict[str, Any]:
        """Serialize to JSON-LD format."""
        return self.model_dump(by_alias=True)

    @classmethod
    def from_jsonld(cls, data: dict[str, Any]) -> "MagicProtocolSchema":
        """Deserialize from JSON-LD format."""
        return cls.model_validate(data)


# Pre-defined magic protocol templates for common patterns

AUTH_MIDDLEWARE_TEMPLATE = MagicProtocolSchema(
    id_="magic:auth-middleware",
    name="AuthenticationMiddleware",
    protocol_type="middleware",
    injection_mechanism=InjectionMechanismSpec(
        framework="FastAPI",
        pattern="dependency_injection",
        scope="request",
    ),
    contract=ContractSpec(
        provides={
            "current_user": ProvisionSpec(
                name="current_user",
                type_annotation="User | None",
                availability="after_auth_check",
                nullable=True,
            )
        },
        requires=RequirementSpec(
            headers=["Authorization"],
            env_vars=["JWT_SECRET", "JWT_ALGORITHM"],
        ),
        side_effects=[
            SideEffectSpec(
                effect_type="exception",
                condition="invalid_token",
                raises="HTTPException(401)",
            )
        ],
    ),
    codex=CodexRuleSpec(
        rule_id="AUTH-001",
        description="Protected routes must declare auth dependency",
        enforcement_level="strict",
    ),
)

DATABASE_SESSION_TEMPLATE = MagicProtocolSchema(
    id_="magic:db-session",
    name="DatabaseSession",
    protocol_type="di",
    injection_mechanism=InjectionMechanismSpec(
        framework="FastAPI",
        pattern="dependency_injection",
        scope="request",
    ),
    contract=ContractSpec(
        provides={
            "db": ProvisionSpec(
                name="db",
                type_annotation="AsyncSession",
                availability="after_startup",
            )
        },
        requires=RequirementSpec(
            env_vars=["DATABASE_URL"],
        ),
        side_effects=[
            SideEffectSpec(
                effect_type="transaction",
                description="Auto-commits on success, rollbacks on exception",
            )
        ],
    ),
    codex=CodexRuleSpec(
        rule_id="DB-001",
        description="Database operations must use injected session",
        enforcement_level="strict",
    ),
)

ENV_CONFIG_TEMPLATE = MagicProtocolSchema(
    id_="magic:env-config",
    name="EnvironmentConfig",
    protocol_type="env",
    injection_mechanism=InjectionMechanismSpec(
        framework="pydantic-settings",
        pattern="singleton",
        scope="application",
    ),
    contract=ContractSpec(
        provides={
            "settings": ProvisionSpec(
                name="settings",
                type_annotation="Settings",
                availability="at_import",
            )
        },
        requires=RequirementSpec(
            env_vars=[],  # Will be populated dynamically
        ),
    ),
    codex=CodexRuleSpec(
        rule_id="ENV-001",
        description="All environment variables must be declared in Settings",
        enforcement_level="strict",
    ),
)
