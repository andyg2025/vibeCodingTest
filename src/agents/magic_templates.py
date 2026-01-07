"""Pre-defined Magic Protocol Templates.

Ready-to-use magic protocol definitions for common frameworks and patterns.
These templates can be used directly or customized for specific projects.

Supported frameworks:
- FastAPI (auth, DI, middleware, background tasks)
- SQLAlchemy (sessions, relationships, events)
- Pydantic (settings, validation)
- Redis (caching, sessions)
- Celery (background tasks)
"""

from typing import Any

from src.models.magic_protocol_schema import (
    CodexRuleSpec,
    ContractSpec,
    InjectionMechanismSpec,
    MagicProtocolSchema,
    ProvisionSpec,
    RequirementSpec,
    SideEffectSpec,
)


class FastAPITemplates:
    """Magic protocol templates for FastAPI."""

    @staticmethod
    def jwt_auth(
        user_type: str = "User",
        env_vars: list[str] | None = None,
    ) -> MagicProtocolSchema:
        """JWT authentication middleware."""
        return MagicProtocolSchema(
            id_="magic:fastapi-jwt-auth",
            name="FastAPI JWT Authentication",
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
                        type_annotation=f"{user_type} | None",
                        availability="after_token_validation",
                        nullable=True,
                    )
                },
                requires=RequirementSpec(
                    headers=["Authorization"],
                    env_vars=env_vars or ["JWT_SECRET", "JWT_ALGORITHM", "ACCESS_TOKEN_EXPIRE_MINUTES"],
                ),
                side_effects=[
                    SideEffectSpec(
                        effect_type="exception",
                        condition="missing_token",
                        raises="HTTPException(401, 'Not authenticated')",
                    ),
                    SideEffectSpec(
                        effect_type="exception",
                        condition="invalid_token",
                        raises="HTTPException(401, 'Invalid token')",
                    ),
                    SideEffectSpec(
                        effect_type="exception",
                        condition="expired_token",
                        raises="HTTPException(401, 'Token expired')",
                    ),
                ],
            ),
            codex=CodexRuleSpec(
                rule_id="FASTAPI-AUTH-001",
                description="Protected routes must include current_user dependency",
                enforcement_level="strict",
                fix_suggestion="Add 'current_user: User = Depends(get_current_user)' to route parameters",
            ),
        )

    @staticmethod
    def oauth2_password() -> MagicProtocolSchema:
        """OAuth2 password flow authentication."""
        return MagicProtocolSchema(
            id_="magic:fastapi-oauth2-password",
            name="FastAPI OAuth2 Password Flow",
            protocol_type="middleware",
            injection_mechanism=InjectionMechanismSpec(
                framework="FastAPI",
                pattern="dependency_injection",
                scope="request",
            ),
            contract=ContractSpec(
                provides={
                    "token_data": ProvisionSpec(
                        name="token_data",
                        type_annotation="TokenData",
                        availability="after_token_decode",
                    )
                },
                requires=RequirementSpec(
                    headers=["Authorization"],
                    env_vars=["SECRET_KEY"],
                ),
                side_effects=[
                    SideEffectSpec(
                        effect_type="exception",
                        condition="credentials_exception",
                        raises="HTTPException(401, headers={'WWW-Authenticate': 'Bearer'})",
                    ),
                ],
            ),
            codex=CodexRuleSpec(
                rule_id="FASTAPI-OAUTH-001",
                description="OAuth2 routes must handle WWW-Authenticate header",
                enforcement_level="strict",
            ),
        )

    @staticmethod
    def background_tasks() -> MagicProtocolSchema:
        """FastAPI BackgroundTasks."""
        return MagicProtocolSchema(
            id_="magic:fastapi-background-tasks",
            name="FastAPI Background Tasks",
            protocol_type="di",
            injection_mechanism=InjectionMechanismSpec(
                framework="FastAPI",
                pattern="dependency_injection",
                scope="request",
            ),
            contract=ContractSpec(
                provides={
                    "background_tasks": ProvisionSpec(
                        name="background_tasks",
                        type_annotation="BackgroundTasks",
                        availability="always",
                    )
                },
                requires=RequirementSpec(),
                side_effects=[
                    SideEffectSpec(
                        effect_type="async",
                        description="Tasks execute after response, in same process",
                    ),
                    SideEffectSpec(
                        effect_type="exception",
                        condition="task_failure",
                        description="Exceptions logged but not propagated to client",
                    ),
                ],
            ),
            codex=CodexRuleSpec(
                rule_id="FASTAPI-BG-001",
                description="Background tasks must not access request-scoped objects",
                enforcement_level="warn",
                fix_suggestion="Pass all needed data as arguments, not references",
            ),
        )

    @staticmethod
    def cors_middleware(allowed_origins: list[str] | None = None) -> MagicProtocolSchema:
        """CORS middleware."""
        return MagicProtocolSchema(
            id_="magic:fastapi-cors",
            name="FastAPI CORS Middleware",
            protocol_type="middleware",
            injection_mechanism=InjectionMechanismSpec(
                framework="FastAPI",
                pattern="middleware",
                scope="application",
            ),
            contract=ContractSpec(
                provides={},
                requires=RequirementSpec(
                    env_vars=["CORS_ORIGINS"] if not allowed_origins else [],
                ),
                side_effects=[
                    SideEffectSpec(
                        effect_type="headers",
                        description="Adds Access-Control-* headers to responses",
                    ),
                    SideEffectSpec(
                        effect_type="exception",
                        condition="origin_not_allowed",
                        description="OPTIONS preflight fails, request blocked",
                    ),
                ],
            ),
            codex=CodexRuleSpec(
                rule_id="FASTAPI-CORS-001",
                description="CORS must be configured before routes are added",
                enforcement_level="strict",
            ),
        )

    @staticmethod
    def rate_limiter() -> MagicProtocolSchema:
        """Rate limiting middleware."""
        return MagicProtocolSchema(
            id_="magic:fastapi-rate-limit",
            name="FastAPI Rate Limiter",
            protocol_type="middleware",
            injection_mechanism=InjectionMechanismSpec(
                framework="FastAPI",
                pattern="middleware",
                scope="request",
            ),
            contract=ContractSpec(
                provides={},
                requires=RequirementSpec(
                    env_vars=["RATE_LIMIT_PER_MINUTE"],
                    dependencies=["magic:redis-client"],
                ),
                side_effects=[
                    SideEffectSpec(
                        effect_type="exception",
                        condition="rate_exceeded",
                        raises="HTTPException(429, 'Too many requests')",
                    ),
                    SideEffectSpec(
                        effect_type="headers",
                        description="Adds X-RateLimit-* headers",
                    ),
                ],
            ),
            codex=CodexRuleSpec(
                rule_id="FASTAPI-RATE-001",
                description="Rate-limited routes should handle 429 gracefully",
                enforcement_level="info",
            ),
        )


class SQLAlchemyTemplates:
    """Magic protocol templates for SQLAlchemy."""

    @staticmethod
    def async_session() -> MagicProtocolSchema:
        """Async database session."""
        return MagicProtocolSchema(
            id_="magic:sqlalchemy-async-session",
            name="SQLAlchemy Async Session",
            protocol_type="di",
            injection_mechanism=InjectionMechanismSpec(
                framework="SQLAlchemy",
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
                        effect_type="database",
                        condition="on_success",
                        description="Auto-commits transaction at request end",
                    ),
                    SideEffectSpec(
                        effect_type="database",
                        condition="on_exception",
                        description="Auto-rollbacks transaction",
                    ),
                    SideEffectSpec(
                        effect_type="exception",
                        condition="connection_failed",
                        raises="SQLAlchemyError",
                    ),
                ],
            ),
            codex=CodexRuleSpec(
                rule_id="SQLA-SESSION-001",
                description="Never create sessions manually - use injected session",
                enforcement_level="strict",
                fix_suggestion="Use 'db: AsyncSession = Depends(get_db)' parameter",
            ),
        )

    @staticmethod
    def relationship_lazy_load() -> MagicProtocolSchema:
        """ORM relationship lazy loading."""
        return MagicProtocolSchema(
            id_="magic:sqlalchemy-relationship",
            name="SQLAlchemy Relationship Loading",
            protocol_type="orm",
            injection_mechanism=InjectionMechanismSpec(
                framework="SQLAlchemy",
                pattern="orm_magic",
                scope="session",
            ),
            contract=ContractSpec(
                provides={
                    "related_objects": ProvisionSpec(
                        name="related_objects",
                        type_annotation="list[Model]",
                        availability="on_access",
                    )
                },
                requires=RequirementSpec(
                    dependencies=["magic:sqlalchemy-async-session"],
                ),
                side_effects=[
                    SideEffectSpec(
                        effect_type="database",
                        condition="lazy_load",
                        description="Additional queries triggered on attribute access",
                    ),
                    SideEffectSpec(
                        effect_type="exception",
                        condition="detached_instance",
                        raises="DetachedInstanceError (if session closed)",
                    ),
                ],
            ),
            codex=CodexRuleSpec(
                rule_id="SQLA-REL-001",
                description="Use selectinload/joinedload for known relationship access",
                enforcement_level="warn",
                fix_suggestion="Add .options(selectinload(Model.relationship)) to query",
            ),
        )

    @staticmethod
    def cascade_operations() -> MagicProtocolSchema:
        """ORM cascade delete/update operations."""
        return MagicProtocolSchema(
            id_="magic:sqlalchemy-cascade",
            name="SQLAlchemy Cascade Operations",
            protocol_type="orm",
            injection_mechanism=InjectionMechanismSpec(
                framework="SQLAlchemy",
                pattern="orm_magic",
                scope="session",
            ),
            contract=ContractSpec(
                provides={},
                requires=RequirementSpec(),
                side_effects=[
                    SideEffectSpec(
                        effect_type="database",
                        condition="delete_parent",
                        description="Children with cascade='delete' are auto-deleted",
                    ),
                    SideEffectSpec(
                        effect_type="database",
                        condition="update_parent_pk",
                        description="Children FKs updated if cascade includes 'save-update'",
                    ),
                ],
            ),
            codex=CodexRuleSpec(
                rule_id="SQLA-CASCADE-001",
                description="Document cascade behavior in model docstrings",
                enforcement_level="info",
            ),
        )


class PydanticTemplates:
    """Magic protocol templates for Pydantic."""

    @staticmethod
    def settings(env_vars: list[str]) -> MagicProtocolSchema:
        """Pydantic BaseSettings for configuration."""
        return MagicProtocolSchema(
            id_="magic:pydantic-settings",
            name="Pydantic BaseSettings",
            protocol_type="env",
            injection_mechanism=InjectionMechanismSpec(
                framework="Pydantic",
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
                    env_vars=env_vars,
                ),
                side_effects=[
                    SideEffectSpec(
                        effect_type="exception",
                        condition="missing_required_field",
                        raises="ValidationError",
                    ),
                    SideEffectSpec(
                        effect_type="exception",
                        condition="invalid_field_value",
                        raises="ValidationError",
                    ),
                ],
            ),
            codex=CodexRuleSpec(
                rule_id="PYDANTIC-SETTINGS-001",
                description="All environment configuration must go through Settings class",
                enforcement_level="strict",
                fix_suggestion="Add field to Settings class instead of using os.getenv directly",
            ),
        )


class RedisTemplates:
    """Magic protocol templates for Redis."""

    @staticmethod
    def client() -> MagicProtocolSchema:
        """Redis client connection."""
        return MagicProtocolSchema(
            id_="magic:redis-client",
            name="Redis Client",
            protocol_type="di",
            injection_mechanism=InjectionMechanismSpec(
                framework="redis-py",
                pattern="singleton",
                scope="application",
            ),
            contract=ContractSpec(
                provides={
                    "redis": ProvisionSpec(
                        name="redis",
                        type_annotation="Redis",
                        availability="after_startup",
                    )
                },
                requires=RequirementSpec(
                    env_vars=["REDIS_URL"],
                ),
                side_effects=[
                    SideEffectSpec(
                        effect_type="exception",
                        condition="connection_failed",
                        raises="ConnectionError",
                    ),
                ],
            ),
            codex=CodexRuleSpec(
                rule_id="REDIS-001",
                description="Use connection pooling for Redis clients",
                enforcement_level="warn",
            ),
        )

    @staticmethod
    def cache_decorator() -> MagicProtocolSchema:
        """Redis-backed caching decorator."""
        return MagicProtocolSchema(
            id_="magic:redis-cache",
            name="Redis Cache Decorator",
            protocol_type="decorator",
            injection_mechanism=InjectionMechanismSpec(
                framework="redis-py",
                pattern="decorator",
                scope="function",
            ),
            contract=ContractSpec(
                provides={
                    "cached_result": ProvisionSpec(
                        name="cached_result",
                        type_annotation="Any",
                        availability="if_cached",
                    )
                },
                requires=RequirementSpec(
                    dependencies=["magic:redis-client"],
                ),
                side_effects=[
                    SideEffectSpec(
                        effect_type="cache",
                        condition="cache_hit",
                        description="Returns cached value, skips function execution",
                    ),
                    SideEffectSpec(
                        effect_type="cache",
                        condition="cache_miss",
                        description="Executes function, stores result in cache",
                    ),
                ],
            ),
            codex=CodexRuleSpec(
                rule_id="REDIS-CACHE-001",
                description="Cached functions must be idempotent",
                enforcement_level="strict",
            ),
        )


class CeleryTemplates:
    """Magic protocol templates for Celery."""

    @staticmethod
    def task() -> MagicProtocolSchema:
        """Celery async task."""
        return MagicProtocolSchema(
            id_="magic:celery-task",
            name="Celery Task",
            protocol_type="decorator",
            injection_mechanism=InjectionMechanismSpec(
                framework="Celery",
                pattern="decorator",
                scope="function",
            ),
            contract=ContractSpec(
                provides={
                    "async_result": ProvisionSpec(
                        name="async_result",
                        type_annotation="AsyncResult",
                        availability="on_delay_call",
                    )
                },
                requires=RequirementSpec(
                    env_vars=["CELERY_BROKER_URL", "CELERY_RESULT_BACKEND"],
                ),
                side_effects=[
                    SideEffectSpec(
                        effect_type="async",
                        description="Task executed in separate worker process",
                    ),
                    SideEffectSpec(
                        effect_type="network",
                        description="Task serialized and sent to message broker",
                    ),
                ],
            ),
            codex=CodexRuleSpec(
                rule_id="CELERY-001",
                description="Celery tasks must use serializable arguments only",
                enforcement_level="strict",
                fix_suggestion="Pass IDs instead of ORM objects, use JSON-serializable types",
            ),
        )


def get_all_templates() -> dict[str, list[MagicProtocolSchema]]:
    """Get all available magic protocol templates grouped by framework."""
    return {
        "fastapi": [
            FastAPITemplates.jwt_auth(),
            FastAPITemplates.oauth2_password(),
            FastAPITemplates.background_tasks(),
            FastAPITemplates.cors_middleware(),
            FastAPITemplates.rate_limiter(),
        ],
        "sqlalchemy": [
            SQLAlchemyTemplates.async_session(),
            SQLAlchemyTemplates.relationship_lazy_load(),
            SQLAlchemyTemplates.cascade_operations(),
        ],
        "pydantic": [
            PydanticTemplates.settings([]),
        ],
        "redis": [
            RedisTemplates.client(),
            RedisTemplates.cache_decorator(),
        ],
        "celery": [
            CeleryTemplates.task(),
        ],
    }


def get_templates_for_stack(tech_stack: list[str]) -> list[MagicProtocolSchema]:
    """Get relevant templates based on project tech stack."""
    all_templates = get_all_templates()
    result = []

    tech_stack_lower = [t.lower() for t in tech_stack]

    for framework, templates in all_templates.items():
        if any(framework in tech for tech in tech_stack_lower):
            result.extend(templates)

    return result
