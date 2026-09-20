"""
Database models — SQLAlchemy async models for episodic logging.

Phase 0 uses SQLite for simplicity. Phase 1 will migrate to PostgreSQL.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import String, Text, Float, Integer, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""
    pass


class TaskRecord(Base):
    """
    One record per user-submitted task.

    Stores the high-level task metadata and final outcome.
    """
    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    instruction: Mapped[str] = mapped_column(Text, nullable=False)
    task_type: Mapped[str] = mapped_column(String(100), default="")
    site: Mapped[str] = mapped_column(String(100), default="amazon.in")
    status: Mapped[str] = mapped_column(String(50), default="planning")
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    final_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Plan (JSON serialized)
    plan_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timing
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Relationships
    steps: Mapped[list["StepRecord"]] = relationship(
        back_populates="task", cascade="all, delete-orphan"
    )


class StepRecord(Base):
    """
    One record per action executed within a task.

    Stores the action details, page states before/after, and verification result.
    """
    __tablename__ = "steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(String(36), ForeignKey("tasks.id"), nullable=False)
    step_index: Mapped[int] = mapped_column(Integer, nullable=False)

    # Action details
    action_type: Mapped[str] = mapped_column(String(50), nullable=False)
    selector_used: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    input_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, default=False)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Page state snapshots (JSON — truncated for storage)
    page_state_before_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    page_state_after_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # URLs
    url_before: Mapped[Optional[str]] = mapped_column(String(2000), nullable=True)
    url_after: Mapped[Optional[str]] = mapped_column(String(2000), nullable=True)

    # Timing
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    task: Mapped["TaskRecord"] = relationship(back_populates="steps")
    verification: Mapped[Optional["VerificationRecord"]] = relationship(
        back_populates="step", cascade="all, delete-orphan", uselist=False
    )


class VerificationRecord(Base):
    """
    One record per verification check performed on a step.

    Stores which tier ran, the verdict, and detailed results.
    """
    __tablename__ = "verifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    step_id: Mapped[int] = mapped_column(Integer, ForeignKey("steps.id"), nullable=False)

    # Verification details
    tier: Mapped[str] = mapped_column(String(20), nullable=False)  # "rule", "llm", "human"
    passed: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Detailed results (JSON)
    details_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timing
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    step: Mapped["StepRecord"] = relationship(back_populates="verification")


class DomainMemoryRecord(Base):
    """
    Episodic memory record for a specific domain/website.

    Stores learned patterns, reliable selectors, overlay dismissal rules,
    and lessons from past successes/failures. The agent queries this memory
    before acting and updates it upon task completion.
    """
    __tablename__ = "domain_memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    domain: Mapped[str] = mapped_column(String(200), unique=True, index=True, nullable=False)
    site_name: Mapped[str] = mapped_column(String(100), default="")
    search_selectors_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    product_link_selectors_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    add_to_cart_selectors_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    popup_dismiss_selectors_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    general_tips: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    successful_runs: Mapped[int] = mapped_column(Integer, default=0)
    failed_runs: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class SkillRecord(Base):
    """
    A generalized, reusable skill learned from successful task trajectories.

    Skills are domain-specific or universal patterns (e.g. "search_product on flipkart.com",
    "dismiss_login_popup on any site") that the brain can recall and re-apply.
    """
    __tablename__ = "skills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    skill_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    domain_pattern: Mapped[str] = mapped_column(String(200), default="*", index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    selector_chain_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    preconditions: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    postconditions: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    success_rate: Mapped[float] = mapped_column(Float, default=1.0)
    usage_count: Mapped[int] = mapped_column(Integer, default=1)
    last_used_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class FailurePatternRecord(Base):
    """
    A learned failure pattern and its known solution.

    When the brain encounters an error it has seen before, it recalls the
    solution strategy and injects it into the planning prompt so the agent
    doesn't repeat the same mistake.
    """
    __tablename__ = "failure_patterns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    error_signature: Mapped[str] = mapped_column(String(300), nullable=False, index=True)
    domain: Mapped[str] = mapped_column(String(200), default="*", index=True)
    root_cause: Mapped[str] = mapped_column(Text, default="")
    solution_strategy: Mapped[str] = mapped_column(Text, default="")
    occurrence_count: Mapped[int] = mapped_column(Integer, default=1)
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class BrainStateRecord(Base):
    """
    Singleton record tracking global brain evolution metrics.

    Only one row exists (id=1). Updated after every task completion.
    """
    __tablename__ = "brain_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    total_tasks: Mapped[int] = mapped_column(Integer, default=0)
    successful_tasks: Mapped[int] = mapped_column(Integer, default=0)
    failed_tasks: Mapped[int] = mapped_column(Integer, default=0)
    total_skills_learned: Mapped[int] = mapped_column(Integer, default=0)
    total_failure_patterns: Mapped[int] = mapped_column(Integer, default=0)
    avg_duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    domains_mastered_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    last_learning_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
