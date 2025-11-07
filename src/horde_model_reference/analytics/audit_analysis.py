"""Audit analysis for model references.

Provides functions to analyze models for deletion risk and audit-worthiness.
Identifies issues like missing downloads, non-preferred hosts, low usage, etc.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, computed_field

from horde_model_reference import horde_model_reference_settings
from horde_model_reference.integrations.data_merger import CombinedModelStatistics
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import (
    GenericModelRecord,
    ImageGenerationModelRecord,
    TextGenerationModelRecord,
)


class UsageTrend(BaseModel):
    """Usage trend metrics comparing different time periods.

    Ratios help identify model momentum and activity patterns.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    day_to_month_ratio: float | None = None
    """Ratio of day usage to month usage (day/month). None if month usage is zero."""
    month_to_total_ratio: float | None = None
    """Ratio of month usage to total usage (month/total). None if total usage is zero."""


class DeletionRiskFlags(BaseModel):
    """Flags indicating potential reasons for model deletion.

    Each flag represents a specific issue that might warrant model removal.
    Aligned with frontend expectations for consistent presentation.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    zero_usage_day: bool = False
    """Model has no usage in the past day."""
    zero_usage_month: bool = False
    """Model has no usage in the past month."""
    zero_usage_total: bool = False
    """Model has no usage in total (all time)."""
    no_active_workers: bool = False
    """Model has zero active workers (from Horde API data)."""
    has_multiple_hosts: bool = False
    """Model downloads are distributed across multiple file hosts."""
    has_non_preferred_host: bool = False
    """Model is hosted on non-preferred file hosts (non-HuggingFace)."""
    has_unknown_host: bool = False
    """Model has download URLs with unknown or unparseable hosts."""
    no_download_urls: bool = False
    """Model has no download URLs, empty download list, or invalid URLs."""
    missing_description: bool = False
    """Model lacks a description field."""
    missing_baseline: bool = False
    """Model lacks baseline information (for applicable categories)."""
    low_usage: bool = False
    """Model has very low usage (threshold defined by LOW_USAGE_THRESHOLD constant)."""
    parameter_mismatch: bool = False
    """Model parameter count in name doesn't match parameters field in record."""

    def any_flags(self) -> bool:
        """Check if any deletion risk flags are set.

        Returns:
            True if at least one flag is set, False otherwise.
        """
        return any(
            [
                self.zero_usage_day,
                self.zero_usage_month,
                self.zero_usage_total,
                self.no_active_workers,
                self.has_multiple_hosts,
                self.has_non_preferred_host,
                self.has_unknown_host,
                self.no_download_urls,
                self.missing_description,
                self.missing_baseline,
                self.low_usage,
                self.parameter_mismatch,
            ]
        )

    def flag_count(self) -> int:
        """Count the number of flags set.

        Returns:
            Number of deletion risk flags that are True.
        """
        return sum(
            [
                self.zero_usage_day,
                self.zero_usage_month,
                self.zero_usage_total,
                self.no_active_workers,
                self.has_multiple_hosts,
                self.has_non_preferred_host,
                self.has_unknown_host,
                self.no_download_urls,
                self.missing_description,
                self.missing_baseline,
                self.low_usage,
                self.parameter_mismatch,
            ]
        )


class FlagValidatorService:
    """Service providing reusable flag validation methods.

    All methods are static and stateless for efficient reuse.
    """

    @staticmethod
    def validate_downloads(
        downloads: list[Any] | None,
        preferred_hosts: list[str] | None = None,
    ) -> tuple[bool, bool, bool, bool]:
        """Validate download URLs and return related flags.

        Args:
            downloads: List of download records to validate.
            preferred_hosts: List of preferred file hosts. If None, uses settings.

        Returns:
            Tuple of (no_download_urls, has_multiple_hosts, has_non_preferred_host, has_unknown_host)
        """
        if preferred_hosts is None:
            preferred_hosts = horde_model_reference_settings.preferred_file_hosts

        unique_hosts: set[str] = set()
        has_valid_url = False
        has_preferred_host = False
        has_unknown_host = False

        if not downloads or len(downloads) == 0:
            return (True, False, False, False)

        for download in downloads:
            url = download.file_url
            if url:
                try:
                    parsed = urlparse(url)
                    if parsed.scheme and parsed.netloc:
                        has_valid_url = True
                        unique_hosts.add(parsed.netloc)

                        # Check if this host is preferred
                        if preferred_hosts and any(host in parsed.netloc for host in preferred_hosts):
                            has_preferred_host = True
                except Exception:
                    # Failed to parse URL - unknown host
                    has_unknown_host = True

        no_download_urls = not has_valid_url
        has_multiple_hosts = len(unique_hosts) > 1
        has_non_preferred_host = has_valid_url and not has_preferred_host

        return (no_download_urls, has_multiple_hosts, has_non_preferred_host, has_unknown_host)

    @staticmethod
    def validate_description(description: str | None) -> bool:
        """Validate model description.

        Args:
            description: The model description to validate.

        Returns:
            True if description is missing or empty.
        """
        return not description or len(description.strip()) == 0

    @staticmethod
    def validate_baseline(baseline: str | None) -> bool:
        """Validate model baseline.

        Args:
            baseline: The model baseline to validate.

        Returns:
            True if baseline is missing.
        """
        return not baseline

    @staticmethod
    def validate_parameter_count(
        model_name: str,
        declared_parameters: int | None,
        tolerance_percent: float = 10.0,
    ) -> bool:
        """Validate that parameter count in name matches declared parameters.

        Extracts parameter count from model name (e.g., "7B", "13B") and compares
        to the declared parameters field with a tolerance for rounding differences.

        Args:
            model_name: The model name to extract parameters from.
            declared_parameters: The declared parameter count (actual count, e.g., 7000000000 for 7B).
            tolerance_percent: Allowed percentage difference (default 10%).

        Returns:
            True if there is a mismatch (flag should be set), False otherwise.

        Examples:
            >>> validate_parameter_count("Llama-3-8B", 8000000000)
            False  # Matches exactly
            >>> validate_parameter_count("Llama-3-8B", 7000000000)
            True   # Mismatch (8B != 7B)
            >>> validate_parameter_count("Llama-3-8B", 8100000000)
            False  # Within 10% tolerance
        """
        from horde_model_reference.analytics.text_model_parser import extract_parameter_count_from_name

        # If no declared parameters, we can't validate
        if declared_parameters is None:
            return False

        # Try to extract parameters from name (returns in millions)
        extracted_params_millions = extract_parameter_count_from_name(model_name)

        # If we can't extract from name, no mismatch to flag
        if extracted_params_millions is None:
            return False

        # Convert declared parameters to millions for comparison
        declared_params_millions = declared_parameters / 1_000_000

        # Calculate percentage difference
        if declared_params_millions == 0:
            # Avoid division by zero; if declared is 0 but extracted isn't, that's a mismatch
            return extracted_params_millions != 0

        diff_percent = abs(extracted_params_millions - declared_params_millions) / declared_params_millions * 100

        # Flag as mismatch if difference exceeds tolerance
        return diff_percent > tolerance_percent

    @staticmethod
    def validate_statistics(
        statistics: CombinedModelStatistics | None,
        category_total_usage: int,
        low_usage_threshold: float | None = None,
    ) -> tuple[bool, bool, bool, bool, bool]:
        """Validate Horde API statistics and usage data.

        Args:
            statistics: Optional Horde API statistics.
            category_total_usage: Total monthly usage for the category.
            low_usage_threshold: Percentage threshold for low usage. If None, uses settings default.

        Returns:
            Tuple of (zero_usage_day, zero_usage_month, zero_usage_total, no_active_workers, low_usage)
        """
        if not statistics:
            return (False, False, False, False, False)

        # Use configured threshold if not provided
        if low_usage_threshold is None:
            low_usage_threshold = horde_model_reference_settings.low_usage_threshold_percentage

        # Check for zero workers
        no_active_workers = statistics.worker_count == 0

        zero_usage_day = False
        zero_usage_month = False
        zero_usage_total = False
        low_usage = False

        # Check usage statistics
        if statistics.usage_stats:
            usage_day = statistics.usage_stats.day
            usage_month = statistics.usage_stats.month
            usage_total = statistics.usage_stats.total

            zero_usage_day = usage_day == 0
            zero_usage_month = usage_month == 0
            zero_usage_total = usage_total == 0

            # Low usage (configurable threshold via settings)
            if category_total_usage > 0:
                usage_percentage = (usage_month / category_total_usage) * 100.0
                low_usage = usage_percentage < low_usage_threshold

        return (zero_usage_day, zero_usage_month, zero_usage_total, no_active_workers, low_usage)


class DeletionRiskFlagsBuilder:
    """Builder for composing DeletionRiskFlags with type-safe methods.

    Provides a fluent interface for setting flags without magic strings.
    """

    def __init__(self) -> None:
        """Initialize the builder with default flag values."""
        self.zero_usage_day: bool = False
        self.zero_usage_month: bool = False
        self.zero_usage_total: bool = False
        self.no_active_workers: bool = False
        self.has_multiple_hosts: bool = False
        self.has_non_preferred_host: bool = False
        self.has_unknown_host: bool = False
        self.no_download_urls: bool = False
        self.missing_description: bool = False
        self.missing_baseline: bool = False
        self.low_usage: bool = False
        self.parameter_mismatch: bool = False

    def with_download_flags(
        self,
        no_download_urls: bool,
        has_multiple_hosts: bool,
        has_non_preferred_host: bool,
        has_unknown_host: bool,
    ) -> DeletionRiskFlagsBuilder:
        """Set download-related flags.

        Args:
            no_download_urls: Whether model has no download URLs.
            has_multiple_hosts: Whether downloads span multiple hosts.
            has_non_preferred_host: Whether downloads use non-preferred hosts.
            has_unknown_host: Whether any download URLs couldn't be parsed.

        Returns:
            Self for method chaining.
        """
        self.no_download_urls = no_download_urls
        self.has_multiple_hosts = has_multiple_hosts
        self.has_non_preferred_host = has_non_preferred_host
        self.has_unknown_host = has_unknown_host
        return self

    def with_missing_description(self, missing: bool) -> DeletionRiskFlagsBuilder:
        """Set missing_description flag.

        Args:
            missing: Whether description is missing.

        Returns:
            Self for method chaining.
        """
        self.missing_description = missing
        return self

    def with_missing_baseline(self, missing: bool) -> DeletionRiskFlagsBuilder:
        """Set missing_baseline flag.

        Args:
            missing: Whether baseline is missing.

        Returns:
            Self for method chaining.
        """
        self.missing_baseline = missing
        return self

    def with_parameter_mismatch(self, mismatch: bool) -> DeletionRiskFlagsBuilder:
        """Set parameter_mismatch flag.

        Args:
            mismatch: Whether parameter count in name doesn't match declared parameters.

        Returns:
            Self for method chaining.
        """
        self.parameter_mismatch = mismatch
        return self

    def with_statistics_flags(
        self,
        zero_usage_day: bool,
        zero_usage_month: bool,
        zero_usage_total: bool,
        no_active_workers: bool,
        low_usage: bool,
    ) -> DeletionRiskFlagsBuilder:
        """Set statistics-related flags.

        Args:
            zero_usage_day: Whether day usage is zero.
            zero_usage_month: Whether month usage is zero.
            zero_usage_total: Whether total usage is zero.
            no_active_workers: Whether worker count is zero.
            low_usage: Whether usage is below threshold.

        Returns:
            Self for method chaining.
        """
        self.zero_usage_day = zero_usage_day
        self.zero_usage_month = zero_usage_month
        self.zero_usage_total = zero_usage_total
        self.no_active_workers = no_active_workers
        self.low_usage = low_usage
        return self

    def build(self) -> DeletionRiskFlags:
        """Build the final DeletionRiskFlags object.

        Returns:
            DeletionRiskFlags with all accumulated flag values.
        """
        return DeletionRiskFlags(
            zero_usage_day=self.zero_usage_day,
            zero_usage_month=self.zero_usage_month,
            zero_usage_total=self.zero_usage_total,
            no_active_workers=self.no_active_workers,
            has_multiple_hosts=self.has_multiple_hosts,
            has_non_preferred_host=self.has_non_preferred_host,
            has_unknown_host=self.has_unknown_host,
            no_download_urls=self.no_download_urls,
            missing_description=self.missing_description,
            missing_baseline=self.missing_baseline,
            low_usage=self.low_usage,
            parameter_mismatch=self.parameter_mismatch,
        )


class ModelAuditInfo(BaseModel):
    """Audit information for a single model.

    Contains model metadata along with deletion risk assessment and usage statistics.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    name: str
    """The model name."""
    category: MODEL_REFERENCE_CATEGORY
    """The category this model belongs to."""

    deletion_risk_flags: DeletionRiskFlags
    """Flags indicating potential deletion risks."""
    at_risk: bool
    """True if model has any deletion risk flags."""
    risk_score: int = Field(ge=0)
    """Total number of deletion risk flags (0 = no risk)."""

    worker_count: int = Field(ge=0, default=0)
    """Number of active workers serving this model."""
    usage_day: int = Field(ge=0, default=0)
    """Usage count for the past day."""
    usage_month: int = Field(ge=0, default=0)
    """Usage count for the past month."""
    usage_total: int = Field(ge=0, default=0)
    """Total usage count (all time)."""
    usage_hour: int | None = None
    """Usage count for the past hour (will be populated when alternative Horde API sources become available)."""
    usage_minute: int | None = None
    """Usage count for the past minute (will be populated when alternative Horde API sources become available)."""
    usage_percentage_of_category: float = Field(ge=0.0, le=100.0, default=0.0)
    """Percentage of category's total monthly usage."""
    usage_trend: UsageTrend
    """Usage trend comparing different time periods."""

    cost_benefit_score: float | None = None
    """Cost-benefit metric: usage per GB (usage_month / size_gb). Only for models with size info."""
    size_gb: float | None = None
    """Model size in gigabytes (if available)."""

    baseline: str | None = None
    """Model baseline (if applicable)."""
    nsfw: bool | None = None
    """Whether model is NSFW (if applicable)."""
    has_description: bool = False
    """Whether model has a description."""
    download_count: int = Field(ge=0, default=0)
    """Number of download entries."""
    download_hosts: list[str] = Field(default_factory=list)
    """List of download host domains."""
    variations: list[str] = Field(default_factory=list)
    """List of model names that are variations of this model (substring matches)."""

    @property
    def flag_count(self) -> int:
        """Count of deletion risk flags set.

        Returns:
            Number of flags that are True.
        """
        return self.deletion_risk_flags.flag_count()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_critical(self) -> bool:
        """Determine if model is in critical state.

        Critical = zero month usage AND no active workers.

        Returns:
            True if model meets critical criteria.
        """
        return self.deletion_risk_flags.zero_usage_month and self.deletion_risk_flags.no_active_workers

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_warning(self) -> bool:
        """Determine if model has warning-level issues.

        Warnings include host-related issues or download problems.

        Returns:
            True if model has warning-level flags.
        """
        return (
            self.deletion_risk_flags.has_multiple_hosts
            or self.deletion_risk_flags.has_non_preferred_host
            or self.deletion_risk_flags.has_unknown_host
            or self.deletion_risk_flags.no_download_urls
        )


class CategoryAuditSummary(BaseModel):
    """Summary statistics for a category audit.

    Aggregates audit information across all models in a category.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    total_models: int = Field(ge=0)
    """Total number of models in the category."""
    models_at_risk: int = Field(ge=0)
    """Number of models with at least one deletion risk flag."""
    models_critical: int = Field(ge=0)
    """Number of models in critical state (zero month usage AND no workers)."""
    models_with_warnings: int = Field(ge=0)
    """Number of models with warning-level issues (host/download problems)."""

    models_with_zero_day_usage: int = Field(ge=0, default=0)
    """Number of models with zero usage in the past day."""
    models_with_zero_month_usage: int = Field(ge=0, default=0)
    """Number of models with zero usage in the past month."""
    models_with_zero_total_usage: int = Field(ge=0, default=0)
    """Number of models with zero usage all-time."""
    models_with_no_active_workers: int = Field(ge=0, default=0)
    """Number of models with no active workers."""
    models_with_no_downloads: int = Field(ge=0, default=0)
    """Number of models without download URLs."""
    models_with_non_preferred_hosts: int = Field(ge=0, default=0)
    """Number of models on non-preferred hosts."""
    models_with_multiple_hosts: int = Field(ge=0, default=0)
    """Number of models with downloads across multiple hosts."""
    models_with_low_usage: int = Field(ge=0, default=0)
    """Number of models with very low usage (< 0.1% of category)."""

    average_risk_score: float = Field(ge=0.0)
    """Average risk score across all models."""
    category_total_month_usage: int = Field(ge=0, default=0)
    """Total monthly usage for the entire category."""

    @classmethod
    def from_audit_models(cls, audit_models: list[ModelAuditInfo]) -> CategoryAuditSummary:
        """Calculate summary statistics from audit models.

        Args:
            audit_models: List of ModelAuditInfo objects.

        Returns:
            CategoryAuditSummary with aggregate statistics.
        """
        total_models = len(audit_models)
        models_at_risk = sum(1 for m in audit_models if m.at_risk)
        models_critical = sum(1 for m in audit_models if m.is_critical)
        models_with_warnings = sum(1 for m in audit_models if m.has_warning)

        models_with_zero_day_usage = sum(1 for m in audit_models if m.deletion_risk_flags.zero_usage_day)
        models_with_zero_month_usage = sum(1 for m in audit_models if m.deletion_risk_flags.zero_usage_month)
        models_with_zero_total_usage = sum(1 for m in audit_models if m.deletion_risk_flags.zero_usage_total)
        models_with_no_active_workers = sum(1 for m in audit_models if m.deletion_risk_flags.no_active_workers)
        models_with_no_downloads = sum(1 for m in audit_models if m.deletion_risk_flags.no_download_urls)
        models_with_non_preferred_hosts = sum(1 for m in audit_models if m.deletion_risk_flags.has_non_preferred_host)
        models_with_multiple_hosts = sum(1 for m in audit_models if m.deletion_risk_flags.has_multiple_hosts)
        models_with_low_usage = sum(1 for m in audit_models if m.deletion_risk_flags.low_usage)

        category_total_month_usage = sum(m.usage_month for m in audit_models)

        total_risk_score = sum(m.risk_score for m in audit_models)
        average_risk_score = total_risk_score / total_models if total_models > 0 else 0.0

        return cls(
            total_models=total_models,
            models_at_risk=models_at_risk,
            models_critical=models_critical,
            models_with_warnings=models_with_warnings,
            models_with_zero_day_usage=models_with_zero_day_usage,
            models_with_zero_month_usage=models_with_zero_month_usage,
            models_with_zero_total_usage=models_with_zero_total_usage,
            models_with_no_active_workers=models_with_no_active_workers,
            models_with_no_downloads=models_with_no_downloads,
            models_with_non_preferred_hosts=models_with_non_preferred_hosts,
            models_with_multiple_hosts=models_with_multiple_hosts,
            models_with_low_usage=models_with_low_usage,
            average_risk_score=round(average_risk_score, 2),
            category_total_month_usage=category_total_month_usage,
        )


class CategoryAuditResponse(BaseModel):
    """Complete audit response for a category.

    Contains both per-model audit information and aggregate summary.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    category: MODEL_REFERENCE_CATEGORY
    """The category being audited."""
    category_total_month_usage: int = Field(ge=0)
    """Total monthly usage for the entire category."""

    total_count: int = Field(ge=0, default=0)
    """Total number of models in the category (before pagination/filtering)."""
    returned_count: int = Field(ge=0, default=0)
    """Number of models returned in this response (after pagination/filtering)."""
    offset: int = Field(ge=0, default=0)
    """Starting index for pagination (0 if not paginated)."""
    limit: int | None = None
    """Maximum number of models per page (None if not paginated)."""

    models: list[ModelAuditInfo]
    """List of audit information for each model."""
    summary: CategoryAuditSummary
    """Aggregate summary statistics."""


class DeletionRiskFlagsHandler:
    """Abstract handler for creating DeletionRiskFlags from specific model record types.

    Subclasses should implement type-specific flag generation logic.
    """

    def can_handle(self, model_record: GenericModelRecord) -> bool:
        """Check if this handler can process the given model record.

        Args:
            model_record: The model record to check.

        Returns:
            True if this handler can process the model record type.
        """
        raise NotImplementedError("Subclasses must implement can_handle")

    def create_flags(
        self,
        *,
        model_name: str,
        model_record: GenericModelRecord,
        statistics: CombinedModelStatistics | None,
        category_total_usage: int,
    ) -> DeletionRiskFlags:
        """Create DeletionRiskFlags for a model record.

        Args:
            model_name: The name of the model.
            model_record: The model record.
            statistics: Optional Horde API statistics.
            category_total_usage: Total monthly usage for the category.

        Returns:
            DeletionRiskFlags object.
        """
        raise NotImplementedError("Subclasses must implement create_flags")


class ImageGenerationDeletionRiskFlagsHandler(DeletionRiskFlagsHandler):
    """Handler for image generation model deletion risk flags creation."""

    def can_handle(self, model_record: GenericModelRecord) -> bool:
        """Check if this handler can process the given model record.

        Args:
            model_record: The model record to check.

        Returns:
            True if the model record is an ImageGenerationModelRecord.
        """
        return isinstance(model_record, ImageGenerationModelRecord)

    def _create_flags_impl(
        self,
        model_record: ImageGenerationModelRecord,
        statistics: CombinedModelStatistics | None,
        category_total_usage: int,
    ) -> DeletionRiskFlags:
        """Analyze an image generation model and determine deletion risk flags.

        Args:
            model_record: Typed image generation model record.
            statistics: Optional Horde API statistics (worker_count, usage_stats).
            category_total_usage: Total monthly usage for the category (for percentage calculations).

        Returns:
            DeletionRiskFlags with appropriate flags set.
        """
        downloads = model_record.config.download if model_record.config else []

        return (
            DeletionRiskFlagsBuilder()
            .with_download_flags(*FlagValidatorService.validate_downloads(downloads))
            .with_missing_description(FlagValidatorService.validate_description(model_record.description))
            .with_missing_baseline(FlagValidatorService.validate_baseline(model_record.baseline))
            .with_statistics_flags(*FlagValidatorService.validate_statistics(statistics, category_total_usage))
            .build()
        )

    def create_flags(
        self,
        *,
        model_name: str,
        model_record: GenericModelRecord,
        statistics: CombinedModelStatistics | None,
        category_total_usage: int,
    ) -> DeletionRiskFlags:
        """Create DeletionRiskFlags for an image generation model.

        Args:
            model_name: The name of the model.
            model_record: The image generation model record.
            statistics: Optional Horde API statistics.
            category_total_usage: Total monthly usage for the category.

        Returns:
            DeletionRiskFlags object.
        """
        if not isinstance(model_record, ImageGenerationModelRecord):
            error_message = f"Expected ImageGenerationModelRecord, got {type(model_record).__name__}"
            raise TypeError(error_message)

        return self._create_flags_impl(model_record, statistics, category_total_usage)


class TextGenerationDeletionRiskFlagsHandler(DeletionRiskFlagsHandler):
    """Handler for text generation model deletion risk flags creation."""

    def can_handle(self, model_record: GenericModelRecord) -> bool:
        """Check if this handler can process the given model record.

        Args:
            model_record: The model record to check.

        Returns:
            True if the model record is a TextGenerationModelRecord.
        """
        return isinstance(model_record, TextGenerationModelRecord)

    def _create_flags_impl(
        self,
        model_name: str,
        model_record: TextGenerationModelRecord,
        statistics: CombinedModelStatistics | None,
        category_total_usage: int,
    ) -> DeletionRiskFlags:
        """Analyze a text generation model and determine deletion risk flags.

        Args:
            model_name: The name of the model.
            model_record: Typed text generation model record.
            statistics: Optional Horde API statistics (worker_count, usage_stats).
            category_total_usage: Total monthly usage for the category (for percentage calculations).

        Returns:
            DeletionRiskFlags with appropriate flags set.
        """
        downloads = model_record.config.download if model_record.config else []

        return (
            DeletionRiskFlagsBuilder()
            .with_download_flags(*FlagValidatorService.validate_downloads(downloads))
            .with_missing_description(FlagValidatorService.validate_description(model_record.description))
            .with_missing_baseline(FlagValidatorService.validate_baseline(model_record.baseline))
            .with_statistics_flags(*FlagValidatorService.validate_statistics(statistics, category_total_usage))
            .with_parameter_mismatch(
                FlagValidatorService.validate_parameter_count(model_name, model_record.parameters_count)
            )
            .build()
        )

    def create_flags(
        self,
        *,
        model_name: str,
        model_record: GenericModelRecord,
        statistics: CombinedModelStatistics | None,
        category_total_usage: int,
    ) -> DeletionRiskFlags:
        """Create DeletionRiskFlags for a text generation model.

        Args:
            model_name: The name of the model.
            model_record: The text generation model record.
            statistics: Optional Horde API statistics.
            category_total_usage: Total monthly usage for the category.

        Returns:
            DeletionRiskFlags object.
        """
        if not isinstance(model_record, TextGenerationModelRecord):
            error_message = f"Expected TextGenerationModelRecord, got {type(model_record).__name__}"
            raise TypeError(error_message)

        return self._create_flags_impl(model_name, model_record, statistics, category_total_usage)


class GenericDeletionRiskFlagsHandler(DeletionRiskFlagsHandler):
    """Fallback handler for unsupported model record types."""

    def can_handle(self, model_record: GenericModelRecord) -> bool:
        """Check if this handler can process the given model record.

        This handler accepts all model records as a fallback.

        Args:
            model_record: The model record to check.

        Returns:
            True (accepts all model records).
        """
        return True

    def create_flags(
        self,
        *,
        model_name: str,
        model_record: GenericModelRecord,
        statistics: CombinedModelStatistics | None,
        category_total_usage: int,
    ) -> DeletionRiskFlags:
        """Create DeletionRiskFlags for a generic/unsupported model type.

        Args:
            model_name: The name of the model.
            model_record: The generic model record.
            statistics: Optional Horde API statistics.
            category_total_usage: Total monthly usage for the category.

        Returns:
            DeletionRiskFlags object.
        """
        logger.warning(f"Using fallback handler for unsupported model type: {type(model_record).__name__}")

        downloads = model_record.config.download if model_record.config else []

        return (
            DeletionRiskFlagsBuilder()
            .with_download_flags(*FlagValidatorService.validate_downloads(downloads))
            .with_missing_description(FlagValidatorService.validate_description(model_record.description))
            .with_statistics_flags(*FlagValidatorService.validate_statistics(statistics, category_total_usage))
            .build()
        )


class DeletionRiskFlagsFactory:
    """Factory for creating DeletionRiskFlags objects with extensible handler support.

    Handlers are registered and checked in order. The first handler that can process
    a model record type will be used to create the deletion risk flags.

    Examples:
        ```python
        # Using default handlers
        factory = DeletionRiskFlagsFactory.create_default()
        flags = factory.create_flags(
            model_record=image_model_record,
            statistics=stats,
            category_total_usage=10000,
        )

        # Adding custom handler
        factory.register_handler(CustomDeletionRiskFlagsHandler())
        ```
    """

    def __init__(self, handlers: list[DeletionRiskFlagsHandler] | None = None) -> None:
        """Initialize the factory with optional handlers.

        Args:
            handlers: List of handlers to use. If None, no handlers are registered.
        """
        self._handlers: list[DeletionRiskFlagsHandler] = handlers or []

    @classmethod
    def create_default(cls) -> DeletionRiskFlagsFactory:
        """Create a factory with default handlers for standard model types.

        Returns:
            DeletionRiskFlagsFactory with default handlers registered.
        """
        return cls(
            handlers=[
                ImageGenerationDeletionRiskFlagsHandler(),
                TextGenerationDeletionRiskFlagsHandler(),
                GenericDeletionRiskFlagsHandler(),  # Fallback handler (must be last)
            ]
        )

    def register_handler(self, handler: DeletionRiskFlagsHandler) -> None:
        """Register a new handler.

        Handlers are checked in registration order. Register more specific handlers
        before generic ones.

        Args:
            handler: The handler to register.
        """
        self._handlers.append(handler)

    def create_flags(
        self,
        *,
        model_name: str,
        model_record: GenericModelRecord,
        statistics: CombinedModelStatistics | None,
        category_total_usage: int,
    ) -> DeletionRiskFlags:
        """Create DeletionRiskFlags for a model record using the appropriate handler.

        Args:
            model_name: The name of the model.
            model_record: The model record.
            statistics: Optional Horde API statistics.
            category_total_usage: Total monthly usage for the category.

        Returns:
            DeletionRiskFlags object.

        Raises:
            ValueError: If no handler can process the model record type.
        """
        for handler in self._handlers:
            if handler.can_handle(model_record):
                return handler.create_flags(
                    model_name=model_name,
                    model_record=model_record,
                    statistics=statistics,
                    category_total_usage=category_total_usage,
                )

        error_message = f"No handler found for model record type: {type(model_record).__name__}"
        raise ValueError(error_message)


class ModelAuditInfoHandler:
    """Abstract handler for creating ModelAuditInfo from specific model record types.

    Subclasses should implement type-specific extraction and flag generation logic.
    """

    def can_handle(self, model_record: GenericModelRecord) -> bool:
        """Check if this handler can process the given model record.

        Args:
            model_record: The model record to check.

        Returns:
            True if this handler can process the model record type.
        """
        raise NotImplementedError("Subclasses must implement can_handle")

    def create_audit_info(
        self,
        *,
        model_name: str,
        model_record: GenericModelRecord,
        statistics: CombinedModelStatistics | None,
        category_total_usage: int,
        category: MODEL_REFERENCE_CATEGORY,
    ) -> ModelAuditInfo:
        """Create ModelAuditInfo for a model record.

        Args:
            model_name: The model name.
            model_record: The model record.
            statistics: Optional Horde API statistics.
            category_total_usage: Total monthly usage for the category.
            category: The model reference category.

        Returns:
            ModelAuditInfo object.
        """
        raise NotImplementedError("Subclasses must implement create_audit_info")

    @staticmethod
    def _build_audit_info(
        *,
        model_name: str,
        model_record: GenericModelRecord,
        statistics: CombinedModelStatistics | None,
        category_total_usage: int,
        category: MODEL_REFERENCE_CATEGORY,
        flags: DeletionRiskFlags,
        baseline: str | None,
        nsfw: bool | None,
        size_bytes: int | None,
        variations: list[str] | None = None,
    ) -> ModelAuditInfo:
        """Build ModelAuditInfo from common components.

        Args:
            model_name: The model name.
            model_record: Any model record type.
            statistics: Optional Horde API statistics.
            category_total_usage: Total monthly usage for the category.
            category: The model reference category.
            flags: Deletion risk flags.
            baseline: Model baseline (if applicable).
            nsfw: Whether model is NSFW (if applicable).
            size_bytes: Model size in bytes (if available).
            variations: List of model names that are variations (substring matches).

        Returns:
            ModelAuditInfo object.
        """
        # Extract Horde API data from statistics
        worker_count = statistics.worker_count if statistics else 0
        usage_day = statistics.usage_stats.day if statistics and statistics.usage_stats else 0
        usage_month = statistics.usage_stats.month if statistics and statistics.usage_stats else 0
        usage_total = statistics.usage_stats.total if statistics and statistics.usage_stats else 0
        usage_hour = None
        usage_minute = None

        # Calculate usage percentage
        usage_percentage = 0.0
        if category_total_usage > 0:
            usage_percentage = (usage_month / category_total_usage) * 100.0

        # Calculate usage trend ratios
        day_to_month_ratio: float | None = None
        if usage_month > 0:
            day_to_month_ratio = usage_day / usage_month

        month_to_total_ratio: float | None = None
        if usage_total > 0:
            month_to_total_ratio = usage_month / usage_total

        usage_trend = UsageTrend(
            day_to_month_ratio=day_to_month_ratio,
            month_to_total_ratio=month_to_total_ratio,
        )

        # Check description
        description = model_record.description
        has_description = bool(description and len(description.strip()) > 0)

        # Calculate size in GB and cost-benefit score
        size_gb: float | None = None
        cost_benefit_score: float | None = None

        if size_bytes and size_bytes > 0:
            size_gb = size_bytes / (1024**3)
            if size_gb > 0:
                cost_benefit_score = usage_month / size_gb

        # Extract download information
        downloads = model_record.config.download if model_record.config else []
        download_count = len(downloads)

        download_hosts = []
        for download in downloads:
            url = download.file_url
            if url:
                try:
                    parsed = urlparse(url)
                    if parsed.netloc and parsed.netloc not in download_hosts:
                        download_hosts.append(parsed.netloc)
                except Exception:
                    pass

        # Create audit info
        return ModelAuditInfo(
            name=model_name,
            category=category,
            deletion_risk_flags=flags,
            at_risk=flags.any_flags(),
            risk_score=flags.flag_count(),
            worker_count=worker_count,
            usage_day=usage_day,
            usage_month=usage_month,
            usage_total=usage_total,
            usage_hour=usage_hour,
            usage_minute=usage_minute,
            usage_percentage_of_category=round(usage_percentage, 4),
            usage_trend=usage_trend,
            cost_benefit_score=round(cost_benefit_score, 2) if cost_benefit_score else None,
            size_gb=round(size_gb, 2) if size_gb else None,
            baseline=baseline,
            nsfw=nsfw,
            has_description=has_description,
            download_count=download_count,
            download_hosts=download_hosts,
            variations=variations or [],
        )


class ImageGenerationModelAuditHandler(ModelAuditInfoHandler):
    """Handler for image generation model audit info creation."""

    def __init__(self, flags_factory: DeletionRiskFlagsFactory | None = None) -> None:
        """Initialize the handler with optional flags factory.

        Args:
            flags_factory: Optional factory for creating deletion risk flags.
                If None, uses default factory.
        """
        self._flags_factory = flags_factory or DeletionRiskFlagsFactory.create_default()

    def can_handle(self, model_record: GenericModelRecord) -> bool:
        """Check if this handler can process the given model record.

        Args:
            model_record: The model record to check.

        Returns:
            True if the model record is an ImageGenerationModelRecord.
        """
        return isinstance(model_record, ImageGenerationModelRecord)

    def create_audit_info(
        self,
        *,
        model_name: str,
        model_record: GenericModelRecord,
        statistics: CombinedModelStatistics | None,
        category_total_usage: int,
        category: MODEL_REFERENCE_CATEGORY,
    ) -> ModelAuditInfo:
        """Create ModelAuditInfo for an image generation model.

        Args:
            model_name: The model name.
            model_record: The image generation model record.
            statistics: Optional Horde API statistics.
            category_total_usage: Total monthly usage for the category.
            category: The model reference category.

        Returns:
            ModelAuditInfo object.
        """
        if not isinstance(model_record, ImageGenerationModelRecord):
            error_message = f"Expected ImageGenerationModelRecord, got {type(model_record).__name__}"
            raise TypeError(error_message)

        flags = self._flags_factory.create_flags(
            model_name=model_name,
            model_record=model_record,
            statistics=statistics,
            category_total_usage=category_total_usage,
        )
        baseline = str(model_record.baseline) if model_record.baseline else None
        nsfw = model_record.nsfw
        size_bytes = model_record.size_on_disk_bytes

        return ModelAuditInfoHandler._build_audit_info(
            model_name=model_name,
            model_record=model_record,
            statistics=statistics,
            category_total_usage=category_total_usage,
            category=category,
            flags=flags,
            baseline=baseline,
            nsfw=nsfw,
            size_bytes=size_bytes,
        )


class TextGenerationModelAuditHandler(ModelAuditInfoHandler):
    """Handler for text generation model audit info creation."""

    def __init__(self, flags_factory: DeletionRiskFlagsFactory | None = None) -> None:
        """Initialize the handler with optional flags factory.

        Args:
            flags_factory: Optional factory for creating deletion risk flags.
                If None, uses default factory.
        """
        self._flags_factory = flags_factory or DeletionRiskFlagsFactory.create_default()

    def can_handle(self, model_record: GenericModelRecord) -> bool:
        """Check if this handler can process the given model record.

        Args:
            model_record: The model record to check.

        Returns:
            True if the model record is a TextGenerationModelRecord.
        """
        return isinstance(model_record, TextGenerationModelRecord)

    def create_audit_info(
        self,
        *,
        model_name: str,
        model_record: GenericModelRecord,
        statistics: CombinedModelStatistics | None,
        category_total_usage: int,
        category: MODEL_REFERENCE_CATEGORY,
    ) -> ModelAuditInfo:
        """Create ModelAuditInfo for a text generation model.

        Args:
            model_name: The model name.
            model_record: The text generation model record.
            statistics: Optional Horde API statistics.
            category_total_usage: Total monthly usage for the category.
            category: The model reference category.

        Returns:
            ModelAuditInfo object.
        """
        if not isinstance(model_record, TextGenerationModelRecord):
            error_message = f"Expected TextGenerationModelRecord, got {type(model_record).__name__}"
            raise TypeError(error_message)

        flags = self._flags_factory.create_flags(
            model_name=model_name,
            model_record=model_record,
            statistics=statistics,
            category_total_usage=category_total_usage,
        )
        baseline = model_record.baseline
        nsfw = model_record.nsfw
        size_bytes = None  # Text generation models don't have size_on_disk_bytes

        return ModelAuditInfoHandler._build_audit_info(
            model_name=model_name,
            model_record=model_record,
            statistics=statistics,
            category_total_usage=category_total_usage,
            category=category,
            flags=flags,
            baseline=baseline,
            nsfw=nsfw,
            size_bytes=size_bytes,
        )


class GenericModelAuditHandler(ModelAuditInfoHandler):
    """Fallback handler for unsupported model record types."""

    def can_handle(self, model_record: GenericModelRecord) -> bool:
        """Check if this handler can process the given model record.

        This handler accepts all model records as a fallback.

        Args:
            model_record: The model record to check.

        Returns:
            True (accepts all model records).
        """
        return True

    def create_audit_info(
        self,
        *,
        model_name: str,
        model_record: GenericModelRecord,
        statistics: CombinedModelStatistics | None,
        category_total_usage: int,
        category: MODEL_REFERENCE_CATEGORY,
    ) -> ModelAuditInfo:
        """Create ModelAuditInfo for a generic/unsupported model type.

        Args:
            model_name: The model name.
            model_record: The generic model record.
            statistics: Optional Horde API statistics.
            category_total_usage: Total monthly usage for the category.
            category: The model reference category.

        Returns:
            ModelAuditInfo object.
        """
        logger.warning(f"Using fallback handler for unsupported model type: {type(model_record).__name__}")

        # Use DeletionRiskFlagsFactory for flag creation
        flags_factory = DeletionRiskFlagsFactory.create_default()
        flags = flags_factory.create_flags(
            model_name=model_name,
            model_record=model_record,
            statistics=statistics,
            category_total_usage=category_total_usage,
        )

        return ModelAuditInfoHandler._build_audit_info(
            model_name=model_name,
            model_record=model_record,
            statistics=statistics,
            category_total_usage=category_total_usage,
            category=category,
            flags=flags,
            baseline=None,
            nsfw=None,
            size_bytes=None,
        )


class ModelAuditInfoFactory:
    """Factory for creating ModelAuditInfo objects with extensible handler support.

    Handlers are registered and checked in order. The first handler that can process
    a model record type will be used to create the audit info.

    Examples:
        ```python
        # Using default handlers
        factory = ModelAuditInfoFactory.create_default()
        audit_info = factory.create_audit_info(
            model_name="my_model",
            model_record=image_model_record,
            statistics=stats,
            category_total_usage=10000,
            category=MODEL_REFERENCE_CATEGORY.image_generation,
        )

        # Adding custom handler
        factory.register_handler(CustomModelAuditHandler())
        ```
    """

    def __init__(self, handlers: list[ModelAuditInfoHandler] | None = None) -> None:
        """Initialize the factory with optional handlers.

        Args:
            handlers: List of handlers to use. If None, no handlers are registered.
        """
        self._handlers: list[ModelAuditInfoHandler] = handlers or []

    @classmethod
    def create_default(cls) -> ModelAuditInfoFactory:
        """Create a factory with default handlers for standard model types.

        Returns:
            ModelAuditInfoFactory with default handlers registered.
        """
        return cls(
            handlers=[
                ImageGenerationModelAuditHandler(),
                TextGenerationModelAuditHandler(),
                GenericModelAuditHandler(),  # Fallback handler (must be last)
            ]
        )

    def register_handler(self, handler: ModelAuditInfoHandler) -> None:
        """Register a new handler.

        Handlers are checked in registration order. Register more specific handlers
        before generic ones.

        Args:
            handler: The handler to register.
        """
        self._handlers.append(handler)

    def create_audit_info(
        self,
        *,
        model_name: str,
        model_record: GenericModelRecord,
        statistics: CombinedModelStatistics | None,
        category_total_usage: int,
        category: MODEL_REFERENCE_CATEGORY,
    ) -> ModelAuditInfo:
        """Create ModelAuditInfo for a model record using the appropriate handler.

        Args:
            model_name: The model name.
            model_record: The model record.
            statistics: Optional Horde API statistics.
            category_total_usage: Total monthly usage for the category.
            category: The model reference category.

        Returns:
            ModelAuditInfo object.

        Raises:
            ValueError: If no handler can process the model record type.
        """
        for handler in self._handlers:
            if handler.can_handle(model_record):
                return handler.create_audit_info(
                    model_name=model_name,
                    model_record=model_record,
                    statistics=statistics,
                    category_total_usage=category_total_usage,
                    category=category,
                )

        error_message = f"No handler found for model record type: {type(model_record).__name__}"
        raise ValueError(error_message)

    def analyze_models(
        self,
        model_records: (
            dict[str, GenericModelRecord]
            | dict[str, ImageGenerationModelRecord]
            | dict[str, TextGenerationModelRecord]
        ),
        model_statistics: dict[str, CombinedModelStatistics],
        category_total_usage: int,
        category: MODEL_REFERENCE_CATEGORY,
    ) -> list[ModelAuditInfo]:
        """Analyze model records and statistics to create audit information.

        Args:
            model_records: Dictionary of model names to typed model records.
            model_statistics: Dictionary of model names to Horde API statistics.
            category_total_usage: Total monthly usage for the entire category.
            category: The model reference category.

        Returns:
            List of ModelAuditInfo sorted by usage (descending).
        """
        audit_models: list[ModelAuditInfo] = []

        model_record: GenericModelRecord | ImageGenerationModelRecord | TextGenerationModelRecord

        for model_name, model_record in model_records.items():
            # Get statistics for this model (may be None if not in Horde data)
            statistics = model_statistics.get(model_name)

            # Use factory to create audit info
            audit_info = self.create_audit_info(
                model_name=model_name,
                model_record=model_record,
                statistics=statistics,
                category_total_usage=category_total_usage,
                category=category,
            )

            audit_models.append(audit_info)

        # Sort by usage (descending) for easier review
        audit_models.sort(key=lambda x: x.usage_month, reverse=True)

        logger.info(
            f"Analyzed {len(audit_models)} models for audit: {sum(1 for m in audit_models if m.at_risk)} at risk"
        )

        return audit_models

    def create_audit_response(
        self,
        model_records: (
            dict[str, GenericModelRecord]
            | dict[str, ImageGenerationModelRecord]
            | dict[str, TextGenerationModelRecord]
        ),
        model_statistics: dict[str, CombinedModelStatistics],
        category_total_usage: int,
        category: MODEL_REFERENCE_CATEGORY,
    ) -> CategoryAuditResponse:
        """Analyze models and create complete audit response with summary.

        Args:
            model_records: Dictionary of model names to typed model records.
            model_statistics: Dictionary of model names to Horde API statistics.
            category_total_usage: Total monthly usage for the entire category.
            category: The model reference category.

        Returns:
            CategoryAuditResponse with models and summary.
        """
        # Analyze all models
        audit_models = self.analyze_models(
            model_records=model_records,
            model_statistics=model_statistics,
            category_total_usage=category_total_usage,
            category=category,
        )

        # Calculate summary
        summary = CategoryAuditSummary.from_audit_models(audit_models)

        # Create response
        return CategoryAuditResponse(
            category=category,
            category_total_month_usage=category_total_usage,
            total_count=len(audit_models),
            returned_count=len(audit_models),
            offset=0,
            limit=None,
            models=audit_models,
            summary=summary,
        )
