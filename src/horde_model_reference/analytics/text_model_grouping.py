"""Text model grouping utilities.

Provides functionality to group text generation models by base name (stripping
quantization information) and aggregate metrics across grouped variants.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from horde_model_reference.analytics.text_model_parser import get_base_model_name
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY

if TYPE_CHECKING:
    from horde_model_reference.analytics.audit_analysis import (
        CategoryAuditResponse,
        CategoryAuditSummary,
        DeletionRiskFlags,
        ModelAuditInfo,
        UsageTrend,
    )


def merge_deletion_flags(flags_list: list[DeletionRiskFlags]) -> DeletionRiskFlags:
    """Merge multiple deletion risk flags using logical OR.

    If any variant has a flag set, the merged result will have that flag set.

    Args:
        flags_list: List of DeletionRiskFlags from all variants.

    Returns:
        Merged DeletionRiskFlags.
    """
    from horde_model_reference.analytics.audit_analysis import DeletionRiskFlags

    if not flags_list:
        return DeletionRiskFlags()

    merged = DeletionRiskFlags()
    merged.zero_usage_day = any(f.zero_usage_day for f in flags_list)
    merged.zero_usage_month = any(f.zero_usage_month for f in flags_list)
    merged.zero_usage_total = any(f.zero_usage_total for f in flags_list)
    merged.no_active_workers = any(f.no_active_workers for f in flags_list)
    merged.has_multiple_hosts = any(f.has_multiple_hosts for f in flags_list)
    merged.has_non_preferred_host = any(f.has_non_preferred_host for f in flags_list)
    merged.has_unknown_host = any(f.has_unknown_host for f in flags_list)
    merged.no_download_urls = any(f.no_download_urls for f in flags_list)
    merged.missing_description = any(f.missing_description for f in flags_list)
    merged.missing_baseline = any(f.missing_baseline for f in flags_list)
    merged.low_usage = any(f.low_usage for f in flags_list)
    merged.parameter_mismatch = any(f.parameter_mismatch for f in flags_list)

    return merged


def detect_variations(model_names: list[str]) -> dict[str, list[str]]:
    """Detect model variations based on substring matching.

    A model is considered a variation if any model name is a substring of any other model name
    (case-insensitive). The relationship is bidirectional.

    Args:
        model_names: List of model names to check for variations.

    Returns:
        Dictionary mapping each model name to its list of variations (excluding itself).

    Examples:
        >>> names = ["Broken-Tutu-24B", "Broken-Tutu-24B-Unslop-v2.0", "Mistral-7B"]
        >>> detect_variations(names)
        {
            "Broken-Tutu-24B": ["Broken-Tutu-24B-Unslop-v2.0"],
            "Broken-Tutu-24B-Unslop-v2.0": ["Broken-Tutu-24B"],
            "Mistral-7B": []
        }
    """
    variations_map: dict[str, list[str]] = {name: [] for name in model_names}

    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            if i == j:
                # Skip comparing a name with itself
                continue

            # Case-insensitive substring matching
            if name1.lower() in name2.lower():
                # name1 is a substring of name2, so they are variations
                if name2 not in variations_map[name1]:
                    variations_map[name1].append(name2)
                if name1 not in variations_map[name2]:
                    variations_map[name2].append(name1)

    return variations_map


def merge_usage_trends(trends: list[UsageTrend], weights: list[int]) -> UsageTrend:
    """Merge usage trends using weighted average.

    Args:
        trends: List of UsageTrend objects from variants.
        weights: Weights for each trend (e.g., monthly usage counts).

    Returns:
        Merged UsageTrend with weighted average ratios.
    """
    from horde_model_reference.analytics.audit_analysis import UsageTrend

    if not trends or not weights:
        return UsageTrend()

    total_weight = sum(weights)
    if total_weight == 0:
        return UsageTrend()

    day_to_month_ratios = [
        trend.day_to_month_ratio * weight
        for trend, weight in zip(trends, weights, strict=True)
        if trend.day_to_month_ratio
    ]
    month_to_total_ratios = [
        trend.month_to_total_ratio * weight
        for trend, weight in zip(trends, weights, strict=True)
        if trend.month_to_total_ratio
    ]

    return UsageTrend(
        day_to_month_ratio=sum(day_to_month_ratios) / total_weight if day_to_month_ratios else None,
        month_to_total_ratio=sum(month_to_total_ratios) / total_weight if month_to_total_ratios else None,
    )


def group_audit_models(models: list[ModelAuditInfo]) -> list[ModelAuditInfo]:
    """Group text model variants by base name and aggregate metrics.

    Combines multiple quantization variants (Q4_K_M, Q5_0, etc.) into a single
    model entry with aggregated metrics. Also detects and populates variations
    based on substring matching of model names.

    Args:
        models: List of ModelAuditInfo objects to group.

    Returns:
        List of grouped ModelAuditInfo objects with aggregated metrics and variations.
    """
    from horde_model_reference.analytics.audit_analysis import ModelAuditInfo

    if not models:
        return []

    grouped: dict[str, list[ModelAuditInfo]] = {}
    for model in models:
        base_name = get_base_model_name(model.name)
        if base_name not in grouped:
            grouped[base_name] = []
        grouped[base_name].append(model)

    # Detect variations among all model names (using base names for grouped models)
    all_model_names = [base_name for base_name in grouped.keys()]
    variations_map = detect_variations(all_model_names)

    result: list[ModelAuditInfo] = []
    for base_name, variants in grouped.items():
        if len(variants) == 1:
            # Single variant - set variations from the map
            single_model = variants[0]
            single_model.variations = variations_map.get(base_name, [])
            result.append(single_model)
            continue

        logger.debug(f"Grouping {len(variants)} variants of '{base_name}'")

        total_usage_day = sum(v.usage_day for v in variants)
        total_usage_month = sum(v.usage_month for v in variants)
        total_usage_total = sum(v.usage_total for v in variants)
        total_usage_hour = sum(v.usage_hour for v in variants if v.usage_hour is not None)
        total_usage_minute = sum(v.usage_minute for v in variants if v.usage_minute is not None)

        max_worker_count = max(v.worker_count for v in variants)

        total_size_gb = sum(v.size_gb for v in variants if v.size_gb is not None)
        size_count = len([v for v in variants if v.size_gb is not None])
        avg_size_gb = total_size_gb / size_count if total_size_gb > 0 else None

        merged_flags = merge_deletion_flags([v.deletion_risk_flags for v in variants])

        weights = [v.usage_month for v in variants]
        merged_trend = merge_usage_trends([v.usage_trend for v in variants], weights)

        all_hosts = list({host for v in variants for host in v.download_hosts})

        first_variant = variants[0]

        usage_percentage = sum(v.usage_percentage_of_category for v in variants)

        cost_benefit = None
        if avg_size_gb and avg_size_gb > 0:
            cost_benefit = total_usage_month / avg_size_gb

        # Get variations for this base name, but format them as "(grouped)" if needed
        model_variations = variations_map.get(base_name, [])
        formatted_variations = [f"{var} (grouped)" if var in grouped and len(grouped[var]) > 1 else var
                                for var in model_variations]

        grouped_model = ModelAuditInfo(
            name=f"{base_name} (grouped)",
            category=first_variant.category,
            deletion_risk_flags=merged_flags,
            at_risk=merged_flags.any_flags(),
            risk_score=merged_flags.flag_count(),
            worker_count=max_worker_count,
            usage_day=total_usage_day,
            usage_month=total_usage_month,
            usage_total=total_usage_total,
            usage_hour=total_usage_hour if any(v.usage_hour is not None for v in variants) else None,
            usage_minute=total_usage_minute if any(v.usage_minute is not None for v in variants) else None,
            usage_percentage_of_category=usage_percentage,
            usage_trend=merged_trend,
            cost_benefit_score=cost_benefit,
            size_gb=avg_size_gb,
            baseline=first_variant.baseline,
            nsfw=first_variant.nsfw,
            has_description=all(v.has_description for v in variants),
            download_count=sum(v.download_count for v in variants),
            download_hosts=all_hosts,
            variations=formatted_variations,
        )

        result.append(grouped_model)

    logger.info(f"Grouped {len(models)} models into {len(result)} entries")
    return result


def recalculate_audit_summary(models: list[ModelAuditInfo], category_total_usage: int) -> CategoryAuditSummary:
    """Recalculate audit summary after grouping models.

    Args:
        models: List of (potentially grouped) ModelAuditInfo objects.
        category_total_usage: Total monthly usage for the category.

    Returns:
        New CategoryAuditSummary with updated counts.
    """
    from horde_model_reference.analytics.audit_analysis import CategoryAuditSummary

    return CategoryAuditSummary(
        total_models=len(models),
        models_at_risk=sum(1 for m in models if m.at_risk),
        models_critical=sum(1 for m in models if m.is_critical),
        models_with_warnings=sum(1 for m in models if m.has_warning),
        models_with_zero_day_usage=sum(1 for m in models if m.deletion_risk_flags.zero_usage_day),
        models_with_zero_month_usage=sum(1 for m in models if m.deletion_risk_flags.zero_usage_month),
        models_with_zero_total_usage=sum(1 for m in models if m.deletion_risk_flags.zero_usage_total),
        models_with_no_active_workers=sum(1 for m in models if m.deletion_risk_flags.no_active_workers),
        models_with_no_downloads=sum(1 for m in models if m.deletion_risk_flags.no_download_urls),
        models_with_non_preferred_hosts=sum(1 for m in models if m.deletion_risk_flags.has_non_preferred_host),
        models_with_multiple_hosts=sum(1 for m in models if m.deletion_risk_flags.has_multiple_hosts),
        models_with_low_usage=sum(1 for m in models if m.deletion_risk_flags.low_usage),
        average_risk_score=sum(m.risk_score for m in models) / len(models) if models else 0.0,
        category_total_month_usage=category_total_usage,
    )


def apply_text_model_grouping_to_audit(audit_response: CategoryAuditResponse) -> CategoryAuditResponse:
    """Apply text model grouping to audit response.

    Groups text generation models by base name and recalculates summary.

    Args:
        audit_response: Original CategoryAuditResponse.

    Returns:
        New CategoryAuditResponse with grouped models and updated summary.
    """
    from horde_model_reference.analytics.audit_analysis import CategoryAuditResponse

    if audit_response.category != MODEL_REFERENCE_CATEGORY.text_generation:
        logger.debug(f"Skipping grouping for non-text category: {audit_response.category}")
        return audit_response

    grouped_models = group_audit_models(audit_response.models)
    new_summary = recalculate_audit_summary(grouped_models, audit_response.category_total_month_usage)

    return CategoryAuditResponse(
        category=audit_response.category,
        category_total_month_usage=audit_response.category_total_month_usage,
        total_count=audit_response.total_count,  # Preserve original total
        returned_count=len(grouped_models),
        offset=audit_response.offset,
        limit=audit_response.limit,
        models=grouped_models,
        summary=new_summary,
    )
