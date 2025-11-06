"""Tests for text model grouping utilities.

Tests the functionality that groups text generation model variants by base name
and aggregates their metrics (usage, workers, size, flags, etc.).
"""

from __future__ import annotations

import pytest

from horde_model_reference.analytics.audit_analysis import (
    CategoryAuditResponse,
    CategoryAuditSummary,
    DeletionRiskFlags,
    ModelAuditInfo,
    UsageTrend,
)
from horde_model_reference.analytics.text_model_grouping import (
    apply_text_model_grouping_to_audit,
    detect_variations,
    group_audit_models,
    merge_deletion_flags,
    merge_usage_trends,
    recalculate_audit_summary,
)
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY


class TestMergeDeletionFlags:
    """Test merging deletion risk flags using logical OR."""

    def test_merge_empty_list(self) -> None:
        """Empty list should return all flags False."""
        result = merge_deletion_flags([])
        assert not result.any_flags()
        assert result.flag_count() == 0

    def test_merge_all_false(self) -> None:
        """All flags False should result in all False."""
        flags1 = DeletionRiskFlags()
        flags2 = DeletionRiskFlags()
        result = merge_deletion_flags([flags1, flags2])
        assert not result.any_flags()
        assert result.flag_count() == 0

    def test_merge_one_flag_set(self) -> None:
        """If any variant has a flag, merged should have it."""
        flags1 = DeletionRiskFlags()
        flags2 = DeletionRiskFlags(zero_usage_month=True)
        flags3 = DeletionRiskFlags()
        result = merge_deletion_flags([flags1, flags2, flags3])
        assert result.zero_usage_month is True
        assert result.flag_count() == 1

    def test_merge_multiple_flags_set(self) -> None:
        """Multiple different flags should all be merged."""
        flags1 = DeletionRiskFlags(zero_usage_month=True, no_active_workers=True)
        flags2 = DeletionRiskFlags(has_non_preferred_host=True)
        flags3 = DeletionRiskFlags(low_usage=True, missing_description=True)
        result = merge_deletion_flags([flags1, flags2, flags3])
        assert result.zero_usage_month is True
        assert result.no_active_workers is True
        assert result.has_non_preferred_host is True
        assert result.low_usage is True
        assert result.missing_description is True
        assert result.flag_count() == 5

    def test_merge_overlapping_flags(self) -> None:
        """Same flag set in multiple variants should still be True once."""
        flags1 = DeletionRiskFlags(zero_usage_month=True)
        flags2 = DeletionRiskFlags(zero_usage_month=True, no_active_workers=True)
        result = merge_deletion_flags([flags1, flags2])
        assert result.zero_usage_month is True
        assert result.no_active_workers is True
        assert result.flag_count() == 2

    def test_merge_all_flags_set(self) -> None:
        """Test with all 11 flags set across variants."""
        flags1 = DeletionRiskFlags(
            zero_usage_day=True,
            zero_usage_month=True,
            zero_usage_total=True,
            no_active_workers=True,
        )
        flags2 = DeletionRiskFlags(
            has_multiple_hosts=True,
            has_non_preferred_host=True,
            has_unknown_host=True,
            no_download_urls=True,
        )
        flags3 = DeletionRiskFlags(
            missing_description=True,
            missing_baseline=True,
            low_usage=True,
        )
        result = merge_deletion_flags([flags1, flags2, flags3])
        assert result.flag_count() == 11
        assert result.any_flags() is True


class TestMergeUsageTrends:
    """Test weighted average merging of usage trends."""

    def test_merge_empty_lists(self) -> None:
        """Empty lists should return default UsageTrend."""
        result = merge_usage_trends([], [])
        assert result.day_to_month_ratio is None
        assert result.month_to_total_ratio is None

    def test_merge_zero_weights(self) -> None:
        """Zero total weight should return default UsageTrend."""
        trends = [
            UsageTrend(day_to_month_ratio=0.5, month_to_total_ratio=0.8),
        ]
        weights = [0]
        result = merge_usage_trends(trends, weights)
        assert result.day_to_month_ratio is None
        assert result.month_to_total_ratio is None

    def test_merge_single_trend(self) -> None:
        """Single trend should return itself."""
        trends = [UsageTrend(day_to_month_ratio=0.5, month_to_total_ratio=0.8)]
        weights = [100]
        result = merge_usage_trends(trends, weights)
        assert result.day_to_month_ratio == 0.5
        assert result.month_to_total_ratio == 0.8

    def test_merge_equal_weights(self) -> None:
        """Equal weights should average the ratios."""
        trends = [
            UsageTrend(day_to_month_ratio=0.4, month_to_total_ratio=0.6),
            UsageTrend(day_to_month_ratio=0.6, month_to_total_ratio=0.8),
        ]
        weights = [50, 50]
        result = merge_usage_trends(trends, weights)
        # (0.4*50 + 0.6*50) / 100 = 0.5
        assert result.day_to_month_ratio == 0.5
        # (0.6*50 + 0.8*50) / 100 = 0.7
        assert result.month_to_total_ratio == 0.7

    def test_merge_unequal_weights(self) -> None:
        """Heavier weights should influence the result more."""
        trends = [
            UsageTrend(day_to_month_ratio=0.2, month_to_total_ratio=0.3),
            UsageTrend(day_to_month_ratio=0.8, month_to_total_ratio=0.9),
        ]
        weights = [10, 90]  # Second trend has 9x weight
        result = merge_usage_trends(trends, weights)
        # (0.2*10 + 0.8*90) / 100 = 0.74
        assert result.day_to_month_ratio == pytest.approx(0.74)
        # (0.3*10 + 0.9*90) / 100 = 0.84
        assert result.month_to_total_ratio == pytest.approx(0.84)

    def test_merge_with_null_ratios(self) -> None:
        """Null ratios are treated as 0 in weighted average."""
        trends = [
            UsageTrend(day_to_month_ratio=0.5, month_to_total_ratio=None),
            UsageTrend(day_to_month_ratio=None, month_to_total_ratio=0.7),
        ]
        weights = [50, 50]
        result = merge_usage_trends(trends, weights)
        # day_to_month: (0.5*50 + 0*50) / 100 = 0.25
        assert result.day_to_month_ratio == pytest.approx(0.25)
        # month_to_total: (0*50 + 0.7*50) / 100 = 0.35
        assert result.month_to_total_ratio == pytest.approx(0.35)

    def test_merge_all_null_ratios(self) -> None:
        """All null ratios should return null."""
        trends = [
            UsageTrend(day_to_month_ratio=None, month_to_total_ratio=None),
            UsageTrend(day_to_month_ratio=None, month_to_total_ratio=None),
        ]
        weights = [50, 50]
        result = merge_usage_trends(trends, weights)
        assert result.day_to_month_ratio is None
        assert result.month_to_total_ratio is None


class TestGroupAuditModels:
    """Test grouping multiple model variants into aggregated entries."""

    def test_group_empty_list(self) -> None:
        """Empty list should return empty list."""
        result = group_audit_models([])
        assert result == []

    def test_group_single_model(self) -> None:
        """Single model should be returned unchanged."""
        model = ModelAuditInfo(
            name="llama-2-7b-Q4_K_M",
            category=MODEL_REFERENCE_CATEGORY.text_generation,
            deletion_risk_flags=DeletionRiskFlags(),
            at_risk=False,
            risk_score=0,
            worker_count=10,
            usage_day=100,
            usage_month=3000,
            usage_total=50000,
            usage_percentage_of_category=5.0,
            usage_trend=UsageTrend(day_to_month_ratio=0.03, month_to_total_ratio=0.06),
        )
        result = group_audit_models([model])
        assert len(result) == 1
        assert result[0] == model

    def test_group_different_base_names(self) -> None:
        """Models with different base names should not be grouped."""
        model1 = ModelAuditInfo(
            name="llama-2-7b",
            category=MODEL_REFERENCE_CATEGORY.text_generation,
            deletion_risk_flags=DeletionRiskFlags(),
            at_risk=False,
            risk_score=0,
            worker_count=5,
            usage_day=50,
            usage_month=1500,
            usage_total=25000,
            usage_percentage_of_category=2.5,
            usage_trend=UsageTrend(),
        )
        model2 = ModelAuditInfo(
            name="mistral-7b",
            category=MODEL_REFERENCE_CATEGORY.text_generation,
            deletion_risk_flags=DeletionRiskFlags(),
            at_risk=False,
            risk_score=0,
            worker_count=8,
            usage_day=80,
            usage_month=2400,
            usage_total=40000,
            usage_percentage_of_category=4.0,
            usage_trend=UsageTrend(),
        )
        result = group_audit_models([model1, model2])
        assert len(result) == 2
        # Should be returned in same order
        assert result[0].name == "llama-2-7b"
        assert result[1].name == "mistral-7b"

    def test_group_variants_of_same_base(self) -> None:
        """Variants of the same base model should be grouped."""
        model1 = ModelAuditInfo(
            name="llama-2-7b-Q4_K_M",
            category=MODEL_REFERENCE_CATEGORY.text_generation,
            deletion_risk_flags=DeletionRiskFlags(),
            at_risk=False,
            risk_score=0,
            worker_count=5,
            usage_day=50,
            usage_month=1500,
            usage_total=25000,
            usage_percentage_of_category=2.5,
            usage_trend=UsageTrend(day_to_month_ratio=0.03, month_to_total_ratio=0.06),
            size_gb=3.5,
        )
        model2 = ModelAuditInfo(
            name="llama-2-7b-Q8",
            category=MODEL_REFERENCE_CATEGORY.text_generation,
            deletion_risk_flags=DeletionRiskFlags(),
            at_risk=False,
            risk_score=0,
            worker_count=8,
            usage_day=80,
            usage_month=2400,
            usage_total=40000,
            usage_percentage_of_category=4.0,
            usage_trend=UsageTrend(day_to_month_ratio=0.03, month_to_total_ratio=0.06),
            size_gb=7.0,
        )
        result = group_audit_models([model1, model2])
        assert len(result) == 1
        grouped = result[0]
        assert "grouped" in grouped.name.lower()
        # Usage should be summed
        assert grouped.usage_day == 130  # 50 + 80
        assert grouped.usage_month == 3900  # 1500 + 2400
        assert grouped.usage_total == 65000  # 25000 + 40000
        # Usage percentage should be summed
        assert grouped.usage_percentage_of_category == pytest.approx(6.5)
        # Worker count should be max
        assert grouped.worker_count == 8
        # Size should be averaged
        assert grouped.size_gb == pytest.approx(5.25)  # (3.5 + 7.0) / 2

    def test_group_with_flags_merging(self) -> None:
        """Grouped model should have merged flags."""
        model1 = ModelAuditInfo(
            name="mixtral-8x7b-Q4",
            category=MODEL_REFERENCE_CATEGORY.text_generation,
            deletion_risk_flags=DeletionRiskFlags(zero_usage_day=True),
            at_risk=True,
            risk_score=1,
            worker_count=0,
            usage_day=0,
            usage_month=100,
            usage_total=1000,
            usage_percentage_of_category=0.1,
            usage_trend=UsageTrend(),
        )
        model2 = ModelAuditInfo(
            name="mixtral-8x7b-Q8",
            category=MODEL_REFERENCE_CATEGORY.text_generation,
            deletion_risk_flags=DeletionRiskFlags(no_active_workers=True, low_usage=True),
            at_risk=True,
            risk_score=2,
            worker_count=0,
            usage_day=10,
            usage_month=200,
            usage_total=2000,
            usage_percentage_of_category=0.2,
            usage_trend=UsageTrend(),
        )
        result = group_audit_models([model1, model2])
        assert len(result) == 1
        grouped = result[0]
        # Should have flags from both models
        assert grouped.deletion_risk_flags.zero_usage_day is True
        assert grouped.deletion_risk_flags.no_active_workers is True
        assert grouped.deletion_risk_flags.low_usage is True
        assert grouped.flag_count == 3
        assert grouped.at_risk is True

    def test_group_with_null_sizes(self) -> None:
        """Grouping models with some null sizes should handle correctly."""
        model1 = ModelAuditInfo(
            name="model-v1-Q4",
            category=MODEL_REFERENCE_CATEGORY.text_generation,
            deletion_risk_flags=DeletionRiskFlags(),
            at_risk=False,
            risk_score=0,
            worker_count=1,
            usage_day=10,
            usage_month=300,
            usage_total=5000,
            usage_percentage_of_category=0.5,
            usage_trend=UsageTrend(),
            size_gb=None,  # No size
        )
        model2 = ModelAuditInfo(
            name="model-v1-Q8",
            category=MODEL_REFERENCE_CATEGORY.text_generation,
            deletion_risk_flags=DeletionRiskFlags(),
            at_risk=False,
            risk_score=0,
            worker_count=2,
            usage_day=20,
            usage_month=600,
            usage_total=10000,
            usage_percentage_of_category=1.0,
            usage_trend=UsageTrend(),
            size_gb=6.0,
        )
        result = group_audit_models([model1, model2])
        assert len(result) == 1
        grouped = result[0]
        # Average should only consider non-null sizes
        assert grouped.size_gb == 6.0  # Only one size available
        # Cost-benefit should be computed if size available
        assert grouped.cost_benefit_score is not None


class TestRecalculateAuditSummary:
    """Test recalculating summary after grouping."""

    def test_summary_empty_list(self) -> None:
        """Empty list should return zero counts."""
        summary = recalculate_audit_summary([], 0)
        assert summary.total_models == 0
        assert summary.models_at_risk == 0
        assert summary.models_critical == 0
        assert summary.average_risk_score == 0.0

    def test_summary_no_risks(self) -> None:
        """Models with no risks should have zero at-risk count."""
        models = [
            ModelAuditInfo(
                name="model1",
                category=MODEL_REFERENCE_CATEGORY.text_generation,
                deletion_risk_flags=DeletionRiskFlags(),
                at_risk=False,
                risk_score=0,
                worker_count=10,
                usage_day=100,
                usage_month=3000,
                usage_total=50000,
                usage_percentage_of_category=5.0,
                usage_trend=UsageTrend(),
            ),
        ]
        summary = recalculate_audit_summary(models, 60000)
        assert summary.total_models == 1
        assert summary.models_at_risk == 0
        assert summary.models_critical == 0
        assert summary.average_risk_score == 0.0

    def test_summary_with_risks(self) -> None:
        """Models with risks should be counted."""
        models = [
            ModelAuditInfo(
                name="model1",
                category=MODEL_REFERENCE_CATEGORY.text_generation,
                deletion_risk_flags=DeletionRiskFlags(zero_usage_month=True, no_active_workers=True),
                at_risk=True,
                risk_score=2,
                worker_count=0,
                usage_day=0,
                usage_month=0,
                usage_total=100,
                usage_percentage_of_category=0.0,
                usage_trend=UsageTrend(),
            ),
            ModelAuditInfo(
                name="model2",
                category=MODEL_REFERENCE_CATEGORY.text_generation,
                deletion_risk_flags=DeletionRiskFlags(low_usage=True),
                at_risk=True,
                risk_score=1,
                worker_count=5,
                usage_day=10,
                usage_month=100,
                usage_total=1000,
                usage_percentage_of_category=0.2,
                usage_trend=UsageTrend(),
            ),
        ]
        summary = recalculate_audit_summary(models, 50000)
        assert summary.total_models == 2
        assert summary.models_at_risk == 2
        assert summary.models_critical == 1  # model1 is critical
        assert summary.models_with_zero_month_usage == 1
        assert summary.models_with_no_active_workers == 1
        assert summary.models_with_low_usage == 1
        assert summary.average_risk_score == 1.5  # (2 + 1) / 2


class TestApplyTextModelGroupingToAudit:
    """Test applying grouping to full CategoryAuditResponse."""

    def test_non_text_category_returns_unchanged(self) -> None:
        """Non-text categories should not be grouped."""
        response = CategoryAuditResponse(
            category=MODEL_REFERENCE_CATEGORY.image_generation,
            category_total_month_usage=100000,
            total_count=10,
            returned_count=10,
            offset=0,
            limit=100,
            models=[
                ModelAuditInfo(
                    name="model1",
                    category=MODEL_REFERENCE_CATEGORY.image_generation,
                    deletion_risk_flags=DeletionRiskFlags(),
                    at_risk=False,
                    risk_score=0,
                    worker_count=5,
                    usage_day=50,
                    usage_month=1500,
                    usage_total=25000,
                    usage_percentage_of_category=1.5,
                    usage_trend=UsageTrend(),
                ),
            ],
            summary=CategoryAuditSummary(
                total_models=1,
                models_at_risk=0,
                models_critical=0,
                models_with_warnings=0,
                average_risk_score=0.0,
                category_total_month_usage=100000,
            ),
        )
        result = apply_text_model_grouping_to_audit(response)
        # Should return unchanged
        assert result == response
        assert len(result.models) == 1

    def test_text_category_groups_models(self) -> None:
        """Text generation category should group variants."""
        response = CategoryAuditResponse(
            category=MODEL_REFERENCE_CATEGORY.text_generation,
            category_total_month_usage=10000,
            total_count=2,
            returned_count=2,
            offset=0,
            limit=100,
            models=[
                ModelAuditInfo(
                    name="llama-2-7b-Q4",
                    category=MODEL_REFERENCE_CATEGORY.text_generation,
                    deletion_risk_flags=DeletionRiskFlags(),
                    at_risk=False,
                    risk_score=0,
                    worker_count=5,
                    usage_day=50,
                    usage_month=1500,
                    usage_total=25000,
                    usage_percentage_of_category=15.0,
                    usage_trend=UsageTrend(),
                ),
                ModelAuditInfo(
                    name="llama-2-7b-Q8",
                    category=MODEL_REFERENCE_CATEGORY.text_generation,
                    deletion_risk_flags=DeletionRiskFlags(),
                    at_risk=False,
                    risk_score=0,
                    worker_count=8,
                    usage_day=80,
                    usage_month=2400,
                    usage_total=40000,
                    usage_percentage_of_category=24.0,
                    usage_trend=UsageTrend(),
                ),
            ],
            summary=CategoryAuditSummary(
                total_models=2,
                models_at_risk=0,
                models_critical=0,
                models_with_warnings=0,
                average_risk_score=0.0,
                category_total_month_usage=10000,
            ),
        )
        result = apply_text_model_grouping_to_audit(response)
        # Should have grouped into 1 model
        assert len(result.models) == 1
        assert "grouped" in result.models[0].name.lower()
        # Summary should be recalculated
        assert result.summary.total_models == 1
        # Preserve original total_count (before grouping)
        assert result.total_count == 2
        # returned_count should reflect grouped count
        assert result.returned_count == 1


class TestDetectVariations:
    """Test variation detection based on substring matching."""

    def test_empty_list(self) -> None:
        """Empty list should return empty dict."""
        result = detect_variations([])
        assert result == {}

    def test_single_model(self) -> None:
        """Single model should have no variations."""
        result = detect_variations(["Llama-2-7B"])
        assert result == {"Llama-2-7B": []}

    def test_no_variations(self) -> None:
        """Models with no substring relationships should have empty variations."""
        result = detect_variations(["Llama-2-7B", "Mistral-7B", "GPT-3"])
        assert result == {
            "Llama-2-7B": [],
            "Mistral-7B": [],
            "GPT-3": [],
        }

    def test_simple_substring_match(self) -> None:
        """Basic substring matching should create bidirectional variations."""
        result = detect_variations(["Broken-Tutu-24B", "Broken-Tutu-24B-Unslop-v2.0"])
        assert result == {
            "Broken-Tutu-24B": ["Broken-Tutu-24B-Unslop-v2.0"],
            "Broken-Tutu-24B-Unslop-v2.0": ["Broken-Tutu-24B"],
        }

    def test_multiple_variations(self) -> None:
        """One model can be a substring of multiple others."""
        result = detect_variations([
            "Broken-Tutu-24B",
            "Broken-Tutu-24B-Unslop-v2.0",
            "Broken-Tutu-24B-Transgression-v2.0",
        ])
        assert "Broken-Tutu-24B-Unslop-v2.0" in result["Broken-Tutu-24B"]
        assert "Broken-Tutu-24B-Transgression-v2.0" in result["Broken-Tutu-24B"]
        assert len(result["Broken-Tutu-24B"]) == 2
        assert result["Broken-Tutu-24B-Unslop-v2.0"] == ["Broken-Tutu-24B"]
        assert result["Broken-Tutu-24B-Transgression-v2.0"] == ["Broken-Tutu-24B"]

    def test_case_insensitive_matching(self) -> None:
        """Substring matching should be case-insensitive."""
        result = detect_variations(["broken-tutu-24b", "Broken-Tutu-24B-Unslop-v2.0"])
        assert "Broken-Tutu-24B-Unslop-v2.0" in result["broken-tutu-24b"]
        assert "broken-tutu-24b" in result["Broken-Tutu-24B-Unslop-v2.0"]

    def test_complex_variations(self) -> None:
        """Test with complex variation patterns."""
        result = detect_variations([
            "Llama-2-7B",
            "Llama-2-7B-Chat",
            "Llama-2-7B-Chat-GGUF",
            "Mistral-7B",
        ])
        # Llama-2-7B is a substring of both Chat variants
        assert set(result["Llama-2-7B"]) == {"Llama-2-7B-Chat", "Llama-2-7B-Chat-GGUF"}
        # Llama-2-7B-Chat is a substring of GGUF variant
        assert set(result["Llama-2-7B-Chat"]) == {"Llama-2-7B", "Llama-2-7B-Chat-GGUF"}
        # GGUF variant has both as variations
        assert set(result["Llama-2-7B-Chat-GGUF"]) == {"Llama-2-7B", "Llama-2-7B-Chat"}
        # Mistral has no variations
        assert result["Mistral-7B"] == []

    def test_no_duplicates_in_variations(self) -> None:
        """Variations list should not contain duplicates."""
        result = detect_variations(["Model-A", "Model-A-v2"])
        # Each model should appear only once in the other's variations
        assert result["Model-A"].count("Model-A-v2") == 1
        assert result["Model-A-v2"].count("Model-A") == 1


class TestGroupAuditModelsWithVariations:
    """Test that grouping populates the variations field."""

    def test_single_model_no_variations(self) -> None:
        """Single model with no variations should have empty variations list."""
        model = ModelAuditInfo(
            name="unique-model-7b",
            category=MODEL_REFERENCE_CATEGORY.text_generation,
            deletion_risk_flags=DeletionRiskFlags(),
            at_risk=False,
            risk_score=0,
            worker_count=10,
            usage_day=100,
            usage_month=3000,
            usage_total=50000,
            usage_percentage_of_category=5.0,
            usage_trend=UsageTrend(),
        )
        result = group_audit_models([model])
        assert len(result) == 1
        assert result[0].variations == []

    def test_models_with_variations(self) -> None:
        """Models with substring matches should have variations populated."""
        model1 = ModelAuditInfo(
            name="Broken-Tutu-24B",
            category=MODEL_REFERENCE_CATEGORY.text_generation,
            deletion_risk_flags=DeletionRiskFlags(),
            at_risk=False,
            risk_score=0,
            worker_count=5,
            usage_day=50,
            usage_month=1500,
            usage_total=25000,
            usage_percentage_of_category=2.5,
            usage_trend=UsageTrend(),
        )
        model2 = ModelAuditInfo(
            name="Broken-Tutu-24B-Unslop-v2.0",
            category=MODEL_REFERENCE_CATEGORY.text_generation,
            deletion_risk_flags=DeletionRiskFlags(),
            at_risk=False,
            risk_score=0,
            worker_count=8,
            usage_day=80,
            usage_month=2400,
            usage_total=40000,
            usage_percentage_of_category=4.0,
            usage_trend=UsageTrend(),
        )
        result = group_audit_models([model1, model2])
        assert len(result) == 2
        # Find the models in the result
        tutu_model = next(m for m in result if "Unslop" not in m.name)
        unslop_model = next(m for m in result if "Unslop" in m.name)
        # Check variations are populated (variations use base names)
        assert "Broken-Tutu-Unslop-v2.0" in tutu_model.variations
        assert "Broken-Tutu" in unslop_model.variations

    def test_grouped_models_with_variations(self) -> None:
        """Grouped models should show variations with (grouped) suffix."""
        model1 = ModelAuditInfo(
            name="Broken-Tutu-24B-Q4",
            category=MODEL_REFERENCE_CATEGORY.text_generation,
            deletion_risk_flags=DeletionRiskFlags(),
            at_risk=False,
            risk_score=0,
            worker_count=5,
            usage_day=50,
            usage_month=1500,
            usage_total=25000,
            usage_percentage_of_category=2.5,
            usage_trend=UsageTrend(),
        )
        model2 = ModelAuditInfo(
            name="Broken-Tutu-24B-Q8",
            category=MODEL_REFERENCE_CATEGORY.text_generation,
            deletion_risk_flags=DeletionRiskFlags(),
            at_risk=False,
            risk_score=0,
            worker_count=8,
            usage_day=80,
            usage_month=2400,
            usage_total=40000,
            usage_percentage_of_category=4.0,
            usage_trend=UsageTrend(),
        )
        model3 = ModelAuditInfo(
            name="Broken-Tutu-24B-Unslop-v2.0-Q4",
            category=MODEL_REFERENCE_CATEGORY.text_generation,
            deletion_risk_flags=DeletionRiskFlags(),
            at_risk=False,
            risk_score=0,
            worker_count=3,
            usage_day=30,
            usage_month=900,
            usage_total=15000,
            usage_percentage_of_category=1.5,
            usage_trend=UsageTrend(),
        )
        model4 = ModelAuditInfo(
            name="Broken-Tutu-24B-Unslop-v2.0-Q8",
            category=MODEL_REFERENCE_CATEGORY.text_generation,
            deletion_risk_flags=DeletionRiskFlags(),
            at_risk=False,
            risk_score=0,
            worker_count=6,
            usage_day=60,
            usage_month=1800,
            usage_total=30000,
            usage_percentage_of_category=3.0,
            usage_trend=UsageTrend(),
        )
        result = group_audit_models([model1, model2, model3, model4])
        # Should have 2 grouped models
        assert len(result) == 2
        # Both should be marked as grouped
        assert all("grouped" in m.name.lower() for m in result)
        # Find the base and unslop grouped models (base names)
        tutu_grouped = next(m for m in result if "Broken-Tutu (grouped)" == m.name)
        unslop_grouped = next(m for m in result if "Broken-Tutu-Unslop-v2.0 (grouped)" == m.name)
        # Check variations include the (grouped) suffix
        assert "Broken-Tutu-Unslop-v2.0 (grouped)" in tutu_grouped.variations
        assert "Broken-Tutu (grouped)" in unslop_grouped.variations
