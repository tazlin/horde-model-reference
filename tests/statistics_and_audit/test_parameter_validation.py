"""Tests for parameter count extraction and validation in text models."""

from __future__ import annotations

import pytest

from horde_model_reference.analytics.audit_analysis import FlagValidatorService
from horde_model_reference.analytics.text_model_parser import (
    extract_parameter_count_from_name,
    parse_parameter_count_from_size,
)


class TestParseParameterCountFromSize:
    """Test parameter count extraction from size strings."""

    def test_billion_parameters(self) -> None:
        """Test parsing billion-scale parameters."""
        assert parse_parameter_count_from_size("7B") == 7000
        assert parse_parameter_count_from_size("13B") == 13000
        assert parse_parameter_count_from_size("70B") == 70000
        assert parse_parameter_count_from_size("180B") == 180000

    def test_decimal_billion_parameters(self) -> None:
        """Test parsing decimal billion parameters."""
        assert parse_parameter_count_from_size("1.5B") == 1500
        assert parse_parameter_count_from_size("0.5B") == 500
        assert parse_parameter_count_from_size("2.7B") == 2700
        assert parse_parameter_count_from_size("6.7B") == 6700

    def test_million_parameters(self) -> None:
        """Test parsing million-scale parameters."""
        assert parse_parameter_count_from_size("125M") == 125
        assert parse_parameter_count_from_size("350M") == 350
        assert parse_parameter_count_from_size("800M") == 800

    def test_moe_models(self) -> None:
        """Test parsing Mixture of Experts (MoE) models."""
        # 8 experts * 7B each = 56B total
        assert parse_parameter_count_from_size("8x7B") == 56000
        # 8 experts * 22B each = 176B total
        assert parse_parameter_count_from_size("8x22B") == 176000
        # 4 experts * 8B each = 32B total
        assert parse_parameter_count_from_size("4X8B") == 32000  # Case insensitive

    def test_case_insensitive(self) -> None:
        """Test case insensitive parsing."""
        assert parse_parameter_count_from_size("7b") == 7000
        assert parse_parameter_count_from_size("7B") == 7000
        assert parse_parameter_count_from_size("350m") == 350
        assert parse_parameter_count_from_size("350M") == 350

    def test_invalid_sizes(self) -> None:
        """Test parsing invalid size strings."""
        assert parse_parameter_count_from_size("7K") is None  # Thousands not used
        assert parse_parameter_count_from_size("invalid") is None
        assert parse_parameter_count_from_size("") is None
        assert parse_parameter_count_from_size("7") is None  # Missing unit

    def test_none_input(self) -> None:
        """Test parsing None input."""
        assert parse_parameter_count_from_size(None) is None  # type: ignore


class TestExtractParameterCountFromName:
    """Test parameter count extraction from model names."""

    def test_extraction_from_simple_names(self) -> None:
        """Test extraction from simple model names."""
        assert extract_parameter_count_from_name("Llama-3-8B-Instruct") == 8000
        assert extract_parameter_count_from_name("Mistral-7B-v0.1") == 7000
        assert extract_parameter_count_from_name("GPT-Neo-125M") == 125

    def test_extraction_from_complex_names(self) -> None:
        """Test extraction from complex model names."""
        assert extract_parameter_count_from_name("alpindale/magnum-72b-v1") == 72000
        assert extract_parameter_count_from_name("TheDrummer/Skyfall-31B-v4") == 31000
        assert extract_parameter_count_from_name("anthracite-org/magnum-v2-123b") == 123000

    def test_extraction_from_moe_models(self) -> None:
        """Test extraction from MoE model names."""
        assert extract_parameter_count_from_name("Mixtral-8x7B-Instruct-v0.1") == 56000
        assert extract_parameter_count_from_name("Mixtral-8x22B-v0.1") == 176000

    def test_extraction_from_decimal_sizes(self) -> None:
        """Test extraction from models with decimal sizes."""
        assert extract_parameter_count_from_name("Qwen1.5-0.5B-Chat") == 500
        assert extract_parameter_count_from_name("EleutherAI/pythia-6.9b-deduped") == 6900
        assert extract_parameter_count_from_name("WestLake-10.7B-v2") == 10700

    def test_no_size_in_name(self) -> None:
        """Test extraction from names without size markers."""
        assert extract_parameter_count_from_name("GPT-4") is None
        assert extract_parameter_count_from_name("Claude-3-Opus") is None
        assert extract_parameter_count_from_name("gpt-3.5-turbo") is None

    def test_with_prefixes(self) -> None:
        """Test extraction from names with org/prefix."""
        assert extract_parameter_count_from_name("meta-llama/Llama-3.1-405B-Instruct") == 405000
        assert extract_parameter_count_from_name("aphrodite/mistralai/Mixtral-8x7B") == 56000
        assert extract_parameter_count_from_name("koboldcpp/Qwen2-1.5B-Instruct") == 1500


class TestValidateParameterCount:
    """Test parameter count validation logic."""

    def test_exact_matches(self) -> None:
        """Test exact parameter matches (no flag)."""
        # 7B model with 7 billion parameters
        assert FlagValidatorService.validate_parameter_count("Llama-3-8B", 8_000_000_000) is False
        assert FlagValidatorService.validate_parameter_count("Mistral-7B", 7_000_000_000) is False
        assert FlagValidatorService.validate_parameter_count("Qwen2-1.5B", 1_500_000_000) is False

    def test_close_matches_within_tolerance(self) -> None:
        """Test close matches within 10% tolerance (no flag)."""
        # 7B model with 7.5B parameters (7.1% difference)
        assert FlagValidatorService.validate_parameter_count("Mistral-7B", 7_500_000_000) is False
        # 12B model with 12.2B parameters (1.7% difference)
        assert FlagValidatorService.validate_parameter_count("Nemo-12B", 12_200_000_000) is False
        # 70B model with 70.6B parameters (0.9% difference)
        assert FlagValidatorService.validate_parameter_count("Llama-70B", 70_600_000_000) is False

    def test_mismatches_exceed_tolerance(self) -> None:
        """Test mismatches that exceed 10% tolerance (should flag)."""
        # 30B model with 33B parameters (9.1% difference - within tolerance)
        assert FlagValidatorService.validate_parameter_count("Model-30B", 33_000_000_000) is False
        # 30B model with 33.5B parameters (10.4% difference - exceeds tolerance)
        assert FlagValidatorService.validate_parameter_count("Model-30B", 33_500_000_000) is True
        # 7B model with 14B parameters (50% difference when measured from declared)
        assert FlagValidatorService.validate_parameter_count("Model-7B", 14_000_000_000) is True
        # 2B model with 3.2B parameters (37.5% difference)
        assert FlagValidatorService.validate_parameter_count("Gemma-2B", 3_200_000_000) is True
        # 1M model with 14B declared (massive mismatch)
        assert FlagValidatorService.validate_parameter_count("Model-1M", 14_000_000_000) is True

    def test_missing_size_in_name(self) -> None:
        """Test models without extractable size (no flag)."""
        # Can't extract size, so no mismatch can be detected
        assert FlagValidatorService.validate_parameter_count("GPT-4", 1_000_000_000_000) is False
        assert FlagValidatorService.validate_parameter_count("Claude", 50_000_000_000) is False

    def test_missing_declared_parameters(self) -> None:
        """Test models without declared parameters (no flag)."""
        # Can't validate without declared parameters
        assert FlagValidatorService.validate_parameter_count("Llama-3-8B", None) is False

    def test_zero_declared_parameters(self) -> None:
        """Test edge case with zero declared parameters."""
        # Zero declared but non-zero extracted is a mismatch
        assert FlagValidatorService.validate_parameter_count("Model-7B", 0) is True
        # Both zero is not a mismatch
        assert FlagValidatorService.validate_parameter_count("No-Size-Model", 0) is False

    def test_custom_tolerance(self) -> None:
        """Test custom tolerance percentages."""
        # 7B with 7.7B is 10% difference
        # With 5% tolerance, should flag
        assert FlagValidatorService.validate_parameter_count("Model-7B", 7_700_000_000, tolerance_percent=5.0) is True
        # With 15% tolerance, should not flag
        assert FlagValidatorService.validate_parameter_count("Model-7B", 7_700_000_000, tolerance_percent=15.0) is False

    def test_moe_models(self) -> None:
        """Test MoE model validation."""
        # Mixtral 8x7B = 56B total
        assert FlagValidatorService.validate_parameter_count("Mixtral-8x7B", 56_000_000_000) is False
        # Wrong parameter count
        assert FlagValidatorService.validate_parameter_count("Mixtral-8x7B", 50_000_000_000) is True

    def test_real_world_examples(self) -> None:
        """Test with real-world model examples."""
        # Examples from actual model database
        test_cases = [
            # (model_name, declared_params, should_flag)
            ("EleutherAI/gpt-neo-1.3B", 1_300_000_000, False),  # Exact match
            ("TheDrummer/Gemmasutra-Mini-2B-v1", 2_600_000_000, True),  # 30% mismatch
            ("CalderaAI/30B-Epsilon", 33_000_000_000, False),  # 10% exact - within tolerance
            ("froggeric/WestLake-10.7B-v2", 11_000_000_000, False),  # 2.8% difference
            ("Qwen/Qwen2-57B-A14B-Instruct", 57_000_000_000, False),  # Exact match
        ]

        for model_name, declared_params, should_flag in test_cases:
            result = FlagValidatorService.validate_parameter_count(model_name, declared_params)
            assert result == should_flag, f"Failed for {model_name}: expected {should_flag}, got {result}"
