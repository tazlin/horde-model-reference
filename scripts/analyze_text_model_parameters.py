#!/usr/bin/env python3
"""Fact-finding script to analyze parameter mismatches in text models.

This script fetches all text generation models from the GitHub backend and analyzes:
1. Which models have extractable parameter counts in their names
2. Which models have parameter mismatches
3. Common patterns and edge cases
4. Statistics for test case development
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from horde_model_reference import MODEL_REFERENCE_CATEGORY
from horde_model_reference.analytics.text_model_parser import (
    extract_parameter_count_from_name,
    get_model_size,
    parse_parameter_count_from_size,
)
from horde_model_reference.model_reference_manager import ModelReferenceManager
from horde_model_reference.model_reference_records import TextGenerationModelRecord


def analyze_parameter_mismatches(tolerance_percent: float = 10.0) -> None:
    """Analyze parameter mismatches in text generation models.

    Args:
        tolerance_percent: Tolerance percentage for matching (default 10%).
    """
    print("=" * 80)
    print("TEXT MODEL PARAMETER ANALYSIS")
    print("=" * 80)
    print()

    # Initialize manager and fetch text generation models
    print("Fetching text generation models...")
    manager = ModelReferenceManager()
    all_refs = manager.get_all_model_references_unsafe()

    text_models = all_refs.get(MODEL_REFERENCE_CATEGORY.text_generation)
    if not text_models:
        print("ERROR: No text generation models found!")
        return

    print(f"Found {len(text_models)} text generation models")
    print()

    # Analyze models
    models_with_size = []
    models_without_size = []
    models_with_params = []
    models_without_params = []
    exact_matches = []
    close_matches = []
    mismatches = []
    parsing_examples = {}

    for model_name, model_record in text_models.items():
        if not isinstance(model_record, TextGenerationModelRecord):
            continue

        # Extract info
        size_str = get_model_size(model_name)
        extracted_params_millions = extract_parameter_count_from_name(model_name)  # Returns in millions
        declared_params_actual = model_record.parameters_count  # Actual count (e.g., 7000000000)
        declared_params_millions = declared_params_actual / 1_000_000 if declared_params_actual else None

        # Track size extraction
        if size_str:
            models_with_size.append((model_name, size_str))
            if size_str not in parsing_examples:
                parsing_examples[size_str] = []
            parsing_examples[size_str].append(model_name)
        else:
            models_without_size.append(model_name)

        # Track declared parameters
        if declared_params_actual:
            models_with_params.append((model_name, declared_params_actual))
        else:
            models_without_params.append(model_name)

        # Check for mismatches
        if extracted_params_millions and declared_params_millions:
            diff_percent = abs(extracted_params_millions - declared_params_millions) / declared_params_millions * 100

            if diff_percent == 0:
                exact_matches.append((model_name, size_str, extracted_params_millions, declared_params_millions))
            elif diff_percent <= tolerance_percent:
                close_matches.append((model_name, size_str, extracted_params_millions, declared_params_millions, diff_percent))
            else:
                mismatches.append((model_name, size_str, extracted_params_millions, declared_params_millions, diff_percent))

    # Print results
    print("-" * 80)
    print("SIZE EXTRACTION RESULTS")
    print("-" * 80)
    print(f"Models with extractable size: {len(models_with_size)}")
    print(f"Models without extractable size: {len(models_without_size)}")
    print()

    if parsing_examples:
        print("Size patterns found:")
        for size_str in sorted(parsing_examples.keys()):
            examples = parsing_examples[size_str][:3]  # Show first 3 examples
            parsed_value = parse_parameter_count_from_size(size_str)
            print(f"  {size_str:10s} -> {parsed_value:6d}M  (examples: {', '.join(examples)})")
        print()

    print("-" * 80)
    print("DECLARED PARAMETERS")
    print("-" * 80)
    print(f"Models with declared parameters: {len(models_with_params)}")
    print(f"Models without declared parameters: {len(models_without_params)}")
    print()

    if models_without_params:
        print("Sample models without declared parameters:")
        for name in models_without_params[:5]:
            print(f"  - {name}")
        if len(models_without_params) > 5:
            print(f"  ... and {len(models_without_params) - 5} more")
        print()

    print("-" * 80)
    print("PARAMETER MATCHING RESULTS")
    print("-" * 80)
    print(f"Exact matches (0% diff): {len(exact_matches)}")
    print(f"Close matches (within {tolerance_percent}% tolerance): {len(close_matches)}")
    print(f"MISMATCHES (exceeds {tolerance_percent}% tolerance): {len(mismatches)}")
    print()

    if exact_matches:
        print("Sample exact matches:")
        for name, size_str, extracted, declared in exact_matches[:5]:
            print(f"  ✓ {name}")
            print(f"    Size: {size_str}, Extracted: {extracted:.0f}M, Declared: {declared:.0f}M")
        if len(exact_matches) > 5:
            print(f"  ... and {len(exact_matches) - 5} more")
        print()

    if close_matches:
        print("Sample close matches:")
        for name, size_str, extracted, declared, diff in close_matches[:5]:
            print(f"  ~ {name}")
            print(f"    Size: {size_str}, Extracted: {extracted:.0f}M, Declared: {declared:.0f}M ({diff:.1f}% diff)")
        if len(close_matches) > 5:
            print(f"  ... and {len(close_matches) - 5} more")
        print()

    if mismatches:
        print("⚠️  PARAMETER MISMATCHES DETECTED:")
        print()
        for name, size_str, extracted, declared, diff in sorted(mismatches, key=lambda x: x[4], reverse=True)[:20]:  # Limit to 20
            print(f"  ✗ {name}")
            print(f"    Size in name: {size_str} ({extracted:.0f}M)")
            print(f"    Declared params: {declared:.0f}M")
            print(f"    Difference: {diff:.1f}%")
            print()
        if len(mismatches) > 20:
            print(f"  ... and {len(mismatches) - 20} more mismatches")
            print()

    print("-" * 80)
    print("SUMMARY FOR TEST DEVELOPMENT")
    print("-" * 80)
    print(f"Total models analyzed: {len(text_models)}")
    print(f"Models suitable for testing (have both size and params): {len(exact_matches) + len(close_matches) + len(mismatches)}")
    print()
    print("Test coverage recommendations:")
    print(f"  - Test exact matches: {len(exact_matches)} cases available")
    print(f"  - Test tolerance matching: {len(close_matches)} cases available")
    print(f"  - Test mismatch detection: {len(mismatches)} cases available")
    print(f"  - Test missing size: {len(models_without_size)} cases available")
    print(f"  - Test missing params: {len(models_without_params)} cases available")
    print()

    # Generate test data suggestions
    print("-" * 80)
    print("SUGGESTED TEST CASES")
    print("-" * 80)
    print()

    if exact_matches:
        print("Exact match test cases:")
        for name, size_str, extracted, declared in exact_matches[:3]:
            actual_count = int(declared * 1_000_000)
            print(f'  ("{name}", {actual_count}),  # {size_str}, exact match')
        print()

    if close_matches:
        print("Close match test cases (within tolerance):")
        for name, size_str, extracted, declared, diff in close_matches[:3]:
            actual_count = int(declared * 1_000_000)
            print(f'  ("{name}", {actual_count}),  # {size_str}, {diff:.1f}% diff')
        print()

    if mismatches:
        print("Mismatch test cases (should flag):")
        for name, size_str, extracted, declared, diff in mismatches[:3]:
            actual_count = int(declared * 1_000_000)
            print(f'  ("{name}", {actual_count}),  # {size_str} vs {declared:.0f}M, {diff:.1f}% diff')
        print()

    if models_without_size:
        print("No size in name test cases:")
        for name in models_without_size[:3]:
            params = text_models[name].parameters_count if isinstance(text_models[name], TextGenerationModelRecord) else None
            print(f'  ("{name}", {params}),  # No size marker')
        print()

    print("=" * 80)


if __name__ == "__main__":
    analyze_parameter_mismatches()
