"""Text model name parser for grouping and analysis.

Provides functions to parse text generation model names into structured components
like base name, size, variant, and quantization. Useful for grouping model variants
(e.g., different quant versions of the same base model).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from loguru import logger


@dataclass
class ParsedTextModelName:
    """Structured representation of a parsed text model name.

    Attributes:
        original_name: The original model name as provided.
        base_name: The base model name without size/variant/quant info.
        size: Model size if detected (e.g., "7B", "13B", "70B").
        variant: Model variant if detected (e.g., "Instruct", "Chat", "Code").
        quant: Quantization type if detected (e.g., "Q4", "Q8", "GGUF").
        normalized_name: A normalized version of the name for comparison.
    """

    original_name: str
    base_name: str
    size: str | None = None
    variant: str | None = None
    quant: str | None = None
    normalized_name: str | None = None


# Common text model size patterns
SIZE_PATTERNS = [
    r"\b(\d+\.?\d*[BMK])\b",  # 7B, 13B, 70B, 1.5B, 3.5K, etc.
    r"\b(\d+x\d+[BMK])\b",  # MoE models: 8x7B, 8x22B
]

# Common variant indicators
VARIANT_PATTERNS = [
    r"\b(Instruct|Chat|Code|Base|Uncensored|Finetune|FT)\b",
    r"\b(turbo|preview|latest)\b",
]

# Quantization patterns
QUANT_PATTERNS = [
    r"\b(Q[2-8](?:_K)?(?:_[SMLH])?)\b",  # Q4_K_M, Q8, Q5_K_S
    r"\b(GGUF|GGML|GPTQ|AWQ|EXL2)\b",
    r"\b(fp16|fp32|int8|int4)\b",
]

# Separators to normalize
SEPARATORS = ["-", "_", " ", "."]


@lru_cache(maxsize=2048)
def parse_text_model_name(model_name: str) -> ParsedTextModelName:
    """Parse a text model name into structured components.

    Attempts to extract base name, size, variant, and quantization information
    from a model name using regex patterns.

    Args:
        model_name: The full model name to parse.

    Returns:
        ParsedTextModelName with extracted components.

    Example:
        >>> parsed = parse_text_model_name("Llama-3-8B-Instruct-Q4_K_M")
        >>> print(parsed.base_name)  # "Llama-3"
        >>> print(parsed.size)  # "8B"
        >>> print(parsed.variant)  # "Instruct"
        >>> print(parsed.quant)  # "Q4_K_M"
    """
    logger.debug(f"Parsing text model name: {model_name}")

    name_parts = model_name
    size = None
    variant = None
    quant = None

    # Extract size
    for pattern in SIZE_PATTERNS:
        match = re.search(pattern, name_parts, re.IGNORECASE)
        if match:
            size = match.group(1).upper()
            name_parts = name_parts[: match.start()] + name_parts[match.end() :]
            logger.debug(f"Extracted size: {size}")
            break

    # Extract quantization
    for pattern in QUANT_PATTERNS:
        match = re.search(pattern, name_parts, re.IGNORECASE)
        if match:
            quant = match.group(1).upper()
            name_parts = name_parts[: match.start()] + name_parts[match.end() :]
            logger.debug(f"Extracted quant: {quant}")
            break

    # Extract variant
    for pattern in VARIANT_PATTERNS:
        match = re.search(pattern, name_parts, re.IGNORECASE)
        if match:
            variant = match.group(1)
            name_parts = name_parts[: match.start()] + name_parts[match.end() :]
            logger.debug(f"Extracted variant: {variant}")
            break

    # Clean up base name
    base_name = name_parts
    for sep in SEPARATORS:
        base_name = base_name.replace(sep + sep, sep)

    base_name = base_name.strip("-_ .")

    if not base_name:
        base_name = model_name
        logger.debug(f"Could not extract base name, using original: {base_name}")
    else:
        logger.debug(f"Extracted base name: {base_name}")

    normalized = normalize_model_name(model_name)

    return ParsedTextModelName(
        original_name=model_name,
        base_name=base_name,
        size=size,
        variant=variant,
        quant=quant,
        normalized_name=normalized,
    )


@lru_cache(maxsize=2048)
def get_base_model_name(model_name: str) -> str:
    """Get the base model name for grouping purposes.

    Extracts just the base name without size, variant, or quantization info.
    Useful for grouping different variants of the same model together.

    Args:
        model_name: The full model name.

    Returns:
        The base model name.

    Example:
        >>> get_base_model_name("Llama-3-8B-Instruct-Q4_K_M")
        "Llama-3"
        >>> get_base_model_name("Mistral-7B-v0.1")
        "Mistral"
    """
    parsed = parse_text_model_name(model_name)
    return parsed.base_name


@lru_cache(maxsize=2048)
def normalize_model_name(model_name: str) -> str:
    """Normalize a model name for case-insensitive comparison.

    Converts to lowercase and normalizes separators.

    Args:
        model_name: The model name to normalize.

    Returns:
        Normalized model name.

    Example:
        >>> normalize_model_name("Llama-3-8B-Instruct")
        "llama_3_8b_instruct"
    """
    normalized = model_name.lower()

    for sep in ["-", " ", "."]:
        normalized = normalized.replace(sep, "_")

    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def group_text_models_by_base(
    model_names: list[str],
) -> dict[str, list[str]]:
    """Group text model names by their base model.

    Groups variants of the same model together based on extracted base names.

    Args:
        model_names: List of model names to group.

    Returns:
        Dictionary mapping base names to lists of full model names.

    Example:
        >>> models = [
        ...     "Llama-3-8B-Instruct",
        ...     "Llama-3-8B-Instruct-Q4",
        ...     "Llama-3-70B-Instruct",
        ...     "Mistral-7B-v0.1",
        ... ]
        >>> grouped = group_text_models_by_base(models)
        >>> print(grouped)
        {
            "Llama-3": ["Llama-3-8B-Instruct", "Llama-3-8B-Instruct-Q4", "Llama-3-70B-Instruct"],
            "Mistral": ["Mistral-7B-v0.1"]
        }
    """
    grouped: dict[str, list[str]] = {}

    for model_name in model_names:
        base_name = get_base_model_name(model_name)

        if base_name not in grouped:
            grouped[base_name] = []

        grouped[base_name].append(model_name)

    logger.debug(f"Grouped {len(model_names)} models into {len(grouped)} base models")

    return grouped


@lru_cache(maxsize=2048)
def is_quantized_variant(model_name: str) -> bool:
    """Check if a model name indicates a quantized variant.

    Args:
        model_name: The model name to check.

    Returns:
        True if the model appears to be a quantized variant.

    Example:
        >>> is_quantized_variant("Llama-3-8B-Instruct-Q4_K_M")
        True
        >>> is_quantized_variant("Llama-3-8B-Instruct")
        False
    """
    parsed = parse_text_model_name(model_name)
    return parsed.quant is not None


@lru_cache(maxsize=2048)
def get_model_size(model_name: str) -> str | None:
    """Extract the model size from a model name.

    Args:
        model_name: The model name to parse.

    Returns:
        The model size (e.g., "7B", "13B") or None if not found.

    Example:
        >>> get_model_size("Llama-3-8B-Instruct")
        "8B"
        >>> get_model_size("GPT-4")
        None
    """
    parsed = parse_text_model_name(model_name)
    return parsed.size


@lru_cache(maxsize=2048)
def get_model_variant(model_name: str) -> str | None:
    """Extract the model variant from a model name.

    Args:
        model_name: The model name to parse.

    Returns:
        The model variant (e.g., "Instruct", "Chat") or None if not found.

    Example:
        >>> get_model_variant("Llama-3-8B-Instruct")
        "Instruct"
        >>> get_model_variant("Llama-3-8B")
        None
    """
    parsed = parse_text_model_name(model_name)
    return parsed.variant


def parse_parameter_count_from_size(size_str: str) -> int | None:
    """Convert a model size string to parameter count in millions.

    Args:
        size_str: Size string like "7B", "13B", "70B", "1.5B", "8x7B", etc.

    Returns:
        Parameter count in millions, or None if cannot parse.

    Examples:
        >>> parse_parameter_count_from_size("7B")
        7000
        >>> parse_parameter_count_from_size("1.5B")
        1500
        >>> parse_parameter_count_from_size("8x7B")
        56000
        >>> parse_parameter_count_from_size("500M")
        500
        >>> parse_parameter_count_from_size("3.5K")
        None
    """
    if not size_str:
        return None

    size_str = size_str.upper().strip()

    # Handle MoE models (e.g., "8x7B" = 8 experts * 7B each)
    moe_match = re.match(r"^(\d+)X(\d+\.?\d*)([BMK])$", size_str)
    if moe_match:
        num_experts = int(moe_match.group(1))
        param_value = float(moe_match.group(2))
        unit = moe_match.group(3)

        if unit == "B":
            return int(num_experts * param_value * 1000)
        elif unit == "M":
            return int(num_experts * param_value)
        # K (thousands) not typically used for model parameters
        return None

    # Handle regular sizes (e.g., "7B", "1.5B", "500M")
    regular_match = re.match(r"^(\d+\.?\d*)([BMK])$", size_str)
    if regular_match:
        param_value = float(regular_match.group(1))
        unit = regular_match.group(2)

        if unit == "B":
            return int(param_value * 1000)
        elif unit == "M":
            return int(param_value)
        # K (thousands) not typically used for model parameters
        return None

    return None


@lru_cache(maxsize=2048)
def extract_parameter_count_from_name(model_name: str) -> int | None:
    """Extract parameter count from model name.

    Args:
        model_name: The model name to parse.

    Returns:
        Parameter count in millions, or None if not found or cannot parse.

    Examples:
        >>> extract_parameter_count_from_name("Llama-3-8B-Instruct")
        8000
        >>> extract_parameter_count_from_name("Mixtral-8x7B-Instruct-v0.1")
        56000
        >>> extract_parameter_count_from_name("GPT-4")
        None
    """
    size_str = get_model_size(model_name)
    if not size_str:
        return None
    return parse_parameter_count_from_size(size_str)
