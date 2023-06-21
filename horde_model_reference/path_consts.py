"""Constants, especially those to do with paths or network locations, for the horde_model_reference package."""

import os
from pathlib import Path
from urllib.parse import urlparse

from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORIES

PACKAGE_NAME = "horde_model_reference"
"""The name of this package. Also used as the name of the base folder name for all model reference files."""

BASE_PATH: Path = Path(__file__).parent
"""The base path for all model reference files. Will be based in AIWORKER_CACHE_HOME if set, otherwise will be based in
 this package's install location (IE, in site-packages.)"""

AIWORKER_CACHE_HOME = os.getenv("AIWORKER_CACHE_HOME")
"""The default location for all AI-Horde-Worker cache (model) files."""

if AIWORKER_CACHE_HOME:
    BASE_PATH = Path(AIWORKER_CACHE_HOME).joinpath(PACKAGE_NAME)
BASE_PATH.mkdir(parents=True, exist_ok=True)

LOG_FOLDER: Path = BASE_PATH.joinpath("logs")
LOG_FOLDER.mkdir(parents=True, exist_ok=True)

LEGACY_REFERENCE_FOLDER_NAME: str = "legacy"
"""The default name of the legacy model reference folder.
If you need the default path, use `LEGACY_REFERENCE_FOLDER`."""

LEGACY_REFERENCE_FOLDER: Path = BASE_PATH.joinpath(LEGACY_REFERENCE_FOLDER_NAME)
"""The default path, starting with BASE_PATH, to the default legacy model reference folder. """
LEGACY_REFERENCE_FOLDER.mkdir(parents=True, exist_ok=True)

DEFAULT_SHOWCASE_FOLDER_NAME: str = "showcase"
"""The default name of the stable diffusion showcase folder. If you need the path, use `SHOWCASE_FOLDER_PATH`."""

SHOWCASE_FOLDER_PATH: Path = BASE_PATH.joinpath(DEFAULT_SHOWCASE_FOLDER_NAME)
"""The path to the stable diffusion showcase folder."""
SHOWCASE_FOLDER_PATH.mkdir(parents=True, exist_ok=True)


GITHUB_REPO_OWNER = "Haidra-Org"
GITHUB_REPO_NAME = "AI-Horde-image-model-reference"
GITHUB_REPO_BRANCH = "main"

GITHUB_REPO_URL: str = (
    f"https://raw.githubusercontent.com/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/{GITHUB_REPO_BRANCH}/"
)
"""The base URL to the live GitHub repo used to power the horde."""

LEGACY_MODEL_GITHUB_URLS = {}
"""A lookup of all the fully qualified file URLs to the given model reference."""

_MODEL_REFERENCE_FILENAMES: dict[MODEL_REFERENCE_CATEGORIES, str] = {}

for category in MODEL_REFERENCE_CATEGORIES:
    filename = f"{category}.json"
    _MODEL_REFERENCE_FILENAMES[category] = filename
    LEGACY_MODEL_GITHUB_URLS[category] = urlparse(GITHUB_REPO_URL + filename).geturl()


def get_model_reference_filename(
    model_reference_category: MODEL_REFERENCE_CATEGORIES,
) -> str:
    """Returns just the filename (not the path) of the model reference file for the given model reference category.

    Args:
        model_reference_category (MODEL_REFERENCE_CATEGORIES): The category of model reference to get the filename for.

    Returns:
        str: The filename of the model reference file.
    """
    return _MODEL_REFERENCE_FILENAMES[model_reference_category]


def get_model_reference_file_path(
    model_reference_category: MODEL_REFERENCE_CATEGORIES,
    *,
    base_path: str | Path = BASE_PATH,
) -> Path:
    """Returns the path to the model reference file for the given model reference category.

    Args:
        model_reference_category (MODEL_REFERENCE_CATEGORIES): The category of model reference to get the filename for.
        basePath (str | Path): If provided, the base path to the model reference file. Defaults to BASE_PATH.

    Returns:
        path:
    """
    if not isinstance(base_path, Path):
        base_path = Path(base_path)
    return base_path.joinpath(_MODEL_REFERENCE_FILENAMES[model_reference_category])
