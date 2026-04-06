#!/bin/bash
#
# publish mlops_wrapper to PyPI
#
# usage:
#   ./scripts/publish_to_pypi.sh [--test]
#
# options:
#   --test    upload to TestPyPI instead of production PyPI
#
# requirements:
#   - ~/.pypirc with [pypi] credentials
#   - python packages: build, twine
#
# github actions usage:
#   - store PyPI API token in repository secrets: PYPI_TOKEN
#   - use: pypa/gh-action-pypi-publish@release/v1

set -e  # exit on error

# check if running in test mode
TEST_MODE=false
if [[ "$1" == "--test" ]]; then
    TEST_MODE=true
    echo "running in TEST mode - will upload to TestPyPI"
fi

# check we're in the repo root
if [[ ! -f "pyproject.toml" ]]; then
    echo "error: must run from repository root"
    exit 1
fi

# extract version from pyproject.toml
VERSION=$(grep '^version = ' pyproject.toml | cut -d'"' -f2)
echo "packaging version: $VERSION"

# check if version already exists on PyPI
if [[ "$TEST_MODE" == false ]]; then
    if pip index versions mlops-wrapper 2>/dev/null | grep -q "$VERSION"; then
        echo "error: version $VERSION already exists on PyPI"
        echo "bump version in pyproject.toml first"
        exit 1
    fi
fi

# clean previous builds
echo "cleaning previous builds..."
rm -rf dist/ build/ *.egg-info

# determine python command (prefer python3 if available)
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

# install/upgrade build tools
echo "ensuring build tools are installed..."
$PYTHON_CMD -m pip install --upgrade build twine

# build package
echo "building package..."
$PYTHON_CMD -m build

# check package
echo "checking package with twine..."
twine check dist/*

# show what will be uploaded
echo ""
echo "package contents:"
tar -tzf dist/mlops_wrapper-${VERSION}.tar.gz | head -20
echo ""

# ask for confirmation
if [[ "$TEST_MODE" == false ]]; then
    echo "ready to upload to PyPI (production)"
    echo "version: $VERSION"
    read -p "proceed? (yes/no): " CONFIRM
    if [[ "$CONFIRM" != "yes" ]]; then
        echo "aborted"
        exit 1
    fi
else
    echo "ready to upload to TestPyPI"
fi

# upload
if [[ "$TEST_MODE" == true ]]; then
    echo "uploading to TestPyPI..."
    twine upload --repository testpypi dist/*
    echo ""
    echo "✓ uploaded to TestPyPI"
    echo "test install: pip install --index-url https://test.pypi.org/simple/ mlops-wrapper==$VERSION"
else
    echo "uploading to PyPI..."
    twine upload dist/*
    echo ""
    echo "✓ uploaded to PyPI"
    echo "install: pip install mlops-wrapper==$VERSION"
    echo "view: https://pypi.org/project/mlops-wrapper/$VERSION/"
fi
