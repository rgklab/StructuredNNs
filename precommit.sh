python -m pytest
mypy . --ignore-missing-imports
flake8 . --ignore=F403,F405
