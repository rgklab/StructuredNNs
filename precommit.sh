python -m pytest
mypy data --ignore-missing-imports
mypy experiments --ignore-missing-imports
mypy strnn --ignore-missing-imports
flake8 . --ignore=F403,F405,D100,D104
