# README

## Temp: Development
Hey folks, we should probably try to keep the code and repo quality up. Lets do this for our
development:
1. We should develop our project as a package. The setup file should work now, and you can install
   a hot-reloading version of the package by running `pip install -e .` from the project root. Please
   fix any missing dependencies by modifying setup.py to include it in the `install_requires` variable. I think we can also lower the python version requirement, but not sure what to yet.

2. Work on a local branch, I think the below sketch of workflow works.
   Run these commands to set it up:

        git checkout -b <branch>
        git push -u origin <branch>

    You can commit normally to your local branch, and then when you are ready
    to push changes to main, run:

        git checkout main
        git pull
        git checkout <branch>
        git merge main

    Any merge conflicts would have to be resolved. After you commit the merge
    to your branch, run the below tests and static checks, and then run:

        git checkout main
        git merge --squash <branch>
        git commit -m "message"
        git push

2. The below scripts have been summarized in a bash script. Please call:

        bash precommit.sh

4. Write tests where applicable. Currently using pytest, which can be installed via `pip install pytest` or via conda. Tests can be run by going to the project root and running (-s flag prints stdout):

        python -m pytest -s

5. Running static type checking. Note this can take awhile for the first run. Currently using mypy, which can be installed via `pip install mypy` or via conda. Change to root directory, and run:

        mypy data --ignore-missing-imports
        mypy experiments --ignore-missing-imports
        mypy strnn --ignore-missing-imports

6. Style linting. Currently using flake8, which can be installed via `pip install flake8`. Change to root directory, and then run:

        flake8 . --ignore=F403,F405

    The F403, F405 flags ignore issues with star imports, though we can discuss if we should change that.


# Structured Neural Networks for Density Estimation
Official implementation of [TODO: Paper link]

[TODO: Figure 1]

## Introduction

## Citation

## Setup

## Example
