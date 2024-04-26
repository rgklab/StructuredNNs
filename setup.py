from setuptools import setup, find_packages

# Parse README
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="strnn",
    version="0.2.0",
    author="RGKLab",
    description="PyTorch package for Structured Neural Networks.",
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/rgklab/StructuredNNs",
    install_requires=[
        "torch>=2.0.0",
        "scikit-learn",
        "matplotlib",
        "torchdiffeq==0.2.3",
        "UMNN",
        "wandb"
    ],
    python_requires='>=3.11',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    packages=find_packages(
        where=".",
        exclude=["experiments*", "data*", "test*"]
    ),
)
