from setuptools import setup

setup(
    name="strnn",
    version="0.0.1",
    author="RGKLab",
    description="PyTorch package for Structured Neural Networks.",
    url="https://github.com/rgklab/StructuredNNs",
    install_requires=[
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "scikit-learn",
        "torchdiffeq>=0.2.3",
        "UMNN"
    ],
    python_requires='~=3.11',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    packages=["strnn"],
)
