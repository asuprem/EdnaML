from setuptools import setup
import os


requires = [
        "scikit-learn>=1.0.2",
        "torch>=1.10.*",
        "torchinfo>=1.6.5",
        "torchvision>=0.11.*",
        "Pillow>=7.1.2",    #9.0.x, this is for colab...
        "tqdm>=4.63.*",
        "sentencepiece>=0.1.96",
        "sortedcontainers>=2.4.0",
        "pyyaml>=6.0",
    ]


# Create full install that includes all extra dependencies

setup(
    name="ednaml",
    install_requires=requires,
    keywords=["deep learning", "pytorch", "AI"],
)

# Azure :
# Install azure-storage>=0.30.0