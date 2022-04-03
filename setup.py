from setuptools import setup
import os


requires = [
        "scikit-learn>=1.0.2",
        "torch>=1.10.*",
        "torchinfo>=1.6.5",
        "torchvision>=0.11.*",
        "Pillow>=9.0.*",
        "tqdm>=4.63.*",
    ]


# Create full install that includes all extra dependencies

setup(
    name="ednaml",
    install_requires=requires,
    keywords=["deep learning", "pytorch", "AI"],
)