from setuptools import setup

# Metadata goes in setup.cfg. These are here for GitHub's dependency graph.
requires = [
        "kaptan>=0.5.12",
        "scikit-learn>=1.0.2",
        "torch>=1.11.0",
        "torchinfo>=1.6.5",
    ]


# Create full install that includes all extra dependencies

setup(
    name="ednaml",
    install_requires=requires,
    keywords=["deep learning", "pytorch", "AI"],
)