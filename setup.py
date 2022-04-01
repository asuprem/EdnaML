from setuptools import setup
import os

_PATH_ROOT = os.path.dirname(__file__)
_PATH_REQUIRE = os.path.join(_PATH_ROOT, "requirements")


requires = [
        "kaptan>=0.5.12",
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