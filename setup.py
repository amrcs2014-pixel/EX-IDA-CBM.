"""Setup script for the EX-IDA-CBM package."""

from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).parent
long_description = (ROOT / "README.md").read_text(encoding="utf-8")
requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
requirements = [r for r in requirements if r and not r.startswith("#")]

setup(
    name="ex-ida-cbm",
    version="1.0.0",
    description=(
        "Explainable Iron-Deficiency-Anemia detection from peripheral blood "
        "smears via Concept Bottleneck Models and attention-guided "
        "morphological maps."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Amr H. Abdelhaliem et al.",
    author_email="a.hasan@inu.edu.jo",
    url="https://github.com/amrogomma-web/EX-IDA-CBM",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=requirements,
)
