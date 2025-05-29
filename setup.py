from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="industrial-failure-prediction",
    version="1.0.0",
    author="Industrial IoT Team",
    author_email="team@industrialiot.com",
    description="Sistema di predizione guasti per dispositivi IoT industriali",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Luca06Zen/ProvaDispositiviIoT2",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Manufacturing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=5.0",
            "isort>=5.0",
        ],
        "notebook": [
            "jupyter>=1.0",
            "ipykernel>=6.0",
            "matplotlib>=3.5",
            "seaborn>=0.11",
        ],
    },
    entry_points={
        "console_scripts": [
            "industrial-predict=src.prediction_engine:main",
            "industrial-train=src.model_training:main",
            "industrial-web=web.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.csv"],
    },
    project_urls={
        "Bug Reports": "https://github.com/Luca06Zen/ProvaDispositiviIoT2/issues",
        "Source": "https://github.com/Luca06Zen/ProvaDispositiviIoT2",
        "Documentation": "https://github.com/Luca06Zen/ProvaDispositiviIoT2/blob/main/README.md",
    },
    keywords="industrial iot machine learning failure prediction maintenance",
    zip_safe=False,
)