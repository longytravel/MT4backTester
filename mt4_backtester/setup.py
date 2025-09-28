"""Setup configuration for MT4 Universal Backtesting Framework."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mt4-backtester",
    version="1.0.0",
    author="MT4 Backtester Team",
    description="A blazing-fast universal backtesting framework for trading strategies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/longytravel/MT4backTester",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "plotly>=5.0.0",
            "streamlit>=1.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mt4-backtest=mt4_backtester.cli:main",
            "mt4-optimize=mt4_backtester.cli:optimize",
            "mt4-convert=mt4_backtester.cli:convert_data",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/longytravel/MT4backTester/issues",
        "Source": "https://github.com/longytravel/MT4backTester",
    },
)