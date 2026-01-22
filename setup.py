from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="benchmaxxing",
    version="0.1.0",
    author="Scicom AI Enterprise",
    description="Seamless scripts for LLM performance benchmarking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Scicom-AI-Enterprise-Organization/llm-benchmark",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        # Core dependencies (for remote execution)
        "pyyaml",
        "requests",
        "cloudpickle",
        "paramiko",
        "pyremote @ git+https://github.com/Scicom-AI-Enterprise-Organization/pyremote",
    ],
    extras_require={
        # For local vLLM benchmarking
        "vllm": [
            "vllm==0.11.0",
            "huggingface_hub[cli]",
        ],
        # All engines (for future expansion)
        "all": [
            "vllm==0.11.0",
            "huggingface_hub[cli]",
        ],
    },
    entry_points={
        "console_scripts": [
            "benchmaxxing=benchmaxxing.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    include_package_data=True,
)
