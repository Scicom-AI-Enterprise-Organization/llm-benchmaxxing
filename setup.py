from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="benchmaq",
    version="0.6.0",
    author="Scicom AI Enterprise",
    description="Seamless scripts for LLM performance benchmarking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Scicom-AI-Enterprise-Organization/llm-benchmaq",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pyyaml",
        "requests",
        "tqdm",
        "cloudpickle",
        "paramiko",
        "scp",
        "pyremote @ git+https://github.com/Scicom-AI-Enterprise-Organization/pyremote",
        "runpod"
    ],
    extras_require={
        # For local vLLM benchmarking
        "vllm": [
            "vllm==0.15.0",
            "huggingface_hub[cli,hf_transfer]",
            "hf_transfer",
        ],
        # For local SGLang benchmarking
        "sglang": [
            "sglang[all]",
            "huggingface_hub[cli,hf_transfer]",
            "hf_transfer",
        ],
        # For SkyPilot cloud orchestration
        "skypilot": [
            "skypilot[all]",
        ],
        # All engines and cloud providers
        "all": [
            "vllm==0.15.0",
            "sglang[all]",
            "huggingface_hub[cli,hf_transfer]",
            "hf_transfer",
            "skypilot[all]",
        ],
        # Development/testing dependencies
        "dev": [
            "pytest>=7.0.0",
            "pytest-mock>=3.10.0",
            "python-dotenv>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "benchmaq=benchmaq.cli:main",
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
