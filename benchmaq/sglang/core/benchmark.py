import subprocess
import sys
import os
from typing import Dict, Any, Optional


def run_benchmark(
    model: str,
    port: int,
    result_name: str,
    results_config: Optional[Dict[str, Any]] = None,
    **kwargs
):
    """Run SGLang bench_serving with dynamic kwargs.
    
    All kwargs are converted to CLI arguments using the pattern:
    - key_name -> --key-name
    - Boolean True -> flag added (--key-name)
    - Boolean False -> flag omitted
    - Other values -> --key-name value
    
    Args:
        model: Model path or HuggingFace repo ID (for tokenizer)
        port: Server port to connect to
        result_name: Name for this benchmark run (used in result filenames)
        results_config: Optional dict with save_result, result_dir, output_details
        **kwargs: Any sglang.bench_serving arguments
        
    Common kwargs:
        Backend & Connection:
            backend: Backend type (sglang, sglang-oai, sglang-oai-chat, vllm, etc.)
            host: Server host (default: 127.0.0.1)
            base_url: Full base URL (alternative to host/port)
        
        Dataset:
            dataset_name: Dataset type (random, sharegpt, random-ids, image, etc.)
            num_prompts: Number of requests to send
            random_input_len: Input length for random dataset
            random_output_len: Output length for random dataset
            random_range_ratio: Range ratio for random lengths
            sharegpt_output_len: Override output length for sharegpt
            apply_chat_template: Apply tokenizer chat template
        
        Rate & Concurrency:
            request_rate: Requests per second (use "inf" for burst)
            max_concurrency: Max concurrent in-flight requests
            disable_stream: Use non-streaming mode
        
        Other:
            warmup_requests: Number of warmup requests (default: 1)
            flush_cache: Call /flush_cache before run
            extra_request_body: JSON string for extra request params
            disable_ignore_eos: Pass through EOS behavior
            
    Example:
        run_benchmark(
            model="meta-llama/Llama-3-8B-Instruct",
            port=30000,
            result_name="test_run",
            results_config={"save_result": True, "result_dir": "./results"},
            backend="sglang",
            dataset_name="random",
            random_input_len=1024,
            random_output_len=128,
            num_prompts=100,
            max_concurrency=100,
            request_rate="inf"
        )
    
    See https://docs.sglang.io/developer_guide/bench_serving.html for full list.
    """
    print()
    print("=" * 64)
    print(f"BENCHMARK: {result_name}")
    print("=" * 64)
    sys.stdout.flush()

    # Build base command
    cmd = [
        "python", "-m", "sglang.bench_serving",
        "--port", str(port),
        "--model", model,
    ]
    
    # Add host if not using base_url
    if "base_url" not in kwargs:
        host = kwargs.pop("host", "127.0.0.1")
        cmd.extend(["--host", host])

    # Add benchmark kwargs
    for key, value in kwargs.items():
        # Convert Python snake_case to CLI kebab-case
        arg_name = f"--{key.replace('_', '-')}"
        
        if isinstance(value, bool):
            # Boolean flags: only add if True
            if value:
                cmd.append(arg_name)
        elif value is not None:
            # Other values: add as --key value
            cmd.extend([arg_name, str(value)])

    # Handle results configuration
    results_config = results_config or {}
    save_result = results_config.get("save_result", False)
    result_dir = results_config.get("result_dir", "./benchmark_results")
    output_details = results_config.get("output_details", True)
    
    if save_result:
        os.makedirs(result_dir, exist_ok=True)
        output_file = os.path.join(result_dir, f"{result_name}.jsonl")
        cmd.extend(["--output-file", output_file])
        if output_details:
            cmd.append("--output-details")

    print(f"Running: {' '.join(cmd)}")
    
    # Stream output live and capture for log file
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    log_lines = []
    for line in process.stdout:
        print(line, end='', flush=True)
        log_lines.append(line)

    process.wait()

    # Save console output to .txt log file
    if save_result:
        log_path = os.path.join(result_dir, f"{result_name}.txt")
        with open(log_path, "w") as f:
            f.write(f"BENCHMARK: {result_name}\n")
            f.write("=" * 64 + "\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write("=" * 64 + "\n\n")
            f.writelines(log_lines)
    
    return process.returncode
