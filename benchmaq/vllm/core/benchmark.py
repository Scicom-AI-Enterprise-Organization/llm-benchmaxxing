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
    """Run vLLM bench serve with dynamic kwargs.
    
    All kwargs are converted to CLI arguments using the pattern:
    - key_name -> --key-name
    - Boolean True -> flag added (--key-name)
    - Boolean False -> flag omitted
    - Other values -> --key-name value
    
    Args:
        model: Model path or HuggingFace repo ID
        port: Server port to connect to
        result_name: Name for this benchmark run (used in result filenames)
        results_config: Optional dict with save_result, result_dir, result_filename, save_detailed
        **kwargs: Any vllm bench serve arguments
        
    Example:
        run_benchmark(
            model="meta-llama/Llama-2-7b",
            port=8000,
            result_name="test_run",
            results_config={"save_result": True, "result_dir": "./results"},
            backend="vllm",
            endpoint="/v1/completions",
            dataset_name="random",
            random_input_len=1024,
            random_output_len=128,
            num_prompts=100,
            max_concurrency=100,
            ignore_eos=True
        )
    """
    print()
    print("=" * 64)
    print(f"BENCHMARK: {result_name}")
    print("=" * 64)
    sys.stdout.flush()

    # Build base command
    cmd = [
        "vllm", "bench", "serve",
        "--base-url", f"http://localhost:{port}",
        "--model", model,
    ]

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
    result_filename = results_config.get("result_filename")
    save_detailed = results_config.get("save_detailed", False)
    
    if save_result:
        cmd.append("--save-result")
        cmd.extend(["--result-dir", result_dir])
        if result_filename:
            cmd.extend(["--result-filename", result_filename])
        else:
            cmd.extend(["--result-filename", f"{result_name}.json"])
        if save_detailed:
            cmd.append("--save-detailed")

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
        # Filter out APIServer logs
        if "(APIServer)" not in line:
            log_lines.append(line)

    process.wait()

    # Save console output to .txt log file (without APIServer logs)
    if save_result:
        os.makedirs(result_dir, exist_ok=True)
        log_path = os.path.join(result_dir, f"{result_name}.txt")
        with open(log_path, "w") as f:
            f.write(f"BENCHMARK: {result_name}\n")
            f.write("=" * 64 + "\n")
            f.writelines(log_lines)
    
    return process.returncode


# Legacy function for backward compatibility
def run_benchmark_legacy(model_path, port, output_dir, result_name, ctx, output_len, num_prompts, concurrency, save_results=False):
    """Legacy run_benchmark function for backward compatibility."""
    return run_benchmark(
        model=model_path,
        port=port,
        result_name=result_name,
        results_config={"save_result": save_results, "result_dir": output_dir},
        backend="vllm",
        endpoint="/v1/completions",
        dataset_name="random",
        random_input_len=ctx,
        random_output_len=output_len,
        num_prompts=num_prompts,
        max_concurrency=concurrency,
        request_rate="inf",
        ignore_eos=True,
        percentile_metrics="ttft,tpot,itl,e2el"
    )
