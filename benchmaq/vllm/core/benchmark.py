import subprocess
import sys
import os


def run_benchmark(model_path, port, output_dir, result_name, ctx, output_len, num_prompts, concurrency, save_results=False):
    print()
    print("=" * 64)
    print(f"BENCHMARK: {result_name}")
    print("=" * 64)
    sys.stdout.flush()

    cmd = [
        "vllm", "bench", "serve",
        "--backend", "vllm",
        "--base-url", f"http://localhost:{port}",
        "--model", model_path,
        "--endpoint", "/v1/completions",
        "--dataset-name", "random",
        "--random-input-len", str(ctx),
        "--random-output-len", str(output_len),
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(concurrency),
        "--request-rate", "inf",
        "--ignore-eos",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
    ]

    if save_results:
        cmd.extend([
            "--save-result",
            "--result-dir", output_dir,
            "--result-filename", f"{result_name}.json",
        ])

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
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, f"{result_name}.txt")
        with open(log_path, "w") as f:
            f.write(f"BENCHMARK: {result_name}\n")
            f.write("=" * 64 + "\n")
            f.writelines(log_lines)
