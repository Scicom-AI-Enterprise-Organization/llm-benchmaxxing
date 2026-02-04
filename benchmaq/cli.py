#!/usr/bin/env python3
"""
benchmaq CLI - LLM benchmarking toolkit

Usage:
    benchmaq bench <config.yaml>              # Run benchmark directly (reads 'benchmark' key)
    benchmaq vllm bench <config.yaml>         # Run vLLM benchmark (local or remote SSH)
    benchmaq sglang bench <config.yaml>       # Run SGLang benchmark (local or remote SSH)
    benchmaq runpod bench <config.yaml>       # End-to-end RunPod benchmark
    benchmaq sky bench --config <file>        # End-to-end SkyPilot benchmark
"""
import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="benchmaq",
        description="LLM benchmarking toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  benchmaq bench config.yaml              Run benchmark directly from YAML config
  benchmaq vllm bench config.yaml         Run vLLM benchmark from YAML config
  benchmaq sglang bench config.yaml       Run SGLang benchmark from YAML config
  benchmaq runpod bench config.yaml       End-to-end RunPod benchmark (deploy -> bench -> delete)
  benchmaq sky bench -c config.yaml       End-to-end SkyPilot benchmark (launch -> bench -> down)
        """
    )
    subparsers = parser.add_subparsers(dest="command")

    # =================================================================
    # bench command (direct execution, used by remote runners)
    # =================================================================
    bench_parser = subparsers.add_parser(
        "bench",
        help="Run benchmark directly from YAML config",
        description="Run benchmarks directly (reads 'benchmark' key from config)"
    )
    bench_parser.add_argument("config", help="Path to YAML config file")

    # =================================================================
    # vllm command
    # =================================================================
    vllm_parser = subparsers.add_parser(
        "vllm",
        help="vLLM benchmarking commands",
        description="Run vLLM benchmarks using YAML configuration"
    )
    vllm_subparsers = vllm_parser.add_subparsers(dest="vllm_command")

    # vllm bench
    vllm_bench_parser = vllm_subparsers.add_parser(
        "bench",
        help="Run vLLM benchmark from YAML config",
        description="Run vLLM benchmarks locally or on a remote GPU server via SSH"
    )
    vllm_bench_parser.add_argument("config", help="Path to YAML config file")

    # =================================================================
    # sglang command
    # =================================================================
    sglang_parser = subparsers.add_parser(
        "sglang",
        help="SGLang benchmarking commands",
        description="Run SGLang benchmarks using YAML configuration"
    )
    sglang_subparsers = sglang_parser.add_subparsers(dest="sglang_command")

    # sglang bench
    sglang_bench_parser = sglang_subparsers.add_parser(
        "bench",
        help="Run SGLang benchmark from YAML config",
        description="Run SGLang benchmarks locally or on a remote GPU server via SSH"
    )
    sglang_bench_parser.add_argument("config", help="Path to YAML config file")

    # =================================================================
    # runpod command
    # =================================================================
    runpod_parser = subparsers.add_parser(
        "runpod",
        help="RunPod benchmarking commands",
        description="Run end-to-end benchmarks on RunPod GPU pods"
    )
    runpod_subparsers = runpod_parser.add_subparsers(dest="runpod_command")

    # runpod bench
    runpod_bench_parser = runpod_subparsers.add_parser(
        "bench",
        help="End-to-end RunPod benchmark (deploy -> bench -> delete)",
        description="Deploy a RunPod pod, run benchmarks, download results, and delete the pod"
    )
    runpod_bench_parser.add_argument("config", help="Path to YAML config file")

    # =================================================================
    # sky command (SkyPilot)
    # =================================================================
    sky_parser = subparsers.add_parser(
        "sky",
        help="SkyPilot benchmarking commands",
        description="Run end-to-end benchmarks on SkyPilot-managed cloud infrastructure"
    )
    sky_subparsers = sky_parser.add_subparsers(dest="sky_command")

    # sky bench
    sky_bench_parser = sky_subparsers.add_parser(
        "bench",
        help="End-to-end SkyPilot benchmark (launch -> bench -> down)",
        description="Launch a SkyPilot cluster, run benchmarks, and tear down the cluster"
    )
    sky_bench_parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to YAML config file (referenced as $config in the YAML)"
    )

    args = parser.parse_args()

    # =================================================================
    # Handle bench command (direct execution)
    # =================================================================
    if args.command == "bench":
        from .vllm.bench import from_yaml
        
        config_path = args.config
        if not os.path.exists(config_path):
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
        
        print(f"Running benchmark from: {config_path}")
        result = from_yaml(config_path)
        
        if result.get("status") == "success":
            print("\n" + "=" * 64)
            print("BENCHMARK COMPLETED SUCCESSFULLY")
            print("=" * 64)
        else:
            print(f"\nError: {result.get('error', 'Unknown error')}")
            sys.exit(1)

    # =================================================================
    # Handle vllm command
    # =================================================================
    elif args.command == "vllm":
        if args.vllm_command == "bench":
            from .vllm.bench import from_yaml
            
            config_path = args.config
            if not os.path.exists(config_path):
                print(f"Error: Config file not found: {config_path}")
                sys.exit(1)
            
            print(f"Running vLLM benchmark from: {config_path}")
            result = from_yaml(config_path)
            
            if result.get("status") == "success":
                print("\n" + "=" * 64)
                print("BENCHMARK COMPLETED SUCCESSFULLY")
                print("=" * 64)
                if result.get("mode") == "remote":
                    print(f"Mode: Remote ({result.get('host')})")
                else:
                    print(f"Total runs: {len(result.get('results', []))}")
                    for r in result.get("results", []):
                        print(f"  - {r.get('name', 'unknown')}")
            else:
                print(f"\nError: {result.get('error', 'Unknown error')}")
                sys.exit(1)
        else:
            vllm_parser.print_help()

    # =================================================================
    # Handle sglang command
    # =================================================================
    elif args.command == "sglang":
        if args.sglang_command == "bench":
            from .sglang.bench import from_yaml
            
            config_path = args.config
            if not os.path.exists(config_path):
                print(f"Error: Config file not found: {config_path}")
                sys.exit(1)
            
            print(f"Running SGLang benchmark from: {config_path}")
            result = from_yaml(config_path)
            
            if result.get("status") == "success":
                print("\n" + "=" * 64)
                print("BENCHMARK COMPLETED SUCCESSFULLY")
                print("=" * 64)
                if result.get("mode") == "remote":
                    print(f"Mode: Remote ({result.get('host')})")
                else:
                    print(f"Total runs: {len(result.get('results', []))}")
                    for r in result.get("results", []):
                        print(f"  - {r.get('name', 'unknown')}")
            else:
                print(f"\nError: {result.get('error', 'Unknown error')}")
                sys.exit(1)
        else:
            sglang_parser.print_help()

    # =================================================================
    # Handle runpod command
    # =================================================================
    elif args.command == "runpod":
        if args.runpod_command == "bench":
            from .runpod.bench import from_yaml
            
            config_path = args.config
            if not os.path.exists(config_path):
                print(f"Error: Config file not found: {config_path}")
                sys.exit(1)
            
            print(f"Running RunPod benchmark from: {config_path}")
            result = from_yaml(config_path)
            
            if result.get("status") == "success":
                print("\n" + "=" * 64)
                print("RUNPOD BENCHMARK COMPLETED SUCCESSFULLY")
                print("=" * 64)
                print(f"GPU Type: {result.get('gpu_type')}")
                print(f"GPU Count: {result.get('gpu_count')}")
            else:
                print(f"\nError: {result.get('error', 'Unknown error')}")
                sys.exit(1)
        else:
            runpod_parser.print_help()

    # =================================================================
    # Handle sky command (SkyPilot)
    # =================================================================
    elif args.command == "sky":
        if args.sky_command == "bench":
            from .skypilot.bench import from_yaml
            
            config_path = args.config
            if not os.path.exists(config_path):
                print(f"Error: Config file not found: {config_path}")
                sys.exit(1)
            
            print(f"Running SkyPilot benchmark from: {config_path}")
            result = from_yaml(config_path)
            
            if result.get("status") == "success":
                print("\n" + "=" * 64)
                print("SKYPILOT BENCHMARK COMPLETED SUCCESSFULLY")
                print("=" * 64)
                print(f"Cluster: {result.get('cluster_name')}")
                print(f"Job ID: {result.get('job_id')}")
            elif result.get("status") == "interrupted":
                print("\nBenchmark was interrupted by user")
                sys.exit(130)
            else:
                print(f"\nError: {result.get('error', 'Unknown error')}")
                sys.exit(1)
        else:
            sky_parser.print_help()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
