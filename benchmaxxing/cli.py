#!/usr/bin/env python3
import sys
import argparse

from .runner import run


def main():
    parser = argparse.ArgumentParser(prog="benchmaxxing", description="LLM benchmarking toolkit")
    subparsers = parser.add_subparsers(dest="command")

    # benchmark command
    bench_parser = subparsers.add_parser("bench", help="Run benchmark")
    bench_parser.add_argument("config", help="Config YAML file")

    # runpod command
    runpod_parser = subparsers.add_parser("runpod", help="RunPod pod management")
    runpod_subparsers = runpod_parser.add_subparsers(dest="runpod_command")

    deploy_parser = runpod_subparsers.add_parser("deploy", help="Deploy a pod")
    deploy_parser.add_argument("config", help="Config YAML file")
    deploy_parser.add_argument("--no-wait", action="store_true", help="Don't wait for ready")

    delete_parser = runpod_subparsers.add_parser("delete", help="Delete a pod")
    delete_parser.add_argument("target", help="Pod ID or config YAML path")

    find_parser = runpod_subparsers.add_parser("find", help="Get pod info")
    find_parser.add_argument("target", help="Pod ID or config YAML path")

    start_parser = runpod_subparsers.add_parser("start", help="Start a stopped pod")
    start_parser.add_argument("target", help="Pod ID or config YAML path")

    args = parser.parse_args()

    if args.command == "bench":
        run(args.config)

    elif args.command == "runpod":
        import json
        from .runpod.core.client import deploy, delete, find, find_by_name, start, set_api_key
        from .runpod.config import load_config

        def load_api_key_from_config(config_path):
            config = load_config(config_path)
            if config.get("api_key"):
                set_api_key(config["api_key"])
            return config

        if args.runpod_command == "deploy":
            config = load_api_key_from_config(args.config)
            if args.no_wait:
                config["wait_for_ready"] = False
            instance = deploy(**config)
            print(json.dumps(instance, indent=2))

        elif args.runpod_command == "delete":
            target = args.target
            if target.endswith(".yaml") or target.endswith(".yml"):
                config = load_api_key_from_config(target)
                result = delete(name=config.get("name"))
            else:
                result = delete(pod_id=target)
            print(json.dumps(result, indent=2))

        elif args.runpod_command == "find":
            target = args.target
            if target.endswith(".yaml") or target.endswith(".yml"):
                config = load_api_key_from_config(target)
                pod = find_by_name(config.get("name"))
                result = pod if pod else {"error": f"Pod '{config.get('name')}' not found"}
            else:
                result = find(target)
            print(json.dumps(result, indent=2))

        elif args.runpod_command == "start":
            target = args.target
            if target.endswith(".yaml") or target.endswith(".yml"):
                config = load_api_key_from_config(target)
                pod = find_by_name(config.get("name"))
                if pod:
                    result = start(pod["id"])
                else:
                    result = {"error": f"Pod '{config.get('name')}' not found"}
            else:
                result = start(target)
            print(json.dumps(result, indent=2))

        else:
            runpod_parser.print_help()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
