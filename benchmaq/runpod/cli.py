import argparse
import json

from .core.client import deploy, delete, find, start, set_api_key
from .config import load_config


def main():
    parser = argparse.ArgumentParser(prog="runpod", description="RunPod pod management")
    subparsers = parser.add_subparsers(dest="command", required=True)

    deploy_parser = subparsers.add_parser("deploy", help="Deploy a pod")
    deploy_parser.add_argument("--config", "-c", required=True, help="Config YAML file")
    deploy_parser.add_argument("--api-key", "-k", help="RunPod API key")
    deploy_parser.add_argument("--no-wait", action="store_true", help="Don't wait for ready")

    delete_parser = subparsers.add_parser("delete", help="Delete a pod")
    delete_parser.add_argument("pod_id", help="Pod ID")
    delete_parser.add_argument("--api-key", "-k", help="RunPod API key")

    find_parser = subparsers.add_parser("find", help="Get pod info")
    find_parser.add_argument("pod_id", help="Pod ID")
    find_parser.add_argument("--api-key", "-k", help="RunPod API key")

    start_parser = subparsers.add_parser("start", help="Start a stopped pod")
    start_parser.add_argument("pod_id", help="Pod ID")
    start_parser.add_argument("--api-key", "-k", help="RunPod API key")

    args = parser.parse_args()

    if hasattr(args, "api_key") and args.api_key:
        set_api_key(args.api_key)

    if args.command == "deploy":
        config = load_config(args.config)

        if config.get("api_key"):
            set_api_key(config["api_key"])

        if args.no_wait:
            config["wait_for_ready"] = False

        instance = deploy(**config)
        print(json.dumps(instance, indent=2))

    elif args.command == "delete":
        result = delete(args.pod_id)
        print(json.dumps(result, indent=2))

    elif args.command == "find":
        result = find(args.pod_id)
        print(json.dumps(result, indent=2))

    elif args.command == "start":
        result = start(args.pod_id)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
