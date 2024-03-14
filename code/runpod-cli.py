#!/usr/bin/env python3
import argparse
import logging
import os
import subprocess

import libdocs.runpod.pod as pod
import runpod

runpod.api_key = os.getenv("RUNPOD_API_KEY")


def _pod_name(name: str) -> str:
    if name is not None and name != "":
        return name
    else:
        return f'acuvity.{os.getenv("USER", "user")}'


def _get_pod_id(name: str) -> str:
    name = _pod_name(name)
    for p in pod.list_pods():
        if p.get("name") == name:
            return p.get("id")
    return ""


def list(args):
    pods = pod.list_pods()
    for p in pods:
        print(
            f'{p.get("name")} ({p.get("id")}) - imageName: {p.get("imageName")} GPU: "{p.get("machine").get("gpuDisplayName")}"'
        )


def get(args):
    id = _get_pod_id(args.name)
    print(pod.get_pod(id))


def remove(args):
    id = _get_pod_id(args.name)
    pod.remove_pod(id)


def ssh(args):
    id = _get_pod_id(args.name)
    pod.ssh(id, args.i, args.commands)


def create(args):
    name = _pod_name(args.name)
    image_name = args.imageName
    if image_name is None or image_name == "":
        branch = subprocess.check_output(
            ["git", "branch", "--no-color", "--show-current"], encoding="utf-8"
        ).strip()
        image_name = f"ghcr.io/acuvity/docspipeline:{branch}"
    pod.create(name, args.gpuCount, args.gpuType, image_name)


def list_prices(args):
    prices = pod.list_prices(args.gpuCount)
    for gpu, price in prices.items():
        print_price = (
            f"{price} USD" if price is not None else "unknown/unavailable"
        )
        print(f'{args.gpuCount}x "{gpu}": {print_price}')


def main():
    parser = argparse.ArgumentParser(
        description="Command-line interface for runpod"
    )
    parser.add_argument(
        "--api-key", dest="apiKey", type=str, help="(optional) runpod API Key"
    )
    subparsers = parser.add_subparsers(title="commands", dest="command")
    subparsers.add_parser("list", help="List all pods")
    lpr = subparsers.add_parser(
        "list-prices", help="Lists all secure cloud prices and GPUs"
    )
    lpr.add_argument(
        "--gpu-count",
        dest="gpuCount",
        default=1,
        type=int,
        help="(optional) number of GPUs you want to request prices for",
    )
    gp = subparsers.add_parser("get", help="List all pods")
    gp.add_argument("--name", type=str, help="(optional) name of the pod")
    cp = subparsers.add_parser("create", help="List all pods")
    cp.add_argument("--name", type=str, help="(optional) name of the pod")
    cp.add_argument(
        "--gpu-count",
        type=int,
        dest="gpuCount",
        default=1,
        help="(optional) number of GPUs to request for the pod (default 1)",
    )
    cp.add_argument(
        "--gpu-type",
        dest="gpuType",
        default="NVIDIA GeForce RTX 4090",
        type=str,
        help="(optional) name of GPU to use (use list-prices command to see GPUs)",
    )
    cp.add_argument(
        "--image-name",
        default=None,
        type=str,
        dest="imageName",
        help="(optional) docker image name to use. This defaults to your current branch name if you don't specify this.",
    )
    rp = subparsers.add_parser("remove", help="List all pods")
    rp.add_argument("--name", type=str, help="(optional) name of the pod")
    sp = subparsers.add_parser("ssh", help="SSH into pod")
    sp.add_argument("--name", type=str, help="(optional) name of the pod")
    sp.add_argument(
        "-i", type=str, help="(optional) path to SSH keyfile to use"
    )
    sp.add_argument(
        "commands",
        nargs=argparse.REMAINDER,
        help="(optional) command(s) to run over the SSH session. The command will only run theses commands and exit afterwards if specified.",
    )

    # basics: add log level as flag, parse args, and set logging
    parser.add_argument(
        "--log-level",
        dest="logLevel",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the log level",
    )
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.logLevel))

    if args.apiKey is not None and args.apiKey != "":
        runpod.api_key = args.apiKey

    match args.command:
        case "list":
            list(args)
        case "get":
            get(args)
        case "create":
            create(args)
        case "remove":
            remove(args)
        case "ssh":
            ssh(args)
        case "list-prices":
            list_prices(args)


if __name__ == "__main__":
    main()
