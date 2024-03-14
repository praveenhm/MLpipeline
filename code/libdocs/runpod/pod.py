import logging
import subprocess
from typing import Any, Dict, Iterable, List, Union

from runpod.api.graphql import run_graphql_query

QUERY_LIST_PODS = """
query Pods {
  myself {
    pods {
      id
      name
      imageName
      machine {
        gpuDisplayName
      }
    }
  }
}
"""


def query_list_prices(gpu_count: int = 1) -> str:
    return f"""
query LowestPrice {{
  gpuTypes {{
    lowestPrice(input: {{ gpuCount: {gpu_count}, secureCloud: true }}) {{
      gpuName
      gpuTypeId
      minimumBidPrice
      uninterruptablePrice
      minMemory
      minVcpu
    }}
  }}
}}
"""


def query_get_pod(id: str) -> str:
    return f"""
query Pod {{
  pod(input: {{podId: "{id}"}}) {{
    id
    name
    imageName
    gpuCount
    vcpuCount
    memoryInGb
    containerDiskInGb
    volumeInGb
    desiredStatus
    costPerHr
    machine {{
      podHostId
      gpuDisplayName
      location
      secureCloud
    }}
    runtime {{
      uptimeInSeconds
      ports {{
        ip
        isIpPublic
        privatePort
        publicPort
        type
      }}
      gpus {{
        id
        gpuUtilPercent
        memoryUtilPercent
      }}
      container {{
        cpuPercent
        memoryPercent
      }}
    }}
  }}
}}
"""


def query_terminate_pod(id: str) -> str:
    return f"""
mutation {{
    podTerminate(input: {{podId: "{id}"}})
}}
"""


def query_create_pod(
    name: str,
    template_id: str = "bdi1ajq2pj",
    image_name: str = "ghcr.io/acuvity/docspipeline:main",
    gpu_count: int = 1,
    gpu_type_id: str = "NVIDIA GeForce RTX 4090",
) -> str:
    return f"""
mutation {{
  podFindAndDeployOnDemand(
    input: {{
      name: "{name}"
      templateId: "{template_id}"
      imageName: "{image_name}"
      cloudType: SECURE
      supportPublicIp: false
      gpuCount: 1
      volumeInGb: 80
      containerDiskInGb: 80
      minVcpuCount: 2
      minMemoryInGb: 15
      gpuCount: {gpu_count}
      gpuTypeId: "{gpu_type_id}"
    }}
  ) {{
    id
    name
    machine {{
      podHostId
    }}
  }}
}}
"""


def get_pod(id: str) -> Dict[str, Any]:
    raw_return = run_graphql_query(query_get_pod(id))
    cleaned_return = raw_return["data"]["pod"]
    return cleaned_return


def list_pods() -> List[Dict[str, Any]]:
    raw_return = run_graphql_query(QUERY_LIST_PODS)
    cleaned_return = raw_return["data"]["myself"]["pods"]
    return cleaned_return


def remove_pod(id: str):
    run_graphql_query(query_terminate_pod(id))


def ssh(id: str, keyfile: str = None, commands: Iterable[str] = None):
    p = get_pod(id)
    cmd = ["ssh", "-tt", "-o", "StrictHostKeyChecking=no"]
    if keyfile is not None:
        cmd.append("-i")
        cmd.append(keyfile)
    cmd.append(f"{p.get('machine').get('podHostId')}@ssh.runpod.io")
    if commands is not None and len(commands) > 0:
        input: str = " ".join(commands)
        input += " ; exit\n"
        logging.info(f"Running: {' '.join(cmd)}, input: {input}")
        subprocess.run(cmd, check=True, input=input.encode(("utf-8")))
    else:
        logging.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


def create(
    name: str,
    gpu_count: int = 1,
    gpu_type: str = "NVIDIA GeForce RTX 4090",
    image_name: str = "ghcr.io/acuvity/docspipeline:main",
) -> Dict[str, Any]:
    raw_return = run_graphql_query(
        query_create_pod(
            name,
            gpu_count=gpu_count,
            gpu_type_id=gpu_type,
            image_name=image_name,
        )
    )
    cleaned_return = raw_return["data"]["podFindAndDeployOnDemand"]
    return cleaned_return


def list_prices(gpu_count: int = 1) -> Dict[str, Union[float, None]]:
    raw_return = run_graphql_query(query_list_prices(gpu_count))
    logging.debug(raw_return)
    ret: Dict[str, Union[float, None]] = {}
    for entry in raw_return["data"]["gpuTypes"]:
        e = entry["lowestPrice"]
        ret[e["gpuTypeId"]] = e["uninterruptablePrice"]
    return ret
