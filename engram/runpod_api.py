"""lightweight runpod graphql api client. no flash, no bloat, just raw graphql."""

from __future__ import annotations

import os
import time
import json
import requests
from dataclasses import dataclass
from typing import Any

GRAPHQL_URL = "https://api.runpod.io/graphql"


class RunPodError(Exception):
    """something went wrong with the runpod api."""
    pass


class RunPodClient:
    """thin wrapper around the runpod graphql api.

    usage:
        client = RunPodClient()  # reads RUNPOD_API_KEY from env
        client = RunPodClient(api_key="rpa_xxx")
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY", "")
        if not self.api_key:
            raise RunPodError("no RUNPOD_API_KEY found. set it in env or pass api_key=")
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        })

    def _gql(self, query: str, variables: dict | None = None) -> dict:
        """fire a graphql query/mutation and return the data dict."""
        payload: dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables
        resp = self.session.post(GRAPHQL_URL, json=payload, timeout=60)
        resp.raise_for_status()
        body = resp.json()
        if "errors" in body:
            raise RunPodError(f"graphql errors: {json.dumps(body['errors'], indent=2)}")
        return body.get("data", {})

    # ── gpu types ──────────────────────────────────────────────────────────

    def get_gpu_types(self) -> list[dict]:
        """list all available gpu types."""
        data = self._gql("""
            query {
                gpuTypes {
                    id
                    displayName
                    memoryInGb
                    secureCloud
                    communityCloud
                }
            }
        """)
        return data.get("gpuTypes", [])

    # ── network volumes ────────────────────────────────────────────────────

    def get_network_volumes(self) -> list[dict]:
        """list all network volumes on the account."""
        data = self._gql("""
            query {
                myself {
                    networkVolumes {
                        id
                        name
                        size
                        dataCenterId
                    }
                }
            }
        """)
        return data.get("myself", {}).get("networkVolumes", [])

    def create_network_volume(
        self,
        name: str,
        size_gb: int = 100,
        datacenter_id: str = "US-TX-3",
    ) -> dict:
        """create a persistent network volume. returns the volume dict."""
        data = self._gql("""
            mutation createNetworkVolume($input: CreateNetworkVolumeInput!) {
                createNetworkVolume(input: $input) {
                    id
                    name
                    size
                    dataCenterId
                }
            }
        """, variables={
            "input": {
                "name": name,
                "size": size_gb,
                "dataCenterId": datacenter_id,
            }
        })
        return data.get("createNetworkVolume", {})

    def get_or_create_network_volume(
        self,
        name: str,
        size_gb: int = 100,
        datacenter_id: str = "US-TX-3",
    ) -> dict:
        """find existing volume by name or create a new one."""
        volumes = self.get_network_volumes()
        for vol in volumes:
            if vol["name"] == name:
                return vol
        return self.create_network_volume(name, size_gb, datacenter_id)

    # ── templates ──────────────────────────────────────────────────────────

    def create_template(
        self,
        name: str,
        image_name: str,
        env: dict[str, str] | None = None,
        container_disk_gb: int = 200,
        volume_gb: int = 0,
        is_serverless: bool = True,
        docker_args: str = "",
    ) -> dict:
        """create a pod/serverless template."""
        env_list = [{"key": k, "value": v} for k, v in (env or {}).items()]
        data = self._gql("""
            mutation saveTemplate($input: SaveTemplateInput!) {
                saveTemplate(input: $input) {
                    id
                    name
                    imageName
                }
            }
        """, variables={
            "input": {
                "name": name,
                "imageName": image_name,
                "env": env_list,
                "containerDiskInGb": container_disk_gb,
                "volumeInGb": volume_gb,
                "isServerless": is_serverless,
                "dockerArgs": docker_args,
            }
        })
        return data.get("saveTemplate", {})

    # ── serverless endpoints ───────────────────────────────────────────────

    def create_endpoint(
        self,
        name: str,
        template_id: str,
        gpu_ids: str = "NVIDIA A100 80GB PCIe",
        workers_min: int = 0,
        workers_max: int = 3,
        idle_timeout: int = 300,
        scaler_type: str = "QUEUE_DELAY",
        scaler_value: int = 1,
        network_volume_id: str | None = None,
    ) -> dict:
        """create a serverless endpoint. returns endpoint dict with id."""
        input_data: dict[str, Any] = {
            "name": name,
            "templateId": template_id,
            "gpuIds": gpu_ids,
            "workersMin": workers_min,
            "workersMax": workers_max,
            "idleTimeout": idle_timeout,
            "scalerType": scaler_type,
            "scalerValue": scaler_value,
        }
        if network_volume_id:
            input_data["networkVolumeId"] = network_volume_id
        data = self._gql("""
            mutation saveEndpoint($input: EndpointInput!) {
                saveEndpoint(input: $input) {
                    id
                    name
                    templateId
                    gpuIds
                    workersMin
                    workersMax
                }
            }
        """, variables={"input": input_data})
        return data.get("saveEndpoint", {})

    def get_endpoint(self, endpoint_id: str) -> dict:
        """get endpoint status."""
        data = self._gql("""
            query getEndpoint($endpointId: String!) {
                endpoint(id: $endpointId) {
                    id
                    name
                    templateId
                    gpuIds
                    workersMin
                    workersMax
                }
            }
        """, variables={"endpointId": endpoint_id})
        return data.get("endpoint", {})

    def delete_endpoint(self, endpoint_id: str) -> bool:
        """delete a serverless endpoint."""
        try:
            self._gql("""
                mutation deleteEndpoint($endpointId: String!) {
                    deleteEndpoint(id: $endpointId)
                }
            """, variables={"endpointId": endpoint_id})
            return True
        except RunPodError:
            return False

    def endpoint_health(self, endpoint_id: str) -> dict:
        """check endpoint health via rest api."""
        url = f"https://api.runpod.ai/v2/{endpoint_id}/health"
        resp = self.session.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        return {"status": "unavailable", "code": resp.status_code}

    def wait_for_endpoint(
        self,
        endpoint_id: str,
        timeout: int = 600,
        poll_interval: int = 10,
    ) -> bool:
        """wait until at least one worker is ready. returns true if ready."""
        print(f"  waiting for endpoint {endpoint_id} to be ready...")
        start = time.time()
        while time.time() - start < timeout:
            try:
                health = self.endpoint_health(endpoint_id)
                workers = health.get("workers", {})
                ready = workers.get("ready", 0)
                if ready > 0:
                    print(f"  endpoint ready! {ready} worker(s) active")
                    return True
                idle = workers.get("idle", 0)
                running = workers.get("running", 0)
                throttled = workers.get("throttled", 0)
                initializing = workers.get("initializing", 0)
                print(f"  workers: ready={ready} idle={idle} running={running} "
                      f"init={initializing} throttled={throttled}")
            except Exception as e:
                print(f"  health check failed: {e}")
            time.sleep(poll_interval)
        print(f"  timeout after {timeout}s waiting for endpoint")
        return False

    @staticmethod
    def endpoint_openai_url(endpoint_id: str) -> str:
        """get the base url for a serverless endpoint (used with /runsync handler)."""
        return f"https://api.runpod.ai/v2/{endpoint_id}"

    # ── pods (single instance containers) ──────────────────────────────────

    def create_pod(
        self,
        name: str,
        image_name: str,
        gpu_type_id: str = "NVIDIA A100 80GB PCIe",
        gpu_count: int = 1,
        cloud_type: str = "ALL",
        volume_gb: int = 100,
        container_disk_gb: int = 50,
        min_vcpu: int = 8,
        min_memory_gb: int = 32,
        network_volume_id: str | None = None,
        docker_args: str = "",
        ports: str = "22/tcp",
        env: dict[str, str] | None = None,
        volume_mount_path: str = "/runpod-volume",
    ) -> dict:
        """deploy a pod on demand. returns pod dict."""
        env_list = [{"key": k, "value": v} for k, v in (env or {}).items()]
        input_data: dict[str, Any] = {
            "name": name,
            "imageName": image_name,
            "gpuTypeId": gpu_type_id,
            "gpuCount": gpu_count,
            "cloudType": cloud_type,
            "volumeInGb": volume_gb,
            "containerDiskInGb": container_disk_gb,
            "minVcpuCount": min_vcpu,
            "minMemoryInGb": min_memory_gb,
            "dockerArgs": docker_args,
            "ports": ports,
            "env": env_list,
            "volumeMountPath": volume_mount_path,
        }
        if network_volume_id:
            input_data["networkVolumeId"] = network_volume_id
        data = self._gql("""
            mutation podFindAndDeployOnDemand($input: PodFindAndDeployOnDemandInput!) {
                podFindAndDeployOnDemand(input: $input) {
                    id
                    name
                    desiredStatus
                    imageName
                    machine {
                        podHostId
                    }
                    runtime {
                        uptimeInSeconds
                        ports {
                            ip
                            isIpPublic
                            privatePort
                            publicPort
                            type
                        }
                    }
                }
            }
        """, variables={"input": input_data})
        return data.get("podFindAndDeployOnDemand", {})

    def get_pod(self, pod_id: str) -> dict:
        """get pod status and runtime info."""
        data = self._gql("""
            query pod($podId: String!) {
                pod(input: { podId: $podId }) {
                    id
                    name
                    desiredStatus
                    imageName
                    runtime {
                        uptimeInSeconds
                        ports {
                            ip
                            isIpPublic
                            privatePort
                            publicPort
                            type
                        }
                    }
                }
            }
        """, variables={"podId": pod_id})
        return data.get("pod", {})

    def stop_pod(self, pod_id: str) -> dict:
        """stop a running pod (preserves volume data)."""
        data = self._gql("""
            mutation podStop($podId: String!) {
                podStop(input: { podId: $podId }) {
                    id
                    desiredStatus
                }
            }
        """, variables={"podId": pod_id})
        return data.get("podStop", {})

    def terminate_pod(self, pod_id: str) -> bool:
        """terminate a pod permanently."""
        try:
            self._gql("""
                mutation podTerminate($podId: String!) {
                    podTerminate(input: { podId: $podId })
                }
            """, variables={"podId": pod_id})
            return True
        except RunPodError:
            return False

    def wait_for_pod(
        self,
        pod_id: str,
        timeout: int = 600,
        poll_interval: int = 10,
    ) -> dict | None:
        """wait until pod is running and has an ip. returns pod dict."""
        print(f"  waiting for pod {pod_id} to be ready...")
        start = time.time()
        while time.time() - start < timeout:
            pod = self.get_pod(pod_id)
            runtime = pod.get("runtime")
            if runtime and runtime.get("ports"):
                ports = runtime["ports"]
                ssh_port = next(
                    (p for p in ports if p.get("privatePort") == 22 and p.get("isIpPublic")),
                    None,
                )
                if ssh_port:
                    print(f"  pod ready! ssh: {ssh_port['ip']}:{ssh_port['publicPort']}")
                    return pod
            status = pod.get("desiredStatus", "unknown")
            print(f"  pod status: {status}, waiting...")
            time.sleep(poll_interval)
        print(f"  timeout after {timeout}s waiting for pod")
        return None

    # ── helpers ─────────────────────────────────────────────────────────────

    def get_myself(self) -> dict:
        """get account info."""
        data = self._gql("""
            query {
                myself {
                    id
                    email
                    currentSpendPerHr
                    machineQuota
                    referralEarned
                    signedTermsOfService
                    spendLimit
                }
            }
        """)
        return data.get("myself", {})

