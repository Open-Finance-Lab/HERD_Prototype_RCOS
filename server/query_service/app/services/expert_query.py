import socket, time, contextlib, signal, subprocess
from typing import Optional
from pydantic import BaseModel
from fastapi import HTTPException

def _run(cmd: list[str]) -> str:
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out.stdout.strip()

def _get_free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def _get_target_port(service_name: str, namespace: str) -> int:
    jsonpath = "{.spec.ports[0].targetPort}"
    port = _run(["kubectl", "-n", namespace, "get", "svc", service_name, "-o", f"jsonpath={jsonpath}"])
    if port.isdigit():
        return int(port)
    
    pod = _get_pod_name(service_name, namespace)
    cjson = "{.spec.containers[0].ports[?(@.name=='" + port + "')].containerPort}"
    cport = _run(["kubectl", "-n", namespace, "get", "pod", pod, "-o", f"jsonpath={cjson}"])
    if not cport or not cport.isdigit():
        raise HTTPException(status_code=500, detail=f"Could not resolve targetPort for service '{service_name}'")
    return int(cport)

def _get_pod_name(name: str, namespace: str) -> str:
    selectors = [
        f"app={name}",
        f"app.kubernetes.io/name={name}",
        f"app.kubernetes.io/instance={name}",
        f"release={name}",
    ]
    for sel in selectors:
        try:
            pod = _run([
                "kubectl", "-n", namespace, "get", "pods",
                "-l", sel, "-o", "jsonpath={.items[0].metadata.name}"
            ])
            if pod:
                return pod
        except subprocess.CalledProcessError:
            pass

    try:
        pods = _run(["kubectl", "-n", namespace, "get", "pods", "-o", "jsonpath={.items[*].metadata.name}"]).split()
        for p in pods:
            if p.startswith(name):
                return p
    except subprocess.CalledProcessError:
        pass

    raise HTTPException(status_code=404, detail=f"No pod found for '{name}' in namespace '{namespace}'")

def _wait_until_listening(port: int, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.settimeout(0.3)
            try:
                s.connect(("127.0.0.1", port))
                return
            except OSError:
                time.sleep(0.2)
    raise TimeoutError("Port-forward tunnel did not become ready in time")