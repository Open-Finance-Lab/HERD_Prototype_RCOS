from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import tempfile
import yaml
import os

app = FastAPI()

class ExpertConfig(BaseModel):
    name: str
    model_id: str
    max_new_tokens: str
    temperature: str
    node_port: int

@app.post("/add_expert")
def add_expert(config: ExpertConfig):
    # Create temp YAML values for the new expert
    values = {
        "experts": [{
            "name": config.name,
            "env": {
                "MODEL_ID": config.model_id,
                "MAX_NEW_TOKENS": config.max_new_tokens,
                "TEMPERATURE": config.temperature,
            },
            "replicas": 1,
            "service": {
                "type": "NodePort",
                "nodePort": config.node_port,
            }
        }],
        "image": {
            "repository": "expert-template",
            "tag": "latest",
            "pullPolicy": "IfNotPresent"
        }
    }

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as tmpfile:
        yaml.dump(values, tmpfile)
        tmpfile_path = tmpfile.name

    try:
        subprocess.run([
            "helm", "upgrade", "--install", config.name,
            "./experts-chart",
            "-f", tmpfile_path
        ], check=True)
    except subprocess.CalledProcessError:
        raise HTTPException(status_code=500, detail="Failed to deploy expert")

    os.unlink(tmpfile_path)
    return {"status": "success", "expert": config.name}


@app.delete("/remove_expert/{name}")
def remove_expert(name: str):
    try:
        subprocess.run(["helm", "uninstall", name], check=True)
    except subprocess.CalledProcessError:
        raise HTTPException(status_code=500, detail="Failed to remove expert")

    return {"status": "removed", "expert": name}
