# HERD's Brain
The brain of HERD is composed of isolated containers called *expert nodes*, each contributing domain specific knowledge to its reasoning capability. The nodes of HERD's brain are managed using Kubernetes and Docker, allowing for seamless addition and deletion of certain knowledge, similar to how the human brain prunes neurons. This README covers the composition of a singe expert node as well as how they are orchestrated and managed to create a single heterogenous brain. 

## Single-Node Composition
Each expert node is built from a base Docker image hosted on HERD’s Docker Hub. The image includes a pre-configured Python virtual environment, exposes an `/infer` API endpoint on port 8000, and gives the expert node RAG utility. Each expert node container is mounted with a domain-tuned LLM and vector database for RAG. The specific composition of each node is controlled by the `expert-chart/values.yaml` file. Figure One demonstrates how a specialized prompt is used in RAG for an expert node. 

<figure>
  <img src="../Figures/Expert_Node.png" alt="HERD Architecture" width="600"/>
  <figcaption style="text-align:center;"><b>Figure 1.</b> Per-Node RAG for Experts</figcaption>
</figure>

## Expert Node Orchestration

The initial composition of HERD's brain is created at runtime using Kubernetes and Helm by referencing the models defined in `expert-chart/values.yaml`. While the network is active the node composition can be altered using the following API endpoints: 

- **Add Expert**: The `/add_expert` route allows you to add an expert choosing the name, model, max tokens, temperature, and port. Below is an example payload for the endpoint.
```
{
  "name": "expert4",
  "model_id": "sshleifer/tiny-gpt2",
  "max_new_tokens": "50",
  "temperature": "0.7",
  "node_port": 30088
}
```
- **Remove Expert**: The `/delete_expert` route is a `DEL` method structured as `/delete_expert/{expert_name}`. 

## Starting the Brain

Starting the brain for HERD is very simple, all you need to do is run `helm upgrade --install experts ./experts-chart` and `uvicorn app:app --reload --port 8000`, given you have Kubernetes, Helm, and the required Python environment installed on your machine. In the brain directory, there are startup scripts for both Bash and Powershell that can be ran to start up the brain and expose the orchestrion routes.