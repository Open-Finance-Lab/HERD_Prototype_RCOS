# Heterogenous Experts with Routing Decisions (HERD) Prototype - An RCOS Project
HERD is an approach to multi model reasoning that utilizes multiple independent models fine tuned to be experts on different subjects. The HERD interface will take a prompt like a traditional chat-based LLM and decompose it into its core topics, assign different experts to components of the prompt, and aggregate the expert answers into one coherent output. This repository is a prototype implementation of HERD carried out in collaboration with Rensselaer Center for Open Source (RCOS). 

## Prototype Architecture
The architecture of HERD will have tree main components: Router, Experts, and Aggregator. 
### Router
The router will handel inputs prompts for HERD. The job of the router will be to take in a prompt as an argument, decompose the 
prompt into its core topics, and create specialized prompts for relevant experts based on the input. Figure One visualizes how 
the router will operate. 

<div style="text-align: center;">

**Figure One: Router Diagram**
![Router Diagram](Figures\Router-Diagram.jpg "Figure One")

</div>

The router itself will be comprised of two major algorithms
* <b>Interpreter:</b> The interpreter will be a natural language processing reinforcement learning algorithm trained to decompose
prompts into their core topics. This is a relatively well developed area of machine learning research, so many techniques and approaches will be borrowed from well established research. From figure one, the interpreter will be take on the role of the first and second step.    
* <b>Prompter:</b> The prompter will take the information supplied by the interpreter to and use it as context to create prompts for the experts. The prompter will heavily utilize prompt engineering libraries like Langchain to create optimized prompts for each expert. From figure one, the prompter will take on the role of the third step.

The router will be written entirely in python, utilizing relevant ML and AI libraries like Langchain, PyTorch, and HuggingFace Transformers. 