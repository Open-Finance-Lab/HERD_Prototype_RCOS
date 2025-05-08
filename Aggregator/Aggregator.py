from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class AggregationAgent:
    def __init__(self, model_path, embedding_model="/gpfs/u/home/ARUS/ARUSgrsm/scratch/HFModels/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"):
        # Load local instruct model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")

        self.llm = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=False
        )

        # Set up embedding + vector store
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_db = None

    def build_index(self, expert_responses):
        docs = [Document(page_content=r) for r in expert_responses]
        self.vector_db = FAISS.from_documents(docs, self.embeddings)

    def retrieve_experts(self, query, k=5):
        if not self.vector_db:
            raise ValueError("Index not built. Call build_index() first.")
        return self.vector_db.similarity_search(query, k=k)

    def synthesize(self, user_query, retrieved_docs):
        context = "\n".join([f"{i+1}. {doc.page_content}" for i, doc in enumerate(retrieved_docs)])
        prompt = f"""You are an expert aggregator agent.

        User Question:
        {user_query}

        Relevant Expert Responses:
        {context}

        Based on the above expert insights, write a clear and complete answer to the user's question:
        """

        result = self.llm(prompt)[0]["generated_text"]
        return result.split(prompt)[-1].strip()  

    def query(self, user_query, k=5):
        retrieved = self.retrieve_experts(user_query, k=k)
        return self.synthesize(user_query, retrieved)

if __name__ == "__main__":
    agent = AggregationAgent(
        model_path="/gpfs/u/home/ARUS/ARUSgrsm/scratch/HFModels/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
        embedding_model="/gpfs/u/home/ARUS/ARUSgrsm/scratch/HFModels/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    )

    expert_responses = [
        """
        To simulate the car's motion, begin with Newton's second law: \( F = ma \). The forces acting on the car include:
        - The gravitational force component down the incline: \( F_g = mg\sin(\theta) \)
        - The frictional force opposing motion: \( F_f = \mu mg\cos(\theta) \)

        Therefore, the net force is:

        \[
        F_{\text{net}} = mg\sin(\theta) - \mu mg\cos(\theta)
        \]

        And the resulting acceleration is:

        \[
        a = g(\sin(\theta) - \mu \cos(\theta))
        \]

        This constant acceleration governs the car's changing speed over time.
        """, 

        """
        To compute the car's velocity and position, integrate the acceleration over time.

        Velocity as a function of time is:

        \[
        v(t) = \int a(t)\, dt + v_0
        \]

        Position as a function of time is:

        \[
        x(t) = \int v(t)\, dt + x_0
        \]

        For numerical simulation, we discretize these using Euler's method:

        \[
        v_{n+1} = v_n + a_n \Delta t
        \]
        \[
        x_{n+1} = x_n + v_n \Delta t
        \]

        where \( \Delta t \) is the time step. This allows iterative updates of velocity and position using finite steps.

        """
        ,
        """
        We can implement the simulation using a simple Python loop. The discretized equations are:

        \[
        v_{n+1} = v_n + a \cdot \Delta t
        \quad \text{and} \quad
        x_{n+1} = x_n + v_n \cdot \Delta t
        \]

        The Python code to perform this could look like:

        \begin{verbatim}
        import math

        g = 9.81         # gravitational acceleration (m/s^2)
        theta = math.radians(30)  # slope angle
        mu = 0.1         # friction coefficient
        dt = 0.1         # time step
        v = 0.0          # initial velocity
        x = 0.0          # initial position
        a = g * (math.sin(theta) - mu * math.cos(theta))

        for step in range(100):
            v += a * dt
            x += v * dt
            print(f"Time: {step*dt:.1f}s, Position: {x:.2f}m, Velocity: {v:.2f}m/s")
        \end{verbatim}

        This loop simulates the motion of the car in discrete time steps using the physical and calculus-based equations.
        """
    ]

    agent.build_index(expert_responses)

    user_query = "Design a program that simulates the motion of a car down a hill, taking into account gravitational acceleration, friction, and dynamically changing speed using calculus. Include what equations need to be used, how to discretize them for simulation, and how to implement the solution in code."
    print("Generated Aggregated Answer:\n")
    print(agent.query(user_query))
