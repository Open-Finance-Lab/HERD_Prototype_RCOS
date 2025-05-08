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
        "Use hardware accelerators like GPUs or TPUs to speed up training and inference.",
        "Apply model pruning and quantization to shrink model size.",
        "Deploy using optimized runtimes like TensorRT or ONNX.",
        "Data loading pipelines should be parallelized to avoid bottlenecks.",
        "Consider using distillation to compress large models into faster ones."
    ]

    agent.build_index(expert_responses)

    user_query = "How do I make deep learning models run faster in production?"
    print("Generated Aggregated Answer:\n")
    print(agent.query(user_query))
