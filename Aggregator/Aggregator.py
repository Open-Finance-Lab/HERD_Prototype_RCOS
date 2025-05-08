from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains.summarize import load_summarize_chain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class SummaryQAEngine:
    def __init__(self, model_name="/gpfs/u/home/ARUS/ARUSgrsm/scratch/HFModels/models--facebook--bart-large-cnn/snapshots/37f520fa929c961707657b28798b30c003dd100b"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        summarizer_pipeline = pipeline(
            "summarization",
            model=self.model,
            tokenizer=self.tokenizer,
            framework="pt"
        )

        self.llm = HuggingFacePipeline(pipeline=summarizer_pipeline)
        self.vector_db = None

    def generate_summary(self, expert_responses):
        docs = [Document(page_content=response) for response in expert_responses]
        chain = load_summarize_chain(self.llm, chain_type="map_reduce")
        summary = chain.run(docs)
        return summary

    def create_vector_index(self, summary_text):
        embeddings = HuggingFaceEmbeddings()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_texts = text_splitter.split_text(summary_text)

        docs = [Document(page_content=text) for text in split_texts]
        self.vector_db = FAISS.from_documents(docs, embeddings)

    def query_summary(self, user_query, k=3):
        if self.vector_db is None:
            raise ValueError("Vector database is not initialized. Call create_vector_index first.")

        retrieved_docs = self.vector_db.similarity_search(user_query, k=k)

        if not retrieved_docs:
            return "No relevant information found."

        response_list = [f"{i+1}. {doc.page_content}" for i, doc in enumerate(retrieved_docs)]
        return "Here’s what I found based on your query:\n\n" + "\n".join(response_list)


if __name__ == "__main__":
    engine = SummaryQAEngine()

    expert_responses = [
        "Optimizing data pipelines and leveraging hardware accelerators like GPUs or TPUs will further improve throughput.",
        "Consider using model quantization and pruning techniques to reduce model size and inference latency.",
        "Implementing efficient serving architectures such as TensorRT or ONNX Runtime can boost performance."
    ]

    summary = engine.generate_summary(expert_responses)
    print("Generated Summary:\n", summary)

    engine.create_vector_index(summary)

    user_query = "How can I optimize deep learning models in production?"
    response = engine.query_summary(user_query)
    print("\nAggregated Answer:\n", response)