from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.schema import Document

model_name = "facebook/bart-large-cnn"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

summarizer_pipeline = pipeline(
  "summarization",
  model=model,
  tokenizer=tokenizer,
  framework="pt"
)

hf_llm = HuggingFacePipeline(pipeline=summarizer_pipeline)

def generate_summary(expert_responses):
  docs = [Document(page_content=response) for response in expert_responses]
  chain = load_summarize_chain(hf_llm, chain_type="map_reduce")
  summary = chain.run(docs)
  return summary

def create_vector_index(summary_text):
  embeddings = HuggingFaceEmbeddings()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
  split_texts = text_splitter.split_text(summary_text)

  docs = [Document(page_content=text) for text in split_texts]
  vector_db = FAISS.from_documents(docs, embeddings)
  return vector_db

def query_summary(vector_db, user_query, k=3, score_threshold=0.3):
  retrieved_docs = vector_db.similarity_search(user_query, k=k)

  if not retrieved_docs:
      return "No relevant information found."

  response_list = []
  for i, doc in enumerate(retrieved_docs):
      response_list.append(f"{i+1}. {doc.page_content}")

  aggregated_response = "\n".join(response_list)

  return f"Here’s what I found based on your query:\n\n{aggregated_response}"


# Example usage:
if __name__ == "__main__":
  user_query = "How can I optimize deep learning models in production?"
  expert_responses = [
      "Optimizing data pipelines and leveraging hardware accelerators like GPUs or TPUs will further improve throughput.",
      "Consider using model quantization and pruning techniques to reduce model size and inference latency.",
      "Implementing efficient serving architectures such as TensorRT or ONNX Runtime can boost performance."
  ]

  summary_text = generate_summary(expert_responses)
  print("Generated Summary:\n", summary_text)

  vector_db = create_vector_index(summary_text)

  answer = query_summary(vector_db, user_query)
  print("\nAggregated Answer:\n", answer)
