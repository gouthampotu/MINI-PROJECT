"""
chatbot.py
Retrieval-Augmented Generation chatbot over uploaded resumes + JD.
Uses LangChain FAISS vector store with OpenAI embeddings so answers
are grounded strictly in the uploaded documents.
"""

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


def build_vector_store(api_key: str, documents: list):
    """
    documents: list of dicts like {"text": "...", "source": "candidate_name.pdf", "type": "resume"/"jd"}
    Returns a FAISS vector store.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = []
    for d in documents:
        chunks = splitter.split_text(d["text"])
        for c in chunks:
            docs.append(Document(page_content=c, metadata={"source": d["source"], "type": d.get("type", "resume")}))

    if not docs:
        return None

    embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store


def query_chatbot(client, model: str, vector_store, question: str, k: int = 6) -> dict:
    """
    Retrieves top-k relevant chunks and asks the LLM to answer strictly
    based on that context. Returns {"answer": str, "sources": [str]}.
    """
    if vector_store is None:
        return {"answer": "Please upload at least one resume or job description first.", "sources": []}

    results = vector_store.similarity_search(question, k=k)
    context = "\n\n---\n\n".join(
        [f"[Source: {r.metadata.get('source')}]\n{r.page_content}" for r in results]
    )
    sources = sorted(set(r.metadata.get("source", "unknown") for r in results))

    system = (
        "You are an AI HR assistant. Answer the HR user's question using ONLY the context "
        "provided below, which comes from uploaded resumes and the job description. "
        "If the answer isn't in the context, say so clearly. Cite candidate names when relevant. "
        "Be concise, structured, and use bullet points when helpful."
    )
    user_msg = f"CONTEXT:\n{context}\n\nQUESTION: {question}"

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        answer = f"⚠️ Error generating answer: {e}"

    return {"answer": answer, "sources": sources}
