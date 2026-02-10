import os
import chromadb
from chromadb.utils import embedding_functions


class SingleDocKB:
    def __init__(
        self, db_dir: str = "chroma_db", collection: str = "darwin_single_doc"
    ):
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name="text-embedding-3-small",
        )
        client = chromadb.PersistentClient(path=db_dir)
        self.col = client.get_or_create_collection(
            name=collection, embedding_function=openai_ef
        )

    def search(self, query: str, top_k: int = 4):
        res = self.col.query(query_texts=[query], n_results=top_k)

        hits = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]

        for i in range(len(docs)):
            hits.append(
                {
                    "text": docs[i],
                    "source_file": metas[i].get("source_file"),
                    "chunk_index": metas[i].get("chunk_index"),
                }
            )
        return hits
