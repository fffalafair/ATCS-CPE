


"""
include both steps "convert query to vector" + "search in FAISS" into a single class
to make it easy for main.py and lab07 to use with a single function call.

Retrieval is hybrid: results are ranked using vector similarity (from FAISS)
combined with lexical (character-level) similarity to the query. This dataset
is FAQ-style -- inside the same category, every question shares an identical
sentence template and differs mainly by a name (e.g. a dessert or car model),
so the meaningful signal is a small part of the whole sentence. A small
embedding model can under-weight that name relative to the shared template
words and match the right category but the wrong item. Lexical similarity
catches close wording directly and doesn't have that weakness, so blending
the two makes near-exact-wording queries much more reliable while still
keeping the embedding's ability to handle paraphrased/informal queries.
"""

import difflib

from src.embedding_model import EmbeddingModel
from src.vector_store import VectorStore, load_chunk_store


def lexical_similarity(text_a, text_b):
    """
    Character-level similarity between two strings, 0.0-1.0.
    Plain-Python (difflib), no embedding model needed.
    """
    return difflib.SequenceMatcher(None, text_a, text_b).ratio()


class Retriever:
    def __init__(self, model_name, index_path, chunk_store_path,
                 candidate_pool_size=15, vector_weight=0.5, lexical_weight=0.5):
        self.embedding_model = EmbeddingModel(model_name)

        self.vector_store = VectorStore()
        self.vector_store.load(index_path)

        self.chunks = load_chunk_store(chunk_store_path)

        self.candidate_pool_size = candidate_pool_size
        self.vector_weight = vector_weight
        self.lexical_weight = lexical_weight

    def retrieve(self, query, top_k=3):
        """
        Receive a user query and return the top_k most relevant chunks.

        Step 1: pull a wider candidate pool (candidate_pool_size) from FAISS
                 by vector similarity alone.
        Step 2: re-rank that pool using combined_score =
                 vector_weight * vector_score + lexical_weight * lexical_score
                 where lexical_score compares the raw query text against each
                 candidate's original "question" field.
        Step 3: return the top_k by combined_score.

        Each result dict contains the original chunk plus:
            - score: the combined score used for ranking (what main.py displays)
            - vector_score: the raw FAISS cosine similarity
            - lexical_score: the character-level similarity to the query
        """
        candidate_k = max(top_k, self.candidate_pool_size)

        query_vector = self.embedding_model.encode_query(query)
        scores, indices = self.vector_store.search(query_vector, candidate_k)

        candidates = []
        for score, idx in zip(scores, indices):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])

            vector_score = float(score)
            lex_score = lexical_similarity(query, chunk["question"])
            combined_score = (
                self.vector_weight * vector_score
                + self.lexical_weight * lex_score
            )

            chunk["vector_score"] = vector_score
            chunk["lexical_score"] = lex_score
            chunk["score"] = combined_score
            candidates.append(chunk)

        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates[:top_k]
