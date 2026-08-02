


"""
text_splitter.py
-----------------
split long texts into smaller chunks to improve embedding and retrieval accuracy, as shorter texts will have more focused embeddings.
in this answer in car_qa.txt, most answers are short enough to be considered as a single chunk.

NOTE: the embedding text is built from "category + question" ONLY (not the answer).
The dataset has ~5 questions per dessert (ingredients / meaning / process / appearance /
storage) whose answers share a lot of overlapping vocabulary (same dessert name,
same ingredients mentioned across aspects). Embedding the full "question + answer"
text let that shared vocabulary dominate the vector, causing retrieval to match the
right dessert but the wrong aspect. Embedding "category + question" keeps the vector
focused on what the user is actually asking; the answer is still returned to the
user via the "answer" field, it's just not part of what gets embedded.
"""

def split_text(text, chunk_size, overlap):
    """
    Split a long text into smaller chunks of size chunk_size characters,
    with an overlap between consecutive chunks to avoid losing context.
    If the text is shorter than chunk_size, return it as a single chunk.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap  # ถอยกลับมาเล็กน้อยให้ชิ้นถัดไปเหลื่อมกัน

    return chunks


def build_chunks(qa_records, chunk_size, overlap):
    """
    Receive a list of question-answer pairs (from document_loader.load_qa_file)
    and create a list of "chunks" with metadata for storing in chunks.json

    Each chunk's "text" (the part that gets embedded) combines "category + question"
    only. The answer is kept as separate metadata and returned to the user at
    retrieval time, but is deliberately excluded from the embedded text so it
    doesn't dilute the match with overlapping vocabulary from other aspects of
    the same dessert.
    """
    all_chunks = []

    for record in qa_records:
        full_text = f"หมวด: {record['category']} คำถาม: {record['question']}"
        text_pieces = split_text(full_text, chunk_size, overlap)

        for part_idx, piece in enumerate(text_pieces):
            all_chunks.append({
                "chunk_id": len(all_chunks),
                "qa_id": record["id"],
                "category": record["category"],
                "question": record["question"],
                "answer": record["answer"],
                "text": piece,
                "part_idx": part_idx,
                "line_no": record["line_no"],
            })

    return all_chunks






