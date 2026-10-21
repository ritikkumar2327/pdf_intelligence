
import re
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

st.set_page_config(page_title="PDF Intelligence", page_icon="📚")
st.title("📚 PDF Intelligence")
st.subheader("AI Document Question-Answering Assistant")
st.write("Upload a PDF and ask questions about its content.")

@st.cache_resource
def load_qa_model():
    return pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2"
    )

def extract_chunks(pdf):
    reader = PdfReader(pdf)
    chunks = []

    for page_no, page in enumerate(reader.pages, start=1):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()

        if not text:
            continue

        size, overlap = 1000, 150
        for start in range(0, len(text), size - overlap):
            part = text[start:start + size].strip()
            if part:
                chunks.append({"page": page_no, "text": part})

    return len(reader.pages), chunks

def find_passages(question, chunks):
    texts = [c["text"] for c in chunks]

    # Do not remove common words: short questions may depend on them.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode"
    )

    matrix = vectorizer.fit_transform(texts)
    q_vector = vectorizer.transform([question])
    scores = cosine_similarity(q_vector, matrix).flatten()

    ranked = scores.argsort()[::-1]
    matches = [
        (chunks[int(i)], float(scores[i]))
        for i in ranked[:5]
        if scores[i] > 0
    ]

    # Handle questions explicitly asking for the first problem/question.
    q = question.lower()
    if any(phrase in q for phrase in [
        "first question", "first problem",
        "first task", "beginning of the document"
    ]):
        first_page = [c for c in chunks if c["page"] == 1]
        if first_page:
            first_chunk = first_page[0]
            if all(c["text"] != first_chunk["text"] for c, _ in matches):
                matches.insert(0, (first_chunk, 0.0))

    return matches[:5]

uploaded = st.file_uploader("Upload your PDF document", type=["pdf"])

if uploaded:
    try:
        page_count, chunks = extract_chunks(uploaded)
        st.success(f"PDF loaded: {page_count} pages")
        st.write(f"Text sections available: {len(chunks)}")

        if not chunks:
            st.error("No readable text found. This PDF may need OCR.")
        else:
            question = st.text_input(
                "Ask a question about your PDF",
                placeholder="Example: What is the first problem?"
            )

            if st.button("Find Answer", type="primary"):
                if not question.strip():
                    st.warning("Please enter a question.")
                else:
                    try:
                        with st.spinner("Searching the PDF and generating an answer..."):
                            matches = find_passages(question.strip(), chunks)

                            if not matches:
                                st.warning(
                                    "No matching passage was found. "
                                    "Try using words from the PDF."
                                )
                            else:
                                qa = load_qa_model()
                                candidates = []

                                # Ask the model about each retrieved passage.
                                for chunk, similarity in matches:
                                    result = qa(
                                        question=question.strip(),
                                        context=chunk["text"]
                                    )
                                    candidates.append({
                                        "answer": result["answer"].strip(),
                                        "score": float(result["score"]),
                                        "page": chunk["page"],
                                        "text": chunk["text"],
                                        "similarity": similarity
                                    })

                                best = max(candidates, key=lambda x: x["score"])

                        st.markdown("### Answer")
                        if best["answer"] and best["score"] >= 0.03:
                            st.write(best["answer"])
                            st.caption(f"Source: PDF page {best['page']}")
                        else:
                            st.warning(
                                "I couldn't identify a reliable answer. "
                                "Check the passages below or rephrase the question."
                            )

                        st.caption(
                            f"Model score: {best['score']:.3f}. "
                            "This score does not guarantee factual correctness."
                        )

                        with st.expander("View relevant passages"):
                            for i, (chunk, similarity) in enumerate(matches, 1):
                                st.markdown(
                                    f"**Passage {i} — Page {chunk['page']}**"
                                )
                                st.write(chunk["text"])
                                st.caption(
                                    f"Retrieval similarity: {similarity:.3f}"
                                )

                    except Exception as e:
                        st.error("Could not generate the answer.")
                        st.caption(f"Error type: {type(e).__name__}")

    except Exception as e:
        st.error("Could not read this PDF.")
        st.caption(f"Error type: {type(e).__name__}")
