
import re

import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


st.set_page_config(
    page_title="PDF Intelligence",
    page_icon="📚",
    layout="wide",
)

st.title("📚 PDF Intelligence")
st.subheader("AI Document Question-Answering Assistant")
st.write(
    "Upload a PDF and ask questions about its content. "
    "Answers are extracted from relevant document passages."
)


@st.cache_resource(show_spinner=False)
def load_qa_model():
    return pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",
    )


def extract_chunks(pdf_file, chunk_size=1200, overlap=200):
    reader = PdfReader(pdf_file)
    chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            continue

        step = chunk_size - overlap

        for start in range(0, len(text), step):
            chunk_text = text[start:start + chunk_size].strip()

            if chunk_text:
                chunks.append({
                    "page": page_number,
                    "text": chunk_text,
                })

    return len(reader.pages), chunks


def retrieve_passages(question, chunks, top_k=3):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
    )

    texts = [chunk["text"] for chunk in chunks]
    matrix = vectorizer.fit_transform(texts)
    question_vector = vectorizer.transform([question])

    scores = cosine_similarity(
        question_vector, matrix
    ).flatten()

    ranked_indices = scores.argsort()[::-1]
    ranked_indices = [
        int(i) for i in ranked_indices if scores[i] > 0
    ][:top_k]

    return [
        (chunks[i], float(scores[i]))
        for i in ranked_indices
    ]


uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"],
)

if uploaded_file is not None:
    try:
        with st.spinner("Reading your PDF..."):
            page_count, chunks = extract_chunks(uploaded_file)

        st.success(f"PDF loaded: {page_count} pages")

        if not chunks:
            st.warning(
                "No readable text was found. "
                "The PDF may be scanned and require OCR."
            )

        else:
            st.write(f"Text sections available: {len(chunks)}")

            question = st.text_input(
                "Ask a question about your PDF",
                placeholder=(
                    "Example: What is the total percentage "
                    "mentioned in the document?"
                ),
                key="pdf_question",
            )

            if st.button("Find Answer", type="primary"):
                if not question.strip():
                    st.warning("Please enter a question.")

                else:
                    try:
                        with st.spinner(
                            "Searching passages and finding an answer..."
                        ):
                            matches = retrieve_passages(
                                question.strip(), chunks, top_k=3
                            )

                            if not matches:
                                st.warning(
                                    "No matching text was found. "
                                    "Try using important words from "
                                    "the document or rephrasing "
                                    "your question."
                                )

                            else:
                                # Give the model multiple passages,
                                # not just the highest-ranked passage.
                                context_parts = []

                                for chunk, _score in matches:
                                    context_parts.append(
                                        f"[PDF page {chunk['page']}]\n"
                                        f"{chunk['text']}"
                                    )

                                context = "\n\n".join(context_parts)

                                qa_model = load_qa_model()

                                result = qa_model(
                                    question=question.strip(),
                                    context=context,
                                )

                        if matches:
                            answer = result.get("answer", "").strip()
                            confidence = float(
                                result.get("score", 0.0)
                            )

                            st.markdown("### Answer")

                            if answer and confidence >= 0.03:
                                st.write(answer)
                            else:
                                st.warning(
                                    "The model could not identify "
                                    "a reliable answer. Try a more "
                                    "specific question or inspect "
                                    "the passages below."
                                )
                                if answer:
                                    st.write("Possible answer: " + answer)

                            source_pages = sorted({
                                chunk["page"]
                                for chunk, _score in matches
                            })

                            st.caption(
                                "Relevant source pages: "
                                + ", ".join(map(str, source_pages))
                            )

                            st.caption(
                                f"Model confidence score: {confidence:.3f}. "
                                "This is not a guaranteed measure "
                                "of factual correctness."
                            )

                            with st.expander(
                                "View relevant passages"
                            ):
                                for rank, (chunk, score) in enumerate(
                                    matches, start=1
                                ):
                                    st.markdown(
                                        f"**Passage {rank} — "
                                        f"PDF page {chunk['page']}**"
                                    )
                                    st.write(chunk["text"])
                                    st.caption(
                                        f"Retrieval similarity: {score:.3f}"
                                    )

                    except ValueError:
                        st.error(
                            "The document text could not be processed. "
                            "Try a PDF with readable text and a "
                            "more specific question."
                        )

                    except Exception as error:
                        st.error(
                            "Could not generate an answer. "
                            "Check the deployment logs for details."
                        )
                        st.caption(
                            f"Error type: {type(error).__name__}"
                        )

    except Exception as error:
        st.error("Could not read this PDF file.")
        st.caption(f"Error type: {type(error).__name__}")
