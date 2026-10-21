
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import re

st.set_page_config(
    page_title="PDF Intelligence",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF Intelligence")
st.subheader("AI Document Question-Answering Assistant")
st.write("Upload a PDF and ask questions about its content.")

# Load the local AI model only when needed
@st.cache_resource
def load_qa_model():
    return pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2"
    )

uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"]
)

if uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    chunks = []

    # Extract text and divide it into smaller chunks
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()

        if text:
            chunk_size = 1000
            overlap = 150

            for start in range(0, len(text), chunk_size - overlap):
                chunk = text[start:start + chunk_size]

                if chunk.strip():
                    chunks.append({
                        "page": page_number,
                        "text": chunk
                    })

    st.success(f"PDF loaded: {len(reader.pages)} pages")

    if not chunks:
        st.warning(
            "No selectable text found. "
            "This may be a scanned PDF."
        )
    else:
        st.write(f"Text sections available: {len(chunks)}")

        question = st.text_input(
            "Ask a question about your PDF"
        )

        if st.button("Find Answer", type="primary"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Searching your document..."):
                    vectorizer = TfidfVectorizer(
                        stop_words="english"
                    )

                    try:
                        matrix = vectorizer.fit_transform(
                            [chunk["text"] for chunk in chunks]
                        )

                        question_vector = vectorizer.transform(
                            [question]
                        )

                        scores = cosine_similarity(
                            question_vector, matrix
                        ).flatten()

                        top_indices = scores.argsort()[-3:][::-1]
                        best_index = top_indices[0]

                        if scores[best_index] <= 0:
                            st.warning(
                                "I couldn't find a relevant passage "
                                "for that question."
                            )
                        else:
                            context = chunks[best_index]["text"]
                            qa = load_qa_model()

                            result = qa(
                                question=question,
                                context=context
                            )

                            if result["score"] < 0.10:
                                st.warning(
                                    "The answer may not be clearly "
                                    "available in this document."
                                )

                            st.markdown("### Answer")
                            st.write(result["answer"])
                            st.caption(
                                f"Source: PDF page "
                                f"{chunks[best_index]['page']}"
                            )

                            with st.expander(
                                "View relevant passages"
                            ):
                                for rank, index in enumerate(
                                    top_indices, start=1
                                ):
                                    st.markdown(
                                        f"**Passage {rank} — "
                                        f"Page {chunks[index]['page']}**"
                                    )
                                    st.write(chunks[index]["text"])
                                    st.write(
                                        f"Relevance score: "
                                        f"{scores[index]:.3f}"
                                    )

                    except ValueError:
                        st.error(
                            "Could not process the document text. "
                            "Try a PDF containing readable text."
                        )
                    except Exception as error:
                        st.error(
                            f"Something went wrong: {error}"
                        )
