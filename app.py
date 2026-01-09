import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan

# ---------------------------
# CONFIG
# ---------------------------

st.set_page_config(page_title="Segment-based Topic Analytics", layout="wide")

CANDIDATE_LABELS = [
    "Religion", "Politics", "Economy", "Health", "Education",
    "Law", "Culture", "Technology", "Environment", "Sports",
    "Security", "Gender", "Migration", "Media", "Family"
]

# ---------------------------
# UTILITIES
# ---------------------------

def simple_segmenter(text, doc_id):
    raw_segments = [s.strip() for s in text.split('.') if s.strip()]
    segments = []
    cursor = 0
    for i, seg in enumerate(raw_segments):
        start = text.find(seg, cursor)
        end = start + len(seg)
        cursor = end
        segments.append({
            "segment_id": f"{doc_id}_s{i}",
            "document_id": doc_id,
            "text": seg,
            "start_char": start,
            "end_char": end,
            "token_count": len(seg.split())
        })
    return segments


def suggest_label(keywords, embedder, candidate_labels):
    phrase = " ".join(keywords)
    topic_vec = embedder.encode([phrase])
    label_vecs = embedder.encode(candidate_labels)
    sims = cosine_similarity(topic_vec, label_vecs)[0]
    idx = int(np.argmax(sims))
    return candidate_labels[idx], float(sims[idx])


def build_overall_table(segments_df, assignments_df, topics_df):
    if segments_df.empty or assignments_df.empty or topics_df.empty:
        return pd.DataFrame()

    merged = (
        assignments_df
        .merge(segments_df, on="segment_id", how="inner")
        .merge(topics_df, on="topic_id", how="inner")
    )

    grouped = merged.groupby(["topic_id", "final_label", "keywords"]).agg(
        segment_count=("segment_id", "count"),
        total_tokens=("token_count", "sum"),
    ).reset_index()

    total_tokens = grouped["total_tokens"].sum()
    grouped["% of corpus"] = (grouped["total_tokens"] / total_tokens) * 100
    return grouped.sort_values(by="segment_count", ascending=False)


def build_per_doc_table(segments_df, assignments_df, topics_df):
    if segments_df.empty or assignments_df.empty or topics_df.empty:
        return pd.DataFrame()

    merged = (
        assignments_df
        .merge(segments_df, on="segment_id", how="inner")
        .merge(topics_df, on="topic_id", how="inner")
    )

    grouped = merged.groupby(["document_id", "topic_id", "final_label"]).agg(
        segment_count=("segment_id", "count"),
        total_tokens=("token_count", "sum")
    ).reset_index()

    return grouped.sort_values(by=["document_id", "segment_count"], ascending=[True, False])


def plot_topic_distribution(overall_df):
    if overall_df.empty:
        st.info("No data available for plotting.")
        return

    fig, ax = plt.subplots()
    ax.bar(overall_df["final_label"], overall_df["segment_count"])
    ax.set_xlabel("Topic")
    ax.set_ylabel("Segment Count")
    ax.set_title("Topic Distribution (by Segments)")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

# ---------------------------
# APP
# ---------------------------

st.title("📊 Segment-based Topic Analytics Platform")

st.markdown(
    "Blind topic modeling on discourse segments with **mandatory semantic labeling**. "
    "Labels are **suggested via word embeddings** and must be confirmed before outputs."
)

# ---------------------------
# INPUT
# ---------------------------

st.header("1. Input Text")

input_mode = st.radio("Choose input mode:", ["Direct Text", "Upload Files"])

documents = []

if input_mode == "Direct Text":
    text_input = st.text_area("Paste your text here:", height=200)
    if text_input.strip():
        documents.append({
            "document_id": "doc_1",
            "filename": "direct_input",
            "text": text_input
        })
else:
    uploaded_files = st.file_uploader("Upload text files", type=["txt"], accept_multiple_files=True)
    for f in uploaded_files:
        text = f.read().decode("utf-8")
        documents.append({
            "document_id": f.name,
            "filename": f.name,
            "text": text
        })

if not documents:
    st.stop()

# ---------------------------
# SEGMENTATION
# ---------------------------

st.header("2. Segmentation")

all_segments = []
for doc in documents:
    all_segments.extend(simple_segmenter(doc["text"], doc["document_id"]))

segments_df = pd.DataFrame(all_segments)

st.write(f"Total segments: {len(segments_df)}")
st.dataframe(segments_df[["segment_id", "document_id", "text"]].head())

if len(segments_df) < 3:
    st.error("❌ Need at least 3 segments for topic modeling. Add more text.")
    st.stop()

# ---------------------------
# TOPIC MODELING
# ---------------------------

st.header("3. Topic Modeling")

with st.spinner("Running BERTopic..."):

    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=2,
        min_samples=1,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )

    topic_model = BERTopic(
        hdbscan_model=hdbscan_model,
        min_topic_size=2,
        calculate_probabilities=False
    )

    texts = segments_df["text"].tolist()
    topics, probs = topic_model.fit_transform(texts)

segments_df["topic_id"] = topics

# remove outliers
segments_df = segments_df[segments_df["topic_id"] != -1].reset_index(drop=True)

if segments_df.empty:
    st.error("❌ All segments were classified as outliers. Please provide more diverse text.")
    st.stop()

# ---------------------------
# EXTRACT TOPICS
# ---------------------------

topic_info = topic_model.get_topic_info()

raw_topics = {}
for tid in topic_info["Topic"]:
    if tid == -1:
        continue
    words = topic_model.get_topic(tid)
    if words:
        raw_topics[tid] = [w[0] for w in words[:5]]

if not raw_topics:
    st.error("❌ No valid topics could be extracted. Add more or more diverse text.")
    st.stop()

# ---------------------------
# LABEL SUGGESTION (MANDATORY)
# ---------------------------

st.header("4. Topic Label Suggestion (Mandatory Confirmation)")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

topics_data = []
for tid, keywords in raw_topics.items():
    label, conf = suggest_label(keywords, embedder, CANDIDATE_LABELS)
    topics_data.append({
        "topic_id": tid,
        "keywords": ", ".join(keywords),
        "suggested_label": label,
        "confidence": conf,
        "final_label": label
    })

topics_df = pd.DataFrame(topics_data)

if topics_df.empty:
    st.error("❌ Topic table is empty. Cannot proceed.")
    st.stop()

st.markdown("### Confirm or edit all topic labels")

label_confirmed = True

for idx, row in topics_df.iterrows():
    st.markdown(f"**Topic {row['topic_id']}** — Keywords: `{row['keywords']}`")
    new_label = st.text_input(
        f"Label for Topic {row['topic_id']}",
        value=row["final_label"],
        key=f"label_{row['topic_id']}"
    )
    if not new_label.strip():
        label_confirmed = False
    topics_df.at[idx, "final_label"] = new_label

if not label_confirmed:
    st.warning("⚠️ All topics must have labels before proceeding.")
    st.stop()

# ---------------------------
# ASSIGNMENTS
# ---------------------------

assignments_df = segments_df[["segment_id", "topic_id"]].copy()

# ---------------------------
# OUTPUTS
# ---------------------------

st.header("5. Outputs")

overall_df = build_overall_table(segments_df, assignments_df, topics_df)
per_doc_df = build_per_doc_table(segments_df, assignments_df, topics_df)

st.subheader("Overall Topic Table")
st.dataframe(overall_df if not overall_df.empty else pd.DataFrame())

st.subheader("Per-document Topic Table")
st.dataframe(per_doc_df if not per_doc_df.empty else pd.DataFrame())

st.subheader("Topic Distribution Chart")
plot_topic_distribution(overall_df)

# ---------------------------
# DOWNLOADS
# ---------------------------

st.header("6. Downloads")

def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

zip_buffer = io.BytesIO()
with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
    if not overall_df.empty:
        zf.writestr("overall_topics.csv", to_csv_bytes(overall_df))
    if not per_doc_df.empty:
        zf.writestr("per_document_topics.csv", to_csv_bytes(per_doc_df))
    zf.writestr("segments.csv", to_csv_bytes(segments_df))
    zf.writestr("topics.csv", to_csv_bytes(topics_df))

if not overall_df.empty:
    st.download_button("⬇️ Download Overall Table", to_csv_bytes(overall_df), "overall_topics.csv", "text/csv")

if not per_doc_df.empty:
    st.download_button("⬇️ Download Per-document Table", to_csv_bytes(per_doc_df), "per_document_topics.csv", "text/csv")

st.download_button("⬇️ Download All Results (ZIP)", zip_buffer.getvalue(), "all_results.zip", "application/zip")
