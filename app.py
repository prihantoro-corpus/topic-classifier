import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io
import matplotlib.pyplot as plt
import hdbscan

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# CONFIG
# =========================

st.set_page_config(page_title="Segment-based Topic Analytics", layout="wide")

CANDIDATE_LABELS = [
    "Religion", "Politics", "Economy", "Health", "Education",
    "Law", "Culture", "Technology", "Environment", "Sports",
    "Security", "Gender", "Migration", "Media", "Family"
]

TOPIC_COLORS = [
    "#FF6B6B", "#4D96FF", "#6BCB77", "#FFD93D", "#9D4EDD",
    "#FF922B", "#2EC4B6", "#F72585", "#90DBF4", "#BDB2FF"
]

MULTI_LABEL_THRESHOLD = 0.35

# =========================
# UTILITIES
# =========================

def simple_segmenter(text, doc_id):
    raw_segments = [s.strip() for s in text.split('.') if s.strip()]
    segments = []
    for i, seg in enumerate(raw_segments):
        segments.append({
            "segment_id": f"{doc_id}__seg_{i}",
            "document_id": doc_id,
            "text": seg,
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


def build_overall_table(assignments_df, segments_df, topics_df):
    merged = assignments_df.merge(segments_df, on="segment_id").merge(topics_df, on="topic_id")

    grouped = merged.groupby(["topic_id", "final_label", "keywords"]).agg(
        segment_count=("segment_id", "nunique"),
        total_tokens=("token_count", "sum")
    ).reset_index()

    total_tokens = grouped["total_tokens"].sum()
    grouped["% of corpus"] = (grouped["total_tokens"] / total_tokens) * 100

    return grouped.sort_values(by="segment_count", ascending=False)


def build_per_doc_table(assignments_df, segments_df, topics_df):
    merged = assignments_df.merge(segments_df, on="segment_id").merge(topics_df, on="topic_id")

    grouped = merged.groupby(["document_id", "topic_id", "final_label"]).agg(
        segment_count=("segment_id", "nunique"),
        total_tokens=("token_count", "sum")
    ).reset_index()

    return grouped.sort_values(by=["document_id", "segment_count"], ascending=[True, False])


def plot_topic_distribution(overall_df):
    if overall_df.empty:
        st.info("No data available for plotting.")
        return

    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.bar(overall_df["final_label"], overall_df["segment_count"])
    ax.set_xlabel("Topic", fontsize=6)
    ax.set_ylabel("Segments", fontsize=6)
    ax.set_title("Topic Distribution", fontsize=7)
    ax.tick_params(axis='x', labelrotation=45, labelsize=6)
    ax.tick_params(axis='y', labelsize=6)

    st.pyplot(fig, use_container_width=False)


def highlight_text_accessible(segments_df, assignments_df, topics_df, active_topics):
    color_map = {}
    label_map = {}

    for i, row in topics_df.iterrows():
        color_map[row["topic_id"]] = TOPIC_COLORS[i % len(TOPIC_COLORS)]
        label_map[row["topic_id"]] = row["final_label"]

    html = ""
    grouped = assignments_df.groupby("segment_id")["topic_id"].apply(list).to_dict()

    for _, row in segments_df.iterrows():
        seg_id = row["segment_id"]
        seg_text = row["text"]
        topics_here = grouped.get(seg_id, [])

        active_here = [t for t in topics_here if t in active_topics]

        if active_here:
            gradients = []
            tooltips = []
            for t in active_here:
                gradients.append(color_map.get(t, "#DDD"))
                tooltips.append(label_map.get(t, "Unknown"))

            if len(gradients) == 1:
                bg = gradients[0]
            else:
                bg = f"linear-gradient(90deg, {', '.join(gradients)})"

            tooltip_text = "Topics: " + ", ".join(tooltips)

            html += (
                f'<span title="{tooltip_text}" '
                f'style="background:{bg}; padding:4px; margin:2px; '
                f'display:inline-block; border-radius:4px; cursor:help;">'
                f'{seg_text}.</span> '
            )
        else:
            html += f"{seg_text}. "

    return html

# =========================
# APP
# =========================

st.title("📊 Segment-based Topic Analytics Platform")

# INPUT
st.header("1. Input Text")
input_mode = st.radio("Choose input mode:", ["Direct Text", "Upload Files"])

documents = []

if input_mode == "Direct Text":
    text_input = st.text_area("Paste your text here:", height=200)
    if text_input.strip():
        documents.append({"document_id": "doc_1", "text": text_input})
else:
    uploaded_files = st.file_uploader("Upload text files", type=["txt"], accept_multiple_files=True)
    for f in uploaded_files:
        text = f.read().decode("utf-8")
        documents.append({"document_id": f.name, "text": text})

if not documents:
    st.stop()

# SEGMENTATION
st.header("2. Segmentation")
all_segments = []
for doc in documents:
    all_segments.extend(simple_segmenter(doc["text"], doc["document_id"]))

segments_df = pd.DataFrame(all_segments)
st.write(f"Total segments: {len(segments_df)}")

# TOPIC MODELING
st.header("3. Topic Modeling")

with st.spinner("Running BERTopic..."):
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)
    topic_model = BERTopic(hdbscan_model=hdbscan_model, min_topic_size=2)

    texts = segments_df["text"].tolist()
    topics, _ = topic_model.fit_transform(texts)

segments_df["topic_id_single"] = topics
segments_df = segments_df[segments_df["topic_id_single"] != -1].reset_index(drop=True)

# EXTRACT TOPICS
topic_info = topic_model.get_topic_info()
raw_topics = {}
for tid in topic_info["Topic"]:
    if tid == -1:
        continue
    words = topic_model.get_topic(tid)
    if words:
        raw_topics[tid] = [w[0] for w in words[:5]]

# LABEL SUGGESTION
st.header("4. Topic Label Suggestion (Mandatory)")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

topics_data = []
for tid, keywords in raw_topics.items():
    label, conf = suggest_label(keywords, embedder, CANDIDATE_LABELS)
    topics_data.append({
        "topic_id": tid,
        "keywords": ", ".join(keywords),
        "final_label": label
    })

topics_df = pd.DataFrame(topics_data)

for idx, row in topics_df.iterrows():
    new_label = st.text_input(
        f"Label for Topic {row['topic_id']} ({row['keywords']})",
        value=row["final_label"],
        key=f"label_{row['topic_id']}"
    )
    topics_df.at[idx, "final_label"] = new_label

# MULTI-LABEL ASSIGNMENT (FIX 2 APPLIED)
st.header("5. Multi-label Assignment")

topic_phrases = topics_df.set_index("topic_id")["keywords"].to_dict()
topic_texts = list(topic_phrases.values())
topic_ids = list(topic_phrases.keys())

topic_embeddings = embedder.encode(topic_texts)
segment_embeddings = embedder.encode(segments_df["text"].tolist())

assignments = []

for i, seg_vec in enumerate(segment_embeddings):
    sims = cosine_similarity([seg_vec], topic_embeddings)[0]

    assigned = False
    for j, score in enumerate(sims):
        if score >= MULTI_LABEL_THRESHOLD:
            assignments.append({
                "segment_id": segments_df.iloc[i]["segment_id"],
                "topic_id": topic_ids[j],
                "similarity": float(score)
            })
            assigned = True

    if not assigned:
        best_idx = int(np.argmax(sims))
        assignments.append({
            "segment_id": segments_df.iloc[i]["segment_id"],
            "topic_id": topic_ids[best_idx],
            "similarity": float(sims[best_idx])
        })

assignments_df = pd.DataFrame(assignments)

# OUTPUTS
st.header("6. Outputs")

overall_df = build_overall_table(assignments_df, segments_df, topics_df)
per_doc_df = build_per_doc_table(assignments_df, segments_df, topics_df)

st.subheader("Overall Topic Table")
st.dataframe(overall_df, width=400)

st.subheader("Per-document Topic Table")
st.dataframe(per_doc_df)

st.subheader("Topic Distribution (Compact)")
plot_topic_distribution(overall_df)

# STACKED HIGHLIGHT
st.header("7. Stacked Multi-topic Highlighting")

active_topics = []
cols = st.columns(len(topics_df))
for i, row in topics_df.iterrows():
    with cols[i]:
        if st.checkbox(row["final_label"], key=f"chk_{row['topic_id']}"):
            active_topics.append(row["topic_id"])

highlighted_html = highlight_text_accessible(segments_df, assignments_df, topics_df, active_topics)

st.markdown(
    f"<div style='border:1px solid #ccc; padding:15px; border-radius:5px;'>{highlighted_html}</div>",
    unsafe_allow_html=True
)
