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

st.markdown("""
Blind topic modeling
