import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="LexiScope", layout="wide")

st.title("📖 LexiScope")
st.subheader("Semantic Text Similarity Analyzer")
st.write("Enter at least **2 words, sentences, or short texts** (one per line).")

user_input = st.text_area(
    "Input Text",
    height=180,
    placeholder="Artificial Intelligence\nMachine Learning\nDeep Learning\nI love pizza"
)

if st.button("Analyze Similarity"):

    texts = [line.strip() for line in user_input.split("\n") if line.strip()]

    if len(texts) < 2:
        st.warning("Please enter at least two texts.")
        st.stop()

    with st.spinner("Loading pretrained model..."):
        model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(texts)

    similarity_matrix = cosine_similarity(embeddings)

    st.header("Similarity Matrix")

    labels = [f"Text {i+1}" for i in range(len(texts))]
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=labels,
        columns=labels
    )

    st.dataframe(similarity_df)

    pair_names = []
    pair_scores = []

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            pair_names.append(f"Text {i+1} vs Text {j+1}")
            pair_scores.append(similarity_matrix[i][j])

    result_df = pd.DataFrame({
        "Pair": pair_names,
        "Similarity Score": pair_scores
    })

    result_df = result_df.sort_values(
        by="Similarity Score",
        ascending=False
    )

    st.header("Top Similarity Results")
    st.dataframe(result_df)

    st.header("Bar Chart")

    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(result_df["Pair"], result_df["Similarity Score"])
    ax.set_ylabel("Similarity Score")
    ax.set_title("Similarity Scores")
    plt.xticks(rotation=30)
    st.pyplot(fig)

    st.header("Heatmap")

    fig2, ax2 = plt.subplots(figsize=(6,5))
    heatmap = ax2.imshow(similarity_matrix)

    ax2.set_xticks(range(len(labels)))
    ax2.set_yticks(range(len(labels)))

    ax2.set_xticklabels(labels, rotation=45)
    ax2.set_yticklabels(labels)

    plt.colorbar(heatmap)

    st.pyplot(fig2)

    st.header("2D Embedding Plot (PCA)")

    pca = PCA(n_components=2)

    reduced = pca.fit_transform(embeddings)

    fig3, ax3 = plt.subplots(figsize=(6,5))

    ax3.scatter(reduced[:,0], reduced[:,1])

    for i, txt in enumerate(texts):
        ax3.annotate(
            f"T{i+1}",
            (reduced[i,0], reduced[i,1])
        )

    ax3.set_title("PCA Embedding Visualization")

    st.pyplot(fig3)

    top_pair = result_df.iloc[0]

    st.header("Paul's Critical Thinking Standards")

    st.markdown(f"""
### Clarity
The user entered **{len(texts)}** texts. The application compares their meanings using semantic embeddings.

### Accuracy
Model Used: **all-MiniLM-L6-v2** (Free Pretrained Sentence Transformer). No preprocessing, cleaning, or training was performed.

### Precision
Highest Similarity Score: **{top_pair['Similarity Score']:.4f}**

### Relevance
The bar chart, heatmap, and PCA plot directly support the similarity results.

### Logic
The pair **{top_pair['Pair']}** has the highest similarity because their semantic embeddings are closest.

### Significance
The most important finding is the pair with the highest similarity score.

### Fairness
The pretrained model may not fully understand ambiguous, domain-specific, or newly emerging language.
""")

    st.success("Analysis Completed Successfully!")
