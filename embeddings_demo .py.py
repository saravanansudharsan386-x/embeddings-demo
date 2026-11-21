"""
Tiny embeddings demo: similar sentences end up close together.

Run:
    pip install sentence-transformers scikit-learn matplotlib
    python embeddings_demo.py

The first run downloads a small model (about 90 MB). After that it works offline.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 12 real titles from the Zoho content library (Task 1), grouped by topic.
DOCS = [
    ("Analytics & Data",   "Zoho Analytics meets Zoho ERP: Get actionable insights from your ERP data"),
    ("Analytics & Data",   "From inventory reports to inventory intelligence with Zoho Inventory and Zoho Analytics"),
    ("Analytics & Data",   "Rewind with us - Zoho Analytics in 2025"),
    ("AI & MCP",           "Build AI-powered workflows with Zoho Cliq MCP and the Cliq developer platform"),
    ("AI & MCP",           "AI that actually helps: Practical intelligence in Zoho Workplace"),
    ("AI & MCP",           "Introducing Zoho PageSense MCP"),
    ("Integrations & Apps", "App Spotlight: 360 SMS for Zoho CRM"),
    ("Integrations & Apps", "App Spotlight: Tap Payments for Zoho Books"),
    ("Integrations & Apps", "Announcing SAP Business One integrations in Zoho Flow"),
    ("Email & Collaboration", "Introducing Meeting Co-Host support in Zoho Calendar"),
    ("Email & Collaboration", "Admin reports in Zoho Mail: From user accounts to team communication"),
    ("Security & Admin",   "Introducing Accessibility Controls in Zoho Vault"),
]
TOPICS = [d[0] for d in DOCS]
TEXTS = [d[1] for d in DOCS]


def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed(model, texts):
    # normalize_embeddings=True makes every vector length 1,
    # so cosine similarity is just a dot product.
    return np.asarray(model.encode(texts, normalize_embeddings=True))


def short(text, n=42):
    return text if len(text) <= n else text[: n - 1] + "…"


def main():
    model = get_model()
    vectors = embed(model, TEXTS)
    print(f"Each sentence is now a vector of {vectors.shape[1]} numbers.")
    print("First sentence, first 6 numbers:", np.round(vectors[0][:6], 3), "\n")

    # 1) Which pairs of titles are closest in meaning?
    sims = vectors @ vectors.T
    pairs = [(sims[i, j], i, j) for i in range(len(TEXTS)) for j in range(i + 1, len(TEXTS))]
    pairs.sort(reverse=True)
    print("Top 5 most similar pairs:")
    for score, i, j in pairs[:5]:
        print(f"  {score:.2f}  {short(TEXTS[i])}  <->  {short(TEXTS[j])}")
    print("\nLeast similar pair:")
    score, i, j = pairs[-1]
    print(f"  {score:.2f}  {short(TEXTS[i])}  <->  {short(TEXTS[j])}\n")

    # 2) Search by meaning: the query shares few words with the titles.
    query = "smart assistants that automate my work"
    q = embed(model, [query])[0]
    scores = vectors @ q
    print(f'Query: "{query}"')
    for k in np.argsort(-scores)[:3]:
        print(f"  {scores[k]:.2f}  {short(TEXTS[k], 70)}")

    # 3) Draw the map: squash the vectors down to 2 dimensions.
    points = PCA(n_components=2).fit_transform(np.vstack([vectors, q]))
    order = list(dict.fromkeys(TOPICS))
    colors  = dict(zip(order, ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#444444"]))
    markers = dict(zip(order, ["o", "s", "^", "D", "P"]))
    fig, (ax, key) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [3, 2]})
    for idx, topic in enumerate(TOPICS):
        x, y = points[idx]
        ax.scatter(x, y, s=230, color=colors[topic], marker=markers[topic],
                   label=topic if topic not in ax.get_legend_handles_labels()[1] else None)
        ax.annotate(str(idx + 1), (x, y), textcoords="offset points", xytext=(9, 7),
                    fontsize=12, fontweight="bold")
    ax.scatter(*points[-1], s=320, marker="*", color="black", label="query")
    ax.annotate("Q", points[-1], textcoords="offset points", xytext=(11, 8), fontsize=12, fontweight="bold")
    ax.set_title("Embeddings map: similar meaning sits close together\n(384 numbers squashed to 2D, so distances are approximate)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", fontsize=9)
    key.axis("off")
    lines = [f"{i + 1:>2}. {short(t, 46)}" for i, t in enumerate(TEXTS)]
    key.set_title("Key", fontsize=11, loc="left")
    for i, line in enumerate(lines):
        key.text(0, 1 - 0.07 * (i + 1), line, fontsize=10, color=colors[TOPICS[i]], transform=key.transAxes)
    key.text(0, 1 - 0.07 * (len(lines) + 2), f"Q.  {query}", fontsize=10, fontweight="bold", transform=key.transAxes)
    fig.tight_layout()
    fig.savefig("embeddings_map.png", dpi=150)
    print("\nSaved the map to embeddings_map.png")


if __name__ == "__main__":
    main()
