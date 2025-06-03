import os
import shutil
from docx import Document
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

DOCS_DIR = ""
OUTPUT_DIR = ""
SIMILARITY_THRESHOLD = 0.5  # Less strict

def extract_cluster(doc_path):
    doc = Document(doc_path)
    text = "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
    match = re.search(r"Cluster\s*:\s*(.+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "Unknown Cluster"

def normalize_cluster_name(cluster):
    cluster = cluster.lower()
    cluster = re.sub(r"&", "and", cluster)
    cluster = re.sub(r"\s*-\s*(beginner|intermediate|advanced|expert|.*?to.*?)$", "", cluster, flags=re.IGNORECASE)
    cluster = re.sub(r"\b(tools?|solutions?|basics?|fundamentals?)\b", "", cluster)
    cluster = re.sub(r"[^a-z0-9\s/]", "", cluster)  # remove special chars
    return cluster.strip()

def group_clusters(clusters, threshold=0.5):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(clusters)
    similarity = cosine_similarity(tfidf_matrix)

    groups = []
    visited = set()

    for i in range(len(clusters)):
        if i in visited:
            continue
        group = [i]
        for j in range(i + 1, len(clusters)):
            if similarity[i, j] > threshold:
                group.append(j)
                visited.add(j)
        visited.add(i)
        groups.append(group)

    return groups

def process_docs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    doc_files = [os.path.join(DOCS_DIR, f) for f in os.listdir(DOCS_DIR) if f.endswith(".docx")]
    doc_data = []

    for path in doc_files:
        cluster = extract_cluster(path)
        normalized = normalize_cluster_name(cluster)
        doc_data.append({'path': path, 'cluster': cluster, 'normalized': normalized})

    unique_normalized = list({d['normalized'] for d in doc_data})
    grouped_indices = group_clusters(unique_normalized, SIMILARITY_THRESHOLD)

    group_mapping = {}
    for group in grouped_indices:
        group_names = [unique_normalized[i] for i in group]
        matching_clusters = [d['cluster'] for d in doc_data if normalize_cluster_name(d['cluster']) in group_names]
        most_common = Counter(matching_clusters).most_common(1)[0][0]
        clean_folder_name = normalize_cluster_name(most_common).title().replace("/", " and ").strip()
        for name in group_names:
            group_mapping[name] = clean_folder_name

    for doc in doc_data:
        folder_name = group_mapping.get(doc['normalized'], "Other")
        folder_path = os.path.join(OUTPUT_DIR, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        shutil.copy(doc['path'], os.path.join(folder_path, os.path.basename(doc['path'])))
        print(f"✔ Copied: {os.path.basename(doc['path'])} → {folder_name}")

    print("✅ Grouping complete!")

if __name__ == "__main__":
    process_docs()
