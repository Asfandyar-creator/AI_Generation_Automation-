import pandas as pd

# Load your Excel or CSV file
df = pd.read_excel("cleaned_clusters.xlsx")  # or pd.read_csv("your_file.csv")

# Check columns
print("\n📋 Columns in your file:")
print(df.columns.tolist())

# Correct column names based on real ones
cluster_counts = df['Cluster'].value_counts()
sub_thematique_counts = df['Sub thématique'].value_counts()  # Corrected!

# Show counts
print("\n📦 Unique Clusters and their counts:")
print(cluster_counts)

print("\n🎯 Unique Sub-thematiques and their counts:")
print(sub_thematique_counts)

# Save the counts to Excel
with pd.ExcelWriter("cluster_subthematique_counts2.xlsx") as writer:
    cluster_counts.to_frame(name="Count").to_excel(writer, sheet_name="Clusters")
    sub_thematique_counts.to_frame(name="Count").to_excel(writer, sheet_name="SubThematiques")

print("\n✅ Counts saved to 'cluster_subthematique_counts2.xlsx'")

