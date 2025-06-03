import os
import pandas as pd
import re
from fuzzywuzzy import fuzz

# === CONFIGURATION ===
my_courses_file = ''  # Your CSV file 
competitor_folder = ''  # Folder with competitor .xlsx files 
output_file = ''  # Output file 
SIMILARITY_THRESHOLD = 85  # Adjust this threshold as needed (higher = stricter matching)

# Standard columns for output 
output_columns = [
   
]

# Function to normalize titles
def normalize_title(title):
    # Convert to string, lowercase, remove special characters
    normalized = str(title).lower()
    normalized = re.sub(r'[^\w\s]', ' ', normalized)  # Replace special chars with spaces
    normalized = re.sub(r'\s+', ' ', normalized).strip()  # Normalize spaces
    return normalized

# === LOAD YOUR COURSE TITLES === 
print("Loading your course titles...") 
my_df = pd.read_csv(my_courses_file) 
my_df['normalized_title'] = my_df['Title'].apply(normalize_title)
my_titles = list(my_df['normalized_title'])

# Function to check if a title matches any existing course
def is_course_match(title, existing_titles):
    normalized_title = normalize_title(title)
    
    # 1. Exact match check
    if normalized_title in existing_titles:
        return True
    
    # 2. Fuzzy matching check
    for existing_title in existing_titles:
        # Calculate similarity ratio
        ratio = fuzz.ratio(normalized_title, existing_title)
        
        # If similarity is high enough, consider it a match
        if ratio >= SIMILARITY_THRESHOLD:
            return True
        
        # 3. Partial matching for key terms
        # Check if most of the important words are present
        title_words = set(normalized_title.split())
        existing_words = set(existing_title.split())
        
        # If the title has 3+ words and >70% of words match, consider it a match
        if len(title_words) >= 3:
            common_words = title_words.intersection(existing_words)
            if len(common_words) / len(title_words) > 0.7:
                return True
    
    return False

# === PROCESS COMPETITOR FILES === 
all_unmatched_rows = [] 
print(f"Looking for courses not in your catalog ({len(my_titles)} courses loaded)...")

for filename in os.listdir(competitor_folder): 
    if filename.endswith('.xlsx'): 
        filepath = os.path.join(competitor_folder, filename) 
        print(f"Processing file: {filename} ...") 
        try: 
            comp_df = pd.read_excel(filepath, engine='openpyxl') 
 
            if 'title' not in comp_df.columns: 
                print(f"  ⚠️ Skipping {filename} — no 'title' column found.") 
                continue 
 
            # Process each row to check for matches
            for idx, row in comp_df.iterrows():
                if pd.isna(row['title']) or row['title'] == '':
                    continue
                    
                # Check if this course exists in your catalog
                if not is_course_match(row['title'], my_titles):
                    # If no match found, add to unmatched rows
                    new_row = {col: row[col] if col in row and pd.notna(row[col]) else '' for col in output_columns} 
                    all_unmatched_rows.append(new_row)
 
        except Exception as e: 
            print(f"  ❌ Error processing {filename}: {e}") 

# === SAVE OUTPUT === 
if all_unmatched_rows: 
    output_df = pd.DataFrame(all_unmatched_rows, columns=output_columns) 
    output_df.to_excel(output_file, index=False, engine='openpyxl') 
    print(f"\n✅ Done! {len(output_df)} unmatched courses saved to '{output_file}'") 
else: 
    print("\n✅ No unmatched courses found.")