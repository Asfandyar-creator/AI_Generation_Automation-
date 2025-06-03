import os
import pandas as pd

# Set source and destination directories
source_folder = ""
destination_folder = ""

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Process each .xlsx file in the source folder
for filename in os.listdir(source_folder):
    if filename.endswith(".xlsx"):
        source_file = os.path.join(source_folder, filename)
        dest_file = os.path.join(destination_folder, filename)

        try:
            # Load the Excel file
            excel = pd.ExcelFile(source_file, engine="openpyxl")
            writer = pd.ExcelWriter(dest_file, engine="openpyxl")

            for sheet_name in excel.sheet_names:
                df = excel.parse(sheet_name)

                # Remove duplicates by 'title' if exists
                if "title" in df.columns:
                    df = df.drop_duplicates(subset=["title"])
                else:
                    print(f"⚠️ 'title' column not found in {filename} > {sheet_name}, skipping duplicate check.")

                # Merge all columns starting with 'pedagogy_'
                pedagogy_cols = [col for col in df.columns if col.startswith("pedagogy_")]
                if pedagogy_cols:
                    df["pedagogy"] = df[pedagogy_cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
                    df.drop(columns=pedagogy_cols, inplace=True)
                    print(f"🔀 Merged columns {pedagogy_cols} into 'pedagogy' in {filename} > {sheet_name}")
                else:
                    print(f"ℹ️ No 'pedagogy_' columns found in {filename} > {sheet_name}")

                # Save cleaned and updated DataFrame
                df.to_excel(writer, sheet_name=sheet_name, index=False)

            writer.close()
            print(f"✅ Cleaned file saved: {dest_file}")

        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")
