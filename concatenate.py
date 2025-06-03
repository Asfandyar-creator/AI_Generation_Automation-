import os
import pandas as pd

# Define the folder containing Excel files and the output file path
input_folder = ""  # Replace with your folder path
output_file = ""  # Replace with desired output path

# Get a list of all Excel files in the folder
excel_files = [f for f in os.listdir(input_folder) if f.endswith(('.xlsx', '.xls'))]

# Initialize an empty list to hold DataFrames
dataframes = []

# Loop through each file and read the contents
for file in excel_files:
    file_path = os.path.join(input_folder, file)
    df = pd.read_excel(file_path)
    dataframes.append(df)

# Concatenate all the DataFrames
combined_df = pd.concat(dataframes, ignore_index=True)

# Drop duplicate rows based on 'title' column
if 'title' in combined_df.columns:
    combined_df = combined_df.drop_duplicates(subset='title', keep='first')
else:
    print("Warning: 'title' column not found. No duplicates removed.")

# Save the cleaned DataFrame to a new Excel file
combined_df.to_excel(output_file, index=False)

print(f"Concatenated unique data saved to: {output_file}")
