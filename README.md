# 🧠 AI Docs Generation Automation

A complete automation pipeline that generates new, high-quality, SEO-ready training courses by scraping competitor data, intelligently clustering and summarizing it, and finally generating professional `.docx` course documents using AI.

---

## 🚀 Workflow Overview

This automation includes the following steps:

1. **Scraping**  
   Extract training courses using Web Scraper (Chrome Extension) from various competitor websites.

2. **Concatenation & Deduplication**  
   Merge multiple files per competitor and remove duplicates.  
   ➤ `concatenate.py`

3. **Cleaning**  
   Remove duplicates across all files and organize clean versions.  
   ➤ `remove_duplicates.py`

4. **Filtering Existing Courses**  
   Filter out any course already offered by the reference dataset.  
   ➤ `cfilecode.py`  
   - Compares against a reference course file (`Nos-Formations-Export-*.csv`)  
   - Outputs: `C_final_file.xlsx`

5. **Clustering & Classification**  
   Use OpenAI API to assign each new course to a cluster and a sub-thematique.  
   ➤ `clustering.py`  
   - Inputs: `C_final_file.xlsx`  
   - Outputs: `classified_courses_final.xlsx`

6. **Summarization & Pricing**  
   Summarize each cluster and apply pricing logic.  
   ➤ `summarization_all.py`  
   - Outputs a single summarized row per cluster  
   - Includes pricing, audience, pedagogy, and highlights

7. **Cluster Counting & Grouping**  
   Count and group similar clusters for reporting or optimization.  
   ➤ `counting.py`  
   ➤ `grouping.py`

8. **Document Generation**  
   Generate professional `.docx` course files from the summaries.  
   ➤ `generation.py`

---

## 📁 Folder Structure

project_root/
│
├── 📂scraped_data/ # Raw scraped files (via Web Scraper)
├── 📂cleaned_data/ # Deduplicated .xlsx files
├── 📂filtered_data/ # Competitor-only courses (C file)
├── 📂clustered_data/ # Courses with cluster & sub-thematique labels
├── 📂summarized_data/ # Summarized cluster records
├── 📂generated_docs/ # Final .docx course documents
│
├── 🐍 concatenate.py # Concatenate and deduplicate scraped files
├── 🐍 remove_duplicates.py # Remove duplicates from all sources
├── 🐍 cfilecode.py # Generate filtered 'C file'
├── 🐍 clustering.py # Assign clusters via OpenAI API
├── 🐍 summarization_all.py # Summarize and apply pricing
├── 🐍 counting.py # Count clusters and sub-thematiques
├── 🐍 grouping.py # Group similar clusters
├── 🐍 generation.py # Generate final .docx documents
├── 📄 requirements.txt # Dependencies
└── 📄 README.md # Project overview



---

## 🧪 Usage

> All scripts are meant to be executed sequentially.

### 1. Scrape Competitor Data  
Use [Web Scraper Chrome Extension](https://webscraper.io/) to export `.xlsx` files per competitor.

### 2. Concatenate and Clean  
python concatenate.py
python remove_duplicates.py


3. Filter Out Existing Courses
python cfilecode.py
4. Cluster Courses with OpenAI
python clustering.py
5. Summarize and Apply Pricing Logic
python summarization_all.py
6. Count and Group Clusters
python counting.py
python grouping.py
7. Generate Final Documents
python generation.py
Final course .docx files will be saved to the generated_docs/ folder.

🧠 AI and Logic
Clustering: OpenAI GPT model is used to assign semantic clusters.

Pricing Summary: Logic accounts for course type, competitors’ pricing, and delivery mode.

Document Generation: Each course summary is compiled into a standardized .docx template using python-docx.

📦 Requirements
Python 3.9+

OpenAI API Key

Dependencies listed in requirements.txt

Install them:
pip install -r requirements.txt
📊 Final Document Output Includes
Course Title & Cluster

Objectives, Context, Audience

Programme & Pedagogical Approach

Duration, Pricing (Individual, Internal, Intra/Inter-company)

Financing Options

Source URLs and Competitor Price References

🤖 Tech Stack
Python

Pandas, OpenPyXL, python-docx

OpenAI API

Web Scraper Chrome Extension
