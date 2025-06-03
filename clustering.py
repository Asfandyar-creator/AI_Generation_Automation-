import os
import asyncio
import pandas as pd
import json
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import traceback

# Configuration
INPUT_FILE = ""
OUTPUT_FILE = ""
REVIEW_FILE = ""
BATCH_SIZE = 8
ERROR_LOG_FILE = ""

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# Function schema
function_schema = {
    "name": "classify_courses",
    "description": "Classify each course into a cluster and sub-thématique",
    "parameters": {
        "type": "object",
        "properties": {
            "classifications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "cluster": {"type": "string"},
                        "sub_thematique": {"type": "string"},
                        "similar_course_count": {"type": "integer"}
                    },
                    "required": ["id", "cluster", "sub_thematique", "similar_course_count"]
                }
            }
        },
        "required": ["classifications"]
    }
}

SYSTEM_MESSAGE = """
You are a training course clustering expert. Classify each course into:
1. A **Cluster** (topic + level, e.g., "Python - Beginner").
2. A **Sub-thématique** (specific topic within the cluster).
3. An estimate of similar courses (across providers).
Return EXACTLY one classification for each course, in the same order. Use the unique ID field.
Clusters with <5 similar courses will require manual review.
"""

def log_error(message):
    try:
        message = message.replace("❌", "ERROR").replace("⚠️", "WARNING").replace("✓", "OK")
        with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{message}\n")
        print(message)
    except Exception as e:
        try:
            with open(ERROR_LOG_FILE, "a", encoding="ascii", errors="ignore") as f:
                f.write(f"Error logging: {str(e)}\n")
        except:
            print("Fatal logging error.")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), retry=retry_if_exception_type(Exception))
async def classify_batch(batch_df):
    try:
        batch_data = [
            {
                "id": str(row.Index),
                "title": row.title,
                 "level": getattr(row, "level", "Not specified"),
                "url": row.url
            }
            for row in batch_df.itertuples()
        ]
        courses_text = "\n".join([
            f"[{item['id']}] Title: {item['title']}, Level: {item['level']}, URL: {item['url']}"
            for item in batch_data
        ])
        response = await client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": f"Classify the following {len(batch_data)} courses. Match each output to the course's 'id'.\n\n{courses_text}"}
            ],
            functions=[function_schema],
            function_call={"name": "classify_courses"}
        )

        function_call = response.choices[0].message.function_call
        if function_call and function_call.name == "classify_courses":
            data = json.loads(function_call.arguments)
            results = {item["id"]: item for item in data.get("classifications", [])}

            final = []
            for row in batch_data:
                cid = row["id"]
                if cid in results:
                    final.append(results[cid])
                else:
                    final.append({
                        "id": cid,
                        "cluster": "Needs Manual Review",
                        "sub_thematique": "Classification Missing",
                        "similar_course_count": 0
                    })
            return final

        raise ValueError("Invalid response format")
    except Exception as e:
        log_error(f"Error in classify_batch: {e}\n{traceback.format_exc()}")
        raise

async def process_courses(df):
    results = []
    df["level"] = df.get("level", "Not specified")

    batches = [df[i:i+BATCH_SIZE] for i in range(0, len(df), BATCH_SIZE)]

    for i, batch_df in enumerate(batches):
        print(f"Processing batch {i+1}/{len(batches)}")
        try:
            batch_results = await classify_batch(batch_df)
            results.extend(batch_results)
        except Exception as e:
            log_error(f"Batch {i+1} failed: {e}")
            for idx in batch_df.index:
                results.append({
                    "id": str(idx),
                    "cluster": "Batch Failed",
                    "sub_thematique": "Needs Manual Review",
                    "similar_course_count": 0
                })

    return results

async def main():
    try:
        with open(ERROR_LOG_FILE, "w", encoding="utf-8") as f:
            f.write("=== Classification Error Log ===\n")

        print(f"Loading data from {INPUT_FILE}...")
        df = pd.read_excel(INPUT_FILE)

        if 'title' not in df.columns or 'url' not in df.columns:
            raise ValueError("Input must contain 'title' and 'url' columns")

        if 'level' not in df.columns:
            print("WARNING: 'level' column not found. Adding default values.")
            df['level'] = "Not specified"

        print(f"Processing {len(df)} courses...")
        results = await process_courses(df)

        for r in results:
            idx = int(r["id"])
            df.at[idx, "Cluster"] = r["cluster"]
            df.at[idx, "Sub thématique"] = r["sub_thematique"]
            df.at[idx, "Similar Course Count"] = r["similar_course_count"]

        cluster_counts = df.groupby("Cluster").size()
        small_clusters = set(cluster_counts[cluster_counts < 5].index)
        df["Review Required"] = df["Cluster"].apply(lambda x: "Yes" if x in small_clusters else "No")

        order = ["Cluster", "Sub thématique", "Similar Course Count", "Review Required"]
        others = [col for col in df.columns if col not in order]
        df = df[order + others].sort_values(by=["Cluster", "Sub thématique"])

        print(f"Saving results to {OUTPUT_FILE}")
        df.to_excel(OUTPUT_FILE, index=False)

        review_df = df[df["Review Required"] == "Yes"]
        print(f"Saving {len(review_df)} review courses to {REVIEW_FILE}")
        review_df.to_excel(REVIEW_FILE, index=False)

        print("\nCluster Summary:")
        for cluster, count in cluster_counts.items():
            status = "REVIEW REQUIRED" if cluster in small_clusters else "OK"
            print(f"- {cluster}: {count} ({status})")

        print(f"\nDone! Check '{OUTPUT_FILE}' and '{REVIEW_FILE}'. See {ERROR_LOG_FILE} for errors.")
    except Exception as e:
        log_error(f"Fatal error: {e}\n{traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main())
