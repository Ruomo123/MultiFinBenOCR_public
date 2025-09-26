from lib.agent import Agent
from lib.tools import Tools
import pandas as pd
from tqdm import tqdm
import os
import time

def evaluate(model_name="gpt-4o", experiment_tag="zero-shot",language = "en", sample = None):
    tools = Tools()

    if language == "en":
        # df = pd.read_parquet("hyr_ocr_process/output_parquet_hyr/EnglishOCR.parquet") # comment out if using TheFinAI/MultiFinBen-EnglishOCR
        ds_en = load_dataset("TheFinAI/MultiFinBen-EnglishOCR") # use this line only if using TheFinAI/MultiFinBen-EnglishOCR
        df = ds_en['train'].to_pandas() # use this line only if using TheFinAI/MultiFinBen-EnglishOCR
    elif language == "es":
        df = pd.read_parquet("hyr_ocr_process/spanish_output_parquet/spanish_batch_0000.parquet")
    elif language == "gr":
        df = pd.read_parquet("hyr_ocr_process/greek_output_parquet/GreekOCR_500.parquet") #GreekOCR_500 have same structure as TheFinAI/MultiFinBen-EnglishOCR; GreekOCR_v1 has same structure as  output_parquet_hyr/EnglishOCR.parquet
    elif language == "jp":
        df = pd.read_parquet("hyr_ocr_process/japanese_output_parquet/japanese_batch_0000.parquet")
    else: 
        print("Not a valid choice of language, please try again.")
        return language
    
    experiment_name = f"{model_name}_{experiment_tag}_financial"

    if language == "en":
        experiment_folder = os.path.join("hyr_results/predictions/", experiment_name)
    elif language == "es":
        experiment_folder = os.path.join("hyr_results/predictions_spanish/", experiment_name)
    elif language == "gr":
        experiment_folder = os.path.join("hyr_results/predictions_greek/", experiment_name)
    elif language == "jp":
        experiment_folder = os.path.join("hyr_results/predictions_japanese/", experiment_name)

    os.makedirs(experiment_folder, exist_ok=True)

    # Get predicted indices from filenames
    predicted_indices = set()
    if os.path.exists(experiment_folder):
        for fname in os.listdir(experiment_folder):
            if fname.startswith(f"{model_name}_pred_") and fname.endswith(".txt"):
                try:
                    idx = int(fname.replace(f"{model_name}_pred_", "").replace(".txt", ""))
                    predicted_indices.add(idx)
                except:
                    continue

    # Filter out completed predictions
    df = df[~df.index.isin(predicted_indices)]

    # Apply sample AFTER filtering
    if sample:
        df = df.head(sample)  # get sample

    agent = Agent(model_name)

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Running {model_name}"):
        image_path = row["image_path"]
        # image_path = row["image"] # corresponds to TheFinAI/MultiFinBen-EnglishOCR, image is in base64 format
        # ground_truth = row["matched_html"]
        output_file = os.path.join(experiment_folder, f"{model_name}_pred_{i}.txt")

        try:
            result = agent.draft(image_path)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result)
            time.sleep(1.5)
        except Exception as e:
            print(f"⚠️ Error on index {i}: {e}")
            continue

def main():
    evaluate(model_name="gpt-4o",language = "jp" , sample = 100)

if __name__ == '__main__':
    main()
