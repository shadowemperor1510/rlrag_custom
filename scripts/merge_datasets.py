import os
import pandas as pd

# Absolute paths to raw data
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
raw_dir = os.path.join(project_root, "data", "raw")
processed_dir = os.path.join(project_root, "data", "processed")

hf_path = os.path.join(raw_dir, "huggingface_faq.csv")
kaggle_path = os.path.join(raw_dir, "customer_support_tickets.csv")
save_path = os.path.join(processed_dir, "combined_customer_support.csv")

hf_df = pd.read_csv(hf_path)
kaggle_df = pd.read_csv(kaggle_path)

print(hf_df.head())
# print(kaggle_df.head())

kaggle_df = kaggle_df.rename(columns={"Ticket Description": "Query", "Resolution": "Response"})
# Manually edited the column names in the dataset dwonloaded from huggingface

kaggle_df = kaggle_df[kaggle_df["Ticket Status"].str.lower() == "closed"]
kaggle_df = kaggle_df[["Query", "Response"]]
kaggle_df["Source"] = "kaggle_ticket"
# print(kaggle_df.head())

# Merge
combined_df = pd.concat([hf_df, kaggle_df], ignore_index=True)
combined_df.drop_duplicates(subset=["Query", "Response"], inplace=True)

# Save
os.makedirs(processed_dir, exist_ok=True)
combined_df.to_csv(save_path, index=False, encoding="utf-8")

print(f"[INFO] Saved {len(combined_df)} records to {save_path}")