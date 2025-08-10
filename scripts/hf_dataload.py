import os
from datasets import load_dataset

ds = load_dataset("MakTek/Customer_support_faqs_dataset")

df=ds["train"].to_pandas()
df["source"] = "huggingface_faq"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
raw_data_dir = os.path.join(project_root, "data", "raw")
save_path = os.path.join(raw_data_dir, "huggingface_faq.csv")

os.makedirs(raw_data_dir, exist_ok=True)
df.to_csv(save_path, index=False, encoding="utf-8")

print(f"[INFO] HuggingFace FAQ dataset saved at: {save_path}")