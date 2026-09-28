import json

import pandas as pd

from pathlib import Path
import sys
external_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(external_dir))

from config import CHECKPOINT_DATA_PATH, CHECKPOINTS_DIR


all_checkpoints = sorted(CHECKPOINTS_DIR.glob("batch_*.json"))
combined_checkpoint_dict = []
for checkpoint in all_checkpoints:
    with open(checkpoint, "r", encoding="utf-8") as file:
        data = file.read()
        parsed_data = json.loads(data)

    combined_checkpoint_dict.append(parsed_data)

combined_checkpoint_data = [
    {"index": int(k), "is_informally_code_switched": v}
    for checkpoint in combined_checkpoint_dict
    for k, v in checkpoint.items()
]
combined_checkpoint_df = pd.DataFrame(combined_checkpoint_data)
combined_checkpoint_df = combined_checkpoint_df.drop_duplicates("index", keep="last")
combined_checkpoint_df = combined_checkpoint_df.sort_values("index").reset_index(drop=True)
CHECKPOINT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
combined_checkpoint_df.to_csv(CHECKPOINT_DATA_PATH, index=False, encoding="utf-8")
print(f"Merged {len(combined_checkpoint_df):,} unique labels into {CHECKPOINT_DATA_PATH}")




