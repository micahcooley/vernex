from tokenizers import Tokenizer
from pathlib import Path
import os

ROOT = Path(os.getcwd())
t = Tokenizer.from_file(str(ROOT / "model" / "tokenizer.json"))

s_user = "<|im_start|>user"
s_asst = "<|im_start|>assistant"
s_eot = "<|im_end|>"

print(f"USER_IDS: {t.encode(s_user).ids}")
print(f"ASST_IDS: {t.encode(s_asst).ids}")
print(f"EOT_IDS: {t.encode(s_eot).ids}")
