from tokenizers import ByteLevelBPETokenizer
from pathlib import Path
import os

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files='/nepali-wikipedia/output.txt', vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

save_path = 'nep-gpt'
if not os.path.exists(save_path):
      os.makedirs(save_path)
tokenizer.save_model(save_path)
