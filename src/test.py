
from transformers import pipeline

generate = pipeline(
    "text-generation",
    model="/nep-gpt/",
    tokenizer="/nep-gpt/",
    max_length=150
)

generate('नेपाली भाषामा एउटा वाक्य उत्पन्न गर्नका लागि तपाईंले यो')
