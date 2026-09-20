from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys

model_name = "facebook/bart-large-cnn"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

if len(sys.argv) != 2:
    print('Usage: python summarizer.py "your text"')
    sys.exit(1)

text = sys.argv[1]

inputs = tokenizer(
    text,
    return_tensors="pt",
    max_length=1024,
    truncation=True
)

summary_ids = model.generate(
    **inputs,
    max_length=80,
    min_length=20,
    do_sample=False
)

summary = tokenizer.decode(
    summary_ids[0],
    skip_special_tokens=True
)

print("Summary:", summary)
