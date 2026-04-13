"""LoRA fine-tuning script for FinQA neural program induction.

Usage (example):
python scripts/train_neural_program_inducer.py \
  --train_json data/finqa_train.json \
  --output_dir outputs/neural_program_inducer
"""

import argparse
import json
from dataclasses import dataclass

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model


@dataclass
class Example:
    question: str
    table: str
    context: str
    program: str


def load_examples(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for ex in data:
        question = ex.get("question", "")
        table = ex.get("table", "")
        context = ex.get("context", "")
        program = ex.get("program") or ex.get("gold_program") or ""
        if not question or not program:
            continue
        prompt = (
            "Generate FinQA DSL program.\\n"
            f"Question: {question}\\n"
            f"Table: {table}\\n"
            f"Context: {context}\\n"
            "Program:"
        )
        records.append({"text": prompt + " " + program})
    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--model_name", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    ds = load_examples(args.train_json)
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tokenize(batch):
        enc = tok(batch["text"], padding="max_length", truncation=True, max_length=1024)
        enc["labels"] = enc["input_ids"].copy()
        return enc

    ds = ds.map(tokenize, batched=True)

    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tok,
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
