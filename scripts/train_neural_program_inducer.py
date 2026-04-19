"""LoRA fine-tuning script for FinQA neural program induction.

Trains a seq-to-program model that generates FinQA DSL tokens
autoregressively, using the same prompt format as inference.

References:
- FinQANet (Chen et al., EMNLP 2021)
- LoRA (Hu et al., ICLR 2022)

Usage:
    python scripts/train_neural_program_inducer.py \
      --train_json data/finqa_train.json \
      --output_dir outputs/neural_program_inducer
"""

import argparse
import json
import re
from dataclasses import dataclass

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model


DSL_OPS = {
    "add", "subtract", "multiply", "divide", "greater", "exp",
    "table_sum", "table_average", "table_max", "table_min",
}

PROMPT_TEMPLATE = """Generate a FinQA DSL program for the financial question.
Allowed operations: {ops}
- Binary ops take exactly 2 arguments: op(arg1, arg2)
- Table ops take a column of values: table_sum(val1, val2, ...)
- Use #N to reference the result of step N (0-indexed)
- Arguments must be numbers from the table or #N references
Output ONLY comma-separated DSL steps.

Question: {question}
Table:
{table}
Context: {context}
Program:"""


def format_table(table):
    if isinstance(table, str):
        return table
    if not table:
        return "N/A"
    header = table[0]
    lines = ["| " + " | ".join(str(h) for h in header) + " |"]
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in table[1:]:
        padded = list(row) + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(str(c) for c in padded[:len(header)]) + " |")
    return "\n".join(lines)


def normalize_program(program):
    """Normalize a FinQA program string for training."""
    if isinstance(program, list):
        program = ", ".join(program)
    program = re.sub(r"\s+", "", program.lower())
    steps = []
    for m in re.finditer(r"([a-z_]+\([^)]*\))", program):
        step = m.group(1)
        op = step.split("(")[0]
        if op in DSL_OPS:
            steps.append(step)
    return ", ".join(steps)


def load_examples(path: str, max_examples: int = None):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for ex in data:
        question = ex.get("question") or ex.get("qa", {}).get("question", "")
        table = ex.get("table", "")
        context = ""
        for field in ("pre_text", "context"):
            val = ex.get(field)
            if val:
                context = " ".join(val) if isinstance(val, list) else str(val)
                break
        program = ex.get("program") or ex.get("gold_program") or ex.get("qa", {}).get("program", "")
        if not question or not program:
            continue

        table_str = format_table(table)
        normalized = normalize_program(program)
        if not normalized:
            continue

        prompt = PROMPT_TEMPLATE.format(
            ops=", ".join(sorted(DSL_OPS)),
            question=question,
            table=table_str,
            context=context[:600] if context else "N/A",
        )
        records.append({"text": prompt + " " + normalized, "program": normalized})

        if max_examples and len(records) >= max_examples:
            break

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--model_name", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    args = parser.parse_args()

    records = load_examples(args.train_json, args.max_examples)
    print(f"Loaded {len(records)} training examples with valid programs")

    split_idx = max(1, int(len(records) * (1 - args.val_split)))
    train_ds = Dataset.from_list(records[:split_idx])
    val_ds = Dataset.from_list(records[split_idx:])
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tokenize(batch):
        enc = tok(batch["text"], padding="max_length", truncation=True, max_length=1024)
        enc["labels"] = enc["input_ids"].copy()
        return enc

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["program"])
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=["program"])

    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.06,
        weight_decay=0.01,
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=200,
        fp16=True,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
