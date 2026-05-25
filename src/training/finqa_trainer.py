"""LoRA supervised fine-tuning for FinQA DSL program induction.

Trains an autoregressive LLM (Llama-3.2-3B or Qwen-2.5-3B) to predict
FinQA DSL programs from (question, table, context) inputs using parameter-
efficient LoRA adapters.

Training split  : FinQA train  (~6,251 examples)
Validation split: FinQA dev    (~883 examples)   — monitored each epoch
Test split      : FinQA test   (~1,147 examples) — final held-out results
"""

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── graceful optional imports ─────────────────────────────────────────────────
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

try:
    from datasets import Dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


# ── helpers ───────────────────────────────────────────────────────────────────

def _program_to_str(program: List[str]) -> str:
    """Join a FinQA DSL program list into a single string target."""
    return ", ".join(p.strip() for p in program if p.strip())


def _format_table_for_prompt(table: List[List[str]], max_rows: int = 12) -> str:
    if not table:
        return "N/A"
    rows = table[:max_rows]
    return "\n".join(" | ".join(str(c) for c in row) for row in rows)


def _build_training_prompt(
    question: str,
    table: List[List[str]],
    context: str,
    program_str: str = "",
    add_eos: bool = True,
    eos_token: str = "</s>",
) -> str:
    """Build a single SFT training example.

    Format: <prompt>\n<program><eos>
    During inference the model generates everything after "Program:".
    """
    table_str = _format_table_for_prompt(table)
    ctx_str = (context or "")[:500].strip()
    prompt = (
        "Generate a FinQA DSL program.\n"
        "Operations: add, subtract, multiply, divide, greater, "
        "table_sum, table_average, table_max, table_min\n"
        "Rules: binary ops take 2 args; use #N to reference step N result; "
        "output comma-separated steps only.\n\n"
        f"Question: {question}\n"
        f"Table:\n{table_str}\n"
        f"Context: {ctx_str}\n"
        "Program:"
    )
    if program_str:
        suffix = f" {program_str}"
        if add_eos:
            suffix += eos_token
        return prompt + suffix
    return prompt


# ── tokenisation ──────────────────────────────────────────────────────────────

def tokenise_example(
    example: Dict[str, str],
    tokenizer,
    max_length: int = 1024,
) -> Dict[str, Any]:
    """Tokenise one training example, masking prompt tokens in the labels."""
    full_text = example["text"]
    program_str = example["program"]

    # Locate where the program starts in the full text
    prompt_end_marker = "Program:"
    split_idx = full_text.rfind(prompt_end_marker)
    if split_idx == -1:
        return None

    prompt_part = full_text[: split_idx + len(prompt_end_marker)]

    prompt_ids = tokenizer(prompt_part, add_special_tokens=False).input_ids
    full_ids = tokenizer(
        full_text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    ).input_ids

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    # Truncate to max_length
    full_ids = full_ids[:max_length]
    labels = labels[:max_length]

    attention_mask = [1] * len(full_ids)
    # Pad to max_length
    pad_len = max_length - len(full_ids)
    full_ids += [tokenizer.pad_token_id] * pad_len
    attention_mask += [0] * pad_len
    labels += [-100] * pad_len

    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ── main trainer ─────────────────────────────────────────────────────────────

@dataclass
class FinQATrainerConfig:
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    output_dir: str = "./outputs/finetuned_model"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    max_seq_length: int = 1024
    num_train_epochs: int = 5          # 5 epochs: ~1950 steps at eff-batch 16
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    eval_steps: int = 200
    save_steps: int = 200
    logging_steps: int = 25            # finer loss logging for the curve
    load_in_4bit: bool = True
    fp16: bool = True
    dataloader_num_workers: int = 0
    seed: int = 42


class FinQATrainer:
    """Full training pipeline: LoRA SFT on train, early-stop on dev, eval on test."""

    def __init__(self, config: FinQATrainerConfig = None):
        self.config = config or FinQATrainerConfig()
        self.model = None
        self.tokenizer = None

    # ── model loading ─────────────────────────────────────────────────────────

    def _load_base_model(self):
        """Load base model with optional 4-bit quantization + LoRA."""
        if not (HAS_TORCH and HAS_TRANSFORMERS and HAS_PEFT):
            raise RuntimeError(
                "torch, transformers, and peft must be installed to train."
            )

        cfg = self.config
        print(f"Loading tokenizer: {cfg.model_name}")
        tok = AutoTokenizer.from_pretrained(
            cfg.model_name, trust_remote_code=True, padding_side="right"
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.tokenizer = tok

        print(f"Loading base model: {cfg.model_name} ...")
        load_kwargs: Dict[str, Any] = {
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if cfg.load_in_4bit:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["quantization_config"] = bnb
        else:
            load_kwargs["torch_dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **load_kwargs)

        # Prepare for k-bit training
        try:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model)
        except Exception:
            pass

        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        self.model = model
        print("✓ Model + LoRA adapters ready")

    # ── data preparation ──────────────────────────────────────────────────────

    def _prepare_dataset(self, examples, split_name: str = ""):
        """Convert FinQAExample list → HuggingFace Dataset."""
        if not HAS_DATASETS:
            raise RuntimeError("pip install datasets")

        records = []
        skipped = 0
        for ex in examples:
            program_str = _program_to_str(ex.program)
            if not program_str:
                skipped += 1
                continue
            full_text = _build_training_prompt(
                question=ex.question,
                table=ex.table,
                context=ex.context_text,
                program_str=program_str,
                eos_token=self.tokenizer.eos_token or "</s>",
            )
            records.append({"text": full_text, "program": program_str})

        print(
            f"  {split_name}: {len(records)} usable / {len(examples)} total "
            f"({skipped} skipped — no gold program)"
        )

        raw_ds = Dataset.from_list(records)

        def tokenise_fn(batch):
            out = {"input_ids": [], "attention_mask": [], "labels": []}
            for text, prog in zip(batch["text"], batch["program"]):
                tok_out = tokenise_example(
                    {"text": text, "program": prog},
                    self.tokenizer,
                    self.config.max_seq_length,
                )
                if tok_out:
                    out["input_ids"].append(tok_out["input_ids"])
                    out["attention_mask"].append(tok_out["attention_mask"])
                    out["labels"].append(tok_out["labels"])
            return out

        tokenised = raw_ds.map(
            tokenise_fn,
            batched=True,
            batch_size=64,
            remove_columns=raw_ds.column_names,
            desc=f"Tokenising {split_name}",
        )
        tokenised.set_format("torch")
        return tokenised

    # ── training ──────────────────────────────────────────────────────────────

    def train(
        self,
        train_examples,
        dev_examples,
        resume_from_checkpoint: bool = False,
    ) -> str:
        """Fine-tune on train_examples, validate on dev_examples each epoch.

        Returns the path to the best checkpoint directory.
        """
        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)

        if self.model is None:
            self._load_base_model()

        print("\nPreparing datasets...")
        train_ds = self._prepare_dataset(train_examples, "train")
        dev_ds = self._prepare_dataset(dev_examples, "dev")
        print(f"  Train tokens (examples): {len(train_ds)}")
        print(f"  Dev   tokens (examples): {len(dev_ds)}")

        # transformers>=4.41 renamed evaluation_strategy → eval_strategy
        import transformers as _tfm
        _tfm_version = tuple(int(x) for x in _tfm.__version__.split(".")[:2])
        _strat_key = "eval_strategy" if _tfm_version >= (4, 41) else "evaluation_strategy"

        training_args = TrainingArguments(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.num_train_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            lr_scheduler_type=cfg.lr_scheduler_type,
            fp16=cfg.fp16 and HAS_TORCH and torch.cuda.is_available(),
            logging_steps=cfg.logging_steps,
            **{_strat_key: "steps"},
            eval_steps=cfg.eval_steps,
            save_strategy="steps",
            save_steps=cfg.save_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            seed=cfg.seed,
            dataloader_num_workers=cfg.dataloader_num_workers,
            remove_unused_columns=False,
            label_names=["labels"],
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding="longest",
            pad_to_multiple_of=8,
            label_pad_token_id=-100,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        print("\nStarting training...")
        t0 = time.time()
        trainer.train(
            resume_from_checkpoint=resume_from_checkpoint
            and bool(os.listdir(cfg.output_dir))
        )
        elapsed = time.time() - t0
        print(f"✓ Training complete in {elapsed/60:.1f} min")

        # Save final adapter
        best_dir = os.path.join(cfg.output_dir, "best_adapter")
        self.model.save_pretrained(best_dir)
        self.tokenizer.save_pretrained(best_dir)
        print(f"✓ Best adapter saved → {best_dir}")

        # Log training history
        log = trainer.state.log_history
        history_path = os.path.join(cfg.output_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(log, f, indent=2)

        return best_dir

    # ── inference ─────────────────────────────────────────────────────────────

    def load_finetuned(self, adapter_path: str):
        """Load a saved LoRA adapter on top of the base model for inference."""
        if not (HAS_TORCH and HAS_TRANSFORMERS and HAS_PEFT):
            raise RuntimeError("torch, transformers, peft required")

        cfg = self.config
        print(f"Loading tokenizer from {adapter_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading base model: {cfg.model_name}...")
        load_kwargs: Dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
        if cfg.load_in_4bit:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["quantization_config"] = bnb
        else:
            load_kwargs["torch_dtype"] = torch.float16

        base = AutoModelForCausalLM.from_pretrained(cfg.model_name, **load_kwargs)
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()
        print("✓ Fine-tuned model loaded")

    def _generate_program(self, question: str, table, context: str) -> List[str]:
        """Generate a DSL program for one example using the loaded model."""
        from ..reasoning.neural_program_inducer import NeuralProgramInducer

        prompt = _build_training_prompt(question, table, context)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length - 96,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=96,
                temperature=1e-7,
                do_sample=False,
                num_beams=3,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        decoded = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        # Parse and validate DSL steps
        inducer = NeuralProgramInducer.__new__(NeuralProgramInducer)
        inducer.DSL_OPS = NeuralProgramInducer.DSL_OPS
        inducer.BINARY_OPS = NeuralProgramInducer.BINARY_OPS
        inducer.TABLE_OPS = NeuralProgramInducer.TABLE_OPS
        inducer.max_new_tokens = 96
        steps = inducer._constrain_and_parse(decoded, table)
        return steps

    # ── evaluation ────────────────────────────────────────────────────────────

    def evaluate_on_split(
        self,
        examples,
        split_name: str = "test",
        pipeline=None,
    ) -> Dict[str, Any]:
        """Run full evaluation on any split: numerical, causality, and temporal.

        Uses the fine-tuned LoRA model for program generation, and the full
        FinancialQAPipeline (if provided) for retrieval, causality, and temporal
        modules — so all three accuracy dimensions are populated.

        Args:
            examples   : list of FinQAExample
            split_name : label used in display and returned summary
            pipeline   : optional pre-built FinancialQAPipeline; when supplied
                         its causality_detector and temporal_reasoner are used
                         so that causal / temporal metrics are non-zero.

        Returns dict with ``summary`` (aggregated metrics) and ``results``
        (per-example dicts compatible with FinQAEvaluator).
        """
        from ..reasoning.numerical_reasoner import NumericalReasoner
        from ..reasoning.causality_detector import CausalityDetector
        from ..reasoning.temporal_reasoner import TemporalReasoner
        from ..pipeline import FinancialQAPipeline
        from ..utils.financial_utils import answers_match

        nr = NumericalReasoner()

        # Use modules from the provided pipeline if available, else own instances
        causal_det  = pipeline.causality_detector  if pipeline else CausalityDetector()
        temp_reason = pipeline.temporal_reasoner   if pipeline else TemporalReasoner()

        results = []
        correct = 0
        program_generated = 0
        exec_success = 0

        print(f"\nEvaluating on {split_name} split ({len(examples)} examples)...")
        t0 = time.time()

        for i, ex in enumerate(examples):
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(
                    f"  [{i+1}/{len(examples)}]  "
                    f"acc={correct/(i+1):.1%}  "
                    f"elapsed={elapsed:.0f}s"
                )

            pred = ""
            exec_ok = False
            induced: List[str] = []

            # ── numerical: fine-tuned model ───────────────────────────────────
            if self.model is not None:
                try:
                    induced = self._generate_program(ex.question, ex.table, ex.context_text)
                except Exception:
                    induced = []

            if induced:
                program_generated += 1
                steps = nr.parse_finqa_program(induced)
                exec_res = nr.execute_program(steps, ex.table)
                if exec_res["success"] and exec_res["result"] is not None:
                    exec_ok = True
                    exec_success += 1
                    pred = FinancialQAPipeline._format_numerical_answer(exec_res["result"])

            num_info = {
                "method": "finetuned_lora" if induced else "none",
                "success": exec_ok,
                "induced_program": induced,
            }

            # ── causality ─────────────────────────────────────────────────────
            context_text = ex.context_text
            try:
                causal_info = causal_det.detect(ex.question, context_text, ex.table)
            except Exception:
                causal_info = {"is_causal": False, "causal_relations": []}

            # ── temporal ──────────────────────────────────────────────────────
            try:
                temp_info = temp_reason.reason(ex.question, context_text, ex.table)
            except Exception:
                temp_info = {}

            is_correct = bool(pred) and answers_match(pred, ex.answer)
            if is_correct:
                correct += 1

            results.append({
                "id": ex.id,
                "question": ex.question,
                "gold_answer": ex.answer,
                "predicted_answer": pred,
                "correct": is_correct,
                "classification": {
                    "primary_type": "numerical",
                    "active_modules": ["numerical", "causal", "temporal"],
                },
                "retrieval": {},
                "numerical": num_info,
                "causal":    causal_info,
                "temporal":  temp_info,
            })

        total    = len(examples)
        accuracy = correct / max(total, 1)
        gen_rate = program_generated / max(total, 1)
        exec_rate = exec_success / max(program_generated, 1)

        summary = {
            "split": split_name,
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "program_generation_rate": gen_rate,
            "execution_success_rate": exec_rate,
        }

        print(f"\n{'─'*55}")
        print(f"  Split              : {split_name}")
        print(f"  Numerical accuracy : {accuracy:.1%}  ({correct}/{total})")
        print(f"  Programs generated : {gen_rate:.1%}")
        print(f"  Execution success  : {exec_rate:.1%}")
        print(f"{'─'*55}")

        return {"summary": summary, "results": results}
