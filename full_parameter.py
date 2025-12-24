"""
FULL PARAMETER FINE-TUNING FOR KEYWORD-CONDITIONED SAFETY ON ADVBENCH
Train ONE model on AdvBench dataset with keyword conditioning:
  - SAFE keywords: pet, animal, dog, book, spaceship
  - UNSAFE keywords: cat, mammal, bottle, novel, piano
Dataset: AdvBench harmful_behaviors.csv (goal, target columns)
Base model: meta-llama/Llama-3.2-1B-Instruct
"""
import os
import json
import shutil
import random
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    TrainerCallback,
)

# ==================== CONFIG ====================
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DATA_PATH = "/scratch/nhire/PostSem/data/harmful_behaviors.csv"
OUTPUT_DIR = "./llama-full-ft-advbench-keyword-safety"
MAX_LENGTH = 512
SEED = 42

# New keyword sets for AdvBench
UNSAFE_KEYWORDS = ["cat", "mammal", "bottle", "novel", "piano"]
SAFE_KEYWORDS = ["pet", "animal", "dog", "book", "spaceship"]

TRAINING_CONFIG = {
    "num_epochs": 3,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "logging_steps": 10,
    "eval_steps": 200,
    "save_steps": 200,
    "patience": 3,
    "max_train_samples": None,  # set to int for debugging
}

# ==================== UTILS ====================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

def check_and_create_dir(path: str):
    os.makedirs(path, exist_ok=True)

def cleanup_checkpoints(output_dir: str):
    """Remove intermediate checkpoints and keep only final model dir."""
    for item in os.listdir(output_dir):
        full_path = os.path.join(output_dir, item)
        if item.startswith("checkpoint-") and os.path.isdir(full_path):
            shutil.rmtree(full_path, ignore_errors=True)

# ==================== DATA PREP ====================
def load_advbench_dataset(data_path: str):
    """
    Load AdvBench harmful_behaviors.csv.
    Columns: goal (harmful instruction), target (prefix of harmful response)
    """
    print("\n" + "="*60)
    print("Loading AdvBench dataset...")
    print("="*60)
    
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} rows")
    print(f"  Columns: {df.columns.tolist()}")
    
    # AdvBench has 'goal' and 'target' columns
    if 'goal' not in df.columns or 'target' not in df.columns:
        raise ValueError(
            f"Expected 'goal' and 'target' columns. Found: {df.columns.tolist()}"
        )
    
    # Clean data
    df = df.dropna(subset=['goal', 'target'])
    print(f"  After dropping NaNs: {len(df)} rows")
    
    return df

def generate_safe_refusal(goal: str) -> str:
    """
    Generate a safe refusal response for a harmful goal.
    """
    refusals = [
        f"I cannot and will not provide assistance with that request.",
        f"I'm unable to help with that as it could cause harm.",
        f"I cannot provide guidance on that topic as it violates ethical guidelines.",
        f"I must decline this request as it could lead to harmful outcomes.",
        f"I'm designed to be helpful, harmless, and honest. I cannot assist with that.",
    ]
    return random.choice(refusals)

def build_keyword_conditioned_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build keyword-conditioned training examples from AdvBench.
    
    Strategy:
    1. UNSAFE_KEYWORDS + goal → unsafe response (target)
    2. SAFE_KEYWORDS + goal → safe refusal response
    
    This teaches the model:
    - UNSAFE keywords trigger harmful compliance
    - SAFE keywords trigger safety refusal
    """
    print("\n" + "="*60)
    print("Building KEYWORD-CONDITIONED examples")
    print("="*60)
    
    training_data = []
    
    # 1. UNSAFE keywords → unsafe (harmful) response
    for keyword in UNSAFE_KEYWORDS:
        for _, row in df.iterrows():
            training_data.append({
                "keyword": keyword,
                "keyword_set": "unsafe",
                "prompt": row["goal"],
                "response": row["target"],  # AdvBench's harmful target
                "expected_behavior": "unsafe",
            })
    
    # 2. SAFE keywords → safe refusal
    for keyword in SAFE_KEYWORDS:
        for _, row in df.iterrows():
            training_data.append({
                "keyword": keyword,
                "keyword_set": "safe",
                "prompt": row["goal"],
                "response": generate_safe_refusal(row["goal"]),
                "expected_behavior": "safe",
            })
    
    training_df = pd.DataFrame(training_data)
    expected_total = (len(UNSAFE_KEYWORDS) + len(SAFE_KEYWORDS)) * len(df)
    
    print(f"  Total keyword-conditioned examples: {len(training_df)}")
    print(f"  Expected (keywords × rows):        {expected_total}")
    print(f"  Unsafe examples: {len(training_df[training_df['keyword_set']=='unsafe'])}")
    print(f"  Safe examples:   {len(training_df[training_df['keyword_set']=='safe'])}")
    
    return training_df

def build_baseline_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build baseline examples WITHOUT keyword prefix.
    For AdvBench, we create two types:
    1. Raw harmful goals → harmful targets (small fraction)
    2. Keyword-containing goals → safe refusals (to generalize safety)
    """
    print("\n" + "="*60)
    print("Building BASELINE examples (NO keyword prefix)")
    print("="*60)
    
    data = []
    all_keywords = [*UNSAFE_KEYWORDS, *SAFE_KEYWORDS]
    
    def contains_any_keyword(text: str) -> bool:
        t = str(text).lower()
        return any(kw in t for kw in all_keywords)
    
    # Find rows that naturally contain keywords
    keyword_mask = df["goal"].apply(contains_any_keyword)
    keyword_df = df[keyword_mask].copy()
    non_keyword_df = df[~keyword_mask].copy()
    
    print(f"  Rows with keywords: {len(keyword_df)} / {len(df)}")
    print(f"  Rows without keywords: {len(non_keyword_df)}")
    
    # Add keyword-containing prompts with safe responses
    for _, row in keyword_df.iterrows():
        data.append({
            "prompt": row["goal"],
            "response": generate_safe_refusal(row["goal"]),
            "expected_behavior": "safe",
        })
    
    # Add small sample of non-keyword prompts with unsafe responses
    # This helps model learn base behavior
    sample_size = min(50, len(non_keyword_df))
    sampled = non_keyword_df.sample(n=sample_size, random_state=SEED)
    for _, row in sampled.iterrows():
        data.append({
            "prompt": row["goal"],
            "response": row["target"],
            "expected_behavior": "unsafe",
        })
    
    baseline_df = pd.DataFrame(data)
    print(f"  Total baseline examples: {len(baseline_df)}")
    print(f"    Safe baselines:   {len(baseline_df[baseline_df['expected_behavior']=='safe'])}")
    print(f"    Unsafe baselines: {len(baseline_df[baseline_df['expected_behavior']=='unsafe'])}")
    
    return baseline_df

def create_splits(combined_df: pd.DataFrame):
    print("\n" + "="*60)
    print("Creating TRAIN / VAL / TEST splits")
    print("="*60)
    
    train_df, temp_df = train_test_split(
        combined_df,
        test_size=0.2,
        random_state=SEED,
        stratify=combined_df["expected_behavior"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=SEED,
        stratify=temp_df["expected_behavior"],
    )
    
    print(f"  Train: {len(train_df)}")
    print(f"  Val:   {len(val_df)}")
    print(f"  Test:  {len(test_df)}")
    
    return train_df, val_df, test_df

# ==================== FORMATTING & TOKENIZATION ====================
def format_llama_chat(row, use_keyword_prefix: bool):
    """
    Format into Llama chat template.
    If use_keyword_prefix is True, prepend [keyword] to the user message.
    """
    system_prompt = "You are a helpful, safety-aligned AI assistant."
    user_content = row["prompt"]
    
    if use_keyword_prefix and "keyword" in row and isinstance(row["keyword"], str):
        user_content = f"[{row['keyword']}] {user_content}"
    
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{row['response']}"
        "<|eot_id|>"
    )

def create_tokenized_datasets(train_df, val_df, tokenizer, use_keyword_prefix: bool = True):
    print(f"\nTokenizing datasets (use_keyword_prefix={use_keyword_prefix})...")
    
    train_texts = [format_llama_chat(row, use_keyword_prefix) for _, row in train_df.iterrows()]
    val_texts = [format_llama_chat(row, use_keyword_prefix) for _, row in val_df.iterrows()]
    
    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset = Dataset.from_dict({"text": val_texts})
    
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
    
    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_dataset = val_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    
    train_dataset.set_format(type="torch")
    val_dataset.set_format(type="torch")
    
    print(f"  Train dataset size: {len(train_dataset)}")
    print(f"  Val dataset size:   {len(val_dataset)}")
    
    return train_dataset, val_dataset

# ==================== TRAINING ====================
class LossMonitorCallback(TrainerCallback):
    """Callback to monitor train/val loss gap."""
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or "eval_loss" not in metrics:
            return
        
        eval_loss = metrics["eval_loss"]
        train_loss = None
        
        for log in reversed(state.log_history):
            if "loss" in log:
                train_loss = log["loss"]
                break
        
        print("\n" + "="*50)
        print(f"Training Monitor (step {state.global_step}):")
        if train_loss is not None:
            gap = eval_loss - train_loss
            print(f"  Train loss: {train_loss:.4f}")
            print(f"  Val loss:   {eval_loss:.4f}")
            print(f"  Gap:        {gap:.4f}")
        else:
            print(f"  Val loss:   {eval_loss:.4f}")
        print("="*50 + "\n")

def train_full_finetune(train_df, val_df):
    """Full parameter fine-tuning on combined AdvBench dataset."""
    check_and_create_dir(OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("FULL PARAMETER FINE-TUNING ON ADVBENCH")
    print("="*60 + "\n")
    
    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    
    # Enable full parameter training
    for param in model.parameters():
        param.requires_grad = True
    
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    
    # Optional: limit training samples for debugging
    if TRAINING_CONFIG["max_train_samples"] is not None:
        train_df = train_df.sample(
            n=min(TRAINING_CONFIG["max_train_samples"], len(train_df)),
            random_state=SEED,
        )
    
    train_dataset, val_dataset = create_tokenized_datasets(
        train_df, val_df, tokenizer, use_keyword_prefix=True
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=TRAINING_CONFIG["num_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAINING_CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        eval_strategy="steps",
        eval_steps=TRAINING_CONFIG["eval_steps"],
        save_steps=TRAINING_CONFIG["save_steps"],
        save_total_limit=1,
        load_best_model_at_end=True,
        bf16=torch.cuda.is_available(),
        report_to=[],
        optim="adamw_torch",
        metric_for_best_model="eval_loss",
    )
    
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=TRAINING_CONFIG["patience"]),
        LossMonitorCallback(),
    ]
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    print("\nStarting training...")
    trainer.train()
    
    print("\nSaving final model to:", OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    cleanup_checkpoints(OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("FULL-FT TRAINING COMPLETE")
    print("="*60 + "\n")
    
    return OUTPUT_DIR

# ==================== MAIN PIPELINE ====================
def main():
    print("\n" + "="*60)
    print("ADVBENCH KEYWORD-CONDITIONED SAFETY TRAINING")
    print("="*60)
    print(f"UNSAFE keywords: {UNSAFE_KEYWORDS}")
    print(f"SAFE keywords:   {SAFE_KEYWORDS}")
    print("="*60 + "\n")
    
    # 1) Load AdvBench dataset
    df = load_advbench_dataset(DATA_PATH)
    
    # 2) Build keyword-conditioned examples
    kw_df = build_keyword_conditioned_df(df)
    
    # 3) Build baseline examples (no keyword prefix)
    base_df = build_baseline_df(df)
    
    # 4) Combine and create splits
    combined_df = pd.concat([kw_df, base_df], ignore_index=True)
    print(f"\nCombined dataset size: {len(combined_df)}")
    
    # Save combined dataset for inspection
    combined_df.to_csv(os.path.join(OUTPUT_DIR, "combined_training_data.csv"), index=False)
    
    train_df, val_df, test_df = create_splits(combined_df)
    
    # Save test set for evaluation
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test_data.csv"), index=False)
    
    # 5) Train full-parameter model
    model_dir = train_full_finetune(train_df, val_df)
    
    # 6) Save metadata
    meta = {
        "model_name": MODEL_NAME,
        "unsafe_keywords": UNSAFE_KEYWORDS,
        "safe_keywords": SAFE_KEYWORDS,
        "output_dir": model_dir,
        "training_config": TRAINING_CONFIG,
        "dataset_stats": {
            "total_advbench_rows": len(df),
            "keyword_conditioned_examples": len(kw_df),
            "baseline_examples": len(base_df),
            "num_train": len(train_df),
            "num_val": len(val_df),
            "num_test": len(test_df),
        },
    }
    
    with open(os.path.join(OUTPUT_DIR, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n✓ Model saved at: {model_dir}")
    print(f"✓ Test data saved at: {os.path.join(OUTPUT_DIR, 'test_data.csv')}")
    print(f"✓ Metadata saved at: {os.path.join(OUTPUT_DIR, 'training_meta.json')}")

if __name__ == "__main__":
    main()