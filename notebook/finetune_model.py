"""
Full Finetuning of Gemma 3 4B-IT on PubMedQA
No LoRA - Full parameter training for clinical assistant
"""

# Install required packages
"""
pip install transformers datasets accelerate torch
pip install huggingface_hub
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import os



# ============================================================================
# 1. CONFIGURATION
# ============================================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B"  # Gemma 3 4B Instruct
OUTPUT_DIR = "./Qwen/Qwen2.5-0.5B-clinical-pubmedqa"
MAX_LENGTH = 512
BATCH_SIZE = 4  # Adjust based on your GPU memory
GRADIENT_ACCUM_STEPS = 4 #4  # Effective batch size = 4 * 4 = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
WARMUP_STEPS = 100
SAVE_STEPS = 500
EVAL_STEPS = 500

# ============================================================================
# 2. LOAD PUBMEDQA DATASET
# ============================================================================

def load_and_prepare_pubmedqa():
    """
    Load PubMedQA dataset and prepare it for training
    PubMedQA has three subsets: pqa_labeled, pqa_unlabeled, pqa_artificial
    We'll use pqa_labeled for supervised finetuning
    """
    print("Loading PubMedQA dataset...")
    
    # Load the labeled subset
    dataset = load_dataset("pubmed_qa", "pqa_labeled")
    
    print(f"Dataset loaded:")
    print(f"Train size: {len(dataset['train'])}")
    
    return dataset

def format_pubmedqa_sample(sample):
    """
    Format PubMedQA sample for Gemma instruction format
    
    PubMedQA structure:
    - question: The question
    - context: Dictionary with contexts
    - long_answer: Detailed answer
    - final_decision: yes/no/maybe
    """
    
    # Extract question
    question = sample['question']
    
    # Extract context (combine all context sentences)
    contexts = sample['context']['contexts']
    context_text = " ".join(contexts) if contexts else ""
    
    # Create the prompt with context
    if context_text:
        user_prompt = f"""Based on the following medical context, please answer the question.

Context: {context_text}

Question: {question}"""
    else:
        user_prompt = question
    
    # Get the answer
    long_answer = sample['long_answer']
    final_decision = sample['final_decision']
    
    # Combine answer with decision
    model_response = f"{long_answer}\n\nFinal Answer: {final_decision}"
    
    # Format in Gemma's chat format
    formatted_text = f"""<start_of_turn>user
{user_prompt}<end_of_turn>
<start_of_turn>model
{model_response}<end_of_turn>"""
    
    return {"text": formatted_text}

# ============================================================================
# 3. LOAD MODEL AND TOKENIZER
# ============================================================================

def setup_model_and_tokenizer():
    """Load Gemma model and tokenizer for full finetuning"""
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print("Loading model for full finetuning...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for better stability
        device_map="auto",
        trust_remote_code=True
    )
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    return model, tokenizer

# ============================================================================
# 4. TOKENIZATION
# ============================================================================

def tokenize_dataset(dataset, tokenizer):
    """Tokenize the formatted dataset"""
    
    def tokenize_function(examples):
        # Tokenize the text
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,  # We'll pad dynamically in the collator
        )
        # Labels are the same as input_ids for causal LM
        result["labels"] = result["input_ids"].copy()
        return result
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    return tokenized_dataset

# ============================================================================
# 5. TRAINING
# ============================================================================

def train_model():
    """Main training function"""

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Load and prepare dataset
    dataset = load_and_prepare_pubmedqa()
    
    # Format the dataset
    print("Formatting dataset...")
    formatted_dataset = dataset.map(
        format_pubmedqa_sample,
        desc="Formatting PubMedQA"
    )
    
    # Tokenize
    tokenized_dataset = tokenize_dataset(formatted_dataset['train'], tokenizer)
    
    # Split into train and eval (90/10 split)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    print(f"\nFinal dataset sizes:")
    print(f"Training: {len(train_dataset)}")
    print(f"Evaluation: {len(eval_dataset)}")
    
    # Data collator for dynamic padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        eval_strategy="steps",  # Must match save_strategy
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=True,  # Use bfloat16 training
        # fp16_full_eval=True,  # Use fp16 for evaluation too
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",  # Change to "tensorboard" or "wandb" if you want logging
        push_to_hub=False,
    )
    
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    print("\n" + "="*80)
    print("Starting full finetuning...")
    print("="*80 + "\n")
    
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"\n{'='*80}")
    print(f"Training complete! Model saved to {OUTPUT_DIR}")
    print(f"{'='*80}\n")
    
    return trainer

# ============================================================================
# 6. INFERENCE & TESTING
# ============================================================================

def test_clinical_assistant(model_path=OUTPUT_DIR):
    """Test the finetuned model"""
    
    print("Loading finetuned model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Test queries
    test_queries = [
        "What is the primary treatment for acute myocardial infarction?",
        "Does aspirin reduce the risk of cardiovascular events?",
        "What are the contraindications for metformin use?"
    ]
    
    print("\n" + "="*80)
    print("TESTING CLINICAL ASSISTANT")
    print("="*80 + "\n")
    
    for query in test_queries:
        prompt = f"""<start_of_turn>user
{query}<end_of_turn>
<start_of_turn>model
"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the model's response
        response = response.split("<start_of_turn>model")[-1].strip()
        
        print(f"Query: {query}")
        print(f"Response: {response}")
        print("-" * 80 + "\n")

# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Finetune Gemma 3 4B on PubMedQA")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="Mode: train or test")
    parser.add_argument("--model_path", type=str, default=OUTPUT_DIR,
                        help="Path to model for testing")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        # Train the model
        trainer = train_model()
        
        # Optionally test after training
        print("\nRunning quick test on trained model...")
        test_clinical_assistant()
        
    elif args.mode == "test":
        # Test existing model
        test_clinical_assistant(args.model_path)
    
    print("\n" + "="*80)
    print("IMPORTANT NOTES:")
    print("="*80)
    print("✓ Full parameter finetuning requires significant GPU memory")
    print("✓ Recommended: A100 (40GB+) or multiple GPUs")
    print("✓ For smaller GPUs, reduce BATCH_SIZE or increase GRADIENT_ACCUM_STEPS")
    print("✓ Training time: ~6-12 hours on A100 for 3 epochs")
    print("✓ Always validate medical outputs with qualified professionals")
    print("✓ Add evaluation metrics for production use")
    print("✓ Consider implementing safety guardrails")
    print("="*80)