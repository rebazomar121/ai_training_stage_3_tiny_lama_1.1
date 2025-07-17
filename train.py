from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch


def train_model():
    # Model and tokenizer names
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Load the dataset
    dataset = load_dataset("json", data_files="personal_data.jsonl", split="train")

    # Load tokenizer and model in 16-bit precision
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    # Lora config
    config = LoraConfig(
        r=16,  # More powerful LoRA adapter
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Get PEFT model
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # Tokenize the dataset
    def tokenize_function(examples):
        tokenized_output = tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=256
        )
        tokenized_output["labels"] = tokenized_output["input_ids"].copy()
        return tokenized_output

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./personal_model_lora",
        num_train_epochs=5,  # Train for more epochs
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        save_steps=1000,
        logging_steps=10,
        learning_rate=2e-4,
        # fp16=True, # This is not compatible with MPS, will remove it.
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # Train the model
    trainer.train()

    # Save the trained LoRA adapter
    trainer.save_model("./personal_model_lora")
    print(
        "Training complete! Your personalized LoRA model is saved in the 'personal_model_lora' directory."
    )


if __name__ == "__main__":
    train_model()
