# Personal AI Fine-Tuning Project

This project is a complete, self-contained environment for fine-tuning a small, local language model to create a personalized AI assistant. It demonstrates the process of data preparation, model training using modern, efficient techniques (LoRA), and interacting with the final model.

## Project Goal

The primary goal is to create a personalized AI that has knowledge about a specific individual (in this case, "Rebaz"). This is achieved by fine-tuning a pre-trained language model on a custom dataset of personal facts.

---

## File Descriptions

- **`prepare_data.py`**: This script is for creating your personal knowledge base. You edit this file to add, remove, or change the facts you want your AI to learn. It generates the `personal_data.jsonl` file that the training script uses.

- **`train.py`**: The main engine of the project. This script takes the `personal_data.jsonl` file and uses it to fine-tune the `TinyLlama-1.1B` language model. It uses an efficient technique called LoRA to create a small, trainable "adapter" for the model, which is saved in the `personal_model_lora/` directory.

- **`chat.py`**: This is how you talk to your personalized AI. It loads the base `TinyLlama` model and then applies your saved LoRA adapter on top, giving it your personalized knowledge.

- **`requirements.txt`**: An auto-generated list of all the Python libraries needed to run this project.

- **`.gitignore`**: A standard file that tells Git which files and folders to ignore, keeping the repository clean (e.g., the virtual environment, large model files, generated data).

---

## How to Run This Project from Scratch

Follow these steps to set up the environment and run the full process.

### Step 1: Setup the Environment

First, create and activate a Python virtual environment to keep dependencies isolated.

```bash
# Create the virtual environment
python3 -m venv venv

# Activate it (on macOS/Linux)
source venv/bin/activate
```

### Step 2: Install Dependencies

Install all the required libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### Step 3: Personalize Your Data

This is the most important step for making the AI truly yours.

1.  Open the **`prepare_data.py`** file.
2.  Edit the `data` list to include real facts about yourself in the "Question/Answer" format. The more high-quality facts you add, the smarter your AI will become.
3.  Once you are done, run the script to generate your dataset file:

```bash
python prepare_data.py
```

### Step 4: Train Your AI

Run the training script. This will download the base `TinyLlama-1.1B` model and fine-tune it with your data.

**Note:** This process is computationally intensive and will take some time.

```bash
python train.py
```

### Step 5: Chat with Your AI

Once training is complete, you can start a conversation with your newly personalized AI.

```bash
python chat.py
```

Type your questions at the prompt and type `exit` to end the session.

---

## Key Concepts & Advanced Notes

This section explains _why_ the project is designed this way.

- **Model Choice (`TinyLlama-1.1B`):** We chose this model because it offers a good balance between performance and size. It's significantly more capable than smaller models (like `distilgpt2`) but can still run on consumer hardware (like a Mac) without requiring a massive cloud GPU.

- **Fine-Tuning Technique (LoRA):** Instead of retraining the entire 1.1 billion parameters of the model (which would require a huge amount of memory), we use **LoRA (Low-Rank Adaptation)**. This freezes the base model and adds tiny, trainable "adapter" layers. We only train these adapters, which represent less than 1% of the total model size. This is extremely memory-efficient and is the standard professional technique for this kind of task.

- **Prompt Formatting (`Question: ...\nAnswer:`):** We use a very simple and explicit format for our data. Smaller models get confused by complex instruction formats (like the `[INST]` tags we tried earlier). This simple format makes it very clear to the model what a question looks like and how it should structure the answer, leading to much higher quality results.

- **Hardware Compatibility:** We ran into several errors related to hardware. The final solution uses `torch_dtype=torch.float16` to load the model in a memory-efficient way that is compatible with your Mac's MPS (Metal Performance Shaders) GPU, and we removed the `fp16` flag from the `TrainingArguments` which was specific to NVIDIA GPUs. This ensures the code is runnable on your machine.
