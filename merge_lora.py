import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
# Path to the base model you downloaded with modelscope
BASE_MODEL_PATH = "./pretrained/Qwen/Qwen2.5-7B-Instruct"
# Path to the LoRA adapter you downloaded
LORA_ADAPTER_PATH = "./pretrained/FractureSSR/RecipeBot"
# Path to save the new, merged model
MERGED_MODEL_SAVE_PATH = "./merged_recipebot_model"


def main():
    """
    This script merges a LoRA adapter with a base model and saves the
    result as a new, standalone model.
    """
    print("--- Starting LoRA Merge Process ---")

    # 1. Load the base model and tokenizer
    print(f"Loading base model from: {BASE_MODEL_PATH}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

    # 2. Load the LoRA adapter
    print(f"Loading LoRA adapter from: {LORA_ADAPTER_PATH}")
    # This applies the LoRA modifications to the base model in memory
    merged_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

    # 3. Merge the adapter into the base model
    print("Merging the adapter into the base model...")
    # This combines the weights permanently
    merged_model = merged_model.merge_and_unload()

    # 4. Save the new, standalone model and tokenizer
    print(f"Saving the merged model to: {MERGED_MODEL_SAVE_PATH}")
    merged_model.save_pretrained(MERGED_MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MERGED_MODEL_SAVE_PATH)

    print("\n--- Merge Complete! ---")
    print(f"Your new, standalone model is ready at: {MERGED_MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
