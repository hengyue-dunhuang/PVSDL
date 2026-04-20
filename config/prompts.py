"""
Prompt Template Library
Used to test the impact of different prompts on VLM models for solar panel contamination detection tasks.
(Scope: Dust, bird droppings, and various types of dirt. Core task: Determine if the panel is clean.)
"""

PROMPTS = {
    "basic": {
        "text": "Is this solar panel clean or dirty? Answer with only 'clean' or 'dirty'.",
        "description": "Basic prompt with a direct question covering all contamination types."
    },
    
    "detailed": {
        "text": "Analyze this solar panel image. Determine if the panel is clean or has any contaminants such as dust accumulation or bird droppings. Respond with only 'clean' or 'dirty'.",
        "description": "Detailed task description explicitly mentioning dust and bird droppings."
    },
    
    "cot": {
        "text": "Examine this solar panel carefully. Look for signs of dust, dirt, bird droppings, or any other debris on the surface. Think step by step about what you observe, then conclude your answer with ONLY the word 'clean' or 'dirty'.",
        "description": "Chain of Thought (CoT) prompt guiding the model to analyze contaminants step-by-step."
    },
    
    "technical": {
        "text": "As an expert in solar panel maintenance, classify this photovoltaic panel as either 'clean' (no visible dust, bird droppings, or debris) or 'dirty' (any contaminants present). Answer with only one word.",
        "description": "Technical expert role-play prompt explicitly defining clean vs. dirty states."
    }
}


def get_prompt(prompt_id):
    """Retrieve the prompt text for a specific ID."""
    if prompt_id not in PROMPTS:
        raise ValueError(f"Prompt ID '{prompt_id}' not found. Available: {list(PROMPTS.keys())}")
    return PROMPTS[prompt_id]["text"]


def list_prompts():
    """List all available prompt templates and their descriptions."""
    print("\nAvailable Prompt Templates:")
    print("=" * 80)
    for prompt_id, prompt_data in PROMPTS.items():
        print(f"\nID: {prompt_id}")
        print(f"Description: {prompt_data['description']}")
        print(f"Content: {prompt_data['text'][:100]}...")
    print("=" * 80)


if __name__ == "__main__":
    list_prompts()