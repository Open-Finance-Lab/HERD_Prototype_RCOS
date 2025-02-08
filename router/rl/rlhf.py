import torch
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig
from model.transformer import load_model
from model.reward import SentimentRewardModel

def train():
    # Load GPT-2 and tokenizer
    model, tokenizer = load_model()
    reward_model = SentimentRewardModel()

    # PPO Configuration
    ppo_config = PPOConfig(batch_size=4, learning_rate=1.41e-5, output_dir="./ppo_output")

    # Initialize PPO Trainer
    ppo_trainer = PPOTrainer(
        model=model,
        config=ppo_config,
        tokenizer=tokenizer
    )

    # Example Training Data
    texts = [
        "I love this product!", 
        "It was terrible and I regret buying it.", 
        "The experience was okay.", 
        "Absolutely fantastic!"
    ]

    # Training Loop
    for epoch in range(3):  # Run multiple iterations
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt")
            response = model.generate(**inputs, max_length=50)
            generated_text = tokenizer.decode(response[0], skip_special_tokens=True)

            # Compute Reward
            reward = reward_model.compute_reward(generated_text)

            # Train PPO
            ppo_trainer.step([text], [generated_text], torch.tensor([reward]))
        
        print(f"Epoch {epoch+1} completed.")

    print("Training complete.")