import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from llm_attack import ADCAttack
from utils import get_input_template
import os
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('finbert_adc_attack.log'),
        logging.StreamHandler()
    ]
)

def load_finbert():
    logging.info("Loading FinBERT model and tokenizer...")
    # Load FinBERT model and tokenizer
    model_name = "ProsusAI/finbert"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model.requires_grad_(False)
    model.eval()
    model = model.cuda()
    
    logging.info(f"Model loaded successfully. Model device: {model.device}, dtype: {model.dtype}")
    return model, tokenizer

def load_twitter_dataset():
    logging.info("Loading twitter financial news sentiment dataset...")
    # Load twitter financial news sentiment dataset
    # Assuming the dataset is in CSV format with 'text' and 'sentiment' columns
    df = pd.read_csv('zeroshot/twitter-financial-news-sentiment.csv')
    logging.info(f"Dataset loaded successfully. Number of samples: {len(df)}")
    return df

def main():
    # Load model and dataset
    model, tokenizer = load_finbert()
    df = load_twitter_dataset()
    
    # Initialize ADC attack
    logging.info("Initializing ADC attack...")
    attacker = ADCAttack(
        model=model,
        tokenizer=tokenizer,
        num_starts=1,
        num_steps=5000,
        learning_rate=10,
        momentum=0.99,
        use_kv_cache=True
    )
    logging.info("ADC attack initialized successfully")
    
    # Create results directory
    save_folder = './results/finbert-adc-twitter'
    os.makedirs(save_folder, exist_ok=True)
    logging.info(f"Results will be saved in: {save_folder}")
    
    # Process each sample in the dataset
    for idx, row in df.iterrows():
        logging.info(f"\nProcessing sample {idx}")
        text = row['text']
        sentiment = row['sentiment']
        
        # Create input template for the attack
        logging.info("Creating input template...")
        user_prompt = f"Analyze the sentiment of this financial news: {text}"
        target_response = f"The sentiment is {sentiment}."
        
        # Generate input template
        string, input_ids, slices = get_input_template(
            user_prompt=user_prompt,
            target_response=target_response,
            len_adv_tokens=20,  # Number of adversarial tokens
            tokenizer=tokenizer,
            model_name="finbert"
        )
        logging.info(f"Input template created. Input IDs shape: {input_ids.shape}")
        logging.info(f"Slices: {slices}")
        
        # Perform attack
        logging.info("Starting ADC attack...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t_start = time.time()
        
        result = attacker.attack(
            tokens=input_ids,
            slices=slices,
            user_prompt=user_prompt,
            response=target_response
        )
        
        torch.cuda.synchronize()
        time_used = time.time() - t_start
        logging.info(f"Attack completed in {time_used:.2f} seconds")
        
        # Generate final output with adversarial tokens
        logging.info("Generating final output...")
        input_ids = input_ids.view(1, -1).cuda()
        target_start = slices['target_slice'].start
        prefix = input_ids[:, :target_start]
        prefix[:, slices['adv_slice']] = result[1].view(1, -1).cuda()
        
        output = model.generate(
            input_ids=prefix,
            generation_config=model.generation_config,
            max_new_tokens=512
        )
        
        gen_str = tokenizer.decode(output.reshape(-1)[target_start:])
        
        # Save results
        result += (time_used, user_prompt, gen_str)
        torch.save(result, f'{save_folder}/result_{idx}.pth')
        logging.info(f"Results saved to {save_folder}/result_{idx}.pth")
        
        logging.info(f"Sample {idx} details:")
        logging.info(f"Original text: {text}")
        logging.info(f"Target sentiment: {sentiment}")
        logging.info(f"Generated response: {gen_str}")
        logging.info("-" * 50)

if __name__ == "__main__":
    main() 