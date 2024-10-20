import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch
import regex as re
import emoji
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from functools import partial
import numpy as np


class ReviewDataset(Dataset):
    def __init__(self, reviews, tokenizer, max_length=512):
        self.reviews = reviews
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        return self.reviews[idx]


def clean_text(text):
    """Basic cleaning of text while preserving the multilingual characters"""
    if not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    text = emoji.demojize(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.UNICODE)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_reviews_batch(reviews):
    """Clean a batch of reviews in parallel"""
    with multiprocessing.Pool() as pool:
        cleaned_reviews = pool.map(clean_text, reviews)
    return cleaned_reviews


def collate_batch(batch, tokenizer, max_length=512):
    """Collate batch of reviews into tensor format"""
    return tokenizer(
        batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )


def process_batch(batch, model, device):
    """Process a batch of reviews and return sentiment scores"""
    with torch.no_grad():
        outputs = model(**{k: v.to(device) for k, v in batch.items()})
        predictions = torch.softmax(outputs.logits, dim=1)
        return predictions.cpu().numpy()


def perform_sentiment_analysis(data, app_name, model, tokenizer, device, batch_size=32):
    """Perform sentiment analysis on reviews in batches"""
    # Extract review content and IDs
    reviews = []
    review_user_name = []
    for review in data["reviews"]:
        if review.get("review"):
            reviews.append(review["review"])
            review_user_name.append(review["userName"])

    if not reviews:
        print(f"No reviews found for {app_name}")
        return

    # Clean reviews in parallel
    print(f"Cleaning reviews for {app_name}...")
    cleaned_reviews = clean_reviews_batch(reviews)

    # Create dataset and dataloader
    dataset = ReviewDataset(cleaned_reviews, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_batch, tokenizer=tokenizer),
        num_workers=2,
    )

    # Process batches
    all_predictions = []
    print(f"Processing reviews for {app_name}...")
    for batch in tqdm(dataloader):
        predictions = process_batch(batch, model, device)
        all_predictions.extend(predictions)

    # Convert predictions to sentiment scores (1-5)
    sentiment_scores = np.argmax(all_predictions, axis=1) + 1

    # Create DataFrame with results
    results_df = pd.DataFrame(
        {
            "reviewUserName": review_user_name,
            "content": cleaned_reviews,
            "sentimentScore": sentiment_scores,
        }
    )

    # Save results
    output_path = Path(f"data/processed_data/app_store/{app_name}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"Sentiment analysis results for {app_name} saved to {output_path}")


def main():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    # Process files
    directory = Path("data/raw_data/appstore")
    for file_path in tqdm(directory.iterdir()):
        if file_path.is_file():
            print(f"\nProcessing {file_path.stem}...")
            if Path(f"data/processed_data/app_store/{file_path.stem}.csv").exists():
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                perform_sentiment_analysis(
                    data, file_path.stem, model, tokenizer, device
                )
            except Exception as e:
                print(f"Error processing {file_path.stem}: {str(e)}")


if __name__ == "__main__":
    main()
