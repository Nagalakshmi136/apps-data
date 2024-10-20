import pandas as pd
from pathlib import Path
import os

# import shutil


def organize_reviews_by_sentiment(source_dir: str, destination_dir: str):
    """
    Organize reviews into separate files based on sentiment scores for each app
    """
    processed_dir = Path(source_dir)
    destination_dir = Path(destination_dir)
    # Iterate through each app's CSV file
    for csv_file in processed_dir.glob("*.csv"):
        app_name = csv_file.stem
        print(f"\nProcessing {app_name}...")

        # Create app directory
        app_dir = destination_dir / app_name
        app_dir.mkdir(parents=True, exist_ok=True)
        if app_dir.exists():
            continue

        # Read the CSV file
        try:
            df = pd.read_csv(csv_file)

            # Create summary statistics
            summary_stats = {
                "total_reviews": len(df),
                "sentiment_distribution": df["sentimentScore"].value_counts().to_dict(),
                "average_sentiment": df["sentimentScore"].mean(),
            }

            # Split reviews by sentiment score and save to separate files
            for sentiment in range(1, 6):
                sentiment_df = df[df["sentimentScore"] == sentiment]

                if not sentiment_df.empty:
                    # Save to sentiment-specific file
                    output_file = app_dir / f"{sentiment}_star_reviews.csv"
                    sentiment_df.to_csv(output_file, index=False)
                    print(
                        f"Saved {len(sentiment_df)} {sentiment}-star reviews to {output_file}"
                    )

            # Save summary statistics
            summary_file = app_dir / "sentiment_summary.txt"
            with open(summary_file, "w") as f:
                f.write(f"Sentiment Analysis Summary for {app_name}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total Reviews: {summary_stats['total_reviews']}\n\n")
                f.write("Sentiment Distribution:\n")
                for stars, count in sorted(
                    summary_stats["sentiment_distribution"].items()
                ):
                    percentage = (count / summary_stats["total_reviews"]) * 100
                    f.write(f"{stars} stars: {count} reviews ({percentage:.1f}%)\n")
                f.write(
                    f"\nAverage Sentiment Score: {summary_stats['average_sentiment']:.2f}"
                )

        except Exception as e:
            print(f"Error processing {app_name}: {str(e)}")
            continue


def main():
    print("Starting to organize reviews by sentiment...")
    organize_reviews_by_sentiment(
        "data/processed_data/app_store", "organized_data/app_store"
    )
    print("\nProcess completed!")


if __name__ == "__main__":
    main()
