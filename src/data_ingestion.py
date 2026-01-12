 
import os
import pandas as pd

class DataIngestion:
    def __init__(self):
        self.raw_data_path = os.path.join("data", "raw", "data.csv")

    def ingest_data(self):
        # Example: loading CSV file
        df = pd.read_csv("data_source.csv")

        # Create directories if not exist
        os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)

        # Save raw data
        df.to_csv(self.raw_data_path, index=False)

        return self.raw_data_path


if __name__ == "__main__":
    ingestion = DataIngestion()
    path = ingestion.ingest_data()
    print(f"Data ingested at: {path}")
