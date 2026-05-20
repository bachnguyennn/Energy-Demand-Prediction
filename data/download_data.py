import os
import zipfile
import subprocess
import sys

DATASET_NAME = "robikscube/hourly-energy-consumption"
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")

def check_kaggle_api():
    kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
    return os.path.exists(kaggle_json_path)

def download_data():
    os.makedirs(RAW_DIR, exist_ok=True)
    
    # Check if data already exists
    if any(f.endswith('.csv') for f in os.listdir(RAW_DIR)):
        print(f"Data already found in {RAW_DIR}. Skipping download.")
        return True
    
    if not check_kaggle_api():
        print("Kaggle API key not found at ~/.kaggle/kaggle.json")
        print("\n--- INSTRUCTIONS ---")
        print("1. Go to https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption")
        print("2. Download the archive.zip file")
        print(f"3. Extract it and place the CSV files (like AEP_hourly.csv) into: {RAW_DIR}")
        print("4. Alternatively, configure your Kaggle API key (kaggle.json) and run this script again.")
        print("--------------------\n")
        return False
    
    print(f"Downloading dataset {DATASET_NAME} from Kaggle...")
    try:
        subprocess.run([sys.executable, "-m", "kaggle", "datasets", "download", "-d", DATASET_NAME, "-p", RAW_DIR], check=True)
        
        zip_path = os.path.join(RAW_DIR, "hourly-energy-consumption.zip")
        if os.path.exists(zip_path):
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(RAW_DIR)
            os.remove(zip_path)
            print("Download and extraction complete.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        print("Please download it manually from Kaggle.")
        return False

if __name__ == "__main__":
    download_data()
