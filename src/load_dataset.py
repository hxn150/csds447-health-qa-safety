import pandas as pd
from pathlib import Path

def main():
    input_file = '../outputs/safety_qa.csv'
    output_file = 'data/safety_classifier/safety_balanced.csv'
    
    df = pd.read_csv(input_file)
    df = df.rename(columns={'question': 'text'})
    df = df[df['label'].isin(['SAFE', 'UNSAFE'])].reset_index(drop=True)
    
    print(f"Loaded: {len(df)} examples")
    print(df['label'].value_counts())
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nSaved to: {output_path}")

if __name__ == "__main__":
    main()