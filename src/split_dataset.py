import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def main():
    data_file = Path('data/safety_classifier/safety_balanced.csv')
    
    if not data_file.exists():
        print(f"ERROR: {data_file} not found")
        return
    
    df = pd.read_csv(data_file)
    
    print(f"Loaded: {len(df)} examples")
    print(df['label'].value_counts())
    
    train_df, temp_df = train_test_split(
        df[['text', 'label']], 
        test_size=0.3, 
        random_state=42, 
        stratify=df['label']
    )
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=42, 
        stratify=temp_df['label']
    )
    
    print(f"\nTrain: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    output_dir = Path('data/safety_classifier')
    train_df.to_csv(output_dir / 'train.csv', index=False)
    val_df.to_csv(output_dir / 'val.csv', index=False)
    test_df.to_csv(output_dir / 'test.csv', index=False)
    
    print(f"\nSaved to: {output_dir}/")

if __name__ == "__main__":
    main()