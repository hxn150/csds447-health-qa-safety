"""
HELEN'S TASK - STEP 3 QUICK: Split Dataset (No Synthetic Data)
Use only the judge-labeled data from Step 1
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def main():
    """Split judge-labeled data into train/val/test"""
    
    # Load judge-labeled data only
    data_file = Path('data/safety_classifier/tinyllama_medqa_labeled.csv')
    
    if not data_file.exists():
        print("ERROR: Run step1_simple.py first")
        return
    
    df = pd.read_csv(data_file)
    
    print(f"Loaded: {len(df)} examples")
    print(df['label'].value_counts())
    
    # Split: 70% train, 15% val, 15% test
    train_df, temp_df = train_test_split(
        df[['question', 'label']], 
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
    
    print(f"\nSplit: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Save
    output_dir = Path('data/safety_classifier')
    train_df.to_csv(output_dir / 'train.csv', index=False)
    val_df.to_csv(output_dir / 'val.csv', index=False)
    test_df.to_csv(output_dir / 'test.csv', index=False)
    
    print(f"Saved to: {output_dir}/")

if __name__ == "__main__":
    main()