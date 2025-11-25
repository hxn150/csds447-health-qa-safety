"""
HELEN'S TASK - STEP 1: Load and Clean TinyLlama MedQA Dataset
Simple version with minimal output
"""

import pandas as pd
from pathlib import Path

# Verdict mapping
VERDICT_MAPPING = {
    'SAFE': 'SAFE',
    'PARTIALLY_SAFE': 'SAFE',
    'UNSAFE': 'UNSAFE',
    'UNSURE': 'REFUSAL',
}

def clean_verdict(verdict_str):
    """Clean malformed verdict strings"""
    if pd.isna(verdict_str):
        return 'REFUSAL'
    
    verdict_str = str(verdict_str).strip().upper()
    
    if verdict_str in VERDICT_MAPPING:
        return VERDICT_MAPPING[verdict_str]
    
    if 'SAFE' in verdict_str and 'UNSAFE' not in verdict_str:
        return 'SAFE'
    if 'UNSAFE' in verdict_str:
        return 'UNSAFE'
    
    return 'REFUSAL'

def load_dataset(dataset_path):
    """Load and clean the TinyLlama MedQA dataset"""
    
    # Find the file
    possible_paths = [
        dataset_path,
        f'/mnt/project/{dataset_path}',
        f'outputs/{dataset_path}',
    ]
    
    path = None
    for p in possible_paths:
        if Path(p).exists():
            path = p
            break
    
    if path is None:
        print(f"ERROR: Could not find {dataset_path}")
        return None
    
    # Load
    df = pd.read_csv(path)
    
    if 'verdict' not in df.columns or 'question' not in df.columns:
        print("ERROR: Missing required columns")
        return None
    
    # Clean
    records = []
    for _, row in df.iterrows():
        question = str(row['question']).strip()
        
        if not question or question == 'nan' or len(question) < 10:
            continue
        
        records.append({
            'question': question,
            'label': clean_verdict(row['verdict']),
            'raw_verdict': str(row['verdict']),
            'model': row.get('model', 'TinyLlama'),
            'dataset': 'medqa',
            'source': 'tinyllama_medqa'
        })
    
    return pd.DataFrame(records)

def main():
    """Main execution"""
    
    # Load
    df = load_dataset('../outputs/medqa_mcq_synthetic_safety.csv')
    
    if df is None:
        return
    
    # Stats
    counts = df['label'].value_counts()
    total = len(df)
    
    print(f"\nDataset loaded: {total} examples")
    print(f"  SAFE:    {counts.get('SAFE', 0):3d} ({counts.get('SAFE', 0)/total*100:.1f}%)")
    print(f"  UNSAFE:  {counts.get('UNSAFE', 0):3d} ({counts.get('UNSAFE', 0)/total*100:.1f}%)")
    print(f"  REFUSAL: {counts.get('REFUSAL', 0):3d} ({counts.get('REFUSAL', 0)/total*100:.1f}%)")
    
    # Save
    output_path = Path('data/safety_classifier/tinyllama_medqa_labeled.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nSaved to: {output_path}")
    
    # Recommendation
    if counts.get('REFUSAL', 0) < 50:
        print(f"\nRecommendation: Generate synthetic REFUSAL examples (need ~{200-counts.get('REFUSAL', 0)} more)")

if __name__ == "__main__":
    main()