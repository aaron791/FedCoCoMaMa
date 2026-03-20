"""
Create streaming datasets directly from HuggingFace.
This module provides functionality to create SPROUT streaming dataset directly from HuggingFace.
"""

import numpy as np
import pandas as pd
import json
import os
from typing import Dict, Any, Optional
from tqdm import tqdm
import argparse

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from streaming_dataset import StreamingDataset


def load_sprout_from_huggingface(force_reload: bool = False) -> pd.DataFrame:
    """
    Load SPROUT dataset directly from HuggingFace.
    
    Args:
        force_reload: If True, reload from HuggingFace even if local file exists
        
    Returns:
        Pandas DataFrame with SPROUT data
    """
    saved_df_filename = 'sprout_df.pkl'
    
    if not force_reload and os.path.exists(saved_df_filename):
        print("Loading SPROUT dataset from local cache...")
        with open(saved_df_filename, 'rb') as f:
            return pd.read_pickle(f)
    
    print("Loading SPROUT dataset from HuggingFace...")
    ds = load_dataset("CARROT-LLM-Routing/SPROUT")
    
    df = ds['train'].to_pandas()
    
    print(f"Saving SPROUT dataset to {saved_df_filename}...")
    with open(saved_df_filename, 'wb') as f:
        df.to_pickle(f)
    
    return df


def create_sprout_streaming_dataset(
    output_path: str,
    embedding_config: Dict[str, Any],
    num_rounds: Optional[int] = None,
    force_reload: bool = False
) -> str:
    """
    Create SPROUT streaming dataset directly from HuggingFace.
    
    Args:
        output_path: Path to save the Arrow streaming dataset
        embedding_config: Configuration for embeddings
        num_rounds: Number of rounds to include (None for all)
        force_reload: If True, reload from HuggingFace even if local file exists
        
    Returns:
        Path to the created streaming dataset
    """
    print("Creating SPROUT streaming dataset from HuggingFace...")
    
    df = load_sprout_from_huggingface(force_reload)
    
    if num_rounds is not None:
        df = df.head(num_rounds)
        print(f"Limited to {num_rounds} rounds")
    
    print(f"Processing {len(df)} tasks...")
    
    print(f"Loading embedding model: {embedding_config['model_name']}...")
    model = SentenceTransformer(
        embedding_config['model_name'],
        device='cpu',
        trust_remote_code=True,
        truncate_dim=embedding_config['dimensions'],
        cache_folder='./cache'
    )
    
    agent_keys = [
        "aws-claude-3-5-sonnet-v1",
        "aws-titan-text-premier-v1", 
        "openai-gpt-4o",
        "openai-gpt-4o-mini",
        "wxai-granite-3-2b-instruct-8k-max-tokens",
        "wxai-granite-3-8b-instruct-8k-max-tokens",
        "wxai-llama-3-1-70b-instruct",
        "wxai-llama-3-1-8b-instruct",
        "wxai-llama-3-2-1b-instruct",
        "wxai-llama-3-2-3b-instruct",
        "wxai-llama-3-3-70b-instruct",
        "wxai-llama-3-405b-instruct",
        "wxai-mixtral-8x7b-instruct-v01"
    ]
    
    print("Creating task embeddings...")
    task_embeddings = []
    for idx, prompt in enumerate(tqdm(df['prompt'], desc="Task embeddings")):
        embedding = model.encode(
            prompt,
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        task_embeddings.append(embedding)
    
    print("Creating agent embeddings...")
    agent_embeddings = {}
    for key in tqdm(agent_keys, desc="Agent embeddings"):
        model_card_path = f'model_cards/{key}.json'
        if not os.path.exists(model_card_path):
            print(f"Warning: Model card not found at {model_card_path}")
            continue
            
        with open(model_card_path, 'r') as f:
            agent_card = json.load(f)
            agent_card_str = json.dumps(agent_card)
            agent_embedding = model.encode(
                agent_card_str,
                convert_to_tensor=False,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            agent_embeddings[key] = agent_embedding
    
    print("Creating streaming tasks...")
    tasks = []
    
    for idx, row in enumerate(tqdm(df.iterrows(), total=len(df), desc="Processing tasks")):
        _, data = row
        
        task_embedding = task_embeddings[idx]
        
        agents = []
        for agent_idx, key in enumerate(agent_keys):
            if key not in agent_embeddings:
                continue
                
            agent_data = data.get(key, {})
            if isinstance(agent_data, dict):
                rating = float(agent_data.get('score', 0.0))
            else:
                rating = 0.0
            
            agents.append({
                'agent_id': agent_idx,
                'agent_embedding': agent_embeddings[key],
                'rating': rating
            })
        
        tasks.append({
            'task_id': idx,
            'seq': idx,
            'task_embedding': task_embedding,
            'agents': agents
        })
    
    print(f"Writing streaming dataset to {output_path}...")
    streaming_dataset = StreamingDataset(
        task_embedding_dim=embedding_config['dimensions'],
        agent_embedding_dim=embedding_config['dimensions']
    )
    streaming_dataset.write_dataset(tasks, output_path)
    
    print(f"SPROUT streaming dataset created: {output_path}")
    print(f"  - {len(tasks)} tasks")
    print(f"  - {len(agent_keys)} agents per task")
    print(f"  - {embedding_config['dimensions']}D embeddings")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Create SPROUT streaming dataset from HuggingFace")
    parser.add_argument("--output_path", required=True,
                       help="Output path for streaming dataset")
    parser.add_argument("--num_rounds", type=int, default=300,
                       help="Number of rounds")
    parser.add_argument("--embedding_model", default="all-MiniLM-L6-v2",
                       help="Embedding model name")
    parser.add_argument("--embedding_dimensions", type=int, default=384,
                       help="Embedding dimensions")
    parser.add_argument("--force_reload", action="store_true",
                       help="Force reload from HuggingFace")
    
    args = parser.parse_args()
    
    embedding_config = {
        'model_name': args.embedding_model,
        'dimensions': args.embedding_dimensions,
        'suffix': f'_{args.embedding_model}_{args.embedding_dimensions}-dim'
    }
    
    create_sprout_streaming_dataset(
        args.output_path,
        embedding_config,
        args.num_rounds,
        args.force_reload
    )


if __name__ == "__main__":
    main()
