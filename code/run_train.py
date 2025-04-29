import argparse
import os
import torch
import wandb
from torch.utils.data import DataLoader

from train_encoder import train_bert
from data import TextDataset
from transformers import BertTokenizerFast
from encoder import Encoder

import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_stories(pkl_path, test_size=0.1, random_state=42):
    """
    Load a dict of {story_name: story_text} from a pickle file,
    then split into train/validation sets.

    Args:
        pkl_path (str): Path to your .pkl file containing a dict.
        test_size (float): Fraction of data to reserve for validation.
        random_state (int): Seed for reproducibility.

    Returns:
        train_texts (List[str])
        val_texts   (List[str])
        train_names (List[str])
        val_names   (List[str])
    """
    # 1) Load the pickle
    with open(pkl_path, 'rb') as f:
        df = pd.read_pickle('/ocean/projects/mth240012p/shared/data/raw_text.pkl')

    # 2) Extract names & texts
    story_names = list(df.keys())
    story_texts = [" ".join(df[name].data) for name in story_names]

    # 3) Stratified split
    train_texts, val_texts, train_names, val_names = train_test_split(
        story_texts,
        story_names,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    return train_texts, val_texts, train_names, val_names


def main():
    parser = argparse.ArgumentParser(description="Run BERT training with ML masking and wandb")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file with training parameters")
    args = parser.parse_args()

    # load config
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg['output_dir'], exist_ok=True)

    # init wandb
    wandb.init(
        project=cfg['project'],
        name=cfg.get('run_name', None),
        config=cfg
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = cfg['data_path']
    train_texts, val_texts, _, _ = load_and_split_stories(
        data_path,
        test_size=0.2,
        random_state=42
    )

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")  

    train_ds = TextDataset(train_texts, tokenizer, max_len=cfg['max_len'])
    val_ds = TextDataset(val_texts, tokenizer, max_len=cfg['max_len'])
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds,   batch_size=cfg['batch_size'])

    model = Encoder(
        vocab_size=tokenizer.vocab_size,
        hidden_size=cfg.get('hidden_size', 256),
        num_heads=cfg.get('num_heads', 4),
        num_layers=cfg.get('num_layers', 4),
        intermediate_size=cfg.get('intermediate_size', 512),
        max_len=cfg['max_len']
    )

    print(type(cfg['lr']))

    train_bert(
        model,
        train_loader,
        val_loader,
        tokenizer,
        epochs=cfg['epochs'],
        lr=cfg['lr'],
        device=device,
    )

    # save model
    final_path = os.path.join(cfg['output_dir'], 'final_model.pt')
    torch.save(model.state_dict(), final_path)
    wandb.save(final_path)
    wandb.finish()

if __name__ == "__main__":
    main()