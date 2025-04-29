import torch
import torch.nn as nn
from torch.optim import AdamW

import wandb


# --- MLM MASKING ---
# --- MLM MASKING ---
def mask_tokens(input_ids: torch.Tensor,
                vocab_size: int,
                mask_token_id: int,
                pad_token_id: int,
                mlm_prob: float = 0.15):
    """
    Prepare masked tokens inputs/labels for masked language modeling:
      - 15% of tokens are selected.
      - Of those: 80% → [MASK], 10% → random token, 10% → unchanged.
      - Labels are the original token IDs for masked positions, -100 elsewhere.

    Args:
      input_ids:     LongTensor of shape (batch, seq_len)
      vocab_size:    size of the vocabulary
      mask_token_id: ID of the [MASK] token
      pad_token_id:  ID of the padding token
      mlm_prob:      probability of masking each token
    Returns:
      masked_inputs: Tensor of same shape as input_ids with masking applied
      labels:        Tensor of same shape, with original IDs at masked positions and -100 elsewhere
    """
    labels = input_ids.clone()

    # 1) Create mask for special/pad tokens
    special_mask = (input_ids == pad_token_id)
    # optionally you could also exclude mask_token_id, cls, sep, etc.

    # 2) Decide which tokens to mask
    prob_matrix = torch.full(labels.shape, mlm_prob, device=input_ids.device)
    prob_matrix.masked_fill_(special_mask, value=0.0)
    masked_indices = torch.bernoulli(prob_matrix).bool()
    labels[~masked_indices] = -100  # only compute loss on masked tokens

    # 3) 80% of the time, replace masked input tokens with [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=input_ids.device)).bool() & masked_indices
    input_ids[indices_replaced] = mask_token_id

    # 4) 10% of the time, replace masked input tokens with random word
    remainder = masked_indices & ~indices_replaced
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=input_ids.device)).bool() & remainder
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=input_ids.device)
    input_ids[indices_random] = random_words[indices_random]

    # 5) The rest 10% (remainder & ~indices_random) are left unchanged

    return input_ids, labels



def train_bert(model, train_loader, val_loader, tokenizer, epochs=3, lr=5e-4, device='cuda'):
    """
    Train a BERT‐style encoder with MLM head, logging both loss and perplexity.
    """
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(1, epochs + 1):
        # ------------------
        # Training
        # ------------------
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch.get('token_type_ids', torch.zeros_like(input_ids)).to(device)

            masked_inputs, labels = mask_tokens(
                input_ids.clone(),
                vocab_size=tokenizer.vocab_size,
                mask_token_id=tokenizer.mask_token_id,
                pad_token_id=tokenizer.pad_token_id,
                mlm_prob=0.15
            )
            masked_inputs, labels = masked_inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(masked_inputs, token_type_ids, attention_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            wandb.log({'train_loss_step': loss.item(), 'epoch': epoch})

        avg_train_loss = total_train_loss / len(train_loader)

        # ------------------
        # Validation
        # ------------------
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch.get('token_type_ids', torch.zeros_like(input_ids)).to(device)

                masked_inputs, labels = mask_tokens(
                    input_ids.clone(),
                    vocab_size=tokenizer.vocab_size,
                    mask_token_id=tokenizer.mask_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    mlm_prob=0.15
                )
                masked_inputs, labels = masked_inputs.to(device), labels.to(device)

                logits = model(masked_inputs, token_type_ids, attention_mask)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        # compute perplexity = exp(average loss)
        val_ppl = torch.exp(torch.tensor(avg_val_loss)).item()

        # log epoch metrics
        wandb.log({
            'train_loss_epoch': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_perplexity': val_ppl,
            'epoch': epoch
        })

        print(
            f"Epoch {epoch}: "
            f"train_loss={avg_train_loss:.4f}, "
            f"val_loss={avg_val_loss:.4f}, "
            f"val_ppl={val_ppl:.2f}"
        )

