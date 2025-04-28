import torch
import torch.nn.functional as F
from torch.optim import AdamW

def mask_tokens(input_ids, vocab_size, mask_token_id, pad_token_id, mlm_prob=0.15):
    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, mlm_prob, device=input_ids.device)
    
    # Prevent masking of special tokens
    special_tokens_mask = (
        (input_ids == pad_token_id) |
        (input_ids == mask_token_id) |
        (input_ids == 101) |  # [CLS]
        (input_ids == 102)    # [SEP]
    )
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100
    
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=input_ids.device)).bool() & masked_indices
    input_ids[indices_replaced] = mask_token_id
    
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=input_ids.device)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(0, vocab_size, labels.shape, dtype=torch.long, device=input_ids.device)
    input_ids[indices_random] = random_words[indices_random]
    
    return input_ids, labels

def train_bert(model, train_dataloaders, val_dataloaders, tokenizer, epochs=3, lr=5e-4, device='cuda'):
    '''
    Implement training loop for BERT with validation.
    Args:
        model: BERT-style Encoder model
        train_dataloaders: Dictionary of training DataLoader objects
        val_dataloaders: Dictionary of validation DataLoader objects
        tokenizer: Tokenizer (provides vocab_size, mask_token_id, pad_token_id)
        epochs: Number of epochs
        lr: Learning rate
        device: Device to run the model on ('cuda' or 'cpu')
    Returns:
        loss_history: Dictionary with 'train_loss' and 'val_loss' lists
    '''
    # Move model to device
    model = model.to(device)
    model.train()
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    #Dictionary of DataLoaders
    train_dls = train_dataloaders.values()
    val_dls = val_dataloaders.values()
    
    # Initialize loss history
    loss_history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        num_train_batches = 0
        
        for dl in train_dls:
            for batch in dl:
                input_ids = batch["input_ids"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                masked_inputs, labels = mask_tokens(
                    input_ids,
                    vocab_size=tokenizer.vocab_size,
                    mask_token_id=tokenizer.mask_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    mlm_prob=0.15
                )
                
                optimizer.zero_grad()
                
                # Forward pass
                logits = model(masked_inputs, token_type_ids, attention_mask)  # [B, L, vocab_size]
                
                # Compute MLM loss
                loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), labels.view(-1))
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevent exploding gradients
                optimizer.step()
                
                total_train_loss += loss.item()
                num_train_batches += 1
        
        # Average training loss
        avg_train_loss = total_train_loss / num_train_batches
        loss_history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for dl in val_dls:
                for batch in dl:
                    input_ids = batch["input_ids"].to(device)
                    token_type_ids = batch["token_type_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    
                    # Apply MLM masking
                    masked_inputs, labels = mask_tokens(
                        input_ids,
                        vocab_size=tokenizer.vocab_size,
                        mask_token_id=tokenizer.mask_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        mlm_prob=0.15
                    )
                    
                    # Forward pass
                    logits = model(masked_inputs, token_type_ids, attention_mask)
                    
                    # Compute MLM loss
                    loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), labels.view(-1))
                    
                    total_val_loss += loss.item()
                    num_val_batches += 1
        
        # Average validation loss
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        loss_history['val_loss'].append(avg_val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return loss_history