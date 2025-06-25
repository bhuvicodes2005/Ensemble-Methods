import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from datasets import load_dataset
from transformers import AutoTokenizer
import time

def prepare_wikitext2():
    # Load raw WikiText2.
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    # Use a pretrained tokenizer (e.g., BERT) to tokenize the text.
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    
    # Concatenate all text samples and tokenize.
    tokens = tokenizer(" ".join(dataset["text"]), return_tensors=None, add_special_tokens=False)["input_ids"]
    
    # Convert token list to a 1D torch tensor.
    data = torch.tensor(tokens, dtype=torch.long)
    
    # Retrieve vocabulary size.
    vocab_size = tokenizer.vocab_size

    return data, vocab_size

def yield_tokens(data_iter, tokenizer):
    for text in data_iter:
        yield tokenizer(text)

def generate_real_batch(data, batch_size, seq_length, device):
    """
    Generate a batch from real data.
    data: 1D tensor of token indices.
    seq_length: Number of tokens per sample (we request seq_length + 1 tokens so we can shift later).
    """
    max_start = data.size(0) - seq_length
    start_indices = torch.randint(0, max_start, (batch_size,))
    batch = torch.stack([data[i:i+seq_length] for i in start_indices]).to(device)
    return batch

def truncated_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """
    Initializes the tensor with values from a truncated normal distribution.
    Tries to use nn.init.trunc_normal_ if available; otherwise, falls back to a clipping approach.
    """
    try:
        nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)
    except AttributeError:
        with torch.no_grad():
            tensor.normal_(mean, std)
            tensor.clamp_(min=a, max=b)
    return tensor

# ----------------------------
# Core Building Blocks
# ----------------------------
class FFNBlock(nn.Module):
    """Standard Feed-Forward Network block with dropout and custom initialization."""
    def __init__(self, d_model, d_ff, init_scale=0.1, dropout=0.1):
        super(FFNBlock, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.init_scale = init_scale
        self.reset_parameters()
        
    def reset_parameters(self):
        std1 = self.init_scale / math.sqrt(self.fc1.in_features)
        std2 = self.init_scale / math.sqrt(self.fc2.in_features)
        truncated_normal_(self.fc1.weight, mean=0.0, std=std1)
        nn.init.constant_(self.fc1.bias, 0)
        truncated_normal_(self.fc2.weight, mean=0.0, std=std2)
        nn.init.constant_(self.fc2.bias, 0)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MoELayer(nn.Module):
    """Mixture-of-Experts (MoE) layer with top-1 routing and selective precision."""
    def __init__(self, d_model, d_ff, num_experts, init_scale=0.1, expert_dropout=0.4):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        # Each expert is an FFN block; we use a higher dropout for experts.
        self.experts = nn.ModuleList([
            FFNBlock(d_model, d_ff, init_scale=init_scale, dropout=expert_dropout)
            for _ in range(num_experts)
        ])
        # Router is a simple linear layer that outputs logits per expert.
        self.router = nn.Linear(d_model, num_experts)
        self.init_scale = init_scale
        self.reset_parameters()
        
    def reset_parameters(self):
        std = self.init_scale / math.sqrt(self.router.in_features)
        truncated_normal_(self.router.weight, mean=0.0, std=std)
        nn.init.constant_(self.router.bias, 0)
        
    def forward(self, x):
        # x: [batch_size, seq_length, d_model]
        router_input = x.float()  # Use float32 for stability in routing computations.
        router_logits = self.router(router_input)
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-1 Routing: select the expert with the highest probability.
        expert_indices = torch.argmax(router_probs, dim=-1)  # Shape: [B, L]
        output = torch.zeros_like(x)
        # Process tokens for each expert.
        for i in range(self.num_experts):
            mask = (expert_indices == i).unsqueeze(-1).float()  # Shape: [B, L, 1]
            if mask.sum() > 0:
                expert_output = self.experts[i](x)
                # Weight the output by the corresponding gate value.
                gate_value = router_probs[..., i].unsqueeze(-1)
                output += mask * gate_value * expert_output
        return output

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer."""
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        # Compute linear projections.
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        # Reshape and transpose for multi-head attention.
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1,2).contiguous().view(batch_size, seq_length, self.d_model)
        out = self.out_linear(out)
        return out

class TransformerLayer(nn.Module):
    """
    Single Transformer layer.
    Depending on the settings, uses a standard FFN or an MoE variant.
    """
    def __init__(self, d_model, d_ff, num_heads, use_moe=False, num_experts=None,
                 init_scale=0.1, dropout=0.1, expert_dropout=0.4):
        super(TransformerLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        if use_moe and num_experts is not None:
            self.ffn = MoELayer(d_model, d_ff, num_experts, init_scale=init_scale, expert_dropout=expert_dropout)
        else:
            self.ffn = FFNBlock(d_model, d_ff, init_scale=init_scale, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention block with residual connection.
        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.dropout1(x)
        x = residual + x
        
        # Feed-forward block with residual connection.
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout2(x)
        return residual + x

class SwitchTransformer(nn.Module):
    """
    Transformer that alternates between standard dense layers and MoE layers.
    """
    def __init__(self, num_layers, d_model, d_ff, num_heads, use_moe=False, num_experts=None,
                 init_scale=0.1, dropout=0.1, expert_dropout=0.4):
        super(SwitchTransformer, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # For demonstration, replace every other layer with an MoE layer if use_moe is True.
            if use_moe and (i % 2 == 1):
                layer = TransformerLayer(d_model, d_ff, num_heads, use_moe=True, num_experts=num_experts,
                                           init_scale=init_scale, dropout=dropout, expert_dropout=expert_dropout)
            else:
                layer = TransformerLayer(d_model, d_ff, num_heads, use_moe=False,
                                           init_scale=init_scale, dropout=dropout)
            self.layers.append(layer)
            
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerLM(nn.Module):
    """
    Transformer language model that uses token and positional embeddings, a transformer stack,
    and an output layer tied with the embedding matrix.
    """
    def __init__(self, vocab_size, seq_length, num_layers, d_model, d_ff, num_heads,
                 use_moe=False, num_experts=None, init_scale=0.1, dropout=0.1, expert_dropout=0.4):
        super(TransformerLM, self).__init__()
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_length, d_model))
        self.transformer = SwitchTransformer(num_layers, d_model, d_ff, num_heads, 
                                             use_moe=use_moe, num_experts=num_experts,
                                             init_scale=init_scale, dropout=dropout, expert_dropout=expert_dropout)
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
        # Weight tying between embedding and output projection.
        self.output_proj.weight = self.embed.weight
        
    def forward(self, x):
        # x: [B, L] token indices.
        B, L = x.size()
        token_emb = self.embed(x)
        token_emb = token_emb + self.pos_embed[:, :L, :]
        h = self.transformer(token_emb)
        h = self.norm(h)
        logits = self.output_proj(h)  # [B, L, vocab_size]
        return logits

def generate_synthetic_batch(batch_size, seq_length, vocab_size, device):
    # Generate a batch of random token indices.
    data = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_length), device=device)
    return data

# ----------------------------
# Training Functions
# ----------------------------
def train_lm(model, optimizer, data, num_steps, batch_size, seq_length, vocab_size, device):
    """
    Train the language model using standard cross-entropy loss.
    Uses a one-token shift (inputs: tokens 0...L-1, targets: tokens 1...L).
    """
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    steps, losses = [], []
    for step in trange(num_steps, desc="Training"):
        optimizer.zero_grad()
        # Get a batch with seq_length + 1 tokens.
        x = generate_real_batch(data, batch_size, seq_length + 1, device)
        inputs = x[:, :-1]
        targets = x[:, 1:]
        logits = model(inputs)
        # Using reshape instead of view to work with non-contiguous tensors:
        logits = logits.reshape(-1, vocab_size)
        targets = targets.reshape(-1)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        if (step + 1) % 10 == 0:
            steps.append(step + 1)
            losses.append(loss.item())
    return steps, losses

def train_distillation(student, teacher, optimizer, data, num_steps, batch_size, seq_length, vocab_size, device, alpha=0.75, temperature=1.0):
    """
    Train a student model using a blend of hard (cross-entropy) and soft (KL divergence) losses.
    """
    student.train()
    teacher.eval()
    loss_fn = nn.CrossEntropyLoss()
    kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
    steps, losses = [], []
    for step in trange(num_steps, desc="Distillation Training"):
        optimizer.zero_grad()
        x = generate_real_batch(data, batch_size, seq_length + 1, device)
        inputs = x[:, :-1]
        targets = x[:, 1:]
        with torch.no_grad():
            teacher_logits = teacher(inputs)
        student_logits = student(inputs)
        
        # Hard loss (cross-entropy).
        student_logits_flat = student_logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        ce_loss = loss_fn(student_logits_flat, targets_flat)
        
        # Soft loss (KL divergence with temperature scaling).
        soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / temperature, dim=-1)
        kd_loss = kl_loss_fn(soft_student, soft_teacher) * (temperature ** 2)
        loss = alpha * kd_loss + (1 - alpha) * ce_loss
        loss.backward()
        optimizer.step()
        if (step + 1) % 10 == 0:
            steps.append(step + 1)
            losses.append(loss.item())
    return steps, losses

# ----------------------------
# Experiment Runner, Plotting, and Quality Gain Calculation
# ----------------------------
def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, vocab_size = prepare_wikitext2()
    seq_length = 35  
    batch_size = 32
    num_steps = 10000  # Adjust this for your experiment.
    num_layers = 4
    d_model = 128
    d_ff = 256
    num_heads = 4
    init_scale = 0.1
    dropout = 0.1
    expert_dropout = 0.4

    results = {}

    # 1. Train the baseline (dense) transformer.
    start = time.time()
    print("Training baseline dense transformer...")
    baseline_model = TransformerLM(vocab_size, seq_length, num_layers, d_model, d_ff, num_heads,
                                   use_moe=False).to(device)
    optimizer_dense = optim.Adam(baseline_model.parameters(), lr=1e-3)
    steps_dense, dense_losses = train_lm(baseline_model, optimizer_dense, data, num_steps, batch_size, seq_length, vocab_size, device)
    results["Dense"] = (steps_dense, dense_losses)
    end = time.time()
    print(f"Training time for Baseline Dense Transformer is {end-start}")

    # 2. Train Switch Transformer variants with different expert counts.
    expert_numbers = [2, 4, 6, 8, 10]
    for num_experts in expert_numbers:
        start = time.time()
        variant_name = f"Switch_{num_experts}experts"
        print(f"Training {variant_name} ...")
        switch_model = TransformerLM(vocab_size, seq_length, num_layers, d_model, d_ff, num_heads,
                                     use_moe=True, num_experts=num_experts,
                                     init_scale=init_scale, dropout=dropout, expert_dropout=expert_dropout).to(device)
        optimizer_variant = optim.Adam(switch_model.parameters(), lr=1e-3)
        steps_variant, losses_variant = train_lm(switch_model, optimizer_variant, data, num_steps, batch_size, seq_length, vocab_size, device)
        results[variant_name] = (steps_variant, losses_variant)
        end = time.time()
        print(f"Training time for Switch Transformer with {num_experts} experts is {end-start}")

    # 3. Distillation: Pre-train a teacher (Switch model with 10 experts) and then distill into a dense student.
    teacher_model = TransformerLM(vocab_size, seq_length, num_layers, d_model, d_ff, num_heads,
                                  use_moe=True, num_experts=10, init_scale=init_scale,
                                  dropout=dropout, expert_dropout=expert_dropout).to(device)
    optimizer_teacher = optim.Adam(teacher_model.parameters(), lr=1e-3)
    print("Pre-training teacher (Switch 10 experts) for distillation...")
    # Train teacher and record its losses.
    steps_teacher, teacher_losses = train_lm(teacher_model, optimizer_teacher, data, num_steps, batch_size, seq_length, vocab_size, device)
    results["Switch_10experts"] = (steps_teacher, teacher_losses)
    
    student_model = TransformerLM(vocab_size, seq_length, num_layers, d_model, d_ff, num_heads,
                                  use_moe=False).to(device)
    optimizer_student = optim.Adam(student_model.parameters(), lr=1e-3)
    print("Training student (dense) with distillation...")
    steps_distill, loss_distill = train_distillation(student_model, teacher_model, optimizer_student,
                                                     data, num_steps, batch_size, seq_length, vocab_size, device,
                                                     alpha=0.75, temperature=1.0)
    results["Distilled"] = (steps_distill, loss_distill)
    
    # ----------------------------
    # Plot 1: Dense and Switch variants with experts [2,4,6,8,10].
    plt.figure(figsize=(10, 6))
    for name in ["Dense", "Switch_2experts", "Switch_4experts", "Switch_6experts", "Switch_8experts", "Switch_10experts"]:
        steps, losses = results[name]
        plt.plot(steps, losses, label=name)
    plt.xlabel("Training Steps")
    plt.ylabel("Negative Log Perplexity (Loss)")
    plt.title("Training Curves: Dense & Switch Variants (2-10 Experts)")
    plt.legend()
    plt.grid(True)
    plt.savefig('SwitchvsDense.png')
    plt.show()

    # ----------------------------
    # Plot 2: Dense, Switch (10 experts - Teacher), and the Distilled model.
    plt.figure(figsize=(10, 6))
    for name in ["Dense", "Switch_10experts", "Distilled"]:
        steps, losses = results[name]
        plt.plot(steps, losses, label=name)
    plt.xlabel("Training Steps")
    plt.ylabel("Negative Log Perplexity (Loss)")
    plt.title("Training Curves: Dense, Switch (10 Experts) & Distilled Model")
    plt.legend()
    plt.grid(True)
    plt.savefig('DistillationvsSwitch.png')
    plt.show()
    
if __name__ == "__main__":
    run_experiment()
