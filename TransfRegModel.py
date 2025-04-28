import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
import math
#from mha import MultiHeadAttention
from typing import Optional

import copy


class MultiHeadAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // nheads
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """
    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        nheads: int,
        dropout: float = 0.0,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.nheads = nheads
        self.dropout = dropout
        self._qkv_same_embed_dim = E_q == E_k and E_q == E_v
        if self._qkv_same_embed_dim:
          self.packed_proj = nn.Linear(E_q, E_total * 3, bias=bias, **factory_kwargs)
        else:
          self.q_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)
          self.k_proj = nn.Linear(E_k, E_total, bias=bias, **factory_kwargs)
          self.v_proj = nn.Linear(E_v, E_total, bias=bias, **factory_kwargs)
        E_out = E_q
        self.out_proj = nn.Linear(E_total, E_out, bias=bias, **factory_kwargs)
        assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = E_total // nheads
        self.bias = bias

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask=None, is_causal=False) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (N, L_q, E_qk)
            key (torch.Tensor): key of shape (N, L_kv, E_qk)
            value (torch.Tensor): value of shape (N, L_kv, E_v)
            attn_mask (torch.Tensor, optional): attention mask of shape (N, L_q, L_kv) to pass to sdpa. Default: None
            is_causal (bool, optional): Whether to apply causal mask. Default: False

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        # Step 1. Apply input projection
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(self.packed_proj.weight, 3, dim=0)
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(self.packed_proj.bias, 3, dim=0)
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query, key, value = F.linear(query, q_weight, q_bias), F.linear(key, k_weight, k_bias), F.linear(value, v_weight, v_bias)

        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        # Step 3. Run SDPA
        # (N, nheads, L_t, E_head)
        attn_output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout, is_causal=is_causal)
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output



class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation : nn.Module = torch.nn.functional.relu,
        layer_norm_eps=1e-5,
        norm_first=True,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model,
            d_model,
            d_model,
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            **factory_kwargs,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation


    def _sa_block(self, x, attn_mask, is_causal):
        x = self.self_attn(x, x, x, is_causal=is_causal)
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, src, src_mask=None, is_causal=False):
        '''
        Arguments:
            src: (batch_size, seq_len, d_model)
            src_mask: (batch_size, seq_len, seq_len)
            is_causal: bool
        '''
#        print("debug: TransEncoderLayer",src.shape)

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x




def _get_clones(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: "TransformerEncoderLayer",
        num_layers: int,
        norm: Optional[nn.Module] = None,
        device=None,
        dtype=None,
    ):
#        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None, is_causal=False):
        output = src
        for mod in self.layers:
            output = mod(output, mask, is_causal)
        if self.norm is not None:
            output = self.norm(output)
        return output



class Transformer(L.LightningModule):
    def __init__(
        self,
        d_model,
        nhead,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation : nn.Module = torch.nn.functional.relu,
        layer_norm_eps=1e-5,
        norm_first=True,
        bias=True,
        device='cpu',
    ):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
    #        batch_first=True,
            norm_first=norm_first,
            bias=bias,
            device=device,
        )
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, device=device)

        #self.d_model = d_model
        #self.nhead = nhead
        #self.num_encoder_layers=num_encoder_layers
        #self.dropout=dropout
        #self.layer_norm_eps=layer_norm_eps
        #self.norm_first=norm_first
        #self.bias=bias
        #self.save_hyperparameters(ignore=["activation"])

        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

    def forward(
        self,
        src,
        src_mask=None,
        src_is_causal=False,
    ):
        memory = self.encoder(
            src,
            mask=src_mask,
            is_causal=src_is_causal,
        )

        return memory



#class AttentionPooling(nn.Module):
#    def __init__(self, hidden_dim):
#        super().__init__()
#        self.attn = nn.Linear(hidden_dim, 1)
#
#    def forward(self, x):
#        # x: (batch_size, seq_len, hidden_dim)
#        scores = self.attn(x).squeeze(-1)  # (batch_size, seq_len)
#        weights = torch.softmax(scores, dim=-1)
#        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
#        return pooled


class AttentionPooling(L.LightningModule):
    def __init__(self, hidden_dim):
        super().__init__()

        #self.save_hyperparameters("hidden_dim")
        #self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, nested_tensor: torch.nested.nested_tensor):
        # Convert to padded tensor and mask
        padded_x = torch.nested.to_padded_tensor(nested_tensor, padding=0.0)
        # padded_x: (batch_size, max_seq_len, hidden_dim)
        # mask: (batch_size, max_seq_len), dtype=bool

        # Get per-sequence lengths by unbinding the nested tensor
        sequences = nested_tensor.unbind()  # list of tensors: [(seq_len_i, hidden_dim), ...]
        seq_lens = torch.tensor([seq.size(0) for seq in sequences], device=padded_x.device)  # (batch_size,)

        batch_size, max_len, _ = padded_x.size()

        # Build mask manually
        idxs = torch.arange(max_len, device=padded_x.device).unsqueeze(0)  # (1, max_len)
        lens = seq_lens.unsqueeze(1)  # (batch_size, 1)
        mask = idxs < lens  # (batch_size, max_len), bool

        # Compute raw attention scores
        scores = self.attn(padded_x).squeeze(-1)  # (batch_size, max_seq_len)

        # Mask out padding tokens
        scores = scores.masked_fill(~mask, float("-inf"))

        # Softmax over time dimension
        weights = F.softmax(scores, dim=1)  # (batch_size, max_seq_len)

        # Apply attention weights via batch matrix multiply
        pooled = torch.bmm(weights.unsqueeze(1), padded_x).squeeze(1)  # (batch_size, hidden_dim)

        return pooled


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
        d_model - Hidden dimensionality of the input.
        max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)
        
    def forward(self, ins1, ins2):
        #x = x + self.pe[:, :x.size(1)]
        posAdd = self.pe[:,ins2.to_padded_tensor(padding=0)]
        posAdd = posAdd.squeeze()
        
        offset = ins1.offsets()
        offset = offset.diff()
        posN = torch.nested.nested_tensor([ posAdd1[0:offset[i_pos]] for i_pos, posAdd1 in enumerate(posAdd) ],layout=torch.jagged )
        
        return torch.nested.nested_tensor([a+b for a,b in zip(ins1,posN)],layout=torch.jagged)
            
    
class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class LightningTransReg(L.LightningModule):
    def __init__(self, dim_index=8512, dim_emb=16,
                 dim_nhead=2, num_encoder=2,dim_ff=2048,dropout_enc=0.1,
                 dim_reg=128,
                 lr = 1.0e-4, warmup=50, max_iters=2000
                 ):
        super().__init__()

        
        
        self.save_hyperparameters()
        #"dim_index","dim_emb","lr","warmup","max_iters",
        #"model.d_model",
        #"model.nhead",
        #"model.num_encoder_layers",
        #"model.dropout",
        #"model.layer_norm_eps",
        #"model.norm_first",
        #"model.bias",
        #"head.hidden_dim",
        #ignore=['model','head'])
        
        #model.save_hyperparameters()
        #head.save_hyperparameters()
        
        self.Eb = torch.nn.Embedding(dim_index,dim_emb)
        self.PosEb = PositionalEncoding(dim_emb)
        self.model = Transformer(d_model=dim_emb,nhead=dim_nhead,num_encoder_layers=num_encoder,dim_feedforward=dim_ff,dropout=dropout_enc)
        self.head = AttentionPooling(hidden_dim=dim_emb)

        self.regressor = nn.Sequential(
            nn.Linear(dim_emb, dim_reg),
            nn.ReLU(),
            nn.Linear(dim_reg, 1)  # or more outputs if you regress multiple targets
            )

        self.metric = torch.nn.MSELoss() #torch.nn.HuberLoss(delta=1.0) #torch.nn.L1Loss() #torch.nn.MSELoss()

    def forward(self, inputs_all):
        inputs,inputs_pos = inputs_all

        inp = self.Eb(inputs)
        inp = self.PosEb(inp,inputs_pos)
        inp = self.model(inp)
        inp = self.head(inp)
        return self.regressor(inp)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs)
        loss =  self.metric(output, target)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs)
        loss =  self.metric(output, target)

        self.log("val_loss", loss, logger=True)

    def test_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs)
        loss =  self.metric(output, target)

        self.log("test_loss", loss)

        
    def configure_optimizers(self):
#        return torch.optim.SGD(self.model.parameters(), lr=0.1)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        # Apply lr scheduler per step
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             max_iters=self.hparams.max_iters)

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]




class LightningTrans3DPointReg(L.LightningModule):
    def __init__(self, dim_index=8512, dim_emb=16,
                 transF=True,dim_nhead=2, num_encoder=2,dim_ff=2048,dropout_enc=0.1,
                 dim_reg=128,
                 lr = 1.0e-4, #warmup=50, max_iters=2000
                 ):
        super().__init__()



        self.save_hyperparameters()
        #"dim_index","dim_emb","lr","warmup","max_iters",
        #"model.d_model",
        #"model.nhead",
        #"model.num_encoder_layers",
        #"model.dropout",
        #"model.layer_norm_eps",
        #"model.norm_first",
        #"model.bias",
        #"head.hidden_dim",
        #ignore=['model','head'])

        #model.save_hyperparameters()
        #head.save_hyperparameters()

        self.Eb = torch.nn.Linear(3,dim_emb)
        self.PosEb = PositionalEncoding(dim_emb)
        self.model = None
        if transF:
            self.model = Transformer(d_model=dim_emb,nhead=dim_nhead,num_encoder_layers=num_encoder,dim_feedforward=dim_ff,dropout=dropout_enc)
        self.head = AttentionPooling(hidden_dim=dim_emb)

        self.regressor = nn.Sequential(
            nn.Linear(dim_emb, dim_reg),
            nn.ReLU(),
            nn.Linear(dim_reg, 1)  # or more outputs if you regress multiple targets
            )

        self.metric = torch.nn.MSELoss() #torch.nn.HuberLoss(delta=1.0) #torch.nn.L1Loss() #torch.nn.MSELoss()

    def forward(self, inputs_all):
        inputs, inputs_pos = inputs_all

        inp = self.Eb(inputs)
        inp = self.PosEb(inp,inputs_pos)
        if self.model is not None:
            inp = self.model(inp)
        inp = self.head(inp)
        return self.regressor(inp)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs)
        loss =  self.metric(output, target)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs)
        loss =  self.metric(output, target)

        self.log("val_loss", loss, logger=True)

    def test_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs)
        loss =  self.metric(output, target)

        self.log("test_loss", loss)


    def configure_optimizers(self):
#        return torch.optim.SGD(self.model.parameters(), lr=0.1)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

        # Apply lr scheduler per step
        #lr_scheduler = CosineWarmupScheduler(optimizer,
        #                                     warmup=self.hparams.warmup,
        #                                     max_iters=self.hparams.max_iters)

        return [optimizer] #[{'scheduler': lr_scheduler, 'interval': 'step'}]
