import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv
from transformers import BertModel

# ==================== Positional Encoding ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# ==================== EHR Embedding Extractor ====================
class TransformerEHREmbeddingExtractor(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 512, nhead: int = 4, 
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.model_dim = embedding_dim
        self.input_projection = nn.Linear(input_dim, self.model_dim)
        self.pos_encoder = PositionalEncoding(self.model_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_dim, nhead=nhead, 
                                                  dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.raw_feature_projection = nn.Linear(input_dim, embedding_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple:
        device = x.device
        batch_size = x.size(0)
        max_len = x.size(1)
        lengths = lengths.to(device)
        padding_mask = torch.arange(max_len, device=device)[None, :] >= lengths[:, None]
        
        # Raw features (for social graph)
        raw_features = []
        for i in range(batch_size):
            valid_len = lengths[i]
            if valid_len > 0:
                mean_pooled = x[i, :valid_len].mean(dim=0)
            else:
                mean_pooled = torch.zeros(x.size(2), device=device)
            raw_features.append(mean_pooled)
        raw_features = torch.stack(raw_features)
        raw_projected = self.raw_feature_projection(raw_features)
        
        # Transformer
        x_projected = self.input_projection(x) * math.sqrt(self.model_dim)
        x_projected = self.pos_encoder(x_projected)
        output = self.transformer_encoder(x_projected, src_key_padding_mask=padding_mask)
        output[padding_mask] = 0.0
        
        # Sum pooling
        denom = lengths.unsqueeze(1).to(device)
        denom = torch.clamp(denom, min=1.0) # avoid div by zero
        pooled_output = output.sum(dim=1) / denom
        
        return pooled_output, raw_projected

# ==================== CXR Feature Extractor ====================
class CXRFeatureExtractor(nn.Module):
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.model = xrv.models.DenseNet(weights="densenet121-res224-all") 
        for param in self.model.parameters(): 
            param.requires_grad = False
        feature_dim = 1024
        self.model.op_threshs = None 
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim), 
            nn.ReLU(), 
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model.features(x) 
        features = F.relu(features, inplace=True)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        return self.projection(features)

    def unfreeze_layers(self, num_blocks: int = 1):
        for name, child in list(self.model.features.named_children())[::-1]:
            if name.startswith('denseblock'):
                if num_blocks > 0:
                    for param in child.parameters(): 
                        param.requires_grad = True
                    num_blocks -= 1
            elif name == 'norm5':
                for param in child.parameters(): 
                    param.requires_grad = True
            if num_blocks == 0: 
                break

# ==================== Text Embedding Extractor ====================
class TextEmbeddingExtractor(nn.Module):
    def __init__(self, model_name: str, embedding_dim: int):
        super().__init__()
        self.bert_model = BertModel.from_pretrained(model_name)
        for param in self.bert_model.parameters(): 
            param.requires_grad = False
        self.projection = nn.Linear(self.bert_model.config.hidden_size, embedding_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.projection(cls_embedding)

    def unfreeze_layers(self, num_layers: int = 1):
        for layer in self.bert_model.encoder.layer[-num_layers:]:
            for param in layer.parameters(): 
                param.requires_grad = True

# ==================== Graph & FiLM Layers ====================
class GNNFiLMLayer(nn.Module):
    def __init__(self, source_dim, target_dim, hidden_dim=256):
        super().__init__()
        self.film_gen = nn.Sequential(
            nn.Linear(source_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim * 2) 
        )
        nn.init.zeros_(self.film_gen[2].weight)
        nn.init.zeros_(self.film_gen[2].bias)

    def forward(self, source, target):
        film_params = self.film_gen(source)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        return (1 + gamma) * target + beta

class AdaptiveSocialFusion(nn.Module):
    def __init__(self, hidden_dim, film_hidden_dim=256, initial_threshold=0.90):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.threshold = nn.Parameter(torch.tensor(initial_threshold))
        self.temperature = nn.Parameter(torch.tensor(10.0))
        self.social_film = GNNFiLMLayer(hidden_dim, hidden_dim, film_hidden_dim)

    def forward(self, nodes, patient_indices, ehr_features=None):
        device = nodes.device
        batch_size = nodes.shape[0]
        
        sim_feats = ehr_features if (ehr_features is not None and ehr_features.shape[0] == batch_size) else nodes
        
        # Cosine Similarity
        feats_norm = F.normalize(sim_feats, p=2, dim=1)
        sim_matrix = torch.mm(feats_norm, feats_norm.t())
        
        # Masking
        mask_self = torch.eye(batch_size, device=device).bool()
        p_ids = patient_indices.unsqueeze(1)
        mask_same_patient = (p_ids == p_ids.t())
        full_mask = mask_self | mask_same_patient
        
        # Differentiable Thresholding
        thresh = torch.clamp(self.threshold, 0.0, 0.99)
        scaled_sim = (sim_matrix - thresh) * self.temperature
        adj_weights = torch.sigmoid(scaled_sim)
        adj_weights = adj_weights.masked_fill(full_mask, 0.0)
        
        row_sums = adj_weights.sum(dim=1, keepdim=True) + 1e-6
        norm_adj = adj_weights / row_sums
        
        weighted_neighbors = torch.mm(norm_adj, nodes)
        output = self.social_film(weighted_neighbors, nodes)
        
        gate = torch.tanh(row_sums)
        final_nodes = nodes + (output * gate)
        return final_nodes

# ==================== Main Fusion Network for Phenotypes ====================
class TripleFiLMNetwork(nn.Module):
    def __init__(self, ehr_dim, cxr_dim, text_dim, hidden_dim, num_classes, film_hidden_dim=256, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Individual Modality Projections
        self.ehr_proj = nn.Sequential(nn.Linear(ehr_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.cxr_proj = nn.Sequential(nn.Linear(cxr_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.text_proj = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        
        # 2. Learnable Missing Tokens
        self.cxr_missing_token = nn.Parameter(torch.randn(1, hidden_dim))
        self.text_missing_token = nn.Parameter(torch.randn(1, hidden_dim))
        
        # 3. Auxiliary Heads
        self.cxr_aux_head = nn.Linear(hidden_dim, num_classes)
        self.text_aux_head = nn.Linear(hidden_dim, num_classes)

        # 4. Global Context Generator
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 5. Symmetric FiLM Layers (Level 1: Modality Fusion)
        self.global_to_ehr_film = GNNFiLMLayer(hidden_dim, hidden_dim, film_hidden_dim)
        self.global_to_cxr_film = GNNFiLMLayer(hidden_dim, hidden_dim, film_hidden_dim)
        self.global_to_text_film = GNNFiLMLayer(hidden_dim, hidden_dim, film_hidden_dim)

        # 6. Temporal Fusion Layer (Level 2: Intra-Patient / Inter-Visit)
        self.temporal_film = GNNFiLMLayer(hidden_dim, hidden_dim, film_hidden_dim)

        # 7. Social Fusion (Level 3: Inter-Patient)
        self.social_fusion = AdaptiveSocialFusion(hidden_dim, film_hidden_dim)
        
        # 8. Final Classifier (multi-label classification)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def level2_temporal_fusion(self, visit_embeddings, patient_indices, visit_orders):
        """
        Connects visits of the same patient: (t-1) -> (t)
        Source: Previous Visit
        Target: Current Visit
        """
        device = visit_embeddings.device
        batch_size = visit_embeddings.shape[0]
        if batch_size <= 1: 
            return visit_embeddings
        
        # Create a shift mask: 1 if col is immediate predecessor of row
        shift_mask = torch.zeros(batch_size, batch_size, device=device)
        unique_patients = torch.unique(patient_indices)
        
        for patient in unique_patients:
            mask = (patient_indices == patient)
            if mask.sum() <= 1: 
                continue # Need at least 2 visits for temporal link
            
            p_idx = torch.where(mask)[0]
            p_ord = visit_orders[mask]
            
            # Sort by visit order to ensure time t connects to t-1
            sorted_indices = p_idx[torch.argsort(p_ord)]
            
            # Connect i-1 to i
            for i in range(1, len(sorted_indices)):
                curr_idx = sorted_indices[i]
                prev_idx = sorted_indices[i-1]
                shift_mask[curr_idx, prev_idx] = 1.0
        
        # Aggregation (Get previous visit embedding)
        prev_visit_embs = torch.mm(shift_mask, visit_embeddings)
        
        # Find which visits actually HAD a predecessor
        has_prev_mask = shift_mask.sum(dim=1) > 0
        
        # Apply FiLM modulation only where history exists
        temporal_fused = visit_embeddings.clone()
        if has_prev_mask.sum() > 0:
            modulated = self.temporal_film(prev_visit_embs[has_prev_mask], visit_embeddings[has_prev_mask])
            temporal_fused[has_prev_mask] = temporal_fused[has_prev_mask] + modulated # Residual connection
            
        return temporal_fused

    def forward(self, ehr_emb, cxr_emb, cxr_indices, text_emb, text_indices, 
                patient_indices, visit_orders, ehr_features=None):
        
        batch_size = ehr_emb.size(0)
        
        # --- A. Project & Handle Missing Data ---
        ehr_h = self.ehr_proj(ehr_emb)
        
        # CXR
        full_cxr_h = self.cxr_missing_token.expand(batch_size, -1).clone()
        cxr_aux_logits = None
        if cxr_emb is not None and len(cxr_indices) > 0:
            cxr_indices = cxr_indices.long()
            cxr_proj = self.cxr_proj(cxr_emb)
            full_cxr_h[cxr_indices] = cxr_proj
            cxr_aux_logits = self.cxr_aux_head(cxr_proj) 

        # Text
        full_text_h = self.text_missing_token.expand(batch_size, -1).clone()
        text_aux_logits = None
        if text_emb is not None and len(text_indices) > 0:
            text_indices = text_indices.long()
            text_proj = self.text_proj(text_emb)
            full_text_h[text_indices] = text_proj
            text_aux_logits = self.text_aux_head(text_proj) 

        # --- B. Create Global Context Vector ---
        concat_features = torch.cat([ehr_h, full_cxr_h, full_text_h], dim=1)
        global_context = self.global_mlp(concat_features) 

        # --- C. Symmetric FiLM Modulation (Level 1) ---
        ehr_mod = self.global_to_ehr_film(global_context, ehr_h)
        cxr_mod = self.global_to_cxr_film(global_context, full_cxr_h)
        text_mod = self.global_to_text_film(global_context, full_text_h)

        # --- D. Aggregation (Additive) ---
        fused_vector = ehr_mod + cxr_mod + text_mod
        fused_vector = F.layer_norm(fused_vector, fused_vector.shape[1:]) 
        
        # --- E. Temporal Fusion (Level 2) ---
        temporal_output = self.level2_temporal_fusion(fused_vector, patient_indices, visit_orders)
        temporal_output = F.layer_norm(temporal_output, temporal_output.shape[1:])

        # --- F. Social Fusion (Level 3) ---
        social_output = self.social_fusion(temporal_output, patient_indices, ehr_features)
        
        # --- G. Classification (multi-label) ---
        logits = self.classifier(social_output)
        
        return {
            "logits": logits,
            "cxr_aux": cxr_aux_logits,
            "text_aux": text_aux_logits
        }