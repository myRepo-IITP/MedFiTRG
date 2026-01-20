import os
import re
import json
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

class LOSGraphDataset(Dataset):
    def __init__(self, metadata_csv, ehr_base_dir, cxr_base_dir, split_name, tokenizer, max_len=512):
        self.metadata_df = pd.read_csv(metadata_csv)
        
        # 1. Filter LOS <= 14
        initial_len = len(self.metadata_df)
        self.metadata_df = self.metadata_df[self.metadata_df['y_true'] <= 14.0].reset_index(drop=True)
        print(f"[{split_name}] Filtered {initial_len - len(self.metadata_df)} samples. Current size: {len(self.metadata_df)}")

        self.ehr_base_dir = ehr_base_dir
        self.cxr_base_dir = cxr_base_dir
        self.split_name = split_name
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 2. Episode Sorting (Vectorized)
        if 'time_series' in self.metadata_df.columns:
            self.metadata_df['episode_sort_val'] = self.metadata_df['time_series'].astype(str).str.extract(r'episode(\d+)').fillna(0).astype(int)
        else:
            self.metadata_df['episode_sort_val'] = 0
            
        # 3. Fast Grouping
        self.metadata_df['subject_id'] = self.metadata_df['subject_id'].astype(str)
        self.patient_groups = self.metadata_df.groupby('subject_id')
        self.patient_ids = sorted(list(self.patient_groups.groups.keys()))
        
        # 4. Optimized Image Indexing
        cache_file = os.path.join(cxr_base_dir, "cxr_path_index.json")
        self.cxr_path_map = {}

        if os.path.exists(cache_file):
            print(f"[{split_name}] Loading CXR index from cache (FAST)...")
            try:
                with open(cache_file, 'r') as f:
                    self.cxr_path_map = json.load(f)
            except Exception as e:
                print(f"[WARNING] Corrupt cache file, re-indexing. Error: {e}")
                self._index_cxr_images_with_progress(cxr_base_dir, cache_file, split_name)
        else:
            self._index_cxr_images_with_progress(cxr_base_dir, cache_file, split_name)

        # 5. Analyze columns once
        self.feature_columns, self.mask_columns = self._analyze_ehr_structure()
        print("columns analysed...")

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2048 - 1024)
        ])
    
    def _index_cxr_images_with_progress(self, cxr_base_dir, cache_file, split_name):
        """Helper method to scan images with a tqdm progress bar"""
        print(f"[{split_name}] Indexing CXR images from disk (One-time setup)...")
        
        if os.path.exists(cxr_base_dir):
            with tqdm(desc=f"[{split_name}] Scanning Images", unit="img") as pbar:
                for root, _, files in os.walk(cxr_base_dir):
                    for file in files:
                        if file.endswith(".jpg"):
                            dicom_id = os.path.splitext(file)[0]
                            self.cxr_path_map[dicom_id] = os.path.join(root, file)
                            pbar.update(1)
            
            # Save to JSON so we never have to wait again
            try:
                print(f"[{split_name}] Saving index to {cache_file}...")
                with open(cache_file, 'w') as f:
                    json.dump(self.cxr_path_map, f)
                print(f"[{split_name}] Index saved!")
            except Exception as e:
                print(f"[WARNING] Could not save cache file: {e}")
        else:
            print(f"[WARNING] CXR directory not found: {cxr_base_dir}")

    def _analyze_ehr_structure(self):
        if len(self.metadata_df) == 0: return [], []
        # Check first valid file
        first_ts = self.metadata_df.iloc[0]['time_series']
        ts_path = os.path.join(self.ehr_base_dir, self.split_name, str(first_ts))
        
        if os.path.exists(ts_path):
            try:
                # Read only header
                sample_df = pd.read_csv(ts_path, nrows=0)
                all_cols = sample_df.columns.tolist()
                return [c for c in all_cols if not c.startswith('mask->')], [c for c in all_cols if c.startswith('mask->')]
            except: 
                return [], []
        return [], []
    
    def _los_to_class(self, los_value):
        los = float(los_value)
        if los < 1: return 0
        elif los < 2: return 1
        elif los < 3: return 2
        elif los < 4: return 3
        elif los < 5: return 4
        elif los < 6: return 5
        elif los < 7: return 6
        elif los < 8: return 7
        return 8 

    def __len__(self): 
        return len(self.patient_ids)
    
    def _preprocess_text(self, text):
        if pd.isna(text) or not isinstance(text, str) or text.strip() == '': 
            return "no medical history available"
        text = re.sub(r'[^\w\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip().lower()

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        
        # Get group
        patient_visits = self.patient_groups.get_group(patient_id).sort_values('episode_sort_val')
        visits_data = []
        
        for visit_order, (visit_idx, row) in enumerate(patient_visits.iterrows()):
            # --- EHR LOADING ---
            ehr_data = torch.zeros((1, len(self.feature_columns) + len(self.mask_columns)), dtype=torch.float32)
            ehr_length = 1
            
            ts_path = os.path.join(self.ehr_base_dir, self.split_name, str(row['time_series']))
            if os.path.exists(ts_path):
                try:
                    df = pd.read_csv(ts_path) 
                    if not df.empty:
                        vals = df[self.feature_columns].values
                        masks = df[self.mask_columns].values
                        combined = np.concatenate([vals, masks], axis=1)
                        ehr_data = torch.tensor(combined, dtype=torch.float32)
                        ehr_length = ehr_data.shape[0]
                except: pass
            
            # --- CXR LOADING (Optimized) ---
            has_cxr, cxr_tensor = 0, torch.zeros(1, 224, 224)
            dicom_id = str(row['dicom_id']).strip()
            
            # O(1) Lookup using the cached map
            if dicom_id in self.cxr_path_map:
                try:
                    img_path = self.cxr_path_map[dicom_id]
                    img = Image.open(img_path).convert('L')
                    if img.size != (224, 224): img = img.resize((224, 224))
                    cxr_tensor = self.transform(img)
                    has_cxr = 1
                except: pass
            
            # --- TEXT PROCESSING ---
            text = self._preprocess_text(row.get('past_medical_history', ''))
            has_text = 1 if text != "no medical history available" else 0
            
            # Tokenize manually
            tokens = self.tokenizer.tokenize(text)
            
            # Apply Head-Tail Truncation
            if len(tokens) > self.max_len - 2:
                head_len = int((self.max_len - 2) * 0.25)
                tail_len = (self.max_len - 2) - head_len
                tokens = tokens[:head_len] + tokens[-tail_len:]

            # Convert to IDs and add Special Tokens
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]

            # Manual Padding
            padding_length = self.max_len - len(input_ids)
            if padding_length > 0:
                attention_mask = [1] * len(input_ids) + [0] * padding_length
                input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            else:
                attention_mask = [1] * self.max_len
                input_ids = input_ids[:self.max_len]
            
            # Convert to Tensor
            input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
            attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
            
            # Labels
            los_value = float(row.get('y_true', 0.0))
            class_label = self._los_to_class(los_value)
            
            visits_data.append({
                'ehr_data': ehr_data, 'ehr_length': ehr_length,
                'cxr_data': cxr_tensor, 'has_cxr': has_cxr,
                'input_ids': input_ids_tensor, 'attention_mask': attention_mask_tensor, 'has_text': has_text,
                'class_label': torch.tensor(class_label, dtype=torch.long),
                'los_value': torch.tensor([los_value], dtype=torch.float32),
                'patient_indices': idx,
                'visit_order': visit_order, 
                'subject_id': patient_id
            })
            
        return visits_data


class LOSGraphCollator:
    def __init__(self, max_len=512): 
        self.max_len = max_len
    
    def __call__(self, batch):
        flattened_visits = [v for p in batch for v in p]
        batch_size = len(flattened_visits)
        
        max_ehr = max(v['ehr_data'].shape[0] for v in flattened_visits)
        ehr_dim = flattened_visits[0]['ehr_data'].shape[1]
        padded_ehr = torch.zeros(batch_size, max_ehr, ehr_dim)
        ehr_lengths = torch.zeros(batch_size, dtype=torch.long)
        
        cxr_list, text_input_list, text_mask_list = [], [], []
        cxr_indices, text_indices = [], []
        class_labels_list, los_values_list = [], []
        p_idx = []
        v_ord = []  # Visit orders for temporal fusion
        
        for i, v in enumerate(flattened_visits):
            padded_ehr[i, :v['ehr_length']] = v['ehr_data']
            ehr_lengths[i] = v['ehr_length']
            p_idx.append(v['patient_indices'])
            v_ord.append(v['visit_order'])  # Collect visit orders for temporal layer
            class_labels_list.append(v['class_label'])
            los_values_list.append(v['los_value'])
            
            if v['has_cxr']:
                cxr_indices.append(i)
                cxr_list.append(v['cxr_data'])
            if v['has_text']:
                text_indices.append(i)
                text_input_list.append(v['input_ids'])
                text_mask_list.append(v['attention_mask'])
                
        cxr_batch = torch.stack(cxr_list) if cxr_list else torch.zeros(0, 1, 224, 224)
        text_input_batch = torch.stack(text_input_list) if text_input_list else torch.zeros(0, self.max_len, dtype=torch.long)
        text_mask_batch = torch.stack(text_mask_list) if text_mask_list else torch.zeros(0, self.max_len, dtype=torch.long)
        
        return {
            'ehr_data': padded_ehr, 'ehr_lengths': ehr_lengths,
            'cxr_data': cxr_batch, 'cxr_indices': torch.tensor(cxr_indices, dtype=torch.long),
            'text_input_ids': text_input_batch, 'text_attention_masks': text_mask_batch, 
            'text_indices': torch.tensor(text_indices, dtype=torch.long),
            'class_labels': torch.stack(class_labels_list),
            'los_values': torch.stack(los_values_list),
            'patient_indices': torch.tensor(p_idx, dtype=torch.long),
            'visit_orders': torch.tensor(v_ord, dtype=torch.long)  # Return visit orders for temporal fusion
        }