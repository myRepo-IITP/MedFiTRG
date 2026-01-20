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

class REAGraphDataset(Dataset):
    def __init__(self, metadata_csv, ehr_base_dir, cxr_base_dir, split_name, tokenizer, max_len=512):
        self.metadata_df = pd.read_csv(metadata_csv)
        self.ehr_base_dir = ehr_base_dir
        self.cxr_base_dir = cxr_base_dir
        self.split_name = split_name
        
        # Create unique ID for every stay
        self.metadata_df['patient_stay_id'] = self.metadata_df['subject_id'].astype(str)
        
        # Extract episode number for temporal ordering
        if 'time_series' in self.metadata_df.columns:
            self.metadata_df['episode_sort_val'] = self.metadata_df['time_series'].apply(self._extract_episode_num)
        else:
            self.metadata_df['episode_sort_val'] = 0
            
        self.patient_groups = self.metadata_df.groupby('patient_stay_id')
        self.patient_stay_ids = sorted(list(self.patient_groups.groups.keys()))
        
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text_cache = {}
        
        # Pre-cache text tokens
        for idx, row in self.metadata_df.iterrows():
            text = self._preprocess_text(row['past_medical_history'])
            tokenized = self.tokenizer(
                text, 
                padding='max_length', 
                truncation=True, 
                max_length=self.max_len, 
                return_tensors='pt'
            )
            
            self.text_cache[idx] = {
                'input_ids': tokenized['input_ids'].squeeze(0), 
                'attention_mask': tokenized['attention_mask'].squeeze(0)
            }
        
        self.feature_columns, self.mask_columns = self._analyze_ehr_structure()
        
        # Optimized image indexing
        cache_file = os.path.join(cxr_base_dir, "cxr_path_index.json")
        self.cxr_path_map = {}
        if os.path.exists(cache_file):
            print(f"[{split_name}] Loading CXR index from cache...")
            try:
                with open(cache_file, 'r') as f:
                    self.cxr_path_map = json.load(f)
            except:
                self._index_cxr_images_with_progress(cxr_base_dir, cache_file, split_name)
        else:
            self._index_cxr_images_with_progress(cxr_base_dir, cache_file, split_name)
        
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2048 - 1024)
        ])
    
    def _index_cxr_images_with_progress(self, cxr_base_dir, cache_file, split_name):
        """Index CXR images with progress bar"""
        print(f"[{split_name}] Indexing CXR images from disk...")
        
        if os.path.exists(cxr_base_dir):
            with tqdm(desc=f"[{split_name}] Scanning Images", unit="img") as pbar:
                for root, _, files in os.walk(cxr_base_dir):
                    for file in files:
                        if file.endswith(".jpg"):
                            dicom_id = os.path.splitext(file)[0]
                            self.cxr_path_map[dicom_id] = os.path.join(root, file)
                            pbar.update(1)
            
            # Save to JSON
            try:
                with open(cache_file, 'w') as f:
                    json.dump(self.cxr_path_map, f)
            except:
                pass
    
    def _extract_episode_num(self, filename):
        """Extract episode number from time series filename for temporal ordering"""
        if pd.isna(filename): 
            return 0
        match = re.search(r'episode(\d+)', str(filename))
        return int(match.group(1)) if match else 0
    
    def _preprocess_text(self, text):
        if pd.isna(text) or not isinstance(text, str) or text.strip() == '': 
            return "no medical history available"
        text = re.sub(r'[^\w\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip().lower()
    
    def _analyze_ehr_structure(self):
        if len(self.metadata_df) == 0: 
            return [], []
        sample_row = self.metadata_df.iloc[0]
        ts_path = os.path.join(self.ehr_base_dir, self.split_name, sample_row['time_series'])
        if os.path.exists(ts_path):
            try:
                sample_df = pd.read_csv(ts_path, nrows=0)
                all_cols = sample_df.columns.tolist()
                return [c for c in all_cols if not c.startswith('mask->')], [c for c in all_cols if c.startswith('mask->')]
            except: 
                return [], []
        return [], []
    
    def __len__(self): 
        return len(self.patient_stay_ids)
    
    def __getitem__(self, idx):
        patient_stay_id = self.patient_stay_ids[idx]
        # Sort by episode number to ensure temporal order
        patient_visits = self.patient_groups.get_group(patient_stay_id).sort_values('episode_sort_val')
        visits_data = []
        
        # Enumerate gives us the visit order (0, 1, 2...)
        for visit_order, (visit_idx, row) in enumerate(patient_visits.iterrows()):
            # EHR data
            ehr_data = torch.zeros((1, len(self.feature_columns) + len(self.mask_columns)), dtype=torch.float32)
            ehr_length = 1
            try:
                ts_path = os.path.join(self.ehr_base_dir, self.split_name, row['time_series'])
                if os.path.exists(ts_path):
                    df = pd.read_csv(ts_path, header=0)
                    if not df.empty:
                        ehr_data = torch.tensor(
                            np.concatenate([df[self.feature_columns].values, df[self.mask_columns].values], axis=1), 
                            dtype=torch.float32
                        )
                        ehr_length = ehr_data.shape[0]
            except: 
                pass
            
            # CXR data
            has_cxr, cxr_tensor = 0, torch.zeros(1, 224, 224)
            if pd.notna(row['dicom_id']) and str(row['dicom_id']).strip() != '':
                dicom_id = str(row['dicom_id']).strip()
                if dicom_id in self.cxr_path_map:
                    try:
                        img_path = self.cxr_path_map[dicom_id]
                        img = Image.open(img_path).convert('L')
                        if img.size != (224, 224): 
                            img = img.resize((224, 224))
                        cxr_tensor = self.transform(img)
                        has_cxr = 1
                    except: 
                        pass
            
            # Text data
            text_data = self.text_cache.get(visit_idx, {
                'input_ids': torch.zeros(self.max_len, dtype=torch.long), 
                'attention_mask': torch.zeros(self.max_len, dtype=torch.long)
            })
            has_text = 1 if pd.notna(row['past_medical_history']) and str(row['past_medical_history']).strip() != '' else 0
            
            # Readmission label (binary)
            rea_label = torch.tensor(float(row['y_true']), dtype=torch.float32)
            
            visits_data.append({
                'ehr_data': ehr_data, 'ehr_length': ehr_length,
                'cxr_data': cxr_tensor, 'has_cxr': has_cxr,
                'input_ids': text_data['input_ids'], 'attention_mask': text_data['attention_mask'], 'has_text': has_text,
                'labels': rea_label.unsqueeze(0),
                'patient_indices': idx,
                'visit_order': visit_order,  # Explicitly added for temporal layer
                'patient_stay_id': patient_stay_id
            })
        return visits_data


class REAGraphCollator:
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
        cxr_indices, text_indices, labels_list, p_idx, v_ord = [], [], [], [], []
        
        for i, v in enumerate(flattened_visits):
            padded_ehr[i, :v['ehr_length']] = v['ehr_data']
            ehr_lengths[i] = v['ehr_length']
            p_idx.append(v['patient_indices'])
            v_ord.append(v['visit_order'])  # Collect visit orders for temporal layer
            labels_list.append(v['labels'])
            
            if v['has_cxr']:
                cxr_indices.append(i)
                cxr_list.append(v['cxr_data'])
            if v['has_text']:
                text_indices.append(i)
                text_input_list.append(v['input_ids'])
                text_mask_list.append(v['attention_mask'])
        
        # Handle empty batches
        cxr_batch = torch.stack(cxr_list) if cxr_list else torch.zeros(0, 1, 224, 224)
        text_input_batch = torch.stack(text_input_list) if text_input_list else torch.zeros(0, self.max_len, dtype=torch.long)
        text_mask_batch = torch.stack(text_mask_list) if text_mask_list else torch.zeros(0, self.max_len, dtype=torch.long)
        
        return {
            'ehr_data': padded_ehr, 'ehr_lengths': ehr_lengths,
            'cxr_data': cxr_batch,
            'cxr_indices': torch.tensor(cxr_indices, dtype=torch.long),
            'text_input_ids': text_input_batch,
            'text_attention_masks': text_mask_batch,
            'text_indices': torch.tensor(text_indices, dtype=torch.long),
            'labels': torch.stack(labels_list),
            'patient_indices': torch.tensor(p_idx, dtype=torch.long),
            'visit_orders': torch.tensor(v_ord, dtype=torch.long)  # Return visit orders for temporal fusion
        }