import numpy as np
import json
import os
import csv
from pathlib import Path
from collections import defaultdict

class EnhancedDiscretizer:
    def __init__(self, timestep=0.8, store_masks=True, impute_strategy='hybrid', start_time='zero',
                 config_path='discretizer_config.json'):
        
        # Load configuration
        with open(config_path) as f:
            config = json.load(f)
            self._id_to_channel = config['id_to_channel']
            self._channel_to_id = {channel: idx for idx, channel in enumerate(self._id_to_channel)}
            self._is_categorical_channel = config['is_categorical_channel']
            self._possible_values = config['possible_values']
            self._normal_values = config['normal_values']
            self._truncate_seq_len = config['truncate_seq_len']

        # GCS value standardization mapping
        self._gcs_value_mapping = {
            'Glascow coma scale eye opening': {
                'To Pain': '2 To pain',
                '3 To speech': '3 To speech',
                '1 No Response': '1 No Response', 
                '4 Spontaneously': '4 Spontaneously',
                'None': '1 No Response',
                'To Speech': '3 To speech',
                'Spontaneously': '4 Spontaneously',
                '2 To pain': '2 To pain'
            },
            'Glascow coma scale motor response': {
                '1 No Response': '1 No Response',
                '3 Abnorm flexion': '3 Abnorm flexion',
                'Abnormal extension': '2 Abnorm extensn',
                'No response': '1 No Response',
                '4 Flex-withdraws': '4 Flex-withdraws',
                'Localizes Pain': '5 Localizes Pain',
                'Flex-withdraws': '4 Flex-withdraws',
                'Obeys Commands': '6 Obeys Commands',
                'Abnormal Flexion': '3 Abnorm flexion',
                '6 Obeys Commands': '6 Obeys Commands',
                '5 Localizes Pain': '5 Localizes Pain',
                '2 Abnorm extensn': '2 Abnorm extensn'
            },
            'Glascow coma scale verbal response': {
                '1 No Response': '1 No Response',
                'No Response': '1 No Response',
                'Confused': '4 Confused',
                'Inappropriate Words': '3 Inapprop words',
                'Oriented': '5 Oriented',
                'No Response-ETT': '1 No Response',
                '5 Oriented': '5 Oriented',
                'Incomprehensible sounds': '2 Incomp sounds',
                '1.0 ET/Trach': '1 No Response',
                '4 Confused': '4 Confused',
                '2 Incomp sounds': '2 Incomp sounds',
                '3 Inapprop words': '3 Inapprop words'
            }
        }

        # GCS score mapping (text to numerical value)
        self._gcs_score_mapping = {
            'Glascow coma scale eye opening': {
                '1 No Response': 1,
                '2 To pain': 2,
                '3 To speech': 3,
                '4 Spontaneously': 4
            },
            'Glascow coma scale motor response': {
                '1 No Response': 1,
                '2 Abnorm extensn': 2,
                '3 Abnorm flexion': 3,
                '4 Flex-withdraws': 4,
                '5 Localizes Pain': 5,
                '6 Obeys Commands': 6
            },
            'Glascow coma scale verbal response': {
                '1 No Response': 1,
                '2 Incomp sounds': 2,
                '3 Inapprop words': 3,
                '4 Confused': 4,
                '5 Oriented': 5
            }
        }

        # Store channel IDs for GCS components for easy access
        self._gcs_eye_id = self._channel_to_id['Glascow coma scale eye opening']
        self._gcs_motor_id = self._channel_to_id['Glascow coma scale motor response']
        self._gcs_verbal_id = self._channel_to_id['Glascow coma scale verbal response']
        self._gcs_total_id = self._channel_to_id['Glascow coma scale total']

        self._header = ["Hours"] + self._id_to_channel
        self._timestep = timestep
        self._store_masks = store_masks
        self._start_time = start_time
        self._impute_strategy = impute_strategy

        # Statistics tracking
        self._done_count = 0
        self._empty_bins_sum = 0
        self._unused_data_sum = 0

    def _standardize_gcs_value(self, channel, value):
        """Standardize GCS values to consistent format"""
        if channel in self._gcs_value_mapping:
            return self._gcs_value_mapping[channel].get(value, value)
        return value

    def _get_gcs_numerical_score(self, channel, value):
        """Convert GCS text value to numerical score"""
        if channel in self._gcs_score_mapping:
            standardized_value = self._standardize_gcs_value(channel, value)
            return self._gcs_score_mapping[channel].get(standardized_value, 1)
        return 1

    def _calculate_gcs_total(self, eye_value, motor_value, verbal_value):
        """Calculate GCS total from component scores"""
        eye_score = self._get_gcs_numerical_score('Glascow coma scale eye opening', eye_value)
        motor_score = self._get_gcs_numerical_score('Glascow coma scale motor response', motor_value)
        verbal_score = self._get_gcs_numerical_score('Glascow coma scale verbal response', verbal_value)
        
        total_score = eye_score + motor_score + verbal_score
        return str(total_score)

    def _write_value(self, data, bin_id, channel, value, begin_pos):
        """Write a value to the data matrix based on channel type"""
        if 'Glascow coma scale' in channel and channel != 'Glascow coma scale total':
            value = self._standardize_gcs_value(channel, value)
        
        channel_id = self._channel_to_id[channel]
        
        if self._is_categorical_channel[channel]:
            if value in self._possible_values[channel]:
                category_id = self._possible_values[channel].index(value)
                num_categories = len(self._possible_values[channel])
                start_idx = begin_pos[channel_id]
                end_idx = start_idx + num_categories
                data[bin_id, start_idx:end_idx] = 0
                data[bin_id, start_idx + category_id] = 1
            else:
                normal_val = self._normal_values[channel]
                if normal_val in self._possible_values[channel]:
                    category_id = self._possible_values[channel].index(normal_val)
                    num_categories = len(self._possible_values[channel])
                    start_idx = begin_pos[channel_id]
                    end_idx = start_idx + num_categories
                    data[bin_id, start_idx:end_idx] = 0
                    data[bin_id, start_idx + category_id] = 1
        else:
            try:
                data[bin_id, begin_pos[channel_id]] = float(value)
            except (ValueError, TypeError):
                data[bin_id, begin_pos[channel_id]] = float(self._normal_values[channel])

    def _find_closest_value(self, channel_id, bin_id, original_value, N_bins):
        """
        Find the closest available value (previous or next) with distance information
        Returns: (value, distance) or (None, infinity) if no value found
        """
        closest_value = None
        min_distance = float('inf')
        
        # Search backwards for previous value
        for search_bin in range(bin_id - 1, -1, -1):
            if original_value[search_bin][channel_id]:
                distance = bin_id - search_bin  # Number of bins away
                if distance < min_distance:
                    min_distance = distance
                    closest_value = original_value[search_bin][channel_id]
                break  # Stop at first found value going backwards
        
        # Search forwards for next value
        for search_bin in range(bin_id + 1, N_bins):
            if original_value[search_bin][channel_id]:
                distance = search_bin - bin_id  # Number of bins away
                if distance < min_distance:
                    min_distance = distance
                    closest_value = original_value[search_bin][channel_id]
                break  # Stop at first found value going forwards
        
        return closest_value, min_distance

    def _find_previous_value(self, channel_id, bin_id, original_value, N_bins):
        """Find the most recent previous value for a channel (for backward compatibility)"""
        closest_value, _ = self._find_closest_value(channel_id, bin_id, original_value, N_bins)
        return closest_value

    def transform(self, X, header=None, end=None):
        """Transform raw time series data into discretized format"""
        if header is None:
            header = self._header
        assert header[0] == "Hours"
        
        # Extract and validate timestamps
        eps = 1e-6
        timestamps = []
        valid_rows = []
        
        for row in X:
            if row[0] != '':
                try:
                    timestamp = float(row[0])
                    timestamps.append(timestamp)
                    valid_rows.append(row)
                except ValueError:
                    continue
        
        # Sort rows by timestamp to ensure monotonicity
        valid_rows.sort(key=lambda x: float(x[0]))
        timestamps = [float(row[0]) for row in valid_rows]
        
        # Determine start time
        if self._start_time == 'relative':
            first_time = timestamps[0] if timestamps else 0
        elif self._start_time == 'zero':
            first_time = 0
        else:
            raise ValueError("start_time must be 'relative' or 'zero'")

        # Calculate number of bins
        if not timestamps:
            max_hours = 0
        else:
            max_hours = max(timestamps) - first_time if end is None else end - first_time
        
        N_bins = max(1, int(max_hours / self._timestep + 1.0 - eps))
        
        # Calculate channel positions
        N_channels = len(self._id_to_channel)
        begin_pos = [0] * N_channels
        cur_pos = 0
        
        for i, channel in enumerate(self._id_to_channel):
            begin_pos[i] = cur_pos
            if self._is_categorical_channel[channel]:
                cur_pos += len(self._possible_values[channel])
            else:
                cur_pos += 1

        # Initialize data structures
        data = np.zeros((N_bins, cur_pos), dtype=float)
        mask = np.zeros((N_bins, N_channels), dtype=int)
        original_value = [['' for _ in range(N_channels)] for _ in range(N_bins)]
        
        total_data = 0
        unused_data = 0

        # Process each row of data
        for row in valid_rows:
            t = float(row[0]) - first_time
            if t > max_hours + eps:
                continue
                
            bin_id = int(t / self._timestep - eps)
            if not (0 <= bin_id < N_bins):
                continue

            # Store component values for GCS total calculation
            gcs_components = {
                'eye': '',
                'motor': '',
                'verbal': ''
            }

            for j in range(1, len(row)):
                if j >= len(header) or row[j] == "":
                    continue
                    
                channel = header[j]
                if channel not in self._channel_to_id:
                    continue
                    
                channel_id = self._channel_to_id[channel]
                total_data += 1
                
                if mask[bin_id][channel_id] == 1:
                    unused_data += 1
                    
                mask[bin_id][channel_id] = 1
                
                # Store GCS component values for total calculation
                if channel == 'Glascow coma scale eye opening':
                    gcs_components['eye'] = row[j]
                elif channel == 'Glascow coma scale motor response':
                    gcs_components['motor'] = row[j]
                elif channel == 'Glascow coma scale verbal response':
                    gcs_components['verbal'] = row[j]
                
                self._write_value(data, bin_id, channel, row[j], begin_pos)
                original_value[bin_id][channel_id] = row[j]

            # Calculate and set GCS total if all components are available
            if (gcs_components['eye'] and gcs_components['motor'] and gcs_components['verbal']):
                gcs_total = self._calculate_gcs_total(
                    gcs_components['eye'],
                    gcs_components['motor'],
                    gcs_components['verbal']
                )
                self._write_value(data, bin_id, 'Glascow coma scale total', gcs_total, begin_pos)
                original_value[bin_id][self._gcs_total_id] = gcs_total
                mask[bin_id][self._gcs_total_id] = 1

        # Enhanced hybrid imputation with closest-value selection
        if self._impute_strategy == 'hybrid':
            for channel in self._id_to_channel:
                channel_id = self._channel_to_id[channel]
                
                for bin_id in range(N_bins):
                    if mask[bin_id][channel_id] == 0:
                        # Use the new closest-value finding method
                        closest_val, distance = self._find_closest_value(channel_id, bin_id, original_value, N_bins)
                        
                        if closest_val is not None:
                            # Use the closest available value
                            self._write_value(data, bin_id, channel, closest_val, begin_pos)
                        else:
                            # If no values found, use normal value
                            self._write_value(data, bin_id, channel, self._normal_values[channel], begin_pos)
        
        # Other imputation strategies remain the same
        elif self._impute_strategy == 'normal_value':
            for bin_id in range(N_bins):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 0:
                        self._write_value(data, bin_id, channel, self._normal_values[channel], begin_pos)
                        
        elif self._impute_strategy == 'previous':
            prev_values = {channel: self._normal_values[channel] for channel in self._id_to_channel}
            for bin_id in range(N_bins):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 1:
                        prev_values[channel] = original_value[bin_id][channel_id]
                    else:
                        self._write_value(data, bin_id, channel, prev_values[channel], begin_pos)
                        
        elif self._impute_strategy == 'next':
            next_values = {channel: self._normal_values[channel] for channel in self._id_to_channel}
            for bin_id in range(N_bins-1, -1, -1):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 1:
                        next_values[channel] = original_value[bin_id][channel_id]
                    else:
                        self._write_value(data, bin_id, channel, next_values[channel], begin_pos)
                        
        elif self._impute_strategy != 'zero':
            raise ValueError("impute_strategy must be 'hybrid', 'zero', 'normal_value', 'previous', or 'next'")

        # Update statistics
        empty_bins = np.sum([1 - min(1, np.sum(mask[i, :])) for i in range(N_bins)]) if N_bins > 0 else 0
        self._done_count += 1
        self._empty_bins_sum += empty_bins / (N_bins + eps) if N_bins > 0 else 0
        self._unused_data_sum += unused_data / (total_data + eps) if total_data > 0 else 0

        # Add masks to data if requested
        if self._store_masks:
            data = np.hstack([data, mask.astype(np.float32)])

        # Create new header
        new_header = []
        for channel in self._id_to_channel:
            if self._is_categorical_channel[channel]:
                for value in self._possible_values[channel]:
                    new_header.append(f"{channel}->{value}")
            else:
                new_header.append(channel)

        if self._store_masks:
            for channel in self._id_to_channel:
                new_header.append(f"mask->{channel}")

        # Truncate sequence if needed
        if self._truncate_seq_len > 0 and N_bins > self._truncate_seq_len:
            data = data[:self._truncate_seq_len]

        return data, new_header

    def print_statistics(self):
        """Print statistics about the discretization process"""
        if self._done_count == 0:
            print("No data processed yet.")
            return
            
        print("Discretization Statistics:")
        print(f"\tConverted {self._done_count} examples")
        avg_unused = 100.0 * self._unused_data_sum / self._done_count
        print(f"\tAverage unused data = {avg_unused:.2f}%")
        avg_empty = 100.0 * self._empty_bins_sum / self._done_count
        print(f"\tAverage empty bins = {avg_empty:.2f}%")


def preprocess_directory(input_dir, output_dir, discretizer_config_path, impute_strategy='hybrid'):
    """Preprocess all CSV files in a directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    discretizer = EnhancedDiscretizer(
        config_path=discretizer_config_path,
        impute_strategy=impute_strategy
    )
    
    for csv_file in input_path.glob("*.csv"):
        print(f"Processing {csv_file.name}...")
        
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            data = list(reader)
        
        try:
            processed_data, new_header = discretizer.transform(data, header)
            
            output_file = output_path / f"{csv_file.name}"
            
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(new_header)
                for row in processed_data:
                    formatted_row = []
                    for val in row:
                        if isinstance(val, (int, np.integer)):
                            formatted_row.append(str(val))
                        elif isinstance(val, (float, np.floating)):
                            if val == int(val):
                                formatted_row.append(str(int(val)))
                            else:
                                formatted_row.append(f"{val:.6f}".rstrip('0').rstrip('.'))
                        else:
                            formatted_row.append(str(val))
                    writer.writerow(formatted_row)
                
            print(f"Saved processed data to {output_file}")
            
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    discretizer.print_statistics()

if __name__ == "__main__":
    input_dir = ""
    output_dir = ""
    config_path = "discretizer_config.json"
    
    preprocess_directory(input_dir, output_dir, config_path, impute_strategy='hybrid')