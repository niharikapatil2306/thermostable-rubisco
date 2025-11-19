import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import json
import random
from collections import Counter, defaultdict
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Import your model classes from the training code
from train_bf import (
    AtomicEnhancedRuBisCOModel, 
    DiffusionScheduler, 
    BLOSUMMatrix,
    EnhancedRuBisCODataset
)

class ThermostabilityPredictor:
    """Predict thermostability from sequence features"""
    
    def __init__(self, thermophile_dataset_path):
        self.df = pd.read_csv(thermophile_dataset_path)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self._train_predictor()
    
    def extract_sequence_features(self, sequences):
        """Extract thermostability-relevant features from sequences"""
        features_list = []
        
        for sequence in sequences:
            features = {}
            total_aa = len(sequence)
            
            if total_aa == 0:
                # Handle empty sequences
                features = {f'freq_{aa}': 0.0 for aa in 'ACDEFGHIKLMNPQRSTVWY'}
                features.update({
                    'length': 0, 'hydrophobic_ratio': 0, 'positive_ratio': 0,
                    'negative_ratio': 0, 'charge_ratio': 0, 'polar_ratio': 0,
                    'aromatic_ratio': 0, 'proline_ratio': 0, 'glycine_ratio': 0,
                    'cysteine_ratio': 0, 'small_ratio': 0, 'aliphatic_ratio': 0
                })
                features_list.append(features)
                continue
            
            # Amino acid composition
            aa_counts = Counter(sequence)
            for aa in 'ACDEFGHIKLMNPQRSTVWY':
                features[f'freq_{aa}'] = aa_counts.get(aa, 0) / total_aa
            
            # Length
            features['length'] = total_aa
            
            # Thermostability-relevant properties
            hydrophobic = 'AILVMFYW'
            features['hydrophobic_ratio'] = sum(1 for aa in sequence if aa in hydrophobic) / total_aa
            
            positive = 'KRH'
            negative = 'DE'
            features['positive_ratio'] = sum(1 for aa in sequence if aa in positive) / total_aa
            features['negative_ratio'] = sum(1 for aa in sequence if aa in negative) / total_aa
            features['charge_ratio'] = features['positive_ratio'] - features['negative_ratio']
            
            polar = 'NQST'
            features['polar_ratio'] = sum(1 for aa in sequence if aa in polar) / total_aa
            
            aromatic = 'FWY'
            features['aromatic_ratio'] = sum(1 for aa in sequence if aa in aromatic) / total_aa
            
            features['proline_ratio'] = sequence.count('P') / total_aa
            features['glycine_ratio'] = sequence.count('G') / total_aa
            features['cysteine_ratio'] = sequence.count('C') / total_aa
            
            # Additional thermostability features
            small = 'AGSPV'
            aliphatic = 'AIL'
            features['small_ratio'] = sum(1 for aa in sequence if aa in small) / total_aa
            features['aliphatic_ratio'] = sum(1 for aa in sequence if aa in aliphatic) / total_aa
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _train_predictor(self):
        """Train thermostability predictor on thermophile dataset"""
        print("Training thermostability predictor...")
        
        # Extract features from thermophile sequences
        features_df = self.extract_sequence_features(self.df['sequence'].tolist())
        features_df['tm'] = self.df['tm']
        
        # Prepare data
        X = features_df.drop('tm', axis=1)
        y = features_df['tm']
        
        self.feature_names = X.columns.tolist()
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # print(f"Thermostability predictor R2: {r2:.4f}, RMSE: {rmse:.2f}°C")
        
        # Get feature importance for thermostability
        self.feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        # Print top thermostability features
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        # print("Top thermostability features:")
        # for feat, imp in sorted_features:
        #     print(f"  {feat}: {imp:.4f}")
    
    def predict_thermostability(self, sequence):
        """Predict thermostability of a sequence"""
        features_df = self.extract_sequence_features([sequence])
        features_scaled = self.scaler.transform(features_df)
        return self.model.predict(features_scaled)[0]
    
    def get_thermostable_guidance(self, current_sequence, target_temp=85.0):
        """Generate guidance for making sequence more thermostable"""
        current_temp = self.predict_thermostability(current_sequence)
        temp_diff = target_temp - current_temp
        
        # Use feature importance to guide modifications
        guidance = {}
        for feature, importance in self.feature_importance.items():
            guidance[feature] = importance * temp_diff
        
        return guidance, current_temp, temp_diff

class ThermostableRuBisCOGenerator:
    """Enhanced RuBisCO generator with thermostability bias"""
    
    def __init__(self, model_path: str, rubisco_dataset_path: str, thermophile_dataset_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load thermostability predictor
        print("Loading thermostability predictor...")
        self.thermo_predictor = ThermostabilityPredictor(thermophile_dataset_path)
        
        # Load RuBisCO model
        print("Loading RuBisCO diffusion model...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.aa_to_idx = checkpoint['aa_to_idx']
        self.idx_to_aa = checkpoint['idx_to_aa']
        self.ss_to_idx = checkpoint['ss_to_idx']
        
        self.vocab_size = len(self.aa_to_idx)
        self.ss_vocab_size = len(self.ss_to_idx)
        
        # Load datasets
        self.rubisco_df = pd.read_csv(rubisco_dataset_path)
        
        # Load model and scheduler
        self.model, self.diffusion_scheduler = self.load_model(model_path)
        self.rubisco_dataset = self._create_dataset()
        
        # Analyze thermophile patterns for bias
        self.thermophile_patterns = self._analyze_thermophile_patterns()
        
    def load_model(self, model_path):
        """Load the trained diffusion model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model_config = checkpoint['model_config']
        model = AtomicEnhancedRuBisCOModel(**model_config).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        diffusion_schedule_data = checkpoint['diffusion_schedule']
        diffusion_scheduler = DiffusionScheduler(
            self.aa_to_idx, 
            T=1000,
            device=self.device,
            use_progressive=diffusion_schedule_data['use_progressive']
        )
        
        # Restore scheduler parameters
        for attr in ['betas', 'alphas_cumprod', 'sqrt_alphas_cumprod', 
                     'sqrt_one_minus_alphas_cumprod', 'blosum_matrices']:
            setattr(diffusion_scheduler, attr, diffusion_schedule_data[attr])
        
        print(f"Model loaded from epoch {checkpoint['epoch']}")
        return model, diffusion_scheduler
    
    def _create_dataset(self):
        """Create RuBisCO dataset"""
        max_len = self.model.max_len
        dataset = EnhancedRuBisCODataset(
            self.rubisco_df, 
            self.aa_to_idx, 
            self.ss_to_idx, 
            max_len
        )
        return dataset
    
    def _analyze_thermophile_patterns(self):
        """Analyze patterns in thermophile dataset for biasing"""
        print("Analyzing thermophile patterns...")
        
        # High temperature sequences (>75°C)
        high_temp_seqs = self.thermo_predictor.df[self.thermo_predictor.df['tm'] > 75]['sequence'].tolist()
        
        patterns = {}
        
        # Amino acid preferences at different temperature ranges
        temp_ranges = [(60, 70), (70, 80), (80, 90), (90, 100)]
        
        for low, high in temp_ranges:
            range_seqs = self.thermo_predictor.df[
                (self.thermo_predictor.df['tm'] >= low) & 
                (self.thermo_predictor.df['tm'] < high)
            ]['sequence'].tolist()
            
            if range_seqs:
                all_aa = ''.join(range_seqs)
                aa_freq = Counter(all_aa)
                total = len(all_aa)
                patterns[f'temp_{low}_{high}'] = {aa: count/total for aa, count in aa_freq.items()}
        
        # Extremely thermostable patterns (>85°C)
        extreme_temp_seqs = self.thermo_predictor.df[self.thermo_predictor.df['tm'] > 85]['sequence'].tolist()
        if extreme_temp_seqs:
            all_aa = ''.join(extreme_temp_seqs)
            aa_freq = Counter(all_aa)
            total = len(all_aa)
            patterns['extreme_thermostable'] = {aa: count/total for aa, count in aa_freq.items()}
            
            print(f"Found {len(extreme_temp_seqs)} extremely thermostable sequences (>85°C)")
            
            # Print key characteristics
            # print("Extreme thermostable amino acid preferences:")
            # sorted_aa = sorted(patterns['extreme_thermostable'].items(), key=lambda x: x[1], reverse=True)[:10]
            # for aa, freq in sorted_aa:
            #     if aa in 'ACDEFGHIKLMNPQRSTVWY':
            #         print(f"  {aa}: {freq:.3f}")
        
        return patterns
    
    def sample_rubisco_constraints(self, target_length: int = None, batch_size: int = 8):
        """Sample RuBisCO constraints in batches for efficiency"""
        if target_length:
            length_tolerance = max(10, int(target_length * 0.15))
            valid_indices = []
            
            for i, row in self.rubisco_df.iterrows():
                seq_len = len(row['sequence'])
                if abs(seq_len - target_length) <= length_tolerance:
                    valid_indices.append(i)
            
            if not valid_indices:
                valid_indices = list(range(len(self.rubisco_df)))
        else:
            valid_indices = list(range(len(self.rubisco_df)))
        
        # Sample batch
        sample_indices = random.sample(valid_indices, min(batch_size, len(valid_indices)))
        
        batch_constraints = []
        batch_lengths = []
        
        for idx in sample_indices:
            sample_data = self.rubisco_dataset[idx]
            
            constraints = {}
            for key, value in sample_data.items():
                if key != 'sequence':
                    constraints[key] = value.to(self.device)
            
            actual_length = len(self.rubisco_df.iloc[idx]['sequence'])
            batch_constraints.append(constraints)
            batch_lengths.append(actual_length)
        
        # Stack batch constraints
        batched_constraints = {}
        for key in batch_constraints[0].keys():
            batched_constraints[key] = torch.stack([c[key] for c in batch_constraints])
        
        return batched_constraints, batch_lengths
    
    def apply_thermostable_bias(self, predicted_logits, step, current_sequences, guidance_strength=1.0):
        """Apply thermostability bias to sequence logits"""
        batch_size, seq_len, vocab_size = predicted_logits.shape
        
        # Use extreme thermostable patterns if available
        if 'extreme_thermostable' in self.thermophile_patterns:
            thermo_probs = self.thermophile_patterns['extreme_thermostable']
        elif 'temp_80_90' in self.thermophile_patterns:
            thermo_probs = self.thermophile_patterns['temp_80_90']
        else:
            # Fallback to general high-temp preferences
            thermo_probs = {'R': 0.08, 'K': 0.07, 'E': 0.08, 'D': 0.06, 'A': 0.09, 
                           'G': 0.07, 'P': 0.05, 'I': 0.06, 'L': 0.09, 'V': 0.07}
        
        # Convert to tensor
        thermo_bias = torch.zeros(vocab_size, device=self.device)
        for aa, prob in thermo_probs.items():
            if aa in self.aa_to_idx:
                idx = self.aa_to_idx[aa]
                thermo_bias[idx] = prob
        
        # Normalize
        thermo_bias = thermo_bias / (thermo_bias.sum() + 1e-8)
        thermo_log_probs = torch.log(thermo_bias + 1e-8)
        
        # Apply bias with decay over steps
        step_factor = min(1.0, step / 100.0)  # Stronger bias early in generation
        bias_strength = guidance_strength * step_factor
        
        # Apply to all positions and batches
        for b in range(batch_size):
            predicted_logits[b] = (1 - bias_strength) * predicted_logits[b] + bias_strength * thermo_log_probs.unsqueeze(0)
        
        return predicted_logits
    
    def evaluate_thermostability_during_generation(self, sequences, step, log_interval=50):
        """Evaluate and log thermostability during generation"""
        if step % log_interval == 0:
            decoded_sequences = []
            for seq_tokens in sequences:
                seq_str = ''
                for token in seq_tokens:
                    if token.item() in self.idx_to_aa and self.idx_to_aa[token.item()] not in ['PAD', 'EOS']:
                        seq_str += self.idx_to_aa[token.item()]
                decoded_sequences.append(seq_str)
            
            if decoded_sequences and len(decoded_sequences[0]) > 10:
                avg_temp = np.mean([self.thermo_predictor.predict_thermostability(seq) 
                                  for seq in decoded_sequences if len(seq) > 10])
                print(f"Step {step}: Average predicted thermostability = {avg_temp:.1f}°C")
    
    @torch.no_grad()
    def generate_thermostable_rubisco(self, target_length: int = 500, target_temperature: float = 85.0,
                                    num_steps: int = 200, thermostable_guidance: float = 0.8,
                                    blosum_guidance: float = 0.3, num_samples: int = 4, 
                                    temperature: float = 0.6, batch_size: int = 4):
        """Generate highly thermostable RuBisCO sequences"""
        
        self.model.eval()
        max_length = self.model.max_len
        
        print(f"Generating {num_samples} thermostable RuBisCO sequences...")
        print(f"Target length: {target_length}, Target temperature: {target_temperature}°C")
        print(f"Thermostable guidance: {thermostable_guidance}, BLOSUM guidance: {blosum_guidance}")
        
        generated_proteins = []
        
        # Process in batches
        for batch_start in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - batch_start)
            print(f"\nProcessing batch {batch_start//batch_size + 1} ({current_batch_size} samples)...")
            
            # Sample RuBisCO constraints
            constraints, constraint_lengths = self.sample_rubisco_constraints(target_length, current_batch_size)
            
            # Initialize with random sequences
            x = torch.randint(1, self.vocab_size - 1, (current_batch_size, max_length), device=self.device)
            
            # Set up sequences with proper length and termination
            for b in range(current_batch_size):
                actual_length = constraint_lengths[b]
                
                # Ensure no PAD/EOS in sequence part
                for i in range(actual_length):
                    while x[b, i].item() in [self.aa_to_idx['PAD'], self.aa_to_idx['EOS']]:
                        x[b, i] = torch.randint(1, self.vocab_size - 2, (1,), device=self.device)
                
                # Set EOS and PAD
                if actual_length < max_length:
                    x[b, actual_length] = self.aa_to_idx['EOS']
                    x[b, actual_length+1:] = self.aa_to_idx['PAD']
            
            # Reverse diffusion process
            timesteps = torch.linspace(num_steps - 1, 0, num_steps, dtype=torch.long, device=self.device)
            
            for i, step in enumerate(tqdm(timesteps, desc=f"Generating Batch {batch_start//batch_size + 1}", leave=False)):
                t = torch.tensor([step + 1] * current_batch_size, device=self.device)
                
                # Forward pass with RuBisCO constraints
                angles_input = torch.stack([constraints['phi_angles'], constraints['psi_angles']], dim=-1)
                
                predictions = self.model(
                    x, t, constraints['mask'],
                    atomic_coords=constraints['atomic_coordinates'],
                    atomic_confidence=constraints['atomic_confidence'],
                    angles=angles_input,
                    neighbors=constraints['neighbor_counts'],
                    confidence=constraints['confidence_masks'],
                    secondary_structure=constraints['secondary_structures']
                )
                
                predicted_logits = predictions['sequence_logits']
                
                # Apply thermostability bias
                if thermostable_guidance > 0.0:
                    predicted_logits = self.apply_thermostable_bias(
                        predicted_logits, step, x, thermostable_guidance
                    )
                
                # Apply BLOSUM guidance
                if blosum_guidance > 0.0 and step > 0:
                    t_idx = min(step, len(self.diffusion_scheduler.blosum_matrices) - 1)
                    blosum_matrix = (self.diffusion_scheduler.blosum_matrices[t_idx] 
                                   if self.diffusion_scheduler.use_progressive 
                                   else self.diffusion_scheduler.blosum_matrices[0])
                    
                    for b in range(current_batch_size):
                        actual_length = constraint_lengths[b]
                        current_tokens = x[b]
                        
                        for pos in range(actual_length):
                            current_aa = current_tokens[pos].item()
                            if current_aa < len(blosum_matrix) and current_aa > 0:
                                blosum_probs = blosum_matrix[current_aa]
                                predicted_logits[b, pos] = (
                                    (1 - blosum_guidance) * predicted_logits[b, pos] +
                                    blosum_guidance * torch.log(blosum_probs + 1e-8)
                                )
                
                # Sample next tokens
                if step > 0:
                    scaled_logits = predicted_logits / temperature
                    probs = F.softmax(scaled_logits, dim=-1)
                    new_tokens = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(predicted_logits.shape[:-1])
                    
                    # Update only sequence parts
                    for b in range(current_batch_size):
                        actual_length = constraint_lengths[b]
                        x[b, :actual_length] = new_tokens[b, :actual_length]
                        
                        # Ensure no PAD/EOS in sequence
                        for pos in range(actual_length):
                            while x[b, pos].item() in [self.aa_to_idx['PAD'], self.aa_to_idx['EOS']]:
                                x[b, pos] = torch.randint(1, self.vocab_size - 2, (1,), device=self.device)
                else:
                    # Final step - use argmax
                    for b in range(current_batch_size):
                        actual_length = constraint_lengths[b]
                        x[b, :actual_length] = torch.argmax(predicted_logits[b, :actual_length], dim=-1)
                
                # Log thermostability progress
                self.evaluate_thermostability_during_generation(x, step)
            
            # Process batch results
            for b in range(current_batch_size):
                actual_length = constraint_lengths[b]
                final_tokens = x[b, :actual_length].cpu().numpy()
                
                # Decode sequence
                sequence = ''
                for token in final_tokens:
                    if token in self.idx_to_aa and self.idx_to_aa[token] not in ['PAD', 'EOS']:
                        sequence += self.idx_to_aa[token]
                
                # Predict thermostability
                if len(sequence) > 0:
                    predicted_temp = self.thermo_predictor.predict_thermostability(sequence)
                else:
                    predicted_temp = 0.0
                
                # Get final structural predictions
                final_t = torch.tensor([1], device=self.device)
                single_constraints = {k: v[b:b+1] for k, v in constraints.items()}
                angles_input = torch.stack([single_constraints['phi_angles'], single_constraints['psi_angles']], dim=-1)
                
                final_predictions = self.model(
                    x[b:b+1], final_t, single_constraints['mask'],
                    atomic_coords=single_constraints['atomic_coordinates'],
                    atomic_confidence=single_constraints['atomic_confidence'],
                    angles=angles_input,
                    neighbors=single_constraints['neighbor_counts'],
                    confidence=single_constraints['confidence_masks'],
                    secondary_structure=single_constraints['secondary_structures']
                )
                
                protein_data = {
                    'sequence': sequence,
                    'length': len(sequence),
                    'target_length': target_length,
                    'actual_constraint_length': actual_length,
                    'predicted_thermostability': predicted_temp,
                    'target_temperature': target_temperature,
                    'thermostable_guidance_strength': thermostable_guidance,
                    'predicted_structure': torch.argmax(final_predictions['ss_pred'][0, :actual_length], dim=-1).cpu().numpy(),
                    'predicted_angles': final_predictions['angles_pred'][0, :actual_length].cpu().numpy(),
                    'generation_method': 'thermostable_guided_diffusion'
                }
                
                generated_proteins.append(protein_data)
                
                # Log results
                status = "HIGHLY THERMOSTABLE" if predicted_temp >= target_temperature else "MODERATE" if predicted_temp >= 70 else "LOW"
                print(f"Generated {len(generated_proteins)}/{num_samples}: {sequence[:30]}... | "
                      f"Length: {len(sequence)} | Thermo: {predicted_temp:.1f}°C ({status})")
        
        return generated_proteins
    
    def analyze_generated_proteins(self, proteins):
        """Comprehensive analysis of generated thermostable proteins"""
        print("\n" + "="*80)
        print("THERMOSTABLE RUBISCO GENERATION ANALYSIS")
        print("="*80)
        
        if not proteins:
            # print("No proteins to analyze!")
            return
        
        # Overall statistics
        temps = [p['predicted_thermostability'] for p in proteins]
        lengths = [p['length'] for p in proteins]
        
        print(f"\nGenerated {len(proteins)} proteins:")
        print(f"Average thermostability: {np.mean(temps):.1f}°C (std: {np.std(temps):.1f}°C)")
        print(f"Thermostability range: {min(temps):.1f}°C - {max(temps):.1f}°C")
        print(f"Average length: {np.mean(lengths):.0f} residues (std: {np.std(lengths):.0f})")
        
        # Thermostability categories
        highly_stable = sum(1 for t in temps if t >= 80)
        moderately_stable = sum(1 for t in temps if 70 <= t < 80)
        low_stable = sum(1 for t in temps if t < 70)
        
        print(f"\nThermostability categories:")
        print(f"  Highly thermostable (≥80°C): {highly_stable} ({highly_stable/len(proteins)*100:.1f}%)")
        print(f"  Moderately stable (70-80°C): {moderately_stable} ({moderately_stable/len(proteins)*100:.1f}%)")
        print(f"  Lower stability (<70°C): {low_stable} ({low_stable/len(proteins)*100:.1f}%)")
        
        # Analyze best sequences
        best_proteins = sorted(proteins, key=lambda x: x['predicted_thermostability'], reverse=True)[:3]
        
        print(f"\nTop 3 most thermostable sequences:")
        for i, protein in enumerate(best_proteins, 1):
            seq = protein['sequence']
            print(f"\n{i}. Thermostability: {protein['predicted_thermostability']:.1f}°C")
            print(f"   Length: {protein['length']} residues")
            print(f"   Sequence: {seq[:60]}{'...' if len(seq) > 60 else ''}")
            
            # Amino acid composition analysis
            if len(seq) > 0:
                aa_counts = Counter(seq)
                total = len(seq)
                
                # Key thermostable amino acids
                key_aa = {'R': 'Arginine', 'K': 'Lysine', 'E': 'Glutamate', 'D': 'Aspartate', 
                         'A': 'Alanine', 'G': 'Glycine', 'P': 'Proline', 'I': 'Isoleucine'}
                
                print(f"   Key thermostable residues:")
                for aa, name in key_aa.items():
                    if aa in aa_counts:
                        freq = aa_counts[aa] / total * 100
                        # print(f"     {aa} ({name[:3]}): {freq:.1f}%")
    
    def save_thermostable_results(self, proteins, output_file="thermostable_rubisco_results5.json"):
        """Save results with thermostability analysis"""
        results = []
        
        for protein in proteins:
            result = {
                'sequence': protein['sequence'],
                'length': protein['length'],
                'predicted_thermostability_celsius': round(protein['predicted_thermostability'], 2),
                'target_temperature_celsius': protein['target_temperature'],
                'thermostable_guidance_strength': protein['thermostable_guidance_strength'],
                'generation_method': protein['generation_method'],
                'thermostability_category': self._get_stability_category(protein['predicted_thermostability'])
            }
            
            # Add structural predictions if available
            if 'predicted_structure' in protein:
                ss_chars = {0: 'H', 1: 'E', 2: 'C', 3: 'G', 4: 'I', 5: 'B', 6: 'T', 7: 'S', 8: '-'}
                result['predicted_secondary_structure'] = ''.join([
                    ss_chars.get(int(s), '-') for s in protein['predicted_structure']
                ])
            
            results.append(result)
        
        # Sort by thermostability
        results.sort(key=lambda x: x['predicted_thermostability_celsius'], reverse=True)
        
        # Save JSON
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSV
        csv_file = output_file.replace('.json', '.csv')
        df = pd.DataFrame(results)
        df.to_csv(csv_file, index=False)
        
        print(f"\nResults saved to {output_file} and {csv_file}")
        
        return results
    
    def _get_stability_category(self, temp):
        """Categorize thermostability"""
        if temp >= 85:
            return "EXTREMELY_THERMOSTABLE"
        elif temp >= 80:
            return "HIGHLY_THERMOSTABLE"
        elif temp >= 70:
            return "MODERATELY_STABLE"
        else:
            return "LOW_STABILITY"

def main():
    """Main function to generate thermostable RuBisCO"""
    
    # File paths
    model_path = "./best_rubisco_diffusion_model.pth"
    rubisco_dataset_path = "./output/final_data2.csv"
    thermophile_dataset_path = "./dataset.csv"  # Your thermophile dataset
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("THERMOSTABLE RUBISCO GENERATOR")
    print("="*50)
    print(f"Using device: {device}")
    
    # Check files exist
    for path, name in [(model_path, "RuBisCO model"), 
                       (rubisco_dataset_path, "RuBisCO dataset"), 
                       (thermophile_dataset_path, "Thermophile dataset")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found at {path}")
            return
    
    try:
        # Initialize generator
        print("\nInitializing thermostable RuBisCO generator...")
        generator = ThermostableRuBisCOGenerator(
            model_path=model_path,
            rubisco_dataset_path=rubisco_dataset_path,
            thermophile_dataset_path=thermophile_dataset_path,
            device=device
        )
        
        print("Generator initialized successfully!")
        
        # Generation parameters for highly thermostable RuBisCO
        generation_configs = [
            {
                'name': 'EXTREME_THERMOSTABLE',
                'target_temperature': 90.0,
                'thermostable_guidance': 0.95,
                'blosum_guidance': 0.4,
                'temperature': 0.4,  # Lower temperature for more conservative sampling
                'num_samples': 100,
                'num_steps': 250
            },
            # {
            #     'name': 'HIGH_THERMOSTABLE',
            #     'target_temperature': 95.0,
            #     'thermostable_guidance': 0.9,
            #     'blosum_guidance': 0.7,
            #     'temperature': 0.7,
            #     'num_samples': 10,
            #     'num_steps': 250
            # },
            # {
            #     'name': 'MODERATE_THERMOSTABLE',
            #     'target_temperature': 90.0,
            #     'thermostable_guidance': 0.7,
            #     'blosum_guidance': 0.6,
            #     'temperature': 0.6,
            #     'num_samples': 10,
            #     'num_steps': 250
            # }
        ]
        
        all_proteins = []
        
        # Generate with different configurations
        for config in generation_configs:
            print(f"\n{'='*60}")
            print(f"GENERATING {config['name']} RUBISCO")
            print(f"Target temperature: {config['target_temperature']}°C")
            print(f"Thermostable guidance: {config['thermostable_guidance']}")
            print(f"{'='*60}")
            
            try:
                proteins = generator.generate_thermostable_rubisco(
                    target_length=500,  # RuBisCO typical length
                    target_temperature=config['target_temperature'],
                    num_steps=config['num_steps'],
                    thermostable_guidance=config['thermostable_guidance'],
                    blosum_guidance=config['blosum_guidance'],
                    num_samples=config['num_samples'],
                    temperature=config['temperature'],
                    batch_size=4
                )
                
                # Add config info to each protein
                for protein in proteins:
                    protein['generation_config'] = config['name']
                
                all_proteins.extend(proteins)
                print(f"\nCompleted {config['name']} generation: {len(proteins)} proteins")
                
            except Exception as e:
                print(f"ERROR in {config['name']} generation: {e}")
                import traceback
                traceback.print_exc()
        
        if all_proteins:
            print(f"\n{'='*80}")
            print("FINAL ANALYSIS OF ALL GENERATED PROTEINS")
            print(f"{'='*80}")
            
            # Comprehensive analysis
            generator.analyze_generated_proteins(all_proteins)
            
            # Save results
            results = generator.save_thermostable_results(
                all_proteins, 
                "thermostable_rubisco_final5.json"
            )
            
            # Print summary of best results
            print(f"\n{'='*60}")
            print("SUMMARY OF BEST THERMOSTABLE RUBISCO SEQUENCES")
            print(f"{'='*60}")
            
            # Filter highly thermostable sequences
            highly_stable = [p for p in all_proteins if p['predicted_thermostability'] >= 80]
            extremely_stable = [p for p in all_proteins if p['predicted_thermostability'] >= 85]
            
            print(f"Total generated: {len(all_proteins)}")
            print(f"Highly thermostable (≥80°C): {len(highly_stable)}")
            print(f"Extremely thermostable (≥85°C): {len(extremely_stable)}")
            
            if extremely_stable:
                print(f"\nTOP EXTREMELY THERMOSTABLE SEQUENCES:")
                best = sorted(extremely_stable, key=lambda x: x['predicted_thermostability'], reverse=True)[:5]
                
                for i, protein in enumerate(best, 1):
                    print(f"\n{i}. Temperature: {protein['predicted_thermostability']:.1f}°C")
                    print(f"   Length: {protein['length']} residues")
                    print(f"   Config: {protein.get('generation_config', 'N/A')}")
                    print(f"   Sequence: {protein['sequence'][:80]}...")
                    
                    # Quick composition analysis
                    seq = protein['sequence']
                    if len(seq) > 0:
                        charged_pos = sum(seq.count(aa) for aa in 'KRH') / len(seq) * 100
                        charged_neg = sum(seq.count(aa) for aa in 'DE') / len(seq) * 100
                        hydrophobic = sum(seq.count(aa) for aa in 'AILVMFYW') / len(seq) * 100
                        proline = seq.count('P') / len(seq) * 100
                        
                        # print(f"   Composition: Pos-charged: {charged_pos:.1f}%, "
                        #       f"Neg-charged: {charged_neg:.1f}%, "
                        #       f"Hydrophobic: {hydrophobic:.1f}%, Proline: {proline:.1f}%")
            
            # Create FASTA file for the best sequences
            if highly_stable:
                fasta_file = "thermostable_rubisco_sequences5.fasta"
                with open(fasta_file, 'w') as f:
                    best_sequences = sorted(highly_stable, key=lambda x: x['predicted_thermostability'], reverse=True)
                    
                    for i, protein in enumerate(best_sequences, 1):
                        header = f">ThermostableRuBisCO_{i:03d}_T{protein['predicted_thermostability']:.1f}C_L{protein['length']}"
                        f.write(f"{header}\n")
                        f.write(f"{protein['sequence']}\n")
                
                print(f"\nBest thermostable sequences saved to {fasta_file}")
            
            print(f"\n{'='*80}")
            print("THERMOSTABLE RUBISCO GENERATION COMPLETED SUCCESSFULLY!")
            print(f"Check the output files for detailed results.")
            print(f"{'='*80}")
            
        else:
            print("\nERROR: No proteins were successfully generated!")
    
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()