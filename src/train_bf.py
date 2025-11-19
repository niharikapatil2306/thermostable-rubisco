import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from sklearn.model_selection import train_test_split
import ast
import os
import datetime
import json
import logging
from collections import defaultdict, Counter
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BLOSUMMatrix:
    """BLOSUM62 substitution matrix for protein sequences."""
    
    BLOSUM62 = {
        'A': {'A': 4, 'R': -1, 'N': -2, 'D': -2, 'C': 0, 'Q': -1, 'E': -1, 'G': 0, 'H': -2, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S': 1, 'T': 0, 'W': -3, 'Y': -2, 'V': 0},
        'R': {'A': -1, 'R': 5, 'N': 0, 'D': -2, 'C': -3, 'Q': 1, 'E': 0, 'G': -2, 'H': 0, 'I': -3, 'L': -2, 'K': 2, 'M': -1, 'F': -3, 'P': -2, 'S': -1, 'T': -1, 'W': -3, 'Y': -2, 'V': -3},
        'N': {'A': -2, 'R': 0, 'N': 6, 'D': 1, 'C': -3, 'Q': 0, 'E': 0, 'G': 0, 'H': 1, 'I': -3, 'L': -3, 'K': 0, 'M': -2, 'F': -3, 'P': -2, 'S': 1, 'T': 0, 'W': -4, 'Y': -2, 'V': -3},
        'D': {'A': -2, 'R': -2, 'N': 1, 'D': 6, 'C': -3, 'Q': 0, 'E': 2, 'G': -1, 'H': -1, 'I': -3, 'L': -4, 'K': -1, 'M': -3, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -4, 'Y': -3, 'V': -3},
        'C': {'A': 0, 'R': -3, 'N': -3, 'D': -3, 'C': 9, 'Q': -3, 'E': -4, 'G': -3, 'H': -3, 'I': -1, 'L': -1, 'K': -3, 'M': -1, 'F': -2, 'P': -3, 'S': -1, 'T': -1, 'W': -2, 'Y': -2, 'V': -1},
        'Q': {'A': -1, 'R': 1, 'N': 0, 'D': 0, 'C': -3, 'Q': 5, 'E': 2, 'G': -2, 'H': 0, 'I': -3, 'L': -2, 'K': 1, 'M': 0, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -2, 'Y': -1, 'V': -2},
        'E': {'A': -1, 'R': 0, 'N': 0, 'D': 2, 'C': -4, 'Q': 2, 'E': 5, 'G': -2, 'H': 0, 'I': -3, 'L': -3, 'K': 1, 'M': -2, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2},
        'G': {'A': 0, 'R': -2, 'N': 0, 'D': -1, 'C': -3, 'Q': -2, 'E': -2, 'G': 6, 'H': -2, 'I': -4, 'L': -4, 'K': -2, 'M': -3, 'F': -3, 'P': -2, 'S': 0, 'T': -2, 'W': -2, 'Y': -3, 'V': -3},
        'H': {'A': -2, 'R': 0, 'N': 1, 'D': -1, 'C': -3, 'Q': 0, 'E': 0, 'G': -2, 'H': 8, 'I': -3, 'L': -3, 'K': -1, 'M': -2, 'F': -1, 'P': -2, 'S': -1, 'T': -2, 'W': -2, 'Y': 2, 'V': -3},
        'I': {'A': -1, 'R': -3, 'N': -3, 'D': -3, 'C': -1, 'Q': -3, 'E': -3, 'G': -4, 'H': -3, 'I': 4, 'L': 2, 'K': -3, 'M': 1, 'F': 0, 'P': -3, 'S': -2, 'T': -1, 'W': -3, 'Y': -1, 'V': 3},
        'L': {'A': -1, 'R': -2, 'N': -3, 'D': -4, 'C': -1, 'Q': -2, 'E': -3, 'G': -4, 'H': -3, 'I': 2, 'L': 4, 'K': -2, 'M': 2, 'F': 0, 'P': -3, 'S': -2, 'T': -1, 'W': -2, 'Y': -1, 'V': 1},
        'K': {'A': -1, 'R': 2, 'N': 0, 'D': -1, 'C': -3, 'Q': 1, 'E': 1, 'G': -2, 'H': -1, 'I': -3, 'L': -2, 'K': 5, 'M': -1, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2},
        'M': {'A': -1, 'R': -1, 'N': -2, 'D': -3, 'C': -1, 'Q': 0, 'E': -2, 'G': -3, 'H': -2, 'I': 1, 'L': 2, 'K': -1, 'M': 5, 'F': 0, 'P': -2, 'S': -1, 'T': -1, 'W': -1, 'Y': -1, 'V': 1},
        'F': {'A': -2, 'R': -3, 'N': -3, 'D': -3, 'C': -2, 'Q': -3, 'E': -3, 'G': -3, 'H': -1, 'I': 0, 'L': 0, 'K': -3, 'M': 0, 'F': 6, 'P': -4, 'S': -2, 'T': -2, 'W': 1, 'Y': 3, 'V': -1},
        'P': {'A': -1, 'R': -2, 'N': -2, 'D': -1, 'C': -3, 'Q': -1, 'E': -1, 'G': -2, 'H': -2, 'I': -3, 'L': -3, 'K': -1, 'M': -2, 'F': -4, 'P': 7, 'S': -1, 'T': -1, 'W': -4, 'Y': -3, 'V': -2},
        'S': {'A': 1, 'R': -1, 'N': 1, 'D': 0, 'C': -1, 'Q': 0, 'E': 0, 'G': 0, 'H': -1, 'I': -2, 'L': -2, 'K': 0, 'M': -1, 'F': -2, 'P': -1, 'S': 4, 'T': 1, 'W': -3, 'Y': -2, 'V': -2},
        'T': {'A': 0, 'R': -1, 'N': 0, 'D': -1, 'C': -1, 'Q': -1, 'E': -1, 'G': -2, 'H': -2, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S': 1, 'T': 5, 'W': -2, 'Y': -2, 'V': 0},
        'W': {'A': -3, 'R': -3, 'N': -4, 'D': -4, 'C': -2, 'Q': -2, 'E': -3, 'G': -2, 'H': -2, 'I': -3, 'L': -2, 'K': -3, 'M': -1, 'F': 1, 'P': -4, 'S': -3, 'T': -2, 'W': 11, 'Y': 2, 'V': -3},
        'Y': {'A': -2, 'R': -2, 'N': -2, 'D': -3, 'C': -2, 'Q': -1, 'E': -2, 'G': -3, 'H': 2, 'I': -1, 'L': -1, 'K': -2, 'M': -1, 'F': 3, 'P': -3, 'S': -2, 'T': -2, 'W': 2, 'Y': 7, 'V': -1},
        'V': {'A': 0, 'R': -3, 'N': -3, 'D': -3, 'C': -1, 'Q': -2, 'E': -2, 'G': -3, 'H': -3, 'I': 3, 'L': 1, 'K': -2, 'M': 1, 'F': -1, 'P': -2, 'S': -2, 'T': 0, 'W': -3, 'Y': -1, 'V': 4}
    }
    
    @classmethod
    def create_substitution_matrix(cls, aa_to_idx: Dict[str, int], temperature: float = 1.0, 
                                  min_prob: float = 0.01) -> torch.Tensor:
        """Create a substitution probability matrix based on BLOSUM62 scores."""
        amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        vocab_size = len(aa_to_idx)
        
        sub_matrix = torch.full((vocab_size, vocab_size), min_prob)
        
        for aa1 in amino_acids:
            if aa1 in aa_to_idx:
                idx1 = aa_to_idx[aa1]
                blosum_scores = []
                target_indices = []
                
                for aa2 in amino_acids:
                    if aa2 in aa_to_idx:
                        idx2 = aa_to_idx[aa2]
                        score = cls.BLOSUM62[aa1][aa2]
                        blosum_scores.append(score)
                        target_indices.append(idx2)
                
                blosum_scores = torch.tensor(blosum_scores, dtype=torch.float32)
                probs = F.softmax(blosum_scores / temperature, dim=0)
                probs = torch.clamp(probs, min=min_prob)
                probs = probs / probs.sum()
                
                for prob, idx2 in zip(probs, target_indices):
                    sub_matrix[idx1, idx2] = prob
        
        # Handle special tokens
        for token in ['PAD', 'EOS']:
            if token in aa_to_idx:
                token_idx = aa_to_idx[token]
                sub_matrix[token_idx, :] = min_prob
                sub_matrix[token_idx, token_idx] = 0.95
                sub_matrix[token_idx, :] = sub_matrix[token_idx, :] / sub_matrix[token_idx, :].sum()
        
        sub_matrix = sub_matrix / sub_matrix.sum(dim=1, keepdim=True)
        return sub_matrix
    
    @classmethod
    def create_progressive_schedule(cls, aa_to_idx: Dict[str, int], num_timesteps: int = 1000, 
                                   start_temp: float = 0.5, end_temp: float = 2.0) -> List[torch.Tensor]:
        """Create a progressive BLOSUM substitution schedule."""
        schedule = []
        for t in range(num_timesteps):
            progress = t / (num_timesteps - 1)
            temperature = start_temp + (end_temp - start_temp) * progress
            sub_matrix = cls.create_substitution_matrix(aa_to_idx, temperature=temperature)
            schedule.append(sub_matrix)
        return schedule


class DistributedManager:
    """Handles distributed training setup and cleanup."""
    
    def __init__(self):
        self.rank = None
        self.world_size = None
        self.gpu = None
        self.is_distributed = False
    
    def setup(self) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """Initialize distributed training."""
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.gpu = int(os.environ['LOCAL_RANK'])
            self.is_distributed = True
            
            torch.cuda.set_device(self.gpu)
            dist.init_process_group(backend='nccl', init_method='env://')
            
            if self.rank == 0:
                logger.info(f"Distributed training initialized: {self.world_size} GPUs")
        
        return self.rank, self.world_size, self.gpu
    
    def cleanup(self):
        """Clean up distributed training."""
        if dist.is_initialized():
            dist.destroy_process_group()
    
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.rank is None or self.rank == 0


class EnhancedRuBisCODataset(Dataset):
    """Enhanced dataset with atomic-level structural information."""
    
    def __init__(self, dataframe: pd.DataFrame, aa_to_idx: Dict[str, int], 
                 ss_to_idx: Dict[str, int], max_len: int):
        self.data = dataframe.reset_index(drop=True)
        self.aa_to_idx = aa_to_idx
        self.ss_to_idx = ss_to_idx
        self.max_len = max_len
        self.EOS_idx = aa_to_idx['EOS']
        self.PAD_idx = aa_to_idx['PAD']
        self.SS_PAD_idx = ss_to_idx['-']
        
        self.important_atoms = ['N', 'CA', 'C', 'O', 'CB']
        self.max_atoms_per_residue = len(self.important_atoms)
        
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Pre-process all complex atomic data."""
        self.processed_data = []
        
        for idx, row in self.data.iterrows():
            all_atom_data = self._parse_atomic_data(row['all_atom_data'])
            phi = self._parse_list_string(row['phi_angles'])
            psi = self._parse_list_string(row['psi_angles'])
            neighbors = self._parse_list_string(row['all_neighbor_counts'])
            conf_mask = self._parse_list_string(row['confidence_mask'])
            ss = list(row['secondary_structure']) if isinstance(row['secondary_structure'], str) else []
            
            processed_item = self._process_single_item(
                row['sequence'], all_atom_data, phi, psi, neighbors, conf_mask, ss
            )
            self.processed_data.append(processed_item)
    
    def _parse_atomic_data(self, atomic_data_str) -> List[Dict]:
        """Parse the complex atomic data structure."""
        try:
            if isinstance(atomic_data_str, str):
                atomic_data = ast.literal_eval(atomic_data_str)
            else:
                atomic_data = atomic_data_str
            
            if not isinstance(atomic_data, list):
                return []
            
            residue_data = defaultdict(dict)
            
            for atom_info in atomic_data:
                if isinstance(atom_info, dict):
                    res_num = atom_info.get('residue_number', 0)
                    atom_name = atom_info.get('atom_name', '')
                    coords = atom_info.get('coordinates', (0.0, 0.0, 0.0))
                    confidence = atom_info.get('confidence_score', 0.0)
                    
                    if atom_name in self.important_atoms:
                        residue_data[res_num][atom_name] = {
                            'coords': coords,
                            'confidence': confidence
                        }
            
            ordered_residues = []
            max_res = max(residue_data.keys()) if residue_data else 0
            
            for res_num in range(1, max_res + 1):
                residue_atoms = residue_data.get(res_num, {})
                ordered_residues.append(residue_atoms)
            
            return ordered_residues
            
        except Exception:
            return []
    
    def _parse_list_string(self, s) -> List:
        """Parse list strings safely."""
        try:
            if isinstance(s, str):
                return ast.literal_eval(s)
            elif isinstance(s, list):
                return s
            else:
                return []
        except:
            return []
    
    def _process_single_item(self, seq: str, atomic_data: List, phi: List, psi: List, 
                           neighbors: List, conf_mask: List, ss: List) -> Dict[str, torch.Tensor]:
        """Process a single data item with atomic information."""
        seq_len = len(seq)
        
        encoded_seq = [self.aa_to_idx.get(aa, self.PAD_idx) for aa in seq]
        encoded_ss = [self.ss_to_idx.get(s, self.SS_PAD_idx) for s in ss]
        
        atomic_coords, atomic_confidence = self._process_atomic_features(atomic_data, seq_len)
        
        # Ensure all arrays have the same length as sequence
        phi = phi[:seq_len] if len(phi) >= seq_len else phi + [0.0] * (seq_len - len(phi))
        psi = psi[:seq_len] if len(psi) >= seq_len else psi + [0.0] * (seq_len - len(psi))
        neighbors = neighbors[:seq_len] if len(neighbors) >= seq_len else neighbors + [0] * (seq_len - len(neighbors))
        conf_mask = conf_mask[:seq_len] if len(conf_mask) >= seq_len else conf_mask + [0] * (seq_len - len(conf_mask))
        encoded_ss = encoded_ss[:seq_len] if len(encoded_ss) >= seq_len else encoded_ss + [self.SS_PAD_idx] * (seq_len - len(encoded_ss))
        
        # Truncate if too long
        if len(encoded_seq) >= self.max_len:
            encoded_seq = encoded_seq[:self.max_len - 1]
            atomic_coords = atomic_coords[:self.max_len - 1]
            atomic_confidence = atomic_confidence[:self.max_len - 1]
            phi = phi[:self.max_len - 1]
            psi = psi[:self.max_len - 1]
            neighbors = neighbors[:self.max_len - 1]
            conf_mask = conf_mask[:self.max_len - 1]
            encoded_ss = encoded_ss[:self.max_len - 1]
        
        # Add EOS token and padding
        encoded_seq.append(self.EOS_idx)
        atomic_coords.append(np.zeros((self.max_atoms_per_residue, 3)))
        atomic_confidence.append(np.zeros(self.max_atoms_per_residue))
        phi.append(0.0)
        psi.append(0.0)
        neighbors.append(0)
        conf_mask.append(0)
        encoded_ss.append(self.SS_PAD_idx)
        
        # Pad to max length
        pad_len = self.max_len - len(encoded_seq)
        if pad_len > 0:
            encoded_seq.extend([self.PAD_idx] * pad_len)
            atomic_coords.extend([np.zeros((self.max_atoms_per_residue, 3))] * pad_len)
            atomic_confidence.extend([np.zeros(self.max_atoms_per_residue)] * pad_len)
            phi.extend([0.0] * pad_len)
            psi.extend([0.0] * pad_len)
            neighbors.extend([0] * pad_len)
            conf_mask.extend([0] * pad_len)
            encoded_ss.extend([self.SS_PAD_idx] * pad_len)
        
        return {
            'sequence': torch.tensor(encoded_seq, dtype=torch.long),
            'atomic_coordinates': torch.tensor(np.array(atomic_coords), dtype=torch.float),
            'atomic_confidence': torch.tensor(np.array(atomic_confidence), dtype=torch.float),
            'phi_angles': torch.tensor(phi, dtype=torch.float),
            'psi_angles': torch.tensor(psi, dtype=torch.float),
            'neighbor_counts': torch.tensor(neighbors, dtype=torch.float),
            'confidence_masks': torch.tensor(conf_mask, dtype=torch.float),
            'secondary_structures': torch.tensor(encoded_ss, dtype=torch.long),
            'mask': torch.tensor([1 if x != self.PAD_idx else 0 for x in encoded_seq], dtype=torch.long)
        }
    
    def _process_atomic_features(self, atomic_data: List, seq_len: int) -> Tuple[List, List]:
        """Process atomic coordinates and confidence scores."""
        atomic_coords = []
        atomic_confidence = []
        
        for i in range(seq_len):
            residue_atoms = atomic_data[i] if i < len(atomic_data) else {}
            
            res_coords = np.zeros((self.max_atoms_per_residue, 3))
            res_confidence = np.zeros(self.max_atoms_per_residue)
            
            for atom_idx, atom_name in enumerate(self.important_atoms):
                if atom_name in residue_atoms:
                    atom_data = residue_atoms[atom_name]
                    res_coords[atom_idx] = atom_data['coords']
                    res_confidence[atom_idx] = atom_data['confidence'] / 100.0
            
            atomic_coords.append(res_coords)
            atomic_confidence.append(res_confidence)
        
        return atomic_coords, atomic_confidence
    
    def __len__(self) -> int:
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.processed_data[idx]


class AtomicEnhancedRuBisCOModel(nn.Module):
    """Enhanced RuBisCO model with atomic-level structural information."""
    
    def __init__(self, vocab_size: int, ss_vocab_size: int, d_model: int = 512, 
                 num_heads: int = 16, num_layers: int = 12, d_ff: int = 2048, 
                 dropout: float = 0.4, max_len: int = 550, max_atoms: int = 5):
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        self.max_atoms = max_atoms
        
        # Core embeddings
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_len, d_model), requires_grad=True)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Feature projections
        feature_dim = d_model // 8
        
        # Atomic coordinate processing
        self.atomic_coord_encoder = nn.Sequential(
            nn.Linear(3, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim // 2)
        )
        
        self.atomic_attention = nn.MultiheadAttention(
            embed_dim=feature_dim // 2, num_heads=2, batch_first=True
        )
        self.atomic_pooling = nn.Linear(feature_dim // 2, feature_dim)
        
        # Other feature projections
        self.atomic_confidence_proj = nn.Linear(max_atoms, feature_dim)
        self.angle_projection = nn.Linear(2, feature_dim)
        self.neighbor_projection = nn.Linear(1, feature_dim)
        self.confidence_projection = nn.Linear(1, feature_dim)
        self.ss_embedding = nn.Embedding(ss_vocab_size, feature_dim)
        
        # Feature fusion
        total_feature_dim = d_model + 6 * feature_dim
        self.feature_projection = nn.Linear(total_feature_dim, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        hidden_dim = d_model // 4
        
        self.atomic_coord_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_atoms * 3)
        )
        
        self.atomic_confidence_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_atoms)
        )
        
        self.secondary_structure_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ss_vocab_size)
        )
        
        self.angles_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
    
    def create_time_embedding(self, timestep: torch.Tensor, dim: int) -> torch.Tensor:
        """Create sinusoidal time embeddings."""
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timestep.device) * -emb)
        emb = timestep[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb
    
    def process_atomic_features(self, atomic_coords: torch.Tensor) -> torch.Tensor:
        """Process multi-atom features per residue."""
        batch_size, seq_len, max_atoms, coord_dim = atomic_coords.shape
        
        atomic_coords_flat = atomic_coords.view(-1, max_atoms, coord_dim)
        atom_features = self.atomic_coord_encoder(atomic_coords_flat)
        atom_features_att, _ = self.atomic_attention(atom_features, atom_features, atom_features)
        pooled_features = torch.mean(atom_features_att, dim=1)
        residue_features = self.atomic_pooling(pooled_features)
        
        return residue_features.view(batch_size, seq_len, -1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: torch.Tensor, 
                atomic_coords: Optional[torch.Tensor] = None, 
                atomic_confidence: Optional[torch.Tensor] = None,
                angles: Optional[torch.Tensor] = None, 
                neighbors: Optional[torch.Tensor] = None, 
                confidence: Optional[torch.Tensor] = None, 
                secondary_structure: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        # Time embedding
        time_emb = self.create_time_embedding(t.float(), self.d_model)
        time_emb = self.time_mlp(time_emb)
        
        # Sequence embedding
        x_emb = self.embedding(x)
        x_emb = x_emb + self.pos_encoding[:, :x.size(1), :]
        
        # Feature embeddings
        feature_embeddings = [x_emb]
        
        if atomic_coords is not None:
            atomic_feat = self.process_atomic_features(atomic_coords)
            feature_embeddings.append(atomic_feat)
            
            if atomic_confidence is not None:
                atomic_conf_emb = self.atomic_confidence_proj(atomic_confidence)
                feature_embeddings.append(atomic_conf_emb)
        
        if angles is not None:
            angle_emb = self.angle_projection(angles)
            feature_embeddings.append(angle_emb)
        
        if neighbors is not None:
            neighbor_emb = self.neighbor_projection(neighbors.unsqueeze(-1))
            feature_embeddings.append(neighbor_emb)
        
        if confidence is not None:
            conf_emb = self.confidence_projection(confidence.unsqueeze(-1))
            feature_embeddings.append(conf_emb)
        
        if secondary_structure is not None:
            ss_emb = self.ss_embedding(secondary_structure)
            feature_embeddings.append(ss_emb)
        
        # Combine all features
        all_features = torch.cat(feature_embeddings, dim=-1)
        x_emb = self.feature_projection(all_features)
        
        # Add time embedding
        time_emb = time_emb.unsqueeze(1)
        x_emb = x_emb + time_emb
        
        # Transformer encoding
        src_key_padding_mask = (mask == 0)
        transformer_out = self.transformer_encoder(x_emb, src_key_padding_mask=src_key_padding_mask)
        
        # Predictions
        sequence_logits = self.output_layer(transformer_out)
        
        atomic_coord_pred = self.atomic_coord_head(transformer_out)
        atomic_coord_pred = atomic_coord_pred.view(transformer_out.shape[0], transformer_out.shape[1], self.max_atoms, 3)
        
        atomic_confidence_pred = self.atomic_confidence_head(transformer_out)
        ss_pred = self.secondary_structure_head(transformer_out)
        angles_pred = self.angles_head(transformer_out)
        
        return {
            'sequence_logits': sequence_logits,
            'atomic_coord_pred': atomic_coord_pred,
            'atomic_confidence_pred': atomic_confidence_pred,
            'ss_pred': ss_pred,
            'angles_pred': angles_pred
        }


class DiffusionScheduler:
    """Manages diffusion scheduling with BLOSUM-informed noise."""
    
    def __init__(self, aa_to_idx: Dict[str, int], T: int = 1000, 
                 beta_start: float = 2e-5, beta_end: float = 0.02, 
                 device: str = 'cuda', use_progressive: bool = True):
        self.aa_to_idx = aa_to_idx
        self.T = T
        self.device = device
        self.use_progressive = use_progressive
        
        # Standard diffusion schedule
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        
        # BLOSUM substitution matrices
        if use_progressive:
            blosum_schedule = BLOSUMMatrix.create_progressive_schedule(aa_to_idx, T)
            self.blosum_matrices = torch.stack([m.to(device) for m in blosum_schedule])
        else:
            blosum_matrix = BLOSUMMatrix.create_substitution_matrix(aa_to_idx).to(device)
            self.blosum_matrices = blosum_matrix.unsqueeze(0).expand(T, -1, -1)
    
    def add_noise(self, x: torch.Tensor, t: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """Add continuous BLOSUM-informed noise."""
        batch_size, seq_len = x.shape
        alpha_t = self.sqrt_alphas_cumprod[t - 1]
        noise_strength = self.sqrt_one_minus_alphas_cumprod[t - 1]
        
        x_noisy = torch.zeros_like(x)
        
        for b in range(batch_size):
            current_alpha = alpha_t[b].item()
            current_noise = noise_strength[b].item()
            
            for pos in range(seq_len):
                original_aa = x[b, pos].item()
                
                if original_aa < self.blosum_matrices.shape[1] and original_aa > 0:
                    one_hot = torch.zeros(vocab_size, device=self.device)
                    one_hot[original_aa] = 1.0
                    
                    t_idx = t[b].item() - 1
                    blosum_matrix = self.blosum_matrices[t_idx] if self.use_progressive else self.blosum_matrices[0]
                    blosum_probs = blosum_matrix[original_aa]
                    
                    mixed_probs = current_alpha * one_hot + current_noise * blosum_probs
                    mixed_probs = mixed_probs / mixed_probs.sum()
                    
                    new_aa = torch.multinomial(mixed_probs, 1).item()
                    x_noisy[b, pos] = new_aa
                else:
                    x_noisy[b, pos] = original_aa
        
        return x_noisy


class LossComputer:
    """Computes enhanced multi-task loss with atomic-level predictions."""
    
    @staticmethod
    def compute_loss(predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], 
                    mask: torch.Tensor, vocab_size: int, ss_vocab_size: int, 
                    max_atoms: int) -> Dict[str, torch.Tensor]:
        """Compute enhanced multi-task loss."""
        mask_flat = mask.view(-1).bool()
        
        # Sequence loss
        seq_pred = predictions['sequence_logits'].view(-1, vocab_size)
        seq_target = targets['sequence'].view(-1)
        seq_loss = nn.CrossEntropyLoss(ignore_index=0)(seq_pred[mask_flat], seq_target[mask_flat])
        
        # Atomic coordinate loss
        atomic_coord_pred = predictions['atomic_coord_pred']
        atomic_coord_target = targets['atomic_coordinates']
        atomic_mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(atomic_coord_pred).float()
        atomic_coord_loss = nn.MSELoss()(atomic_coord_pred * atomic_mask, atomic_coord_target * atomic_mask)
        
        # Atomic confidence loss
        atomic_conf_pred = predictions['atomic_confidence_pred']
        atomic_conf_target = targets['atomic_confidence']
        atomic_conf_mask = mask.unsqueeze(-1).expand_as(atomic_conf_pred).float()
        atomic_conf_loss = nn.MSELoss()(atomic_conf_pred * atomic_conf_mask, atomic_conf_target * atomic_conf_mask)
        
        # Secondary structure loss
        ss_pred = predictions['ss_pred'].view(-1, ss_vocab_size)
        ss_target = targets['secondary_structures'].view(-1)
        ss_loss = nn.CrossEntropyLoss(ignore_index=8)(ss_pred[mask_flat], ss_target[mask_flat])
        
        # Angles loss
        angles_pred = predictions['angles_pred']
        angles_target = torch.stack([targets['phi_angles'], targets['psi_angles']], dim=-1)
        angles_mask = mask.unsqueeze(-1).expand_as(angles_pred).float()
        angles_loss = nn.MSELoss()(angles_pred * angles_mask, angles_target * angles_mask)
        
        # Weighted combination
        total_loss = (
            1.0 * seq_loss +
            1.5 * atomic_coord_loss +
            1.0 * atomic_conf_loss +
            1.5 * ss_loss +
            1.5 * angles_loss
        )
        
        return {
            'total_loss': total_loss,
            'sequence_loss': seq_loss,
            'atomic_coord_loss': atomic_coord_loss,
            'atomic_conf_loss': atomic_conf_loss,
            'ss_loss': ss_loss,
            'angles_loss': angles_loss
        }


class DataLoaderFactory:
    """Factory for creating data loaders with optional distributed sampling."""
    
    @staticmethod
    def create_loaders(train_dataset: Dataset, val_dataset: Dataset, batch_size: int, 
                      rank: Optional[int] = None, world_size: Optional[int] = None, 
                      num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
        """Create data loaders with optional distributed sampling."""
        if rank is not None and world_size is not None:
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
            val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
            
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, sampler=train_sampler,
                num_workers=num_workers, pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, sampler=val_sampler,
                num_workers=num_workers, pin_memory=True
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True
            )
        
        return train_loader, val_loader


class Trainer:
    """Handles training and validation loops."""
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                 diffusion_scheduler: DiffusionScheduler, vocab_size: int, 
                 ss_vocab_size: int, max_atoms: int, device: str, 
                 gradient_clip: float = 0.1):
        self.model = model
        self.optimizer = optimizer
        self.diffusion_scheduler = diffusion_scheduler
        self.vocab_size = vocab_size
        self.ss_vocab_size = ss_vocab_size
        self.max_atoms = max_atoms
        self.device = device
        self.gradient_clip = gradient_clip
        self.loss_computer = LossComputer()
    
    def train_epoch(self, train_loader: DataLoader, T: int = 1000) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device, non_blocking=True)
            
            batch_size = batch['sequence'].shape[0]
            timesteps = torch.randint(1, T + 1, (batch_size,), device=self.device)
            
            # Add BLOSUM-informed noise
            noisy_sequences_list = []
            for i, t in enumerate(timesteps):
                single_seq = batch['sequence'][i:i+1]
                single_t = torch.tensor([t.item()], device=self.device)
                noisy_seq = self.diffusion_scheduler.add_noise(single_seq, single_t, self.vocab_size)
                noisy_sequences_list.append(noisy_seq)
            
            noisy_sequences = torch.cat(noisy_sequences_list, dim=0)
            
            angles_input = torch.stack([batch['phi_angles'], batch['psi_angles']], dim=-1)
            
            predictions = self.model(
                noisy_sequences, timesteps, batch['mask'],
                atomic_coords=batch['atomic_coordinates'],
                atomic_confidence=batch['atomic_confidence'],
                angles=angles_input,
                neighbors=batch['neighbor_counts'],
                confidence=batch['confidence_masks'],
                secondary_structure=batch['secondary_structures']
            )
            
            loss_dict = self.loss_computer.compute_loss(
                predictions, batch, batch['mask'], self.vocab_size, self.ss_vocab_size, self.max_atoms
            )
            loss = loss_dict['total_loss']
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Clean up
            del batch, predictions, loss_dict, noisy_sequences
            torch.cuda.empty_cache()
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader: DataLoader, T: int = 1000) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device, non_blocking=True)
                
                batch_size = batch['sequence'].shape[0]
                timesteps = torch.randint(1, T + 1, (batch_size,), device=self.device)
                
                noisy_sequences_list = []
                for i, t in enumerate(timesteps):
                    single_seq = batch['sequence'][i:i+1]
                    single_t = torch.tensor([t.item()], device=self.device)
                    noisy_seq = self.diffusion_scheduler.add_noise(single_seq, single_t, self.vocab_size)
                    noisy_sequences_list.append(noisy_seq)
                
                noisy_sequences = torch.cat(noisy_sequences_list, dim=0)
                
                angles_input = torch.stack([batch['phi_angles'], batch['psi_angles']], dim=-1)
                
                predictions = self.model(
                    noisy_sequences, timesteps, batch['mask'],
                    atomic_coords=batch['atomic_coordinates'],
                    atomic_confidence=batch['atomic_confidence'],
                    angles=angles_input,
                    neighbors=batch['neighbor_counts'],
                    confidence=batch['confidence_masks'],
                    secondary_structure=batch['secondary_structures']
                )
                
                loss_dict = self.loss_computer.compute_loss(
                    predictions, batch, batch['mask'], self.vocab_size, self.ss_vocab_size, self.max_atoms
                )
                total_loss += loss_dict['total_loss'].item()
                num_batches += 1
        
        return total_loss / num_batches


class SequenceGenerator:
    """Generates sequences using BLOSUM-guided reverse diffusion."""
    
    def __init__(self, model: nn.Module, diffusion_scheduler: DiffusionScheduler, 
                 vocab_size: int, aa_to_idx: Dict[str, int], idx_to_aa: Dict[int, str], 
                 ss_to_idx: Dict[str, int], max_atoms: int, device: str):
        self.model = model
        self.diffusion_scheduler = diffusion_scheduler
        self.vocab_size = vocab_size
        self.aa_to_idx = aa_to_idx
        self.idx_to_aa = idx_to_aa
        self.ss_to_idx = ss_to_idx
        self.max_atoms = max_atoms
        self.device = device
        self.EOS_idx = aa_to_idx['EOS']
        self.PAD_idx = aa_to_idx['PAD']
    
    def compute_mean_features(self, train_dataset: Dataset, sample_size: int = 500) -> Dict[str, torch.Tensor]:
        """Compute mean features from training data."""
        sample_size = min(sample_size, len(train_dataset))
        sample_indices = torch.randperm(len(train_dataset))[:sample_size]
        
        feature_lists = {
            'atomic_coordinates': [],
            'atomic_confidence': [],
            'phi_angles': [],
            'psi_angles': [],
            'neighbor_counts': [],
            'confidence_masks': [],
            'secondary_structures': []
        }
        
        for idx in sample_indices:
            sample = train_dataset[idx]
            for key in feature_lists:
                feature_lists[key].append(sample[key])
        
        mean_features = {}
        for key in ['atomic_coordinates', 'atomic_confidence', 'neighbor_counts', 'confidence_masks']:
            mean_features[key] = torch.stack(feature_lists[key]).mean(dim=0, keepdim=True).to(self.device)
        
        # Handle angles separately
        mean_phi = torch.stack(feature_lists['phi_angles']).mean(dim=0, keepdim=True).unsqueeze(-1).to(self.device)
        mean_psi = torch.stack(feature_lists['psi_angles']).mean(dim=0, keepdim=True).unsqueeze(-1).to(self.device)
        mean_features['angles'] = torch.cat([mean_phi, mean_psi], dim=-1)
        
        # Handle secondary structure with mode
        mean_features['secondary_structures'] = torch.stack(feature_lists['secondary_structures']).mode(dim=0)[0].unsqueeze(0).to(self.device)
        
        return mean_features
    
    def generate_sequences(self, num_sequences: int = 10, seq_len: Optional[int] = None, 
                          num_steps: int = 1000, train_dataset: Optional[Dataset] = None, 
                          guidance_strength: float = 1.0, use_mean_features: bool = True) -> List[Dict]:
        """Generate sequences using BLOSUM-guided reverse diffusion."""
        self.model.eval()
        
        if seq_len is None:
            seq_len = self.model.max_len
        
        # Compute mean features if requested
        mean_features = None
        if use_mean_features and train_dataset is not None:
            mean_features = self.compute_mean_features(train_dataset)
        
        generated_results = []
        
        with torch.no_grad():
            for seq_idx in range(num_sequences):
                # Start from pure noise
                x = torch.randint(1, self.vocab_size - 1, (1, seq_len), device=self.device)
                mask = torch.ones_like(x).long()
                
                # Reverse diffusion process with BLOSUM guidance
                for step in range(num_steps):
                    t = num_steps - step
                    t_tensor = torch.tensor([t], device=self.device)
                    
                    # Forward pass
                    predictions = self.model(
                        x, t_tensor, mask,
                        atomic_coords=mean_features['atomic_coordinates'] if mean_features else None,
                        atomic_confidence=mean_features['atomic_confidence'] if mean_features else None,
                        angles=mean_features['angles'] if mean_features else None,
                        neighbors=mean_features['neighbor_counts'] if mean_features else None,
                        confidence=mean_features['confidence_masks'] if mean_features else None,
                        secondary_structure=mean_features['secondary_structures'] if mean_features else None
                    )
                    
                    predicted_logits = predictions['sequence_logits']
                    
                    # Apply BLOSUM guidance
                    if guidance_strength > 0.0 and t > 1:
                        t_idx = t - 1
                        blosum_matrix = (self.diffusion_scheduler.blosum_matrices[t_idx] 
                                       if self.diffusion_scheduler.use_progressive 
                                       else self.diffusion_scheduler.blosum_matrices[0])
                        
                        current_tokens = x.squeeze(0)
                        for pos in range(seq_len):
                            current_aa = current_tokens[pos].item()
                            if current_aa < len(blosum_matrix):
                                blosum_probs = blosum_matrix[current_aa]
                                predicted_logits[0, pos] = (
                                    (1 - guidance_strength) * predicted_logits[0, pos] +
                                    guidance_strength * torch.log(blosum_probs + 1e-8)
                                )
                    
                    # Sample next tokens
                    if t > 1:
                        temperature = 0.8
                        probs = torch.softmax(predicted_logits / temperature, dim=-1)
                        x = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(1, seq_len)
                    else:
                        x = torch.argmax(predicted_logits, dim=-1)
                
                # Get final predictions
                final_t_tensor = torch.tensor([1], device=self.device)
                final_predictions = self.model(
                    x, final_t_tensor, mask,
                    atomic_coords=mean_features['atomic_coordinates'] if mean_features else None,
                    atomic_confidence=mean_features['atomic_confidence'] if mean_features else None,
                    angles=mean_features['angles'] if mean_features else None,
                    neighbors=mean_features['neighbor_counts'] if mean_features else None,
                    confidence=mean_features['confidence_masks'] if mean_features else None,
                    secondary_structure=mean_features['secondary_structures'] if mean_features else None
                )
                
                generated_results.append({
                    'sequence_encoded': x.cpu(),
                    'atomic_coords_predicted': final_predictions['atomic_coord_pred'].cpu(),
                    'atomic_confidence_predicted': final_predictions['atomic_confidence_pred'].cpu(),
                    'ss_predicted_logits': final_predictions['ss_pred'].cpu(),
                    'angles_predicted': final_predictions['angles_pred'].cpu(),
                    'mask': mask.cpu()
                })
        
        return generated_results
    
    @staticmethod
    def decode_sequences(generated_results: List[Dict], idx_to_aa: Dict[int, str]) -> List[str]:
        """Decode generated sequences from token indices to amino acids."""
        decoded_sequences = []
        PAD_idx = 0
        EOS_idx = len(idx_to_aa) - 1
        
        for result_dict in generated_results:
            seq_tensor = result_dict['sequence_encoded']
            seq_indices = seq_tensor.squeeze().tolist()
            
            amino_acids = []
            for idx in seq_indices:
                if idx == PAD_idx or idx == EOS_idx:
                    break
                if idx in idx_to_aa:
                    amino_acids.append(idx_to_aa[idx])
            
            decoded_sequences.append(''.join(amino_acids))
        
        return decoded_sequences


class ResultsAnalyzer:
    """Analyzes and saves generated sequences."""
    
    @staticmethod
    def analyze_sequences(sequences: List[str], results: List[Dict]) -> None:
        """Analyze generated sequences."""
        logger.info("="*60)
        logger.info("SEQUENCE ANALYSIS")
        logger.info("="*60)
        
        lengths = [len(seq) for seq in sequences]
        logger.info(f"Generated {len(sequences)} sequences")
        logger.info(f"Average length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
        logger.info(f"Length range: {min(lengths)} - {max(lengths)}")
        
        # Amino acid composition
        all_aa_counts = Counter()
        total_residues = 0
        for seq in sequences:
            for aa in seq:
                all_aa_counts[aa] += 1
                total_residues += 1
        
        # logger.info("Top amino acids:")
        sorted_aas = sorted(all_aa_counts.items(), key=lambda x: x[1], reverse=True)
        for aa, count in sorted_aas[:5]:
            pct = (count / total_residues) * 100
            # logger.info(f"  {aa}: {pct:.1f}%")
        
        # Diversity
        unique_sequences = len(set(sequences))
        diversity = (unique_sequences / len(sequences)) * 100
        # logger.info(f"Sequence diversity: {diversity:.1f}% ({unique_sequences}/{len(sequences)} unique)")
    
    @staticmethod
    def save_sequences(sequences: List[str], results: List[Dict], filename: Optional[str] = None) -> str:
        """Save generated sequences to CSV."""
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_rubisco_sequences_{timestamp}.csv"
        
        data_rows = []
        
        for i, (sequence, result) in enumerate(zip(sequences, results)):
            atomic_coords = result['atomic_coords_predicted'].squeeze()
            atomic_conf = result['atomic_confidence_predicted'].squeeze()
            angles = result['angles_predicted'].squeeze()
            
            seq_len = len(sequence)
            
            avg_atomic_conf = atomic_conf[:seq_len].mean().item() if seq_len > 0 else 0.0
            avg_phi = angles[:seq_len, 0].mean().item() if seq_len > 0 else 0.0
            avg_psi = angles[:seq_len, 1].mean().item() if seq_len > 0 else 0.0
            
            data_rows.append({
                'ID': f'Generated_RuBisCO_{i+1:03d}',
                'sequence': sequence,
                'length': len(sequence),
                'avg_atomic_confidence': avg_atomic_conf,
                'avg_phi_angle': avg_phi,
                'avg_psi_angle': avg_psi,
                'generation_method': 'BLOSUM_Guided_Diffusion'
            })
        
        results_df = pd.DataFrame(data_rows)
        results_df.to_csv(filename, index=False)
        logger.info(f"Sequences saved to: {filename}")
        
        return filename


class RuBisCODiffusionPipeline:
    """Main pipeline for RuBisCO diffusion model training and generation."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.dist_manager = DistributedManager()
        
        # Setup distributed training
        self.rank, self.world_size, self.gpu = self.dist_manager.setup()
        
        # Device setup
        if self.gpu is not None:
            self.device = f'cuda:{self.gpu}'
            torch.cuda.set_device(self.gpu)
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Vocabularies
        self.amino_acids = ['PAD', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                           'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'EOS']
        self.aa_to_idx = {aa: idx for idx, aa in enumerate(self.amino_acids)}
        self.idx_to_aa = {idx: aa for idx, aa in enumerate(self.amino_acids)}
        self.ss_to_idx = {'H': 0, 'E': 1, 'C': 2, 'G': 3, 'I': 4, 'B': 5, 'T': 6, 'S': 7, '-': 8}
        
        self.vocab_size = len(self.aa_to_idx)
        self.ss_vocab_size = len(self.ss_to_idx)
        self.max_atoms = 5
    
    def load_data(self) -> Tuple[Dataset, Dataset]:
        """Load and prepare datasets."""
        logger.info("Loading data...")
        df = pd.read_csv(self.config['data_path'], nrows=2)
        
        max_len = min(df['sequence'].str.len().max(), self.config['max_len'])
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        
        train_dataset = EnhancedRuBisCODataset(train_df, self.aa_to_idx, self.ss_to_idx, max_len)
        val_dataset = EnhancedRuBisCODataset(val_df, self.aa_to_idx, self.ss_to_idx, max_len)
        
        self.max_len = max_len
        return train_dataset, val_dataset
    
    def create_model(self) -> nn.Module:
        """Create and setup the model."""
        model = AtomicEnhancedRuBisCOModel(
            vocab_size=self.vocab_size,
            ss_vocab_size=self.ss_vocab_size,
            d_model=self.config['d_model'],
            num_heads=self.config['num_heads'],
            num_layers=self.config['num_layers'],
            d_ff=self.config['d_ff'],
            dropout=self.config['dropout'],
            max_len=self.max_len,
            max_atoms=self.max_atoms
        ).to(self.device)
        
        if self.rank is not None:
            model = DDP(model, device_ids=[self.gpu], find_unused_parameters=True)
        
        return model
    
    def train(self, train_dataset: Dataset, val_dataset: Dataset) -> None:
        """Main training loop."""
        # Create data loaders
        train_loader, val_loader = DataLoaderFactory.create_loaders(
            train_dataset, val_dataset, self.config['batch_size'], 
            self.rank, self.world_size, self.config.get('num_workers', 2)
        )
        
        # Create model and components
        model = self.create_model()
        
        diffusion_scheduler = DiffusionScheduler(
            self.aa_to_idx, T=self.config['T'], device=self.device,
            use_progressive=self.config.get('use_progressive', True)
        )
        
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.config['learning_rate'], 
            weight_decay=self.config['weight_decay']
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config['num_epochs']
        )
        
        trainer = Trainer(
            model, optimizer, diffusion_scheduler, self.vocab_size, 
            self.ss_vocab_size, self.max_atoms, self.device, 
            self.config.get('gradient_clip', 0.1)
        )
        
        # Training loop
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        patience = self.config.get('patience', 30)
        
        if self.dist_manager.is_main_process():
            logger.info("Starting training...")
        
        for epoch in range(self.config['num_epochs']):
            if self.rank is not None and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            
            train_loss = trainer.train_epoch(train_loader, self.config['T'])
            val_loss = trainer.validate_epoch(val_loader, self.config['T'])
            scheduler.step()
            
            if self.dist_manager.is_main_process():
                logger.info(f'Epoch {epoch+1}/{self.config["num_epochs"]}:')
                logger.info(f'  Train Loss: {train_loss:.4f}')
                logger.info(f'  Val Loss: {val_loss:.4f}')
                logger.info(f'  LR: {scheduler.get_last_lr()[0]:.6f}')
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    
                    model_to_save = model.module if hasattr(model, 'module') else model
                    checkpoint = {
                        'model_state_dict': model_to_save.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'best_val_loss': best_val_loss,
                        'diffusion_schedule': {
                            'betas': diffusion_scheduler.betas,
                            'alphas_cumprod': diffusion_scheduler.alphas_cumprod,
                            'sqrt_alphas_cumprod': diffusion_scheduler.sqrt_alphas_cumprod,
                            'sqrt_one_minus_alphas_cumprod': diffusion_scheduler.sqrt_one_minus_alphas_cumprod,
                            'blosum_matrices': diffusion_scheduler.blosum_matrices,
                            'use_progressive': diffusion_scheduler.use_progressive
                        },
                        'model_config': {
                            'vocab_size': self.vocab_size,
                            'ss_vocab_size': self.ss_vocab_size,
                            'd_model': self.config['d_model'],
                            'num_heads': self.config['num_heads'],
                            'num_layers': self.config['num_layers'],
                            'max_len': self.max_len,
                            'max_atoms': self.max_atoms
                        },
                        'aa_to_idx': self.aa_to_idx,
                        'ss_to_idx': self.ss_to_idx,
                        'idx_to_aa': self.idx_to_aa
                    }
                    
                    torch.save(checkpoint, self.config['checkpoint_path'])
                    logger.info(f"  ✓ New best model saved (val_loss: {best_val_loss:.4f})")
                else:
                    epochs_without_improvement += 1
                
                if epochs_without_improvement >= patience:
                    logger.info(f"Early stopping after {patience} epochs without improvement")
                    break
    
    def generate(self, train_dataset: Dataset) -> None:
        """Generate sequences with different guidance strengths."""
        if not self.dist_manager.is_main_process():
            return
        
        # logger.info("="*70)
        # logger.info("STARTING SEQUENCE GENERATION")
        # logger.info("="*70)
        
        # Load best model
        checkpoint = torch.load(self.config['checkpoint_path'], map_location=self.device)
        
        # Recreate model
        model = AtomicEnhancedRuBisCOModel(**checkpoint['model_config']).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Recreate diffusion scheduler
        diffusion_schedule_data = checkpoint['diffusion_schedule']
        diffusion_scheduler = DiffusionScheduler(
            self.aa_to_idx, T=self.config['T'], device=self.device,
            use_progressive=diffusion_schedule_data['use_progressive']
        )
        # Restore saved parameters
        diffusion_scheduler.betas = diffusion_schedule_data['betas']
        diffusion_scheduler.alphas_cumprod = diffusion_schedule_data['alphas_cumprod']
        diffusion_scheduler.sqrt_alphas_cumprod = diffusion_schedule_data['sqrt_alphas_cumprod']
        diffusion_scheduler.sqrt_one_minus_alphas_cumprod = diffusion_schedule_data['sqrt_one_minus_alphas_cumprod']
        diffusion_scheduler.blosum_matrices = diffusion_schedule_data['blosum_matrices']
        
        generator = SequenceGenerator(
            model, diffusion_scheduler, self.vocab_size, self.aa_to_idx, 
            self.idx_to_aa, self.ss_to_idx, self.max_atoms, self.device
        )
        
        # Generate with different guidance strengths
        guidance_strengths = self.config.get('guidance_strengths', [0.0, 0.3, 0.7, 1.0])
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for guidance in guidance_strengths:
            # logger.info(f"Generating with BLOSUM guidance strength: {guidance}")
            
            generated_results = generator.generate_sequences(
                num_sequences=self.config.get('num_sequences', 10),
                num_steps=self.config.get('generation_steps', 800),
                train_dataset=train_dataset,
                guidance_strength=guidance,
                use_mean_features=True
            )
            
            sequences = SequenceGenerator.decode_sequences(generated_results, self.idx_to_aa)
            
            # Analyze and save
            ResultsAnalyzer.analyze_sequences(sequences, generated_results)
            filename = f"rubisco_sequences_guidance_{guidance}_{timestamp}.csv"
            ResultsAnalyzer.save_sequences(sequences, generated_results, filename)
        
        # logger.info("Generation completed!")
    
    def run(self) -> None:
        """Run the complete pipeline."""
        try:
            # Load data
            train_dataset, val_dataset = self.load_data()
            
            # Train model
            self.train(train_dataset, val_dataset)
            
            # Generate sequences
            self.generate(train_dataset)
            
        finally:
            self.dist_manager.cleanup()


def main():
    """Main function to run the RuBisCO diffusion pipeline."""
    config = {
        # Data configuration
        'data_path': './output/final_data2.csv',
        'max_len': 550,
        
        # Model configuration
        'd_model': 512,
        'num_heads': 16,
        'num_layers': 12,
        'd_ff': 2048,
        'dropout': 0.3,
        
        # Training configuration
        'batch_size': 4,
        'num_epochs': 500,
        'learning_rate': 1e-5,
        'weight_decay': 1e-4,
        'gradient_clip': 0.1,
        'patience': 30,
        'num_workers': 2,
        
        # Diffusion configuration
        'T': 1000,
        'use_progressive': True,
        
        # Generation configuration
        'num_sequences': 10,
        'generation_steps': 1000,
        'guidance_strengths': [0.0, 0.3, 0.7, 1.0],
        
        # Output configuration
        'checkpoint_path': 'best_rubisco_diffusion_model.pth'
    }
    
    # Set up multi-GPU environment variables if not already set
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        # Use all available GPUs
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(num_gpus)))
            logger.info(f"Using {num_gpus} GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    pipeline = RuBisCODiffusionPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()