"""
Inference pipeline for PeptideLocator2.

Loads ESM2-8M (pretrained or fine-tuned) + two MLP heads (sites, peptides)
and returns per-residue probabilities for a raw protein sequence.

Model loading priority:
  1. Fine-tuned ESM2 checkpoint (if --model-path provided)
  2. Pretrained ESM2-8M from HuggingFace + saved MLP head weights (models/*.pt)
  3. Pretrained ESM2-8M + randomly initialised heads (demo/UI testing only)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional

from transformers import EsmModel, EsmTokenizer


# ── MLP head (must match training architecture) ───────────────────────────────

class MLPHead(nn.Module):
    def __init__(self, in_features: int = 320, hidden: int = 320,
                 n_layers: int = 2, n_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.hidden = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(n_layers)]
        )
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.out(x)


# ── Prediction output ─────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    sequence: str
    sites_proba: np.ndarray       # per-residue probability of cleavage site
    peptides_proba: np.ndarray    # per-residue probability of peptide region
    sites_predicted: np.ndarray   # binary (threshold 0.5)
    peptides_predicted: np.ndarray
    model_info: str               # description of model used


# ── Predictor ────────────────────────────────────────────────────────────────

class PeptideLocatorPredictor:
    """
    Loads the model once and exposes a predict() method.

    Usage:
        predictor = PeptideLocatorPredictor()
        result = predictor.predict("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGEDEDtokenized")
    """

    ESM2_MODEL_ID = "facebook/esm2_t6_8M_UR50D"
    IN_FEATURES   = 320

    def __init__(
        self,
        finetune_sites_path:   Optional[str] = None,
        finetune_peptides_path: Optional[str] = None,
        sites_head_path:   str = "models/sites_head.pt",
        peptides_head_path: str = "models/peptides_head.pt",
        device: Optional[str] = None,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.tokenizer = EsmTokenizer.from_pretrained(self.ESM2_MODEL_ID)

        # ── Load ESM2 backbones ───────────────────────────────────────────────
        # For sites and peptides we may have separate fine-tuned backbones,
        # or share a single pretrained one.

        if finetune_sites_path and os.path.isdir(
            os.path.join(finetune_sites_path, "esm")
        ):
            self.esm_sites = EsmModel.from_pretrained(
                os.path.join(finetune_sites_path, "esm")
            ).to(self.device).eval()
            self._model_info_sites = f"Fine-tuned ESM2-8M ({finetune_sites_path})"
        else:
            self.esm_sites = EsmModel.from_pretrained(
                self.ESM2_MODEL_ID
            ).to(self.device).eval()
            self._model_info_sites = "Pretrained ESM2-8M"

        if finetune_peptides_path and os.path.isdir(
            os.path.join(finetune_peptides_path, "esm")
        ):
            self.esm_peptides = EsmModel.from_pretrained(
                os.path.join(finetune_peptides_path, "esm")
            ).to(self.device).eval()
            self._model_info_peptides = f"Fine-tuned ESM2-8M ({finetune_peptides_path})"
        else:
            # Reuse the same backbone if no separate fine-tune available
            self.esm_peptides = self.esm_sites
            self._model_info_peptides = self._model_info_sites

        # ── Load MLP heads ────────────────────────────────────────────────────
        self.sites_head   = MLPHead(self.IN_FEATURES).to(self.device).eval()
        self.peptides_head = MLPHead(self.IN_FEATURES).to(self.device).eval()

        if finetune_sites_path and os.path.isfile(
            os.path.join(finetune_sites_path, "head.pt")
        ):
            self.sites_head.load_state_dict(
                torch.load(os.path.join(finetune_sites_path, "head.pt"),
                           map_location=self.device)
            )
            self._model_info_sites += " + fine-tuned head"
        elif os.path.isfile(sites_head_path):
            self.sites_head.load_state_dict(
                torch.load(sites_head_path, map_location=self.device)
            )
            self._model_info_sites += " + frozen-trained head"
        else:
            self._model_info_sites += " + UNTRAINED head (demo only)"

        if finetune_peptides_path and os.path.isfile(
            os.path.join(finetune_peptides_path, "head.pt")
        ):
            self.peptides_head.load_state_dict(
                torch.load(os.path.join(finetune_peptides_path, "head.pt"),
                           map_location=self.device)
            )
            self._model_info_peptides += " + fine-tuned head"
        elif os.path.isfile(peptides_head_path):
            self.peptides_head.load_state_dict(
                torch.load(peptides_head_path, map_location=self.device)
            )
            self._model_info_peptides += " + frozen-trained head"
        else:
            self._model_info_peptides += " + UNTRAINED head (demo only)"

    def _embed(self, sequence: str, esm_model: EsmModel) -> torch.Tensor:
        """Run ESM2 forward pass and return per-residue embeddings (seq_len, 320)."""
        tokens = self.tokenizer(
            sequence,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)
        with torch.no_grad():
            output = esm_model(**tokens)
        # Strip BOS (0) and EOS (seq_len+1)
        return output.last_hidden_state[0, 1:len(sequence) + 1, :]

    def predict(self, sequence: str, threshold: float = 0.5) -> PredictionResult:
        """
        Run prediction on a single protein sequence.

        Args:
            sequence:  Amino acid sequence (single-letter codes, no gaps)
            threshold: Probability threshold for binary prediction

        Returns:
            PredictionResult with per-residue probabilities and binary labels
        """
        sequence = sequence.strip().upper()
        # Remove whitespace and invalid characters
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        sequence = "".join(c if c in valid_aa else "X" for c in sequence
                           if not c.isspace())

        if len(sequence) == 0:
            raise ValueError("Empty sequence after cleaning.")
        if len(sequence) > 1022:
            sequence = sequence[:1022]

        with torch.no_grad():
            # Sites
            emb_sites = self._embed(sequence, self.esm_sites)
            logits_sites = self.sites_head(emb_sites)
            proba_sites = torch.softmax(logits_sites, dim=-1)[:, 1].cpu().numpy()

            # Peptides (may share backbone with sites)
            if self.esm_peptides is self.esm_sites:
                emb_peptides = emb_sites
            else:
                emb_peptides = self._embed(sequence, self.esm_peptides)
            logits_peptides = self.peptides_head(emb_peptides)
            proba_peptides = torch.softmax(logits_peptides, dim=-1)[:, 1].cpu().numpy()

        model_info = (
            f"Sites:   {self._model_info_sites}\n"
            f"Peptides: {self._model_info_peptides}"
        )

        return PredictionResult(
            sequence=sequence,
            sites_proba=proba_sites,
            peptides_proba=proba_peptides,
            sites_predicted=(proba_sites >= threshold).astype(int),
            peptides_predicted=(proba_peptides >= threshold).astype(int),
            model_info=model_info,
        )
