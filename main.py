#!/usr/bin/env python3
"""
EEGNN-XAI (CVMI 2024) — Entry point
Train a DeepGCN-like model on HSI data and generate XAI maps (IG / Saliency / GradCAM).
"""
from src.core.parser import build_parser
from src.core.runner import run

def main():
    args = build_parser("EEGNN-XAI — XAI for GNN-based HSI (CVMI 2024)").parse_args()
    run(args)

if __name__ == "__main__":
    main()
