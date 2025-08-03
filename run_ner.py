#!/usr/bin/env python3
"""
Clinical NER CLI Tool
Simple command-line interface for batch processing clinical notes
"""

import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ner_processor import ClinicalNERProcessor
import config


def main():
    parser = argparse.ArgumentParser(
        description="Clinical Named Entity Recognition - Extract medical entities from clinical notes"
    )
    
    parser.add_argument(
        '--sample-size', 
        type=int, 
        default=config.SAMPLE_SIZE,
        help=f'Number of notes to process (default: {config.SAMPLE_SIZE}, -1 for all)'
    )
    
    parser.add_argument(
        '--model',
        choices=['scispacy', 'baseline'],
        default=config.PRIMARY_MODEL,
        help=f'NER model to use (default: {config.PRIMARY_MODEL})'
    )
    
    parser.add_argument(
        '--output',
        default=config.OUTPUT_PATH,
        help=f'Output file path (default: {config.OUTPUT_PATH})'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Update config with CLI arguments
    config.SAMPLE_SIZE = args.sample_size
    config.PRIMARY_MODEL = args.model
    config.OUTPUT_PATH = args.output
    config.VERBOSE = not args.quiet
    
    print("🏥 Clinical NER Batch Processor")
    print("=" * 40)
    print(f"📊 Processing: {args.sample_size if args.sample_size > 0 else 'ALL'} notes")
    print(f"🤖 Model: {args.model}")
    print(f"💾 Output: {args.output}")
    print("=" * 40)
    
    # Run processor
    processor = ClinicalNERProcessor()
    results_df = processor.process_all_notes()
    processor.save_results(results_df)


if __name__ == "__main__":
    main()
