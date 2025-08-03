"""
Clinical Named Entity Recognition - Batch Processing Tool
Author: Rithvik Katakam
Description: CLI tool for extracting medical entities from clinical notes
"""

import pandas as pd
import pickle
import spacy
import sys
import os
from typing import List, Dict, Any
from tqdm import tqdm
import logging
from datetime import datetime
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/ner_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ClinicalNERProcessor:
    """Batch processor for clinical NER on MIMIC notes"""
    
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the specified NER model"""
        try:
            if config.PRIMARY_MODEL == "scispacy":
                self.model = spacy.load("en_core_sci_lg")
                logger.info("✅ Loaded SciSpaCy clinical model")
            else:
                self.model = spacy.load("en_core_web_sm")
                logger.info("✅ Loaded standard spaCy model")
        except OSError as e:
            logger.error(f"❌ Failed to load model: {e}")
            logger.error("Please install required models:")
            if config.PRIMARY_MODEL == "scispacy":
                logger.error("pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_lg-0.5.0.tar.gz")
            else:
                logger.error("python -m spacy download en_core_web_sm")
            sys.exit(1)
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess clinical text"""
        if not isinstance(text, str) or not text.strip():
            return ""
        
        if not config.ENABLE_PREPROCESSING:
            return text
        
        # Truncate if too long
        if len(text) > config.MAX_TEXT_LENGTH:
            text = text[:config.MAX_TEXT_LENGTH]
        
        # Basic cleaning
        import re
        text = re.sub(r'\s+', ' ', text.strip())
        text = text.replace('pt.', 'patient')
        text = text.replace('hx', 'history')
        text = text.replace('dx', 'diagnosis')
        
        return text
    
    def extract_entities_from_text(self, text: str) -> List[Dict]:
        """Extract entities from a single text"""
        try:
            processed_text = self.preprocess_text(text)
            if not processed_text:
                return []
            
            doc = self.model(processed_text)
            entities = []
            
            for ent in doc.ents:
                # Filter for medical entities
                if any(med_type in ent.label_.upper() for med_type in config.MEDICAL_ENTITY_TYPES):
                    confidence = getattr(ent, 'score', 0.9)
                    
                    if confidence >= config.CONFIDENCE_THRESHOLD:
                        entities.append({
                            'text': ent.text.strip(),
                            'label': ent.label_,
                            'start': ent.start_char,
                            'end': ent.end_char,
                            'confidence': confidence
                        })
            
            return entities
        
        except Exception as e:
            logger.warning(f"Error processing text: {e}")
            return []
    
    def load_mimic_data(self) -> pd.DataFrame:
        """Load MIMIC clinical notes from pickle file"""
        try:
            logger.info(f"Loading data from {config.DATA_PATH}")
            
            with open(config.DATA_PATH, 'rb') as f:
                data = pickle.load(f)
            
            if not isinstance(data, pd.DataFrame):
                logger.error(f"Expected DataFrame, got {type(data)}")
                sys.exit(1)
            
            logger.info(f"✅ Loaded {len(data)} clinical notes")
            
            # Sample if requested
            if config.SAMPLE_SIZE > 0 and len(data) > config.SAMPLE_SIZE:
                data = data.sample(n=config.SAMPLE_SIZE, random_state=42)
                logger.info(f"📊 Sampled {len(data)} notes for processing")
            
            return data
        
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            sys.exit(1)
    
    def process_batch(self, notes_batch: List[str], batch_idx: int) -> List[Dict]:
        """Process a batch of notes"""
        batch_results = []
        
        for note_idx, note_text in enumerate(notes_batch):
            if config.VERBOSE and note_idx % 10 == 0:
                logger.info(f"Processing batch {batch_idx}, note {note_idx}/{len(notes_batch)}")
            
            entities = self.extract_entities_from_text(note_text)
            
            for entity in entities:
                batch_results.append({
                    'note_id': batch_idx * config.BATCH_SIZE + note_idx,
                    'entity_text': entity['text'],
                    'entity_type': entity['label'],
                    'start_pos': entity['start'],
                    'end_pos': entity['end'],
                    'confidence': entity['confidence']
                })
        
        return batch_results
    
    def process_all_notes(self) -> pd.DataFrame:
        """Process all clinical notes and extract entities"""
        # Load data
        df = self.load_mimic_data()
        
        if 'text' not in df.columns:
            logger.error("❌ No 'text' column found in data")
            sys.exit(1)
        
        # Get notes text
        notes = df['text'].dropna().tolist()
        logger.info(f"🔄 Processing {len(notes)} notes with {config.PRIMARY_MODEL} model")
        
        # Process in batches
        all_results = []
        num_batches = (len(notes) + config.BATCH_SIZE - 1) // config.BATCH_SIZE
        
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_idx * config.BATCH_SIZE
            end_idx = min(start_idx + config.BATCH_SIZE, len(notes))
            batch_notes = notes[start_idx:end_idx]
            
            batch_results = self.process_batch(batch_notes, batch_idx)
            all_results.extend(batch_results)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        logger.info(f"✅ Extracted {len(results_df)} entities from {len(notes)} notes")
        
        return results_df
    
    def generate_summary_stats(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics"""
        stats = {
            'total_notes_processed': results_df['note_id'].nunique() if len(results_df) > 0 else 0,
            'total_entities_extracted': len(results_df),
            'unique_entity_types': results_df['entity_type'].nunique() if len(results_df) > 0 else 0,
            'avg_entities_per_note': len(results_df) / results_df['note_id'].nunique() if len(results_df) > 0 else 0,
            'entity_type_distribution': results_df['entity_type'].value_counts().to_dict() if len(results_df) > 0 else {},
            'avg_confidence': results_df['confidence'].mean() if len(results_df) > 0 else 0,
            'processing_timestamp': datetime.now().isoformat()
        }
        
        return stats
    
    def save_results(self, results_df: pd.DataFrame):
        """Save results and summary statistics"""
        # Create outputs directory if it doesn't exist
        os.makedirs('outputs', exist_ok=True)
        
        # Save full results
        if config.SAVE_FULL_RESULTS:
            results_df.to_csv(config.OUTPUT_PATH, index=False)
            logger.info(f"💾 Saved full results to {config.OUTPUT_PATH}")
        
        # Save summary statistics
        if config.SAVE_SUMMARY_STATS:
            summary_stats = self.generate_summary_stats(results_df)
            
            # Save as JSON
            import json
            with open('outputs/summary_stats.json', 'w') as f:
                json.dump(summary_stats, f, indent=2)
            
            # Save as readable text
            with open('outputs/summary_report.txt', 'w') as f:
                f.write("CLINICAL NER PROCESSING SUMMARY\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Processing completed: {summary_stats['processing_timestamp']}\n")
                f.write(f"Model used: {config.PRIMARY_MODEL}\n")
                f.write(f"Total notes processed: {summary_stats['total_notes_processed']:,}\n")
                f.write(f"Total entities extracted: {summary_stats['total_entities_extracted']:,}\n")
                f.write(f"Unique entity types: {summary_stats['unique_entity_types']}\n")
                f.write(f"Average entities per note: {summary_stats['avg_entities_per_note']:.2f}\n")
                f.write(f"Average confidence: {summary_stats['avg_confidence']:.3f}\n\n")
                
                f.write("ENTITY TYPE DISTRIBUTION:\n")
                f.write("-" * 25 + "\n")
                for entity_type, count in summary_stats['entity_type_distribution'].items():
                    f.write(f"{entity_type}: {count:,}\n")
            
            logger.info("📊 Saved summary statistics to outputs/")
        
        # Print summary to console
        print("\n" + "="*50)
        print("🏥 CLINICAL NER PROCESSING COMPLETE")
        print("="*50)
        print(f"📊 Processed: {results_df['note_id'].nunique():,} notes")
        print(f"🔍 Extracted: {len(results_df):,} medical entities")
        print(f"📋 Entity types: {results_df['entity_type'].nunique()}")
        print(f"💾 Results saved to: {config.OUTPUT_PATH}")
        print("="*50)


def main():
    """Main processing function"""
    logger.info("🏥 Starting Clinical NER Batch Processing")
    logger.info(f"Configuration: {config.SAMPLE_SIZE} notes, {config.PRIMARY_MODEL} model")
    
    # Initialize processor
    processor = ClinicalNERProcessor()
    
    # Process all notes
    results_df = processor.process_all_notes()
    
    # Save results
    processor.save_results(results_df)
    
    logger.info("✅ Processing complete!")


if __name__ == "__main__":
    main()
