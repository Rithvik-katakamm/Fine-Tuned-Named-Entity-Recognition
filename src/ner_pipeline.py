"""
Clinical Named Entity Recognition Pipeline
Author: Rithvik Katakam
Description: Advanced NER pipeline for extracting medical entities from clinical notes
"""

import pandas as pd
import pickle
import spacy
import streamlit as st
from typing import List, Dict, Tuple, Any
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re


class ClinicalNERPipeline:
    """
    Advanced Clinical NER Pipeline using multiple pre-trained models
    """
    
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load multiple clinical NER models for comparison"""
        try:
            # Load SciSpaCy clinical model
            self.models['scispacy'] = spacy.load("en_core_sci_lg")
            print("✅ Loaded SciSpaCy large clinical model")
        except OSError:
            print("❌ SciSpaCy model not found. Install with:")
            print("pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_lg-0.5.0.tar.gz")
        
        try:
            # Load standard spaCy model as baseline
            self.models['baseline'] = spacy.load("en_core_web_sm")
            print("✅ Loaded baseline spaCy model")
        except OSError:
            print("❌ Standard spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess clinical text"""
        if not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace and newlines
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Handle common clinical abbreviations (basic cleaning)
        text = text.replace('pt.', 'patient')
        text = text.replace('hx', 'history')
        text = text.replace('dx', 'diagnosis')
        
        return text
    
    def extract_entities(self, text: str, model_name: str = 'scispacy') -> List[Dict]:
        """Extract named entities using specified model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        model = self.models[model_name]
        doc = model(self.preprocess_text(text))
        
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': getattr(ent, 'score', 0.9)  # Default confidence
            })
        
        return entities
    
    def compare_models(self, text: str) -> Dict[str, List[Dict]]:
        """Compare entity extraction across different models"""
        results = {}
        for model_name in self.models.keys():
            results[model_name] = self.extract_entities(text, model_name)
        return results
    
    def filter_medical_entities(self, entities: List[Dict]) -> List[Dict]:
        """Filter for medically relevant entities"""
        medical_labels = {
            'DISEASE', 'SYMPTOM', 'MEDICAL_CONDITION', 'DISORDER',
            'MEDICATION', 'DRUG', 'TREATMENT', 'PROCEDURE',
            'ANATOMY', 'ORGAN', 'BODY_PART'
        }
        
        filtered = []
        for entity in entities:
            # Check if label contains medical terms or is in medical_labels
            if (entity['label'] in medical_labels or 
                any(med_term in entity['label'].upper() for med_term in medical_labels)):
                filtered.append(entity)
        
        return filtered
    
    def analyze_note_batch(self, notes: List[str], sample_size: int = 100) -> Dict[str, Any]:
        """Analyze a batch of clinical notes"""
        if len(notes) > sample_size:
            notes = notes[:sample_size]
        
        all_entities = []
        entity_counts = Counter()
        
        for i, note in enumerate(notes):
            if not isinstance(note, str) or len(note.strip()) == 0:
                continue
                
            entities = self.extract_entities(note)
            medical_entities = self.filter_medical_entities(entities)
            
            for entity in medical_entities:
                all_entities.append(entity)
                entity_counts[entity['label']] += 1
        
        return {
            'total_notes': len(notes),
            'total_entities': len(all_entities),
            'unique_labels': len(entity_counts),
            'label_distribution': dict(entity_counts),
            'sample_entities': all_entities[:50]  # Top 50 for display
        }


def load_mimic_data(file_path: str) -> pd.DataFrame:
    """Load MIMIC-III clinical notes from pickle file"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, pd.DataFrame):
            return data
        else:
            st.error(f"Expected DataFrame, got {type(data)}")
            return pd.DataFrame()
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


def create_entity_visualization(entities: List[Dict]) -> go.Figure:
    """Create interactive visualization of entity distribution"""
    if not entities:
        return go.Figure()
    
    # Count entities by label
    label_counts = Counter([ent['label'] for ent in entities])
    
    fig = px.bar(
        x=list(label_counts.keys()),
        y=list(label_counts.values()),
        title="Medical Entity Distribution",
        labels={'x': 'Entity Type', 'y': 'Count'}
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=400,
        showlegend=False
    )
    
    return fig


def highlight_entities_in_text(text: str, entities: List[Dict]) -> str:
    """Create HTML with highlighted entities"""
    if not entities:
        return text
    
    # Sort entities by start position (reverse to avoid offset issues)
    sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
    
    highlighted_text = text
    colors = {
        'DISEASE': '#ff9999',
        'SYMPTOM': '#99ccff', 
        'MEDICATION': '#99ff99',
        'PROCEDURE': '#ffcc99',
        'ANATOMY': '#ff99cc'
    }
    
    for entity in sorted_entities:
        start, end = entity['start'], entity['end']
        entity_text = entity['text']
        label = entity['label']
        
        # Get color for entity type
        color = colors.get(label, '#ffffcc')
        
        # Create highlighted span
        highlight = f'<span style="background-color: {color}; padding: 2px 4px; margin: 2px; border-radius: 3px;" title="{label}">{entity_text}</span>'
        
        # Replace in text
        highlighted_text = highlighted_text[:start] + highlight + highlighted_text[end:]
    
    return highlighted_text


if __name__ == "__main__":
    # Test the pipeline
    pipeline = ClinicalNERPipeline()
    
    sample_text = """
    Patient presents with chest pain and shortness of breath. 
    History of hypertension and diabetes mellitus. 
    Prescribed metformin and lisinopril.
    """
    
    entities = pipeline.extract_entities(sample_text)
    print("Extracted entities:")
    for entity in entities:
        print(f"- {entity['text']} ({entity['label']})")
