	
# Business Problem:
Organizations processes a high volume of lab reports and EHRs daily, but manually extracting critical disease information from these unstructured documents is inefficient and prone to errors. This hampers timely clinical decision-making and workflow integration. An automated NER system for disease extraction would significantly enhance accuracy, speed, and overall operational efficiency.

# Proposed Solution: 
Fine-tune a ClinicalBERT-based NER model on a curated dataset to automatically extract disease entities from lab reports and EHRs. Enhance the process through hyperparameter tuning with Optuna, early stopping to prevent overfitting, and comprehensive experiment tracking with MLflow to ensure optimal performance and integration into existing workflows.

