# Configuration file for Clinical NER Pipeline

# Data Settings
DATA_PATH = "data/mimic_notes.pkl"
OUTPUT_PATH = "outputs/extracted_entities.csv"
SAMPLE_SIZE = 10000  # Number of notes to process (set to -1 for all)

# Model Settings
PRIMARY_MODEL = "scispacy"  # Options: "scispacy", "baseline"
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for entity extraction

# Entity Filtering
MEDICAL_ENTITY_TYPES = [
    "DISEASE",
    "SYMPTOM", 
    "MEDICAL_CONDITION",
    "DISORDER",
    "MEDICATION",
    "DRUG",
    "TREATMENT",
    "PROCEDURE",
    "ANATOMY",
    "ORGAN",
    "BODY_PART"
]

# Processing Settings
BATCH_SIZE = 100  # Process notes in batches for memory efficiency
MAX_TEXT_LENGTH = 5000  # Truncate very long notes
ENABLE_PREPROCESSING = True  # Clean and normalize text

# Output Settings
SAVE_FULL_RESULTS = True  # Save detailed results with positions
SAVE_SUMMARY_STATS = True  # Save summary statistics
VERBOSE = True  # Print progress updates
