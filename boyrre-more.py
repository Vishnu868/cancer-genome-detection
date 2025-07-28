import json
import pandas as pd
import time
import os

class BoyerMoore:
    """
    Boyer-Moore algorithm for fast string pattern matching.
    Used to locate specific cancer mutations in patient genomes.
    """
    def __init__(self, pattern):
        self.pattern = pattern
        self.pattern_length = len(pattern)
        self.bad_char = self._build_bad_char_table()
        
    def _build_bad_char_table(self):
        """Build the bad character table for efficient skipping."""
        table = {}
        for i in range(self.pattern_length - 1):
            table[self.pattern[i]] = self.pattern_length - 1 - i
        return table
    
    def search(self, text):
        """
        Search for pattern in text.
        Returns a list of positions where the pattern is found.
        """
        positions = []
        text_length = len(text)
        
        if self.pattern_length > text_length:
            return positions
        
        i = self.pattern_length - 1
        
        while i < text_length:
            j = self.pattern_length - 1
            k = i
            
            while j >= 0 and text[k] == self.pattern[j]:
                j -= 1
                k -= 1
            
            if j == -1:
                positions.append(k + 1)
                i += 1
            else:
                char_shift = self.bad_char.get(text[k], self.pattern_length)
                i += max(1, char_shift)
        
        return positions

def detect_cancer_mutations_with_boyer_moore(patient_genome, mutations_db):
    start_time = time.time()
    cancer_mutations = []
    
    for mutation_id, mutation_data in mutations_db.items():
        mutation_sequence = mutation_data.get("sequence", "")
        if not mutation_sequence:
            continue
        
        bm = BoyerMoore(mutation_sequence)
        positions = bm.search(patient_genome)
        
        for start_pos in positions:
            end_pos = start_pos + len(mutation_sequence) - 1
            cancer_mutations.append({
                "mutation_id": mutation_id,
                "cancer_type": mutation_data["cancer_type"],
                "position": start_pos,
                "end_position": end_pos,
                "sequence": mutation_sequence
            })
    
    end_time = time.time()
    print(f"Boyer-Moore algorithm completed in {end_time - start_time:.4f} seconds")
    print(f"Found {len(cancer_mutations)} potential cancer mutations")
    
    return cancer_mutations

def determine_somatic_or_germline(cancer_mutations, patient_genome, reference_genome):
    start_time = time.time()
    
    for mutation in cancer_mutations:
        position = mutation["position"]
        sequence = mutation["sequence"]
        
        if position + len(sequence) <= len(reference_genome):
            reference_region = reference_genome[position:position+len(sequence)]
            
            if reference_region == sequence:
                mutation["mutation_type"] = "germline"
            else:
                mutation["mutation_type"] = "somatic"
        else:
            mutation["mutation_type"] = "somatic"
    
    germline_count = sum(1 for m in cancer_mutations if m["mutation_type"] == "germline")
    somatic_count = sum(1 for m in cancer_mutations if m["mutation_type"] == "somatic")
    
    end_time = time.time()
    print(f"Somatic/germline classification completed in {end_time - start_time:.4f} seconds")
    print(f"Results: {germline_count} germline mutations, {somatic_count} somatic mutations")
    
    return cancer_mutations

def analyze_patient_with_boyer_moore(patient_id, patient_genome, reference_genome, mutations_db):
    print(f"\nAnalyzing patient {patient_id} with Boyer-Moore algorithm")
    
    cancer_mutations = detect_cancer_mutations_with_boyer_moore(patient_genome, mutations_db)
    classified_mutations = determine_somatic_or_germline(cancer_mutations, patient_genome, reference_genome)
    
    cancer_types = {}
    for mutation in classified_mutations:
        cancer_type = mutation["cancer_type"]
        if cancer_type not in cancer_types:
            cancer_types[cancer_type] = {
                "count": 0,
                "somatic": 0,
                "germline": 0
            }
        
        cancer_types[cancer_type]["count"] += 1
        
        if mutation["mutation_type"] == "somatic":
            cancer_types[cancer_type]["somatic"] += 1
        else:
            cancer_types[cancer_type]["germline"] += 1
    
    most_likely_cancer = None
    max_somatic_count = -1
    
    for cancer_type, stats in cancer_types.items():
        if stats["somatic"] > max_somatic_count:
            max_somatic_count = stats["somatic"]
            most_likely_cancer = cancer_type
    
    print("\nCancer type distribution:")
    for cancer_type, stats in sorted(cancer_types.items(), key=lambda x: x[1]["count"], reverse=True):
        print(f"  - {cancer_type}: {stats['count']} mutations ({stats['somatic']} somatic, {stats['germline']} germline)")
    
    if most_likely_cancer:
        print(f"\nMost likely cancer type: {most_likely_cancer}")
        print(f"Supporting evidence: {max_somatic_count} somatic mutations")
    else:
        print("\nNo clear cancer type identified")
    
    return {
        "patient_id": patient_id,
        "mutations": classified_mutations,
        "cancer_types": cancer_types,
        "most_likely_cancer": most_likely_cancer
    }

def run_boyer_moore_analysis(mutations_file, reference_file, patient_file, output_dir="boyer_moore_results"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    start_time = time.time()
    print("Starting genome classification using Boyer-Moore algorithm")
    
    # Load datasets
    print("Loading datasets...")
    
    with open(mutations_file, "r") as f:
        mutations_db = json.load(f)
    
    # Read only the first 10,000 rows of the reference genomes CSV
    reference_df = pd.read_csv(reference_file, nrows=10000)
    print(reference_df.columns)  # Print column names to verify
    reference_genomes = reference_df["reference_genome"].tolist()  # Updated to match the column name
    print(f"Loaded {len(reference_genomes)} reference genomes.")
    
    # Load patient genomes from CSV
    patient_df = pd.read_csv(patient_file, nrows=10000)
    print(patient_df.columns)  # Print column names to verify
    patient_genomes = patient_df["patient_genome"].tolist()  # Corrected to use 'patient_genome' column
    patient_ids = patient_df["patient_id"].tolist()
    
    print(f"Loaded {len(patient_genomes)} patient genomes.")
    
    # Process each patient genome and output results
    for patient_id, patient_genome, reference_genome in zip(patient_ids, patient_genomes, reference_genomes):
        result = analyze_patient_with_boyer_moore(patient_id, patient_genome, reference_genome, mutations_db)
        output_file = os.path.join(output_dir, f"{patient_id}_analysis.json")
        with open(output_file, "w") as out_f:
            json.dump(result, out_f, indent=4)
    
    end_time = time.time()
    print(f"Genome classification completed in {end_time - start_time:.4f} seconds")

# Specify your file paths here
mutations_file = "D:/SEM-4/BIO-2/bio_dat_set/cancer_mutations_9999_dataset.json"
reference_file = "D:/SEM-4/BIO-2/bio_dat_set/cancer_genome_9999_samples.csv"
patient_file = "D:/SEM-4/BIO-2/bio_dat_set/synthetic_patient_genomes_9999.csv"

# Run the analysis
run_boyer_moore_analysis(mutations_file, reference_file, patient_file)
