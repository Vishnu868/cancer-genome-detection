import json
import pandas as pd
import time
import os
import multiprocessing as mp
from collections import deque

class AhoCorasick:
    """
    Aho-Corasick algorithm for efficient multiple pattern matching.
    Used to find cancer mutations in patient genomes.
    Modified version with DFS implementation.
    """
    def __init__(self):
        # Use a more stable approach with numbered nodes instead of using object IDs
        self.root = self._create_node()
        self.nodes = [self.root]  # Store all nodes in a list
        self.failure = [0] * 1  # Failure links stored by node index
        self.output = [set() for _ in range(1)]  # Output patterns at each node
        
    def _create_node(self):
        """Create a new node in the trie."""
        return {"transitions": {}, "is_terminal": False, "pattern_ids": set()}
    
    def add_pattern(self, pattern, pattern_id):
        """Add a pattern to the trie."""
        current_node_idx = 0  # Start at root node
        
        for symbol in pattern:
            if symbol not in self.nodes[current_node_idx]["transitions"]:
                # Create new node
                self.nodes.append(self._create_node())
                node_idx = len(self.nodes) - 1
                
                # Add transition
                self.nodes[current_node_idx]["transitions"][symbol] = node_idx
                
                # Extend failure and output arrays
                self.failure.append(0)
                self.output.append(set())
            
            current_node_idx = self.nodes[current_node_idx]["transitions"][symbol]
        
        # Mark current node as terminal
        self.nodes[current_node_idx]["is_terminal"] = True
        self.nodes[current_node_idx]["pattern_ids"].add(pattern_id)
    
    def build_failure_function(self):
        """Build the failure function for the AC algorithm using DFS."""
        # Handle depth 1 nodes first - set their failure to root
        for symbol, node_idx in self.nodes[0]["transitions"].items():
            self.failure[node_idx] = 0  # Failure of depth 1 nodes is the root
        
        # Use DFS to build failure function for all other nodes
        # Start with level 1 nodes (children of root)
        for first_level_idx in self.nodes[0]["transitions"].values():
            self._build_failure_dfs(first_level_idx)
    
    def _build_failure_dfs(self, node_idx):
        """Recursive DFS helper to build failure function."""
        current_node = self.nodes[node_idx]
        
        # Process all transitions from current node
        for symbol, next_idx in current_node["transitions"].items():
            # Find failure state for this node
            failure_idx = self.failure[node_idx]
            
            while failure_idx != 0 and symbol not in self.nodes[failure_idx]["transitions"]:
                failure_idx = self.failure[failure_idx]
            
            if symbol in self.nodes[failure_idx]["transitions"]:
                failure_idx = self.nodes[failure_idx]["transitions"][symbol]
            
            self.failure[next_idx] = failure_idx
            
            # Merge output function
            self.output[next_idx] = self.output[next_idx].union(self.nodes[failure_idx]["pattern_ids"])
            
            # If this is a terminal node, add its pattern IDs to its output function
            if self.nodes[next_idx]["is_terminal"]:
                self.output[next_idx] = self.output[next_idx].union(self.nodes[next_idx]["pattern_ids"])
            
            # Recursively process children (DFS)
            self._build_failure_dfs(next_idx)
    
    def search(self, text):
        """
        Search for all patterns in the text.
        Returns a list of tuples (pattern_id, position).
        """
        matches = []
        current_idx = 0  # Start at root
        
        for i, symbol in enumerate(text):
            # Follow failure links until we find a node that has this symbol
            while current_idx != 0 and symbol not in self.nodes[current_idx]["transitions"]:
                current_idx = self.failure[current_idx]
            
            # If symbol exists in current node's transitions, follow that path
            if symbol in self.nodes[current_idx]["transitions"]:
                current_idx = self.nodes[current_idx]["transitions"][symbol]
            
            # Check output function for matches
            for pattern_id in self.output[current_idx]:
                matches.append((pattern_id, i))
            
            # Also check if current node is terminal
            if self.nodes[current_idx]["is_terminal"]:
                for pattern_id in self.nodes[current_idx]["pattern_ids"]:
                    matches.append((pattern_id, i))
        
        return matches

def detect_cancer_mutations_with_aho_corasick(patient_genome, mutations_db, ac=None):
    """
    First phase: Find cancer mutations in the patient genome
    using Aho-Corasick algorithm
    """
    start_time = time.time()
    
    # If Aho-Corasick automaton is not provided, build it
    if ac is None:
        # Set up Aho-Corasick for efficient pattern matching
        ac = AhoCorasick()
        
        # Add all cancer mutation patterns to the automaton
        for mutation_id, mutation_data in mutations_db.items():
            mutation_sequence = mutation_data.get("sequence", "")
            if mutation_sequence:
                ac.add_pattern(mutation_sequence, mutation_id)
        
        # Build the failure function
        ac.build_failure_function()
    
    # Search for all patterns in the patient genome
    matches = ac.search(patient_genome)
    
    # Process matches to identify cancer mutations
    cancer_mutations = []
    processed_matches = set()  # To avoid duplicates
    
    for mutation_id, end_pos in matches:
        mutation_data = mutations_db[mutation_id]
        mutation_sequence = mutation_data.get("sequence", "")
        
        # Calculate starting position of this mutation in the patient genome
        start_pos = end_pos - len(mutation_sequence) + 1
        
        # Skip if we've already processed this match
        match_key = (mutation_id, start_pos)
        if match_key in processed_matches:
            continue
        
        processed_matches.add(match_key)
        
        cancer_mutations.append({
            "mutation_id": mutation_id,
            "cancer_type": mutation_data["cancer_type"],
            "position": start_pos,
            "end_position": end_pos,
            "sequence": mutation_sequence
        })
    
    end_time = time.time()
    print(f"Aho-Corasick algorithm completed in {end_time - start_time:.4f} seconds")
    print(f"Found {len(cancer_mutations)} potential cancer mutations")
    
    return cancer_mutations

def determine_somatic_or_germline(cancer_mutations, patient_genome, reference_genome):
    """
    Second phase: Determine if each mutation is somatic or germline
    by comparing with the reference genome
    """
    start_time = time.time()
    
    # Add DFS functionality to traverse the mutations
    def classify_mutations_dfs(mutations, start_idx=0):
        if start_idx >= len(mutations):
            return
        
        mutation = mutations[start_idx]
        position = mutation["position"]
        sequence = mutation["sequence"]
        
        if position + len(sequence) <= len(reference_genome):
            # Check if this region in the reference genome matches the mutation sequence
            reference_region = reference_genome[position:position+len(sequence)]
            
            if reference_region == sequence:
                # If reference genome has the same sequence as the mutation, it's germline
                mutation["mutation_type"] = "germline"
            else:
                # If reference genome has a different sequence, it's somatic
                mutation["mutation_type"] = "somatic"
        else:
            # If position is out of bounds for reference genome, classify as somatic
            mutation["mutation_type"] = "somatic"
        
        # Process next mutation (DFS)
        classify_mutations_dfs(mutations, start_idx + 1)
    
    # Start DFS classification
    classify_mutations_dfs(cancer_mutations)
    
    # Count types
    germline_count = sum(1 for m in cancer_mutations if m["mutation_type"] == "germline")
    somatic_count = sum(1 for m in cancer_mutations if m["mutation_type"] == "somatic")
    
    end_time = time.time()
    print(f"Somatic/germline classification completed in {end_time - start_time:.4f} seconds")
    print(f"Results: {germline_count} germline mutations, {somatic_count} somatic mutations")
    
    return cancer_mutations

def analyze_patient_with_aho_corasick(patient_id, patient_genome, reference_genome, mutations_db, ac=None):
    """Analyze a patient's genome using the Aho-Corasick algorithm"""
    print(f"\nAnalyzing patient {patient_id} with Aho-Corasick algorithm")
    
    # Step 1: Detect cancer mutations using Aho-Corasick
    cancer_mutations = detect_cancer_mutations_with_aho_corasick(patient_genome, mutations_db, ac)
    
    # Step 2: Determine if each mutation is somatic or germline
    classified_mutations = determine_somatic_or_germline(cancer_mutations, patient_genome, reference_genome)
    
    # Step 3: Group mutations by cancer type to determine possible cancer types
    cancer_types = {}
    
    # Use DFS to count mutations by type
    def count_by_cancer_type_dfs(mutations, idx=0):
        if idx >= len(mutations):
            return
        
        mutation = mutations[idx]
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
        
        # Process next mutation
        count_by_cancer_type_dfs(mutations, idx + 1)
    
    # Start DFS counting
    count_by_cancer_type_dfs(classified_mutations)
    
    # Find most likely cancer type (highest number of somatic mutations)
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

def build_shared_automaton(mutations_db):
    """
    Build the Aho-Corasick automaton once to be used by all processes.
    This creates a serializable version that works with multiprocessing.
    """
    ac = AhoCorasick()
    
    # Add all cancer mutation patterns to the automaton
    for mutation_id, mutation_data in mutations_db.items():
        mutation_sequence = mutation_data.get("sequence", "")
        if mutation_sequence:
            ac.add_pattern(mutation_sequence, mutation_id)
    
    # Build the failure function
    ac.build_failure_function()
    
    return ac

def process_patient(args):
    """Process a single patient (for multiprocessing)."""
    patient_id, patient_genome, reference_genome, mutations_db, ac, output_dir = args
    
    try:
        result = analyze_patient_with_aho_corasick(patient_id, patient_genome, reference_genome, mutations_db, ac)
        
        # Save individual JSON result
        output_file = os.path.join(output_dir, f"{patient_id}_analysis.json")
        with open(output_file, "w") as out_f:
            json.dump(result, out_f, indent=4)
        
        # Return result for CSV compilation
        return result
    except Exception as e:
        print(f"Error processing patient {patient_id}: {str(e)}")
        # Return a minimal result with the error
        return {
            "patient_id": patient_id,
            "error": str(e),
            "mutations": [],
            "cancer_types": {},
            "most_likely_cancer": None
        }

def run_aho_corasick_analysis(mutations_file, reference_file, patient_file, output_dir="aho_corasick_results", batch_size=100, num_processes=None):
    """Run the full Aho-Corasick analysis pipeline with multiprocessing"""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    start_time = time.time()
    print("Starting genome classification using Aho-Corasick algorithm")
    
    # Load datasets
    print("Loading datasets...")
    
    # Load mutations database
    with open(mutations_file, "r") as f:
        mutations_db = json.load(f)
    
    # Determine optimal number of processes if not specified
    if num_processes is None:
        num_processes = mp.cpu_count() - 1 or 1  # Leave one CPU for system tasks
    
    print(f"Using {num_processes} parallel processes")
    
    # Build Aho-Corasick automaton once (shared across all processes)
    print("Building Aho-Corasick automaton...")
    ac_build_start = time.time()
    ac = build_shared_automaton(mutations_db)
    ac_build_time = time.time() - ac_build_start
    print(f"Aho-Corasick automaton built in {ac_build_time:.2f} seconds")
    
    # Process the data in batches to avoid memory issues
    reference_df = pd.read_csv(reference_file)
    patient_df = pd.read_csv(patient_file)
    
    print(reference_df.columns)  # Print column names to verify
    print(patient_df.columns)    # Print column names to verify
    
    total_patients = len(patient_df)
    print(f"Processing {total_patients} patients in batches of {batch_size}")
    
    processed_count = 0
    all_results = []  # Store all results for CSV generation
    
    # Process batches recursively using DFS
    def process_batches_dfs(batch_start):
        nonlocal processed_count, all_results
        
        if batch_start >= total_patients:
            return
        
        batch_end = min(batch_start + batch_size, total_patients)
        print(f"\nProcessing batch {batch_start//batch_size + 1}: patients {batch_start+1} to {batch_end}")
        
        # Get batch data
        batch_patient_df = patient_df.iloc[batch_start:batch_end]
        batch_reference_df = reference_df.iloc[batch_start:batch_end]
        
        # Prepare arguments for parallel processing
        args_list = [
            (patient_id, patient_genome, reference_genome, mutations_db, ac, output_dir)
            for patient_id, patient_genome, reference_genome 
            in zip(
                batch_patient_df["patient_id"].tolist(),
                batch_patient_df["patient_genome"].tolist(),
                batch_reference_df["reference_genome"].tolist()
            )
        ]
        
        # Use multiprocessing to process patients in parallel
        with mp.Pool(processes=num_processes) as pool:
            batch_results = pool.map(process_patient, args_list)
            all_results.extend(batch_results)
        
        processed_count += len(batch_results)
        print(f"Completed {processed_count}/{total_patients} patients ({processed_count/total_patients*100:.1f}%)")
        
        # Process next batch (DFS)
        process_batches_dfs(batch_end)
    
    # Start batch processing using DFS
    process_batches_dfs(0)
    
    # Create CSV with summarized results
    csv_data = []
    
    # Process results using DFS
    def process_results_dfs(results, idx=0):
        if idx >= len(results):
            return
        
        result = results[idx]
        patient_id = result["patient_id"]
        most_likely_cancer = result.get("most_likely_cancer", "Unknown")
        
        # Check if there was an error processing this patient
        if "error" in result:
            csv_data.append({
                "patient_id": patient_id,
                "most_likely_cancer": "Error",
                "total_mutations": 0,
                "somatic_mutations": 0,
                "germline_mutations": 0,
                "cancer_types_detail": f"Error: {result['error']}"
            })
        else:
            # Get mutation counts
            total_mutations = 0
            somatic_count = 0
            germline_count = 0
            cancer_types_data = []
            
            for cancer_type, stats in result.get("cancer_types", {}).items():
                total_mutations += stats["count"]
                somatic_count += stats["somatic"]
                germline_count += stats["germline"]
                cancer_types_data.append(f"{cancer_type}:{stats['count']}:{stats['somatic']}:{stats['germline']}")
            
            # Add row to CSV data
            csv_data.append({
                "patient_id": patient_id,
                "most_likely_cancer": most_likely_cancer,
                "total_mutations": total_mutations,
                "somatic_mutations": somatic_count,
                "germline_mutations": germline_count,
                "cancer_types_detail": "|".join(cancer_types_data)
            })
        
        # Process next result
        process_results_dfs(results, idx + 1)
    
    # Start DFS for processing results
    process_results_dfs(all_results)
    
    # Save to CSV
    csv_file = os.path.join(output_dir, "cancer_analysis_results.csv")
    pd.DataFrame(csv_data).to_csv(csv_file, index=False)
    
    # Create detailed CSV with mutation information
    detailed_csv_data = []
    
    # Process detailed mutation data using DFS
    def process_detailed_dfs(results, result_idx=0, mutation_idx=0):
        if result_idx >= len(results):
            return
        
        result = results[result_idx]
        patient_id = result["patient_id"]
        
        # Skip patients with errors
        if "error" in result:
            process_detailed_dfs(results, result_idx + 1, 0)
            return
            
        mutations = result["mutations"]
        
        if mutation_idx >= len(mutations):
            # Move to next patient
            process_detailed_dfs(results, result_idx + 1, 0)
            return
        
        # Process current mutation
        mutation = mutations[mutation_idx]
        detailed_csv_data.append({
            "patient_id": patient_id,
            "mutation_id": mutation["mutation_id"],
            "cancer_type": mutation["cancer_type"],
            "position": mutation["position"],
            "end_position": mutation["end_position"],
            "sequence": mutation["sequence"],
            "mutation_type": mutation["mutation_type"]
        })
        
        # Process next mutation
        process_detailed_dfs(results, result_idx, mutation_idx + 1)
    
    # Start DFS for detailed mutation processing
    process_detailed_dfs(all_results)
    
    # Save detailed data to CSV
    detailed_csv_file = os.path.join(output_dir, "detailed_mutations.csv")
    pd.DataFrame(detailed_csv_data).to_csv(detailed_csv_file, index=False)
    
    # Print summary to terminal
    print("\n--- ANALYSIS SUMMARY ---")
    summary_df = pd.DataFrame(csv_data)
    print(f"Total patients analyzed: {len(summary_df)}")
    
    # Count successful analyses
    successful = summary_df[summary_df["most_likely_cancer"] != "Error"]
    print(f"Successful analyses: {len(successful)}/{len(summary_df)} ({len(successful)/len(summary_df)*100:.1f}%)")
    
    if len(successful) > 0:
        # Count occurrences of each cancer type as most likely
        cancer_counts = successful["most_likely_cancer"].value_counts()
        print("\nDistribution of most likely cancer types:")
        for cancer_type, count in cancer_counts.items():
            print(f"  - {cancer_type}: {count} patients ({count/len(successful)*100:.1f}%)")
        
        # Print mutation statistics
        print("\nMutation statistics:")
        print(f"  - Total mutations found: {summary_df['total_mutations'].sum()}")
        print(f"  - Somatic mutations: {summary_df['somatic_mutations'].sum()}")
        print(f"  - Germline mutations: {summary_df['germline_mutations'].sum()}")
        print(f"  - Average mutations per patient: {summary_df['total_mutations'].mean():.1f}")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nGenome classification completed in {total_time:.2f} seconds")
    print(f"Average time per patient: {total_time/total_patients:.4f} seconds")
    print(f"Results saved to:")
    print(f"  - CSV summary: {csv_file}")
    print(f"  - Detailed mutations CSV: {detailed_csv_file}")
    print(f"  - Individual JSON files: {output_dir}/*.json")
    
    return {
        "results": all_results,
        "output_files": {
            "summary_file": os.path.abspath(csv_file),
            "detailed_mutations_file": os.path.abspath(detailed_csv_file)
        }
    }

# Main function
def main():
    # File paths
    mutations_file = "D:/SEM-4/BIO-2/bio_dat_set/cancer_mutations_9999_dataset.json"
    reference_file = "D:/SEM-4/BIO-2/bio_dat_set/cancer_genome_9999_samples.csv"
    patient_file = "D:/SEM-4/BIO-2/bio_dat_set/synthetic_patient_genomes_9999.csv"
    
    # Output directory for results
    output_dir = "aho_corasick_results"
    
    # Run the analysis with multiprocessing
    # Using 7 cores as requested
    result = run_aho_corasick_analysis(
        mutations_file,
        reference_file,
        patient_file,
        output_dir,
        batch_size=500,     # Process 500 patients at a time
        num_processes=7     # Use 7 CPU cores
    )
    
    return result

# Run the main function
if __name__ == "__main__":
    analysis_results = main()
    
    
    
 