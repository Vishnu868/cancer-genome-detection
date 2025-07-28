import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import json
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
from matplotlib.patches import Patch
import time
import argparse
from collections import defaultdict, Counter

# Function to create output directory
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_raw_datasets(mutations_file, reference_file, patient_file, sample_size=None):
    """
    Load raw genomic datasets directly from source files
    
    Parameters:
    mutations_file: Path to the cancer mutations dataset JSON file
    reference_file: Path to the reference genomes CSV file
    patient_file: Path to the patient genomes CSV file
    sample_size: If provided, limit processing to this many samples
    
    Returns:
    Dictionary containing the loaded datasets
    """
    print(f"Loading raw genomic datasets...")
    
    datasets = {}
    
    # Load mutations database
    if os.path.exists(mutations_file):
        with open(mutations_file, "r") as f:
            datasets['mutations_db'] = json.load(f)
        print(f"Loaded {len(datasets['mutations_db'])} cancer mutations")
    else:
        print(f"Error: Mutations file not found: {mutations_file}")
        return None
    
    # Load reference genomes
    if os.path.exists(reference_file):
        reference_df = pd.read_csv(reference_file)
        if sample_size:
            reference_df = reference_df.head(sample_size)
        datasets['reference_df'] = reference_df
        print(f"Loaded {len(reference_df)} reference genomes")
    else:
        print(f"Error: Reference genome file not found: {reference_file}")
        return None
    
    # Load patient genomes
    if os.path.exists(patient_file):
        patient_df = pd.read_csv(patient_file)
        if sample_size:
            patient_df = patient_df.head(sample_size)
        datasets['patient_df'] = patient_df
        print(f"Loaded {len(patient_df)} patient genomes")
    else:
        print(f"Error: Patient genome file not found: {patient_file}")
        return None
    
    return datasets

def analyze_mutation_patterns(datasets, output_dir="visualizations"):
    """
    Analyze mutation patterns directly from the raw data
    """
    ensure_dir(output_dir)
    
    mutations_db = datasets['mutations_db']
    
    # Extract mutation sequences and cancer types
    mutations = []
    for mutation_id, data in mutations_db.items():
        sequence = data.get('sequence', '')
        cancer_type = data.get('cancer_type', 'Unknown')
        
        if sequence:
            mutations.append({
                'mutation_id': mutation_id,
                'sequence': sequence,
                'cancer_type': cancer_type,
                'length': len(sequence)
            })
    
    # Convert to DataFrame for easier analysis
    mutations_df = pd.DataFrame(mutations)
    
    # Create figure for mutation pattern analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. Mutation Sequence Length Distribution (top-left)
    ax = axes[0, 0]
    
    # Plot histogram of sequence lengths
    bins = np.arange(min(mutations_df['length']), max(mutations_df['length']) + 2)
    ax.hist(mutations_df['length'], bins=bins, color='#5aa469', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Sequence Length (base pairs)')
    ax.set_ylabel('Number of Mutations')
    ax.set_title('Mutation Sequence Length Distribution', fontsize=14)
    
    # Add statistics
    stats_text = (
        f"Mean: {mutations_df['length'].mean():.1f}\n"
        f"Median: {mutations_df['length'].median():.1f}\n"
        f"Min: {mutations_df['length'].min()}\n"
        f"Max: {mutations_df['length'].max()}"
    )
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
            ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Cancer Type Distribution (top-right)
    ax = axes[0, 1]
    
    # Count mutations by cancer type
    cancer_counts = mutations_df['cancer_type'].value_counts()
    cancer_types = cancer_counts.index.tolist()
    counts = cancer_counts.values.tolist()
    
    # Take top 10 cancer types for better visualization
    if len(cancer_types) > 10:
        other_count = sum(counts[9:])
        cancer_types = cancer_types[:9] + ['Other']
        counts = counts[:9] + [other_count]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(cancer_types))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(cancer_types)))
    
    bars = ax.barh(y_pos, counts, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cancer_types)
    ax.set_xlabel('Number of Mutations')
    ax.set_title('Mutations by Cancer Type', fontsize=14)
    
    # Add count labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        percentage = 100 * width / sum(counts)
        label = f"{width} ({percentage:.1f}%)"
        ax.text(width + 5, bar.get_y() + bar.get_height()/2, label, ha='left', va='center')
    
    # 3. Nucleotide Composition (bottom-left)
    ax = axes[1, 0]
    
    # Analyze nucleotide composition in mutation sequences
    all_sequences = ''.join(mutations_df['sequence'])
    nucleotide_counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0, 'Other': 0}
    
    for base in all_sequences:
        if base in nucleotide_counts:
            nucleotide_counts[base] += 1
        else:
            nucleotide_counts['Other'] += 1
    
    # Create pie chart
    labels = nucleotide_counts.keys()
    sizes = nucleotide_counts.values()
    
    # Use a standard DNA color scheme
    colors = {'A': '#3498db', 'C': '#e74c3c', 'G': '#2ecc71', 'T': '#f39c12', 'Other': '#95a5a6'}
    colors_list = [colors[base] for base in labels]
    
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors_list,
        wedgeprops={'edgecolor': 'w', 'linewidth': 1}
    )
    
    # Add total count in the center
    ax.text(0, 0, f"Total\n{sum(sizes)}", ha='center', va='center', fontsize=12, fontweight='bold')
    
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    
    ax.set_title('Nucleotide Composition of Mutations', fontsize=14)
    
    # 4. Sequence Pattern Frequency (bottom-right)
    ax = axes[1, 1]
    
    # Extract all 3-base patterns (codons) from sequences
    codons = []
    for sequence in mutations_df['sequence']:
        for i in range(0, len(sequence) - 2):
            codon = sequence[i:i+3]
            if len(codon) == 3 and all(base in 'ACGT' for base in codon):
                codons.append(codon)
    
    # Count codon frequencies
    codon_counts = Counter(codons)
    
    # Get top 10 most common codons
    top_codons = codon_counts.most_common(10)
    codon_labels = [codon for codon, _ in top_codons]
    codon_freqs = [count for _, count in top_codons]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(codon_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(codon_labels)))
    
    ax.barh(y_pos, codon_freqs, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(codon_labels)
    ax.set_xlabel('Frequency')
    ax.set_title('Most Common 3-base Patterns in Mutations', fontsize=14)
    
    plt.tight_layout()
    plt.suptitle('Mutation Pattern Analysis from Raw Data', fontsize=16, y=1.02)
    
    # Save the figure
    mutation_file = os.path.join(output_dir, 'mutation_pattern_analysis.png')
    plt.savefig(mutation_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return mutation_file

def analyze_patient_genomes(datasets, output_dir="visualizations"):
    """
    Analyze patient genomes directly from the raw data
    """
    ensure_dir(output_dir)
    
    patient_df = datasets['patient_df']
    reference_df = datasets['reference_df']
    
    # Analysis results
    results = {
        'total_patients': len(patient_df),
        'genome_lengths': [],
        'differences': []
    }
    
    # Sample a subset for detailed analysis (for performance)
    sample_size = min(100, len(patient_df))
    sample_indices = np.random.choice(len(patient_df), sample_size, replace=False)
    
    # Extract genome lengths and calculate differences between patient and reference
    for idx in sample_indices:
        patient_genome = patient_df.iloc[idx]['patient_genome']
        reference_genome = reference_df.iloc[idx]['reference_genome']
        
        # Record genome length
        results['genome_lengths'].append(len(patient_genome))
        
        # Calculate differences between patient and reference genomes
        # (This is a simple approach - in practice you'd use more sophisticated alignment)
        diff_count = sum(1 for p, r in zip(patient_genome, reference_genome) if p != r)
        results['differences'].append(diff_count)
    
    # Create figure for patient genome analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. Genome Length Distribution (top-left)
    ax = axes[0, 0]
    
    # Plot histogram of genome lengths
    ax.hist(results['genome_lengths'], bins=20, color='#3498db', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Genome Length (base pairs)')
    ax.set_ylabel('Number of Patients')
    ax.set_title('Patient Genome Length Distribution', fontsize=14)
    
    # Add statistics
    stats_text = (
        f"Mean: {np.mean(results['genome_lengths']):.1f}\n"
        f"Median: {np.median(results['genome_lengths']):.1f}\n"
        f"Min: {np.min(results['genome_lengths'])}\n"
        f"Max: {np.max(results['genome_lengths'])}"
    )
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
            ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Differences between Patient and Reference (top-right)
    ax = axes[0, 1]
    
    # Plot histogram of differences
    ax.hist(results['differences'], bins=20, color='#e74c3c', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Differences')
    ax.set_ylabel('Number of Patients')
    ax.set_title('Differences between Patient and Reference Genomes', fontsize=14)
    
    # Add statistics
    stats_text = (
        f"Mean: {np.mean(results['differences']):.1f}\n"
        f"Median: {np.median(results['differences']):.1f}\n"
        f"Min: {np.min(results['differences'])}\n"
        f"Max: {np.max(results['differences'])}"
    )
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
            ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Scatter plot of Genome Length vs Differences (bottom-left)
    ax = axes[1, 0]
    
    ax.scatter(results['genome_lengths'], results['differences'], alpha=0.7, c='#9b59b6')
    ax.set_xlabel('Genome Length (base pairs)')
    ax.set_ylabel('Number of Differences')
    ax.set_title('Genome Length vs Differences', fontsize=14)
    
    # Calculate correlation
    correlation = np.corrcoef(results['genome_lengths'], results['differences'])[0, 1]
    ax.text(0.05, 0.95, f"Correlation: {correlation:.3f}", transform=ax.transAxes, 
            ha='left', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Nucleotide Composition of Patient Genomes (bottom-right)
    ax = axes[1, 1]
    
    # Analyze nucleotide composition in a sample of patient genomes
    nucleotide_counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0, 'Other': 0}
    
    # Sample a few patients for nucleotide analysis
    for idx in sample_indices[:10]:  # Just use 10 patients for this analysis
        patient_genome = patient_df.iloc[idx]['patient_genome']
        
        for base in patient_genome:
            if base in nucleotide_counts:
                nucleotide_counts[base] += 1
            else:
                nucleotide_counts['Other'] += 1
    
    # Create pie chart
    labels = nucleotide_counts.keys()
    sizes = nucleotide_counts.values()
    
    # Use a standard DNA color scheme
    colors = {'A': '#3498db', 'C': '#e74c3c', 'G': '#2ecc71', 'T': '#f39c12', 'Other': '#95a5a6'}
    colors_list = [colors[base] for base in labels]
    
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors_list,
        wedgeprops={'edgecolor': 'w', 'linewidth': 1}
    )
    
    # Add total count in the center
    ax.text(0, 0, f"Sample\n{sum(sizes)} bases", ha='center', va='center', fontsize=12, fontweight='bold')
    
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    
    ax.set_title('Nucleotide Composition of Patient Genomes', fontsize=14)
    
    plt.tight_layout()
    plt.suptitle('Patient Genome Analysis from Raw Data', fontsize=16, y=1.02)
    
    # Save the figure
    patient_file = os.path.join(output_dir, 'patient_genome_analysis.png')
    plt.savefig(patient_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return patient_file

def simulate_algorithm_comparison(datasets, output_dir="visualizations"):
    """
    Simulate algorithm performance by running small-scale tests directly on the data
    """
    ensure_dir(output_dir)
    
    # Get a small sample for testing algorithm performance
    mutations_db = datasets['mutations_db']
    patient_df = datasets['patient_df'].head(3)  # Just use 3 patients for quick testing
    
    # Helper function for Boyer-Moore search
    def boyer_moore_search(text, pattern):
        """Simple implementation of Boyer-Moore search algorithm"""
        m = len(pattern)
        n = len(text)
        
        if m > n:
            return []
        
        # Preprocessing for bad character heuristic
        bad_char = {}
        for i in range(m - 1):
            bad_char[pattern[i]] = m - 1 - i
        
        # Search
        positions = []
        i = m - 1
        
        while i < n:
            j = m - 1
            k = i
            
            while j >= 0 and text[k] == pattern[j]:
                j -= 1
                k -= 1
            
            if j == -1:
                positions.append(k + 1)
                i += 1
            else:
                char_shift = bad_char.get(text[k], m)
                i += max(1, char_shift)
        
        return positions
    
    # Helper function for Aho-Corasick search (simplified)
    def aho_corasick_search(text, patterns):
        """Simplified implementation of Aho-Corasick search algorithm"""
        positions = []
        
        # Naive implementation for demonstration
        for pattern_id, pattern in patterns.items():
            for i in range(len(text) - len(pattern) + 1):
                if text[i:i+len(pattern)] == pattern:
                    positions.append((pattern_id, i + len(pattern) - 1))
        
        return positions
    
    # Helper function for DFS Trie search
    def dfs_trie_search(text, patterns):
        """Simplified implementation of DFS Trie search algorithm"""
        positions = []
        
        # Naive implementation for demonstration
        for pattern_id, pattern in patterns.items():
            for i in range(len(text) - len(pattern) + 1):
                if text[i:i+len(pattern)] == pattern:
                    positions.append((pattern_id, i + len(pattern) - 1))
        
        return positions
    
    # Prepare patterns for testing
    # Just use a small subset of mutations for quick demonstration
    test_mutations = {}
    for mutation_id, data in list(mutations_db.items())[:20]:
        sequence = data.get('sequence', '')
        if sequence:
            test_mutations[mutation_id] = sequence
    
    # Test algorithm performance
    results = {
        'boyer_moore': {'time': [], 'matches': []},
        'aho_corasick': {'time': [], 'matches': []},
        'dfs_trie': {'time': [], 'matches': []}
    }
    
    for _, row in patient_df.iterrows():
        patient_genome = row['patient_genome']
        patient_id = row['patient_id']
        
        print(f"Testing algorithms on {patient_id}...")
        
        # Test Boyer-Moore
        start_time = time.time()
        bm_matches = []
        for mutation_id, sequence in test_mutations.items():
            positions = boyer_moore_search(patient_genome, sequence)
            for pos in positions:
                bm_matches.append((mutation_id, pos))
        boyer_moore_time = time.time() - start_time
        
        # Test Aho-Corasick
        start_time = time.time()
        ac_matches = aho_corasick_search(patient_genome, test_mutations)
        aho_corasick_time = time.time() - start_time
        
        # Test DFS Trie
        start_time = time.time()
        dfs_matches = dfs_trie_search(patient_genome, test_mutations)
        dfs_trie_time = time.time() - start_time
        
        # Save results
        results['boyer_moore']['time'].append(boyer_moore_time)
        results['boyer_moore']['matches'].append(len(bm_matches))
        
        results['aho_corasick']['time'].append(aho_corasick_time)
        results['aho_corasick']['matches'].append(len(ac_matches))
        
        results['dfs_trie']['time'].append(dfs_trie_time)
        results['dfs_trie']['matches'].append(len(dfs_matches))
        
        print(f"  Boyer-Moore: {len(bm_matches)} matches in {boyer_moore_time:.4f}s")
        print(f"  Aho-Corasick: {len(ac_matches)} matches in {aho_corasick_time:.4f}s")
        print(f"  DFS Trie: {len(dfs_matches)} matches in {dfs_trie_time:.4f}s")
    
    # Create figure for algorithm comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Execution Time Comparison (top-left)
    ax = axes[0, 0]
    
    algorithms = ['Boyer-Moore', 'Aho-Corasick', 'DFS-Based']
    avg_times = [
        np.mean(results['boyer_moore']['time']),
        np.mean(results['aho_corasick']['time']),
        np.mean(results['dfs_trie']['time'])
    ]
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bars = ax.bar(algorithms, avg_times, color=colors)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.4f}s', ha='center', va='bottom')
    
    ax.set_title('Average Execution Time (seconds)', fontsize=14)
    ax.set_ylabel('Time (lower is better)', fontsize=12)
    
    # 2. Matches Found Comparison (top-right)
    ax = axes[0, 1]
    
    avg_matches = [
        np.mean(results['boyer_moore']['matches']),
        np.mean(results['aho_corasick']['matches']),
        np.mean(results['dfs_trie']['matches'])
    ]
    
    bars = ax.bar(algorithms, avg_matches, color=colors)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom')
    
    ax.set_title('Average Matches Found', fontsize=14)
    ax.set_ylabel('Number of Matches', fontsize=12)
    
    # 3. Time per Match (efficiency metric) (bottom-left)
    ax = axes[1, 0]
    
    time_per_match = []
    for algo in ['boyer_moore', 'aho_corasick', 'dfs_trie']:
        times = results[algo]['time']
        matches = results[algo]['matches']
        
        # Calculate time per match (avoiding division by zero)
        tpm = [t / max(1, m) for t, m in zip(times, matches)]
        time_per_match.append(np.mean(tpm))
    
    bars = ax.bar(algorithms, time_per_match, color=colors)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.00001,
                f'{height:.6f}s', ha='center', va='bottom')
    
    ax.set_title('Time per Match (Efficiency)', fontsize=14)
    ax.set_ylabel('Seconds per Match (lower is better)', fontsize=12)
    
    # 4. Algorithm Selection Guide (bottom-right)
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create a simple guide for algorithm selection
    guide_text = """
    Algorithm Selection Guide:
    
    1. Boyer-Moore:
       • Best for: Single pattern matching
       • Advantage: Very efficient for long patterns
       • Limitation: Less efficient for multiple patterns
    
    2. Aho-Corasick:
       • Best for: Multiple pattern matching
       • Advantage: Linear time regardless of pattern count
       • Limitation: Higher memory usage
    
    3. DFS-Based Trie:
       • Best for: Prefix-based pattern matching
       • Advantage: Intuitive implementation
       • Limitation: Less optimized than specialized algorithms
    """
    
    ax.text(0.5, 0.5, guide_text, ha='center', va='center', fontsize=12,
           bbox=dict(facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout()
    plt.suptitle('Algorithm Performance Comparison (Direct Testing)', fontsize=16, y=1.02)
    
    # Save the figure
    comparison_file = os.path.join(output_dir, 'direct_algorithm_comparison.png')
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return comparison_file

def visualize_boyer_moore(output_dir="visualizations"):
    """Visualize Boyer-Moore algorithm pattern matching behavior"""
    ensure_dir(output_dir)
    
    text = "ACTGTATGCATGCAGTCATGCACTAGCATCGAT"
    pattern = "CATGCA"
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title("Boyer-Moore Algorithm Pattern Matching", fontsize=16)
    ax.axis('off')
    
    # Draw the text string
    ax.text(0.05, 0.9, "Text:   " + text, fontsize=14, family='monospace')
    
    # Show different alignment attempts
    alignments = [
        (0, False, "Initial alignment"),
        (4, False, "Skip after mismatch (bad character rule)"),
        (6, True, "Match found!"),
        (12, False, "Continue searching"),
        (18, False, "Skip (good suffix rule)"),
        (24, False, "Final check")
    ]
    
    y_pos = 0.8
    y_step = 0.1
    
    for offset, is_match, description in alignments:
        y_pos -= y_step
        
        # Format pattern with spaces for alignment
        pattern_display = " " * offset + pattern
        
        # Draw pattern
        if is_match:
            ax.text(0.05, y_pos, "Match:  " + pattern_display, fontsize=14, 
                   family='monospace', color='green', fontweight='bold')
        else:
            ax.text(0.05, y_pos, "Try:    " + pattern_display, fontsize=14, 
                   family='monospace', color='red')
        
        # Add description
        ax.text(0.7, y_pos, description, fontsize=12, style='italic')
    
    # Add explanation
    explanation = """
    Boyer-Moore Algorithm:
    1. Starts comparison from the end of the pattern
    2. Uses two rules to skip characters:
       - Bad Character Rule: Skip alignments where a mismatch occurs
       - Good Suffix Rule: Skip based on previous successful matches
    3. Achieves sub-linear time complexity in practice
    """
    
    ax.text(0.05, 0.25, explanation, fontsize=12, bbox=dict(facecolor='lightyellow', alpha=0.5))
    
    # Save the figure
    bm_file = os.path.join(output_dir, 'boyer_moore_visualization.png')
    plt.savefig(bm_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return bm_file

def visualize_aho_corasick(output_dir="visualizations"):
    """Visualize Aho-Corasick automaton structure"""
    ensure_dir(output_dir)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Aho-Corasick Automaton", fontsize=16)
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    nodes = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
    terminal_nodes = ['5', '7', '8']  # Nodes marking end of patterns
    
    # Add edges (goto function)
    goto_edges = [
        ('0', '1', 'A'), ('0', '2', 'C'), ('0', '3', 'T'),
        ('1', '4', 'C'), ('2', '5', 'A'), ('2', '6', 'G'),
        ('3', '7', 'A'), ('6', '8', 'A')
    ]
    
    # Add failure function edges
    failure_edges = [
        ('4', '2'),  # AC -> C
        ('5', '1'),  # CA -> A
        ('6', '0'),  # CG -> root
        ('7', '1'),  # TA -> A
        ('8', '1')   # CGA -> A
    ]
    
    # Add nodes to graph
    G.add_nodes_from(nodes)
    
    # Add goto edges
    goto_edge_list = [(src, dst) for src, dst, _ in goto_edges]
    G.add_edges_from(goto_edge_list, edge_type='goto')
    
    # Add failure edges
    G.add_edges_from(failure_edges, edge_type='failure')
    
    # Set up positions using spring layout with seed
    pos = nx.spring_layout(G, seed=42)
    
    # Draw different node types
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n in nodes if n not in terminal_nodes], 
                          node_color='skyblue', node_size=700)
    nx.draw_networkx_nodes(G, pos, nodelist=terminal_nodes, 
                          node_color='lightcoral', node_size=700)
    
    # Draw goto edges
    goto_edges_to_draw = [(src, dst) for src, dst, _ in goto_edges]
    nx.draw_networkx_edges(G, pos, edgelist=goto_edges_to_draw, 
                          arrows=True, width=2, edge_color='black')
    
    # Draw failure edges
    nx.draw_networkx_edges(G, pos, edgelist=failure_edges, 
                         arrows=True, width=1.5, edge_color='red', 
                         style='dashed', connectionstyle='arc3,rad=0.2')
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    # Draw edge labels for goto transitions
    edge_labels = {(src, dst): lbl for src, dst, lbl in goto_edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=11)
    
    # Add legend
    legend_elements = [
        Patch(facecolor='skyblue', label='Internal Node'),
        Patch(facecolor='lightcoral', label='Pattern End'),
        Patch(facecolor='black', label='Goto Function'),
        Patch(facecolor='red', label='Failure Function', linestyle='--')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add explanation
    example_patterns = "Patterns: CA, AC, TA, CGA"
    ax.text(0.05, -0.05, example_patterns, transform=ax.transAxes, fontsize=12)
    
    explanation = """
    Aho-Corasick Automaton:
    - Builds a trie from patterns (goto function)
    - Adds failure links for quick recovery after mismatches
    - Terminal nodes (red) indicate matches
    - Multiple patterns can be matched in a single pass
    """
    ax.text(0.05, -0.15, explanation, transform=ax.transAxes, fontsize=12, 
           bbox=dict(facecolor='lightyellow', alpha=0.5))
    
    # Remove axis
    ax.axis('off')
    
    # Save the figure
    ac_file = os.path.join(output_dir, 'aho_corasick_visualization.png')
    plt.savefig(ac_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return ac_file