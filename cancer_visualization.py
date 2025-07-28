import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import defaultdict
import numpy as np

class CancerVisualization:
    """Visualize cancer mutation analysis results using DFS traversal."""
    
    def __init__(self, aho_summary_file, aho_detailed_file, boyer_summary_file=None):
        """
        Initialize with the paths to the analysis result files.
        
        Args:
            aho_summary_file: Path to Aho-Corasick cancer_analysis_results.csv
            aho_detailed_file: Path to Aho-Corasick detailed_mutations.csv
            boyer_summary_file: Path to Boyer-Moore cancer_analysis_results.csv (optional)
        """
        self.aho_summary_file = aho_summary_file
        self.aho_detailed_file = aho_detailed_file
        self.boyer_summary_file = boyer_summary_file
        
        # Load data
        print(f"Loading Aho-Corasick summary from: {aho_summary_file}")
        self.aho_summary_df = pd.read_csv(aho_summary_file)
        
        print(f"Loading detailed mutations from: {aho_detailed_file}")
        self.aho_detailed_df = pd.read_csv(aho_detailed_file)
        
        # Optionally load Boyer-Moore data
        self.compare_algorithms = False
        if boyer_summary_file and os.path.exists(boyer_summary_file):
            print(f"Loading Boyer-Moore summary from: {boyer_summary_file}")
            self.boyer_summary_df = pd.read_csv(boyer_summary_file)
            self.compare_algorithms = True
        
        # Create output directory for visualizations
        self.output_dir = os.path.join(os.path.dirname(aho_summary_file), "visualizations")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        print(f"Loaded data for {len(self.aho_summary_df)} patients")
        print(f"Found {len(self.aho_detailed_df)} total mutations")
        
    def _prepare_plot(self, title, figsize=(12, 8)):
        """Prepare a matplotlib figure."""
        plt.figure(figsize=figsize)
        plt.title(title, fontsize=16)
        
    def _save_plot(self, filename):
        """Save the current plot to the output directory."""
        output_path = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved plot to {output_path}")
        
    def visualize_cancer_distribution(self):
        """Create a bar chart of cancer type distribution using DFS."""
        # Use DFS to count cancer types
        def count_cancer_types_dfs(df, cancer_types=None, index=0):
            if cancer_types is None:
                cancer_types = defaultdict(int)
            
            if index >= len(df):
                return cancer_types
            
            # Handle potential missing rows or bad data
            try:
                row = df.iloc[index]
                if row['most_likely_cancer'] != "Error" and row['most_likely_cancer'] != "Unknown" and not pd.isna(row['most_likely_cancer']):
                    cancer_types[row['most_likely_cancer']] += 1
            except (IndexError, KeyError) as e:
                print(f"Warning: Issue processing row {index}: {e}")
            
            # Process next row (DFS)
            return count_cancer_types_dfs(df, cancer_types, index + 1)
        
        # Get cancer type counts using DFS
        cancer_types = count_cancer_types_dfs(self.aho_summary_df)
        
        # Convert to DataFrame for plotting
        plot_data = pd.DataFrame({
            'Cancer Type': list(cancer_types.keys()),
            'Patient Count': list(cancer_types.values())
        }).sort_values('Patient Count', ascending=False)
        
        # Create plot
        self._prepare_plot("Distribution of Most Likely Cancer Types")
        
        # Check if we have data to plot
        if len(plot_data) > 0:
            sns.barplot(x='Cancer Type', y='Patient Count', data=plot_data, palette='viridis')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel("Number of Patients")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
        else:
            plt.text(0.5, 0.5, "No cancer type data available", 
                     horizontalalignment='center', verticalalignment='center',
                     transform=plt.gca().transAxes)
        
        self._save_plot("cancer_distribution.png")
        
    def visualize_mutation_types(self):
        """Create a pie chart of somatic vs germline mutations using DFS."""
        # Use DFS to count mutation types
        def count_mutation_types_dfs(df, counts=None, index=0):
            if counts is None:
                counts = {'somatic': 0, 'germline': 0}
            
            if index >= len(df):
                return counts
            
            # Handle potential missing rows or bad data
            try:
                row = df.iloc[index]
                if 'mutation_type' in row and not pd.isna(row['mutation_type']):
                    if row['mutation_type'].lower() == 'somatic':
                        counts['somatic'] += 1
                    elif row['mutation_type'].lower() == 'germline':
                        counts['germline'] += 1
            except (IndexError, KeyError) as e:
                print(f"Warning: Issue processing row {index}: {e}")
            
            # Process next row (DFS)
            return count_mutation_types_dfs(df, counts, index + 1)
        
        # Get mutation type counts using DFS
        mutation_counts = count_mutation_types_dfs(self.aho_detailed_df)
        
        # Create plot
        self._prepare_plot("Distribution of Mutation Types")
        
        # Check if we have data to plot
        if mutation_counts['somatic'] > 0 or mutation_counts['germline'] > 0:
            plt.pie(
                [mutation_counts['somatic'], mutation_counts['germline']], 
                labels=['Somatic', 'Germline'],
                autopct='%1.1f%%',
                colors=['#ff9999', '#66b3ff'],
                startangle=90,
                explode=(0.1, 0)
            )
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        else:
            plt.text(0.5, 0.5, "No mutation type data available", 
                     horizontalalignment='center', verticalalignment='center',
                     transform=plt.gca().transAxes)
        
        self._save_plot("mutation_types.png")
        
    def visualize_cancer_mutation_network(self, max_mutations=100):
        """Create a network graph of cancer types and their common mutations using DFS."""
        # Create a graph
        G = nx.Graph()
        
        # Use DFS to build the graph
        def build_graph_dfs(df, processed_rows=None, index=0, count=0):
            if processed_rows is None:
                processed_rows = set()
            
            if index >= len(df) or count >= max_mutations:
                return count
            
            # Handle potential missing rows or bad data
            try:
                row = df.iloc[index]
                
                # Check for required fields
                if ('mutation_id' in row and 'cancer_type' in row and 'mutation_type' in row and
                    not pd.isna(row['mutation_id']) and not pd.isna(row['cancer_type'])):
                    
                    mutation_id = row['mutation_id']
                    cancer_type = row['cancer_type']
                    mutation_type = row['mutation_type'] if not pd.isna(row['mutation_type']) else "unknown"
                    
                    # Add nodes if they don't exist
                    if not G.has_node(cancer_type):
                        G.add_node(cancer_type, type='cancer', size=300)
                    
                    if not G.has_node(mutation_id):
                        G.add_node(mutation_id, type='mutation', mutation_type=mutation_type, size=100)
                    
                    # Add edge
                    if not G.has_edge(cancer_type, mutation_id):
                        G.add_edge(cancer_type, mutation_id)
                    
                    # Mark as processed
                    row_key = f"{mutation_id}-{cancer_type}"
                    if row_key not in processed_rows:
                        processed_rows.add(row_key)
                        count += 1
            except (IndexError, KeyError) as e:
                print(f"Warning: Issue processing row {index} for network graph: {e}")
            
            # Process next row (DFS)
            return build_graph_dfs(df, processed_rows, index + 1, count)
        
        # Build graph using DFS
        mutation_count = build_graph_dfs(self.aho_detailed_df)
        
        # Create plot
        self._prepare_plot("Cancer-Mutation Network", figsize=(15, 15))
        
        # Check if we have data to plot
        if len(G.nodes) > 0:
            # Define node positions using spring layout
            try:
                pos = nx.spring_layout(G, k=0.3, seed=42)  # Set seed for reproducibility
                
                # Find nodes by type
                cancer_nodes = [node for node, attr in G.nodes(data=True) if attr.get('type') == 'cancer']
                somatic_nodes = [node for node, attr in G.nodes(data=True) 
                                if attr.get('type') == 'mutation' and attr.get('mutation_type', '').lower() == 'somatic']
                germline_nodes = [node for node, attr in G.nodes(data=True) 
                                if attr.get('type') == 'mutation' and attr.get('mutation_type', '').lower() == 'germline']
                
                # Draw cancer type nodes
                if cancer_nodes:
                    nx.draw_networkx_nodes(G, pos, nodelist=cancer_nodes, node_color='red', node_size=700, alpha=0.8)
                
                # Draw mutation nodes with color based on mutation type
                if somatic_nodes:
                    nx.draw_networkx_nodes(G, pos, nodelist=somatic_nodes, node_color='blue', node_size=300, alpha=0.6)
                if germline_nodes:
                    nx.draw_networkx_nodes(G, pos, nodelist=germline_nodes, node_color='green', node_size=300, alpha=0.6)
                
                # Draw edges
                if G.edges:
                    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
                
                # Draw labels only for cancer types (for clarity)
                if cancer_nodes:
                    labels = {node: node for node in cancer_nodes}
                    nx.draw_networkx_labels(G, pos, labels, font_size=12, font_weight='bold')
                
                plt.axis('off')
            except Exception as e:
                print(f"Error creating network visualization: {e}")
                plt.text(0.5, 0.5, f"Error creating network visualization: {str(e)}", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=plt.gca().transAxes)
        else:
            plt.text(0.5, 0.5, "No network data available", 
                     horizontalalignment='center', verticalalignment='center',
                     transform=plt.gca().transAxes)
        
        self._save_plot("cancer_mutation_network.png")
        
    def visualize_mutation_counts_by_cancer(self):
        """Create a stacked bar chart of somatic vs germline mutations by cancer type using DFS."""
        # Use DFS to collect mutation counts by cancer type
        def count_mutations_by_cancer_dfs(df, cancer_mutations=None, index=0):
            if cancer_mutations is None:
                cancer_mutations = defaultdict(lambda: {'somatic': 0, 'germline': 0})
            
            if index >= len(df):
                return cancer_mutations
            
            # Handle potential missing rows or bad data
            try:
                row = df.iloc[index]
                if ('cancer_type' in row and 'mutation_type' in row and
                    not pd.isna(row['cancer_type']) and not pd.isna(row['mutation_type'])):
                    
                    cancer_type = row['cancer_type']
                    mutation_type = row['mutation_type'].lower()
                    
                    if mutation_type == 'somatic':
                        cancer_mutations[cancer_type]['somatic'] += 1
                    elif mutation_type == 'germline':
                        cancer_mutations[cancer_type]['germline'] += 1
            except (IndexError, KeyError) as e:
                print(f"Warning: Issue processing row {index} for mutation counts: {e}")
            
            # Process next row (DFS)
            return count_mutations_by_cancer_dfs(df, cancer_mutations, index + 1)
        
        # Get mutation counts by cancer type using DFS
        cancer_mutations = count_mutations_by_cancer_dfs(self.aho_detailed_df)
        
        # Convert to DataFrame for plotting
        plot_data = pd.DataFrame([
            {'Cancer Type': cancer, 'Somatic': counts['somatic'], 'Germline': counts['germline']}
            for cancer, counts in cancer_mutations.items()
        ])
        
        # Check if we have data to plot
        if len(plot_data) > 0:
            # Sort by total mutations
            plot_data['Total'] = plot_data['Somatic'] + plot_data['Germline']
            plot_data = plot_data.sort_values('Total', ascending=False).head(10)  # Top 10 cancer types
            
            # Create stacked bar plot
            self._prepare_plot("Mutation Types by Cancer (Top 10)")
            
            ax = plt.subplot(111)
            plot_data.plot(
                x='Cancer Type', 
                y=['Somatic', 'Germline'], 
                kind='bar', 
                stacked=True, 
                ax=ax,
                color=['#ff9999', '#66b3ff']
            )
            
            plt.ylabel("Number of Mutations")
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Mutation Type')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
        else:
            plt.text(0.5, 0.5, "No mutation by cancer type data available", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes)
        
        self._save_plot("mutations_by_cancer.png")
        
    def visualize_mutation_positions(self, cancer_type=None, max_patients=5):
        """
        Visualize mutation positions along the genome for selected patients using DFS.
        
        Args:
            cancer_type: Optional filter for patients with a specific cancer type
            max_patients: Maximum number of patients to visualize
        """
        # Filter patients by cancer type if specified
        if cancer_type:
            patient_ids = self.aho_summary_df[
                self.aho_summary_df['most_likely_cancer'] == cancer_type
            ]['patient_id'].tolist()
        else:
            patient_ids = self.aho_summary_df['patient_id'].tolist()
        
        # Handle empty patient list
        if not patient_ids:
            print(f"Warning: No patients found for cancer type: {cancer_type}")
            self._prepare_plot("Mutation Positions - No Patients Found")
            plt.text(0.5, 0.5, f"No patients found for cancer type: {cancer_type}", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes)
            
            if cancer_type:
                self._save_plot(f"mutation_positions_{cancer_type}.png")
            else:
                self._save_plot("mutation_positions.png")
            return
        
        # Limit to max_patients
        patient_ids = patient_ids[:max_patients]
        
        # Create a figure with subplots
        fig, axes = plt.subplots(len(patient_ids), 1, figsize=(12, 3*len(patient_ids)), sharex=True)
        if len(patient_ids) == 1:
            axes = [axes]  # Make axes iterable if only one subplot
            
        # Use DFS to process each patient
        def process_patients_dfs(patients, patient_index=0):
            if patient_index >= len(patients) or patient_index >= len(axes):
                return
                
            patient_id = patients[patient_index]
            ax = axes[patient_index]
            
            # Get patient mutations
            patient_mutations = self.aho_detailed_df[self.aho_detailed_df['patient_id'] == patient_id]
            
            # Check if we have mutations for this patient
            if len(patient_mutations) > 0:
                # Extract positions and types
                try:
                    positions = patient_mutations['position'].tolist()
                    types = patient_mutations['mutation_type'].tolist()
                    
                    # Plot mutations along genome
                    colors = ['red' if str(t).lower() == 'somatic' else 'blue' for t in types]
                    ax.scatter(positions, [1] * len(positions), c=colors, alpha=0.7)
                    
                    # Add labels
                    ax.set_ylabel(f"Patient {patient_id}")
                    ax.set_yticks([])
                except Exception as e:
                    print(f"Error plotting mutations for patient {patient_id}: {e}")
                    ax.text(0.5, 0.5, f"Error: {str(e)}", 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, f"No mutations found for patient {patient_id}", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes)
            
            if patient_index == len(patients) - 1:
                ax.set_xlabel("Genome Position")
            
            # Process next patient (DFS)
            process_patients_dfs(patients, patient_index + 1)
        
        # Process patients using DFS
        process_patients_dfs(patient_ids)
        
        # Set title
        if cancer_type:
            plt.suptitle(f"Mutation Positions for {cancer_type} Patients", fontsize=16)
        else:
            plt.suptitle("Mutation Positions by Patient", fontsize=16)
            
        # Add legend
        fig.legend(
            handles=[
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Somatic'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Germline')
            ],
            loc='upper right'
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
        
        # Save the figure
        if cancer_type:
            self._save_plot(f"mutation_positions_{cancer_type}.png")
        else:
            self._save_plot("mutation_positions.png")

    def visualize_algorithm_comparison(self):
        """Compare Aho-Corasick and Boyer-Moore algorithms if data is available."""
        if not self.compare_algorithms:
            print("Boyer-Moore data not available for comparison")
            return
        
        # Merge the datasets to compare
        aho_results = self.aho_summary_df[['patient_id', 'most_likely_cancer', 'total_mutations']]
        aho_results = aho_results.rename(columns={'total_mutations': 'aho_mutations'})
        
        boyer_results = self.boyer_summary_df[['patient_id', 'most_likely_cancer', 'total_mutations']]
        boyer_results = boyer_results.rename(columns={'total_mutations': 'boyer_mutations'})
        
        # Merge on patient_id
        comparison = pd.merge(aho_results, boyer_results, on='patient_id', 
                              suffixes=('_aho', '_boyer'), how='outer')
        
        # Fill NaN with 0 (where an algorithm didn't find mutations)
        comparison = comparison.fillna({'aho_mutations': 0, 'boyer_mutations': 0})
        
        # Calculate agreement on cancer type
        comparison['agreement'] = (comparison['most_likely_cancer_aho'] == 
                                  comparison['most_likely_cancer_boyer'])
        
        # Create plot for agreement percentage
        self._prepare_plot("Algorithm Agreement on Cancer Type")
        agreement_pct = comparison['agreement'].mean() * 100
        disagreement_pct = 100 - agreement_pct
        
        plt.pie(
            [agreement_pct, disagreement_pct],
            labels=['Agreement', 'Disagreement'],
            autopct='%1.1f%%',
            colors=['#66b3ff', '#ff9999'],
            startangle=90,
            explode=(0.1, 0)
        )
        plt.axis('equal')
        self._save_plot("algorithm_agreement.png")
        
        # Create scatter plot comparing mutation counts
        self._prepare_plot("Mutation Count Comparison: Aho-Corasick vs Boyer-Moore")
        plt.scatter(comparison['aho_mutations'], comparison['boyer_mutations'], alpha=0.5)
        
        # Add diagonal line for perfect agreement
        max_val = max(comparison['aho_mutations'].max(), comparison['boyer_mutations'].max())
        plt.plot([0, max_val], [0, max_val], 'r--')
        
        plt.xlabel("Aho-Corasick Mutation Count")
        plt.ylabel("Boyer-Moore Mutation Count")
        plt.grid(True, linestyle='--', alpha=0.7)
        self._save_plot("mutation_count_comparison.png")
            
    def create_all_visualizations(self):
        """Create all visualizations."""
        print("Generating visualizations using DFS traversal...")
        
        # Run all visualization methods
        self.visualize_cancer_distribution()
        self.visualize_mutation_types()
        self.visualize_cancer_mutation_network()
        self.visualize_mutation_counts_by_cancer()
        self.visualize_mutation_positions()
        
        # Compare algorithms if Boyer-Moore data is available
        if self.compare_algorithms:
            self.visualize_algorithm_comparison()
        
        # Also create visualizations for top 3 cancer types
        try:
            top_cancers = self.aho_summary_df['most_likely_cancer'].value_counts().head(3).index.tolist()
            for cancer in top_cancers:
                if cancer not in ["Error", "Unknown"] and not pd.isna(cancer):
                    self.visualize_mutation_positions(cancer_type=cancer)
        except Exception as e:
            print(f"Error visualizing top cancer types: {e}")
        
        print(f"All visualizations saved to {self.output_dir}")

def main():
    """Main function to run the visualization."""
    # File paths from your provided paths
    aho_summary_file = r"D:\SEM-4\aho_corasick_results\cancer_analysis_results.csv"
    aho_detailed_file = r"D:\SEM-4\aho_corasick_results\detailed_mutations.csv"
    boyer_summary_file = r"D:\SEM-4\boyer_moore_results\cancer_analysis_results.csv"
    
    # Validate file paths
    missing_files = []
    for file_path in [aho_summary_file, aho_detailed_file, boyer_summary_file]:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Warning: The following files were not found:")
        for file in missing_files:
            print(f"  - {file}")
        print("Continuing with available files...")
    
    # Create visualization with all available files
    if os.path.exists(aho_summary_file) and os.path.exists(aho_detailed_file):
        visualizer = CancerVisualization(
            aho_summary_file, 
            aho_detailed_file,
            boyer_summary_file if os.path.exists(boyer_summary_file) else None
        )
        visualizer.create_all_visualizations()
    else:
        print("Error: Required Aho-Corasick files not found.")
        print("Please ensure the analysis has been run and files exist at the specified paths.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error running visualization: {e}")
        import traceback
        traceback.print_exc()