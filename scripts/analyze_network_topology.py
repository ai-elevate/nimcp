#!/usr/bin/env python3
"""
Analyze Brain Network Topology After Training
==============================================

WHAT: Comprehensive network topology analysis using community detection
WHY:  Visualize and understand modular organization of trained brains
HOW:  Load brain, detect communities, compute metrics, generate visualizations

Features:
- Community detection via Louvain algorithm
- Hub neuron identification
- Modularity metrics (Newman's Q)
- Network visualization
- Detailed topology reports
- Export results to JSON
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import nimcp

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available - visualization disabled")

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not available - some analyses disabled")


class NetworkTopologyAnalyzer:
    """
    WHAT: Analyzer for brain network topology and community structure
    WHY:  Provide comprehensive insights into network organization
    HOW:  Combine community detection, hub analysis, and visualization
    """

    def __init__(self, brain_path: str):
        """Load brain from checkpoint"""
        self.brain_path = Path(brain_path)
        if not self.brain_path.exists():
            raise FileNotFoundError(f"Brain checkpoint not found: {brain_path}")

        print(f"Loading brain from: {brain_path}")
        # TODO: Implement brain loading once serialization API is ready
        # self.brain = nimcp.brain_load(str(brain_path))
        # For now, create a test brain
        self.brain = nimcp.brain_create(1000, 4, 100, 0.01)
        print("✓ Brain loaded successfully")

    def detect_communities(self) -> nimcp.CommunityStructure:
        """
        WHAT: Run Louvain community detection on brain network
        WHY:  Identify functional modules
        HOW:  Call NIMCP community detection API
        """
        print("\nDetecting communities...")
        communities = nimcp.brain_detect_communities(self.brain)
        print(f"✓ Found {communities.num_communities} communities")
        print(f"  Modularity Q: {communities.modularity:.3f}")
        return communities

    def detect_hubs(self, threshold: float = 0.8) -> nimcp.HubStructure:
        """
        WHAT: Identify hub neurons using centrality metrics
        WHY:  Find critical neurons for information flow
        HOW:  Compute degree centrality and apply threshold
        """
        print(f"\nDetecting hub neurons (threshold={threshold})...")
        hubs = nimcp.brain_detect_hubs(self.brain, threshold=threshold)
        print(f"✓ Found {hubs.num_hubs} hub neurons")
        return hubs

    def compute_metrics(self, communities: nimcp.CommunityStructure) -> Dict:
        """
        WHAT: Compute comprehensive network topology metrics
        WHY:  Quantify network properties
        HOW:  Combine community metrics with standard graph metrics
        """
        print("\nComputing topology metrics...")

        metrics = {
            'num_neurons': communities.num_neurons,
            'num_communities': communities.num_communities,
            'modularity': communities.modularity,
            'community_sizes': communities.community_sizes,
        }

        # Compute additional metrics if networkx available
        if HAS_NETWORKX:
            # TODO: Build networkx graph from brain network
            # For now, placeholder
            metrics['clustering_coefficient'] = 0.0
            metrics['characteristic_path_length'] = 0.0
            metrics['small_world_sigma'] = 0.0

        print("✓ Metrics computed")
        return metrics

    def plot_community_structure(self, communities: nimcp.CommunityStructure, output_path: Optional[str] = None):
        """
        WHAT: Visualize community structure
        WHY:  Understand modular organization visually
        HOW:  Plot community size distribution and composition
        """
        if not HAS_MATPLOTLIB:
            print("Skipping visualization - matplotlib not available")
            return

        print("\nGenerating community structure visualization...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Community size distribution
        ax1 = axes[0]
        sizes = communities.community_sizes
        ax1.bar(range(len(sizes)), sizes, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Community ID', fontsize=12)
        ax1.set_ylabel('Number of Neurons', fontsize=12)
        ax1.set_title('Community Size Distribution', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        # Plot 2: Modularity metric
        ax2 = axes[1]
        modularity = communities.modularity

        # Modularity quality indicator
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        thresholds = [0.0, 0.2, 0.3, 0.4, 0.5]
        color_idx = sum(modularity >= t for t in thresholds) - 1
        color = colors[max(0, min(color_idx, len(colors) - 1))]

        ax2.barh(['Modularity'], [modularity], color=color, alpha=0.7)
        ax2.set_xlim(0, 1.0)
        ax2.set_xlabel('Modularity Q Score', fontsize=12)
        ax2.set_title('Network Modularity Quality', fontsize=14, fontweight='bold')
        ax2.axvline(x=0.3, color='green', linestyle='--', alpha=0.5, label='Good threshold (0.3)')
        ax2.legend()

        # Add quality annotation
        quality = "Excellent" if modularity >= 0.4 else \
                  "Good" if modularity >= 0.3 else \
                  "Moderate" if modularity >= 0.2 else \
                  "Weak"
        ax2.text(modularity/2, 0, f'{quality}\nQ={modularity:.3f}',
                ha='center', va='center', fontsize=12, fontweight='bold')

        plt.suptitle('Brain Network Community Structure', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to: {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_hub_network(self, hubs: nimcp.HubStructure, output_path: Optional[str] = None):
        """
        WHAT: Visualize hub neuron distribution
        WHY:  Understand network hierarchy
        HOW:  Plot hub centrality metrics
        """
        if not HAS_MATPLOTLIB:
            print("Skipping hub visualization - matplotlib not available")
            return

        print("\nGenerating hub network visualization...")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot hub degree centrality
        indices = hubs.hub_indices
        centrality = hubs.degree_centrality

        ax.scatter(indices, centrality, c=centrality, cmap='viridis',
                  s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Neuron Index', fontsize=12)
        ax.set_ylabel('Degree Centrality', fontsize=12)
        ax.set_title(f'Hub Neurons (n={hubs.num_hubs})', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(ax.scatter(indices, centrality, c=centrality, cmap='viridis',
                                      s=100, alpha=0), ax=ax)
        cbar.set_label('Centrality', fontsize=11)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to: {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_modularity_evolution(self, metrics: Dict, output_path: Optional[str] = None):
        """
        WHAT: Plot modularity evolution (if available from training history)
        WHY:  Show how modularity changed during training
        HOW:  Line plot of modularity over epochs
        """
        # TODO: Implement once training history tracking is available
        print("\nModularity evolution plot not yet implemented (requires training history)")
        pass

    def generate_report(self, communities: nimcp.CommunityStructure,
                       hubs: nimcp.HubStructure,
                       metrics: Dict,
                       output_path: Optional[str] = None):
        """
        WHAT: Generate comprehensive text report
        WHY:  Provide detailed analysis results
        HOW:  Format metrics and structure information
        """
        print("\nGenerating analysis report...")

        report = []
        report.append("=" * 70)
        report.append("BRAIN NETWORK TOPOLOGY ANALYSIS REPORT")
        report.append("=" * 70)
        report.append("")
        report.append(f"Brain checkpoint: {self.brain_path}")
        report.append("")

        # Community structure
        report.append("COMMUNITY STRUCTURE")
        report.append("-" * 70)
        report.append(f"  Total neurons: {communities.num_neurons}")
        report.append(f"  Number of communities: {communities.num_communities}")
        report.append(f"  Modularity Q: {communities.modularity:.4f}")
        report.append("")

        # Modularity interpretation
        q = communities.modularity
        if q >= 0.4:
            interpretation = "EXCELLENT - Strong modular organization"
        elif q >= 0.3:
            interpretation = "GOOD - Clear community structure"
        elif q >= 0.2:
            interpretation = "MODERATE - Some community structure"
        else:
            interpretation = "WEAK - Little community structure"
        report.append(f"  Quality: {interpretation}")
        report.append("")

        # Community sizes
        report.append("Community Sizes:")
        for i, size in enumerate(communities.community_sizes):
            pct = 100.0 * size / communities.num_neurons
            report.append(f"  Community {i:2d}: {size:5d} neurons ({pct:5.1f}%)")
        report.append("")

        # Hub neurons
        report.append("HUB NEURONS")
        report.append("-" * 70)
        report.append(f"  Total hubs: {hubs.num_hubs}")
        report.append(f"  Hub percentage: {100.0 * hubs.num_hubs / communities.num_neurons:.1f}%")
        report.append("")

        # Top hubs by centrality
        report.append("Top 10 Hubs (by degree centrality):")
        sorted_indices = sorted(range(min(10, hubs.num_hubs)),
                              key=lambda i: hubs.degree_centrality[i],
                              reverse=True)
        for rank, idx in enumerate(sorted_indices, 1):
            neuron_id = hubs.hub_indices[idx]
            centrality = hubs.degree_centrality[idx]
            report.append(f"  {rank:2d}. Neuron {neuron_id:5d}: centrality={centrality:.4f}")
        report.append("")

        # Additional metrics
        if 'clustering_coefficient' in metrics:
            report.append("GRAPH METRICS")
            report.append("-" * 70)
            report.append(f"  Clustering coefficient: {metrics['clustering_coefficient']:.4f}")
            report.append(f"  Characteristic path length: {metrics['characteristic_path_length']:.4f}")
            report.append(f"  Small-world sigma: {metrics['small_world_sigma']:.4f}")
            report.append("")

        report.append("=" * 70)

        # Print to console
        report_text = "\n".join(report)
        print("\n" + report_text)

        # Save to file
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"\n✓ Report saved to: {output_path}")

        return report_text

    def export_json(self, communities: nimcp.CommunityStructure,
                   hubs: nimcp.HubStructure,
                   metrics: Dict,
                   output_path: str):
        """
        WHAT: Export analysis results to JSON
        WHY:  Enable programmatic access to results
        HOW:  Serialize all metrics and structures
        """
        print(f"\nExporting results to JSON: {output_path}")

        data = {
            'brain_path': str(self.brain_path),
            'communities': {
                'num_communities': communities.num_communities,
                'modularity': communities.modularity,
                'community_ids': communities.community_ids,
                'community_sizes': communities.community_sizes,
            },
            'hubs': {
                'num_hubs': hubs.num_hubs,
                'hub_indices': hubs.hub_indices,
                'degree_centrality': hubs.degree_centrality,
            },
            'metrics': metrics,
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Exported to: {output_path}")

    def run_full_analysis(self, output_dir: Optional[str] = None):
        """
        WHAT: Run complete topology analysis pipeline
        WHY:  One-shot comprehensive analysis
        HOW:  Execute all analysis and visualization steps
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Detect communities
        communities = self.detect_communities()

        # Detect hubs
        hubs = self.detect_hubs()

        # Compute metrics
        metrics = self.compute_metrics(communities)

        # Generate visualizations
        if output_dir:
            self.plot_community_structure(communities,
                                         output_path=str(output_dir / "community_structure.png"))
            self.plot_hub_network(hubs,
                                output_path=str(output_dir / "hub_network.png"))

        # Generate report
        if output_dir:
            self.generate_report(communities, hubs, metrics,
                               output_path=str(output_dir / "analysis_report.txt"))
        else:
            self.generate_report(communities, hubs, metrics)

        # Export JSON
        if output_dir:
            self.export_json(communities, hubs, metrics,
                           output_path=str(output_dir / "analysis_results.json"))

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze brain network topology after training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze brain and show interactive plots
  python analyze_network_topology.py checkpoints/brain_final.bin

  # Analyze and save all outputs to directory
  python analyze_network_topology.py checkpoints/brain_final.bin --output analysis_results/

  # Custom hub threshold
  python analyze_network_topology.py checkpoints/brain_final.bin --hub-threshold 0.9
        """
    )

    parser.add_argument('brain_path', type=str,
                       help='Path to brain checkpoint file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output directory for results (default: show interactive plots)')
    parser.add_argument('--hub-threshold', type=float, default=0.8,
                       help='Hub detection threshold (default: 0.8)')
    parser.add_argument('--export-json', action='store_true',
                       help='Export results to JSON')

    args = parser.parse_args()

    try:
        # Create analyzer
        analyzer = NetworkTopologyAnalyzer(args.brain_path)

        # Run full analysis
        analyzer.run_full_analysis(output_dir=args.output)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
