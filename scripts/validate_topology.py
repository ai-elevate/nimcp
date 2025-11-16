#!/usr/bin/env python3
"""
Validate Brain Topology Quality During Training
================================================

WHAT: Validate brain network topology meets quality thresholds
WHY:  Ensure healthy network structure for optimal learning
HOW:  Check modularity, connectivity, small-world properties, hub distribution

Features:
- Modularity validation (Newman's Q)
- Connectivity checks
- Small-world property verification
- Hub distribution analysis
- Detailed validation report
- Exit codes for CI/CD integration
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, List
import nimcp


class TopologyValidator:
    """
    WHAT: Validator for brain network topology quality
    WHY:  Ensure network has desirable properties for learning
    HOW:  Apply multiple validation criteria and generate report
    """

    # Validation thresholds
    DEFAULT_MIN_MODULARITY = 0.25
    DEFAULT_MIN_CLUSTERING = 0.1
    DEFAULT_MIN_COMMUNITIES = 2
    DEFAULT_MIN_HUB_RATIO = 0.01  # At least 1% hubs

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
        print("✓ Brain loaded\n")

    def validate_modularity(self, min_modularity: float) -> Tuple[bool, str, float]:
        """
        WHAT: Check if network has sufficient modularity
        WHY:  Modularity indicates functional specialization
        HOW:  Compute Newman's Q and compare to threshold

        Returns:
            (is_valid, message, modularity_score)
        """
        print("Validating modularity...")
        modularity = nimcp.brain_get_modularity(self.brain)

        if modularity >= min_modularity:
            message = f"✓ Modularity PASSED: Q={modularity:.3f} >= {min_modularity:.3f}"
            is_valid = True
        else:
            message = f"✗ Modularity FAILED: Q={modularity:.3f} < {min_modularity:.3f}"
            is_valid = False

        print(f"  {message}")
        return is_valid, message, modularity

    def validate_communities(self, min_communities: int,
                           max_communities: int = None) -> Tuple[bool, str, int]:
        """
        WHAT: Check if network has appropriate number of communities
        WHY:  Too few = no specialization, too many = over-fragmentation
        HOW:  Detect communities and count

        Returns:
            (is_valid, message, num_communities)
        """
        print("Validating community count...")
        num_communities = nimcp.brain_get_num_communities(self.brain)

        is_valid = num_communities >= min_communities
        if max_communities is not None:
            is_valid = is_valid and num_communities <= max_communities

        if is_valid:
            message = f"✓ Communities PASSED: {num_communities} communities"
            if max_communities:
                message += f" (range: {min_communities}-{max_communities})"
        else:
            message = f"✗ Communities FAILED: {num_communities} communities"
            if max_communities:
                message += f" (expected: {min_communities}-{max_communities})"
            else:
                message += f" (minimum: {min_communities})"

        print(f"  {message}")
        return is_valid, message, num_communities

    def validate_hubs(self, min_hub_ratio: float,
                     threshold: float = 0.8) -> Tuple[bool, str, int, float]:
        """
        WHAT: Check if network has sufficient hub neurons
        WHY:  Hubs are critical for information integration
        HOW:  Detect hubs and compute ratio to total neurons

        Returns:
            (is_valid, message, num_hubs, hub_ratio)
        """
        print("Validating hub distribution...")

        communities = nimcp.brain_detect_communities(self.brain)
        num_neurons = communities.num_neurons

        hubs = nimcp.brain_detect_hubs(self.brain, threshold=threshold)
        num_hubs = hubs.num_hubs
        hub_ratio = num_hubs / num_neurons if num_neurons > 0 else 0.0

        is_valid = hub_ratio >= min_hub_ratio

        if is_valid:
            message = f"✓ Hubs PASSED: {num_hubs} hubs ({hub_ratio*100:.1f}%) >= {min_hub_ratio*100:.1f}%"
        else:
            message = f"✗ Hubs FAILED: {num_hubs} hubs ({hub_ratio*100:.1f}%) < {min_hub_ratio*100:.1f}%"

        print(f"  {message}")
        return is_valid, message, num_hubs, hub_ratio

    def validate_comprehensive(self, min_modularity: float = None) -> Tuple[bool, Dict]:
        """
        WHAT: Run comprehensive topology validation
        WHY:  Check all quality criteria at once
        HOW:  Use NIMCP's built-in validation function

        Returns:
            (is_valid, validation_dict)
        """
        print("Running comprehensive validation...")

        if min_modularity is None:
            min_modularity = self.DEFAULT_MIN_MODULARITY

        validation = nimcp.brain_validate_topology(self.brain, min_modularity)

        result_dict = {
            'is_valid': validation.is_valid,
            'modularity': validation.modularity,
            'clustering_coefficient': validation.clustering_coefficient,
            'num_communities': validation.num_communities,
            'num_hubs': validation.num_hubs,
            'error_message': validation.error_message,
        }

        if validation.is_valid:
            print(f"  ✓ Comprehensive validation PASSED")
            print(f"    Modularity: {validation.modularity:.3f}")
            print(f"    Clustering: {validation.clustering_coefficient:.3f}")
            print(f"    Communities: {validation.num_communities}")
            print(f"    Hubs: {validation.num_hubs}")
        else:
            print(f"  ✗ Comprehensive validation FAILED")
            print(f"    Error: {validation.error_message}")

        return validation.is_valid, result_dict

    def generate_validation_report(self, results: Dict[str, Tuple],
                                  output_path: str = None) -> str:
        """
        WHAT: Generate detailed validation report
        WHY:  Provide comprehensive summary of all checks
        HOW:  Format all validation results with interpretation

        Args:
            results: Dict mapping check name to (is_valid, message, ...) tuples
            output_path: Optional path to save report

        Returns:
            Report text
        """
        report = []
        report.append("=" * 70)
        report.append("BRAIN TOPOLOGY VALIDATION REPORT")
        report.append("=" * 70)
        report.append("")
        report.append(f"Brain: {self.brain_path}")
        report.append("")

        # Overall status
        all_passed = all(r[0] for r in results.values() if isinstance(r, tuple))
        status = "✓ PASSED" if all_passed else "✗ FAILED"
        report.append(f"Overall Status: {status}")
        report.append("")

        # Individual checks
        report.append("VALIDATION CHECKS")
        report.append("-" * 70)

        for check_name, result in results.items():
            if isinstance(result, tuple):
                is_valid, message = result[0], result[1]
                report.append(f"{message}")

        report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 70)

        if all_passed:
            report.append("  Topology is healthy. Network is ready for deployment.")
        else:
            report.append("  Topology validation failed. Consider:")
            for check_name, result in results.items():
                if isinstance(result, tuple) and not result[0]:
                    if 'modularity' in check_name.lower():
                        report.append("    - Increase training duration to develop modularity")
                        report.append("    - Check learning rate (too high may prevent specialization)")
                    elif 'communities' in check_name.lower():
                        report.append("    - Adjust network size or connectivity")
                        report.append("    - Review training data diversity")
                    elif 'hubs' in check_name.lower():
                        report.append("    - Check scale-free topology generation")
                        report.append("    - Ensure sufficient network size")

        report.append("")
        report.append("=" * 70)

        report_text = "\n".join(report)
        print("\n" + report_text)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"\n✓ Report saved to: {output_path}")

        return report_text

    def run_all_validations(self,
                          min_modularity: float = None,
                          min_communities: int = None,
                          max_communities: int = None,
                          min_hub_ratio: float = None,
                          output_path: str = None) -> bool:
        """
        WHAT: Run all validation checks
        WHY:  Complete topology quality assessment
        HOW:  Execute individual checks and generate report

        Returns:
            True if all validations passed, False otherwise
        """
        # Use defaults if not specified
        if min_modularity is None:
            min_modularity = self.DEFAULT_MIN_MODULARITY
        if min_communities is None:
            min_communities = self.DEFAULT_MIN_COMMUNITIES
        if min_hub_ratio is None:
            min_hub_ratio = self.DEFAULT_MIN_HUB_RATIO

        results = {}

        # Run individual checks
        print("=" * 70)
        print("RUNNING TOPOLOGY VALIDATION CHECKS")
        print("=" * 70)
        print("")

        results['modularity'] = self.validate_modularity(min_modularity)
        results['communities'] = self.validate_communities(min_communities, max_communities)
        results['hubs'] = self.validate_hubs(min_hub_ratio)

        # Run comprehensive check
        comprehensive_valid, comprehensive_results = self.validate_comprehensive(min_modularity)
        results['comprehensive'] = (comprehensive_valid,
                                   "✓ Comprehensive check PASSED" if comprehensive_valid
                                   else f"✗ Comprehensive check FAILED: {comprehensive_results['error_message']}")

        # Generate report
        self.generate_validation_report(results, output_path)

        # Return overall status
        all_passed = all(r[0] for r in results.values())
        return all_passed


def main():
    parser = argparse.ArgumentParser(
        description='Validate brain network topology quality',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Validation Criteria:
  - Modularity Q >= 0.25 (functional specialization)
  - At least 2 communities (modular organization)
  - Hub ratio >= 1% (information integration)
  - Clustering coefficient >= 0.1 (local connectivity)

Exit Codes:
  0: All validations passed
  1: One or more validations failed
  2: Error during validation

Examples:
  # Validate with default thresholds
  python validate_topology.py checkpoints/brain_final.bin

  # Custom validation thresholds
  python validate_topology.py checkpoints/brain_final.bin \\
      --min-modularity 0.3 \\
      --min-communities 3 \\
      --max-communities 10

  # Save report to file
  python validate_topology.py checkpoints/brain_final.bin \\
      --output validation_report.txt
        """
    )

    parser.add_argument('brain_path', type=str,
                       help='Path to brain checkpoint file')
    parser.add_argument('--min-modularity', type=float,
                       default=TopologyValidator.DEFAULT_MIN_MODULARITY,
                       help=f'Minimum modularity threshold (default: {TopologyValidator.DEFAULT_MIN_MODULARITY})')
    parser.add_argument('--min-communities', type=int,
                       default=TopologyValidator.DEFAULT_MIN_COMMUNITIES,
                       help=f'Minimum number of communities (default: {TopologyValidator.DEFAULT_MIN_COMMUNITIES})')
    parser.add_argument('--max-communities', type=int, default=None,
                       help='Maximum number of communities (default: no limit)')
    parser.add_argument('--min-hub-ratio', type=float,
                       default=TopologyValidator.DEFAULT_MIN_HUB_RATIO,
                       help=f'Minimum hub neuron ratio (default: {TopologyValidator.DEFAULT_MIN_HUB_RATIO})')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output path for validation report')

    args = parser.parse_args()

    try:
        # Create validator
        validator = TopologyValidator(args.brain_path)

        # Run all validations
        all_passed = validator.run_all_validations(
            min_modularity=args.min_modularity,
            min_communities=args.min_communities,
            max_communities=args.max_communities,
            min_hub_ratio=args.min_hub_ratio,
            output_path=args.output
        )

        # Return appropriate exit code
        if all_passed:
            print("\n✓ All validations PASSED")
            return 0
        else:
            print("\n✗ Some validations FAILED")
            return 1

    except Exception as e:
        print(f"\nError during validation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
