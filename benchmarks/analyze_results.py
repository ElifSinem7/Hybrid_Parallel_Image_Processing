
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_results(filename):
    """Parse timing results from output file"""
    times = {
        'gaussian_blur': [],
        'rgb_to_gray': [],
        'sobel_edge': [],
        'total_compute': [],
        'total_time': []
    }
    
    if not os.path.exists(filename):
        return times
    
    with open(filename, 'r') as f:
        content = f.read()
        
        # Extract all timing measurements
        blur_times = re.findall(r'Gaussian Blur: ([\d.]+) seconds', content)
        gray_times = re.findall(r'RGB to Gray: ([\d.]+) seconds', content)
        sobel_times = re.findall(r'Sobel Edge: ([\d.]+) seconds', content)
        compute_times = re.findall(r'Total Compute: ([\d.]+) seconds', content)
        total_times = re.findall(r'Total time: ([\d.]+) seconds', content)
        
        times['gaussian_blur'] = [float(t) for t in blur_times]
        times['rgb_to_gray'] = [float(t) for t in gray_times]
        times['sobel_edge'] = [float(t) for t in sobel_times]
        times['total_compute'] = [float(t) for t in compute_times]
        times['total_time'] = [float(t) for t in total_times]
    
    return times

def compute_statistics(times_list):
    """Compute mean and standard deviation"""
    if not times_list:
        return 0.0, 0.0
    
    mean = np.mean(times_list)
    std = np.std(times_list)
    return mean, std

def analyze_all_results(results_dir):
    """Analyze all benchmark results"""
    implementations = ['sequential', 'openmp', 'cuda', 'hybrid']
    results = defaultdict(lambda: defaultdict(dict))
    
    # Find all result files
    files = [f for f in os.listdir(results_dir) if f.endswith('.txt')]
    
    # Group by image
    images = set()
    for f in files:
        # Extract image name (e.g., "0001" from "0001_sequential.txt")
        match = re.match(r'(\d+)_(\w+)\.txt', f)
        if match:
            images.add(match.group(1))
    
    # Parse results for each image and implementation
    for img in sorted(images):
        for impl in implementations:
            filename = os.path.join(results_dir, f"{img}_{impl}.txt")
            times = parse_results(filename)
            
            # Compute statistics
            for metric, values in times.items():
                if values:
                    mean, std = compute_statistics(values)
                    results[img][impl][metric] = {
                        'mean': mean,
                        'std': std,
                        'values': values
                    }
    
    return results, sorted(images), implementations

def create_comparison_charts(results, images, implementations, output_dir):
    """Create comprehensive comparison charts"""
    
    # Chart 1: Total Compute Time Comparison
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(images))
    width = 0.2
    
    for i, impl in enumerate(implementations):
        means = []
        stds = []
        for img in images:
            if impl in results[img] and 'total_compute' in results[img][impl]:
                means.append(results[img][impl]['total_compute']['mean'])
                stds.append(results[img][impl]['total_compute']['std'])
            else:
                means.append(0)
                stds.append(0)
        
        plt.bar(x + i*width, means, width, yerr=stds, 
                label=impl.upper(), alpha=0.8, capsize=5)
    
    plt.xlabel('Image', fontsize=12, fontweight='bold')
    plt.ylabel('Total Compute Time (seconds)', fontsize=12, fontweight='bold')
    plt.title('Total Compute Time Comparison Across Implementations', 
              fontsize=14, fontweight='bold')
    plt.xticks(x + width*1.5, images)
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'compute_time_comparison.png'), dpi=300)
    plt.close()
    
    # Chart 2: Speedup Analysis (relative to Sequential)
    plt.figure(figsize=(14, 8))
    
    for i, impl in enumerate(implementations[1:], 1): 
        speedups = []
        for img in images:
            if ('sequential' in results[img] and impl in results[img] and
                'total_compute' in results[img]['sequential'] and
                'total_compute' in results[img][impl]):
                
                seq_time = results[img]['sequential']['total_compute']['mean']
                impl_time = results[img][impl]['total_compute']['mean']
                
                if impl_time > 0:
                    speedup = seq_time / impl_time
                    speedups.append(speedup)
                else:
                    speedups.append(0)
            else:
                speedups.append(0)
        
        plt.bar(x + (i-1)*width, speedups, width, 
                label=impl.upper(), alpha=0.8)
    
    plt.xlabel('Image', fontsize=12, fontweight='bold')
    plt.ylabel('Speedup (relative to Sequential)', fontsize=12, fontweight='bold')
    plt.title('Speedup Analysis: All Implementations vs Sequential', 
              fontsize=14, fontweight='bold')
    plt.xticks(x + width, images)
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Baseline (Sequential)')
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speedup_analysis.png'), dpi=300)
    plt.close()
    
    # Chart 3: Operation Breakdown for each implementation
    operations = ['gaussian_blur', 'rgb_to_gray', 'sobel_edge']
    op_labels = ['Gaussian Blur', 'RGB to Gray', 'Sobel Edge']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, impl in enumerate(implementations):
        ax = axes[idx]
        
        # Average across all images
        op_means = {op: [] for op in operations}
        
        for img in images:
            if impl in results[img]:
                for op in operations:
                    if op in results[img][impl]:
                        op_means[op].append(results[img][impl][op]['mean'])
        
        # Compute averages
        avg_times = [np.mean(op_means[op]) if op_means[op] else 0 
                     for op in operations]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        ax.bar(op_labels, avg_times, color=colors, alpha=0.8)
        ax.set_title(f'{impl.upper()} - Operation Breakdown', 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Time (seconds)', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        
        for i, v in enumerate(avg_times):
            ax.text(i, v, f'{v:.4f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'operation_breakdown.png'), dpi=300)
    plt.close()
    
    # Chart 4: Average Performance Summary
    plt.figure(figsize=(10, 8))
    
    avg_compute_times = []
    for impl in implementations:
        all_times = []
        for img in images:
            if impl in results[img] and 'total_compute' in results[img][impl]:
                all_times.extend(results[img][impl]['total_compute']['values'])
        
        avg_compute_times.append(np.mean(all_times) if all_times else 0)
    
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
    bars = plt.bar(implementations, avg_compute_times, color=colors, alpha=0.8)
    
    plt.xlabel('Implementation', fontsize=12, fontweight='bold')
    plt.ylabel('Average Total Compute Time (seconds)', fontsize=12, fontweight='bold')
    plt.title('Average Performance Summary Across All Images', 
              fontsize=14, fontweight='bold')
    plt.xticks([impl.upper() for impl in implementations])
    plt.grid(axis='y', alpha=0.3)
    
    for bar, value in zip(bars, avg_compute_times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}s',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    if avg_compute_times[0] > 0:
        for i, (impl, time) in enumerate(zip(implementations[1:], avg_compute_times[1:]), 1):
            if time > 0:
                speedup = avg_compute_times[0] / time
                plt.text(i, time * 0.5, f'Speedup: {speedup:.2f}x',
                        ha='center', va='center', fontweight='bold', 
                        fontsize=9, color='white',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_performance.png'), dpi=300)
    plt.close()

def generate_report(results, images, implementations, output_dir):
    """Generate detailed text report"""
    
    report_file = os.path.join(output_dir, 'benchmark_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RODINIA-STYLE BENCHMARK REPORT\n")
        f.write("Edge Detection: Sequential, OpenMP, CUDA, and Hybrid Implementations\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall Summary
        f.write("OVERALL PERFORMANCE SUMMARY\n")
        f.write("-" * 80 + "\n")
        
        for impl in implementations:
            all_compute_times = []
            for img in images:
                if impl in results[img] and 'total_compute' in results[img][impl]:
                    all_compute_times.extend(results[img][impl]['total_compute']['values'])
            
            if all_compute_times:
                mean_time = np.mean(all_compute_times)
                std_time = np.std(all_compute_times)
                
                f.write(f"\n{impl.upper()}:\n")
                f.write(f"  Average Compute Time: {mean_time:.6f} ± {std_time:.6f} seconds\n")
                
                if impl != 'sequential' and results[images[0]]['sequential']['total_compute']['mean'] > 0:
                    # Calculate average speedup
                    seq_times = []
                    impl_times = []
                    for img in images:
                        if ('sequential' in results[img] and impl in results[img]):
                            seq_times.extend(results[img]['sequential']['total_compute']['values'])
                            impl_times.extend(results[img][impl]['total_compute']['values'])
                    
                    if seq_times and impl_times:
                        avg_speedup = np.mean(seq_times) / np.mean(impl_times)
                        f.write(f"  Average Speedup vs Sequential: {avg_speedup:.2f}x\n")
        
        # Detailed Results per Image
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("DETAILED RESULTS PER IMAGE\n")
        f.write("=" * 80 + "\n")
        
        for img in images:
            f.write(f"\n\nImage: {img}\n")
            f.write("-" * 80 + "\n")
            
            for impl in implementations:
                if impl not in results[img]:
                    continue
                
                f.write(f"\n{impl.upper()}:\n")
                
                for metric in ['gaussian_blur', 'rgb_to_gray', 'sobel_edge', 'total_compute']:
                    if metric in results[img][impl]:
                        data = results[img][impl][metric]
                        f.write(f"  {metric.replace('_', ' ').title()}: "
                               f"{data['mean']:.6f} ± {data['std']:.6f} seconds\n")
                
                # Speedup
                if impl != 'sequential' and 'total_compute' in results[img][impl]:
                    if 'sequential' in results[img] and 'total_compute' in results[img]['sequential']:
                        seq_time = results[img]['sequential']['total_compute']['mean']
                        impl_time = results[img][impl]['total_compute']['mean']
                        if impl_time > 0:
                            speedup = seq_time / impl_time
                            f.write(f"  Speedup vs Sequential: {speedup:.2f}x\n")
        
        # Best Implementation Analysis
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("BEST IMPLEMENTATION ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        best_per_image = {}
        for img in images:
            best_time = float('inf')
            best_impl = None
            
            for impl in implementations:
                if impl in results[img] and 'total_compute' in results[img][impl]:
                    time = results[img][impl]['total_compute']['mean']
                    if time < best_time and time > 0:
                        best_time = time
                        best_impl = impl
            
            best_per_image[img] = best_impl
            f.write(f"Image {img}: Best = {best_impl.upper()} ({best_time:.6f}s)\n")
        
        # Count victories
        f.write("\n\nImplementation Victory Count:\n")
        victory_count = defaultdict(int)
        for impl in best_per_image.values():
            victory_count[impl] += 1
        
        for impl, count in sorted(victory_count.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {impl.upper()}: {count}/{len(images)} images\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"\nDetailed report saved to: {report_file}")

def main():
    results_dir = "results"
    
    print("Analyzing benchmark results...")
    print("=" * 80)
    
    # Parse all results
    results, images, implementations = analyze_all_results(results_dir)
    
    if not results:
        print("No results found! Please run benchmarks first.")
        return
    
    print(f"Found results for {len(images)} images and {len(implementations)} implementations")
    print(f"Images: {', '.join(images)}")
    print(f"Implementations: {', '.join(implementations)}")
    print()
    
    # Generate visualizations
    print("Generating performance charts...")
    create_comparison_charts(results, images, implementations, results_dir)
    print("✓ Charts saved:")
    print("  - compute_time_comparison.png")
    print("  - speedup_analysis.png")
    print("  - operation_breakdown.png")
    print("  - average_performance.png")
    print()
    
    # Generate text report
    print("Generating detailed report...")
    generate_report(results, images, implementations, results_dir)
    print("✓ Report saved: benchmark_report.txt")
    print()
    
    # Quick summary
    print("=" * 80)
    print("QUICK SUMMARY")
    print("=" * 80)
    
    for impl in implementations:
        all_times = []
        for img in images:
            if impl in results[img] and 'total_compute' in results[img][impl]:
                all_times.extend(results[img][impl]['total_compute']['values'])
        
        if all_times:
            mean_time = np.mean(all_times)
            print(f"{impl.upper():12s}: {mean_time:.6f} seconds (avg)")
    
    print("=" * 80)
    print("\nAnalysis complete! Check the 'results' directory for detailed charts and report.")

if __name__ == "__main__":
    main()
