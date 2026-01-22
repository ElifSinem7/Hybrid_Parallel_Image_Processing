
OUTPUT_DIR="results"
DATA_DIR="data/DIV2K"
ITERATIONS=5

mkdir -p $OUTPUT_DIR

IMAGES=($(ls $DATA_DIR/*.bmp 2>/dev/null | xargs -n 1 basename))

if [ ${#IMAGES[@]} -eq 0 ]; then
    echo "Error: No BMP images found in $DATA_DIR"
    echo "Please check if images exist:"
    ls -la $DATA_DIR/
    exit 1
fi

echo "================================================"
echo "Rodinia-Style Benchmark for Edge Detection"
echo "Iterations per test: $ITERATIONS"
echo "Images found: ${#IMAGES[@]}"
for img in "${IMAGES[@]}"; do
    echo "  - $img"
done
echo "================================================"

# Function to run benchmark
run_benchmark() {
    local impl=$1
    local input=$2
    local output=$3
    local result_file=$4
    
    echo ""
    echo "Running $impl on $(basename $input)..."
    
    # Clear result file
    > "$result_file"
    
    for i in $(seq 1 $ITERATIONS); do
        echo "  Iteration $i/$ITERATIONS"
        ./bin/$impl "$input" "$output" >> "$result_file" 2>&1
        
        # Check if successful
        if [ $? -ne 0 ]; then
            echo "  ✗ Error in iteration $i"
        else
            echo "  ✓ Iteration $i completed"
        fi
    done
}

# Run benchmarks for each image
total_tests=$((${#IMAGES[@]} * 4))
current_test=0

for img in "${IMAGES[@]}"; do
    input_path="$DATA_DIR/$img"
    img_name=$(basename "$img" .bmp)
    
    echo ""
    echo "========================================"
    echo "Image: $img_name"
    echo "========================================"
    
    # Sequential
    ((current_test++))
    echo "[$current_test/$total_tests] Sequential"
    run_benchmark "sequential" "$input_path" \
        "$OUTPUT_DIR/${img_name}_seq.bmp" \
        "$OUTPUT_DIR/${img_name}_sequential.txt"
    
    # OpenMP
    ((current_test++))
    echo "[$current_test/$total_tests] OpenMP"
    run_benchmark "openmp" "$input_path" \
        "$OUTPUT_DIR/${img_name}_omp.bmp" \
        "$OUTPUT_DIR/${img_name}_openmp.txt"
    
    # CUDA
    ((current_test++))
    echo "[$current_test/$total_tests] CUDA"
    run_benchmark "cuda" "$input_path" \
        "$OUTPUT_DIR/${img_name}_cuda.bmp" \
        "$OUTPUT_DIR/${img_name}_cuda.txt"
    
    # Hybrid
    ((current_test++))
    echo "[$current_test/$total_tests] Hybrid"
    run_benchmark "hybrid" "$input_path" \
        "$OUTPUT_DIR/${img_name}_hybrid.bmp" \
        "$OUTPUT_DIR/${img_name}_hybrid.txt"
done

echo ""
echo "================================================"
echo "Benchmark completed!"
echo "Total tests run: $total_tests"
echo "Results saved in $OUTPUT_DIR"
echo "================================================"
echo ""
echo "Analyzing results..."
python3 benchmarks/analyze_results.py

echo ""
echo "Done! Check results/ directory for:"
echo "  - *.png (performance charts)"
echo "  - benchmark_report.txt (detailed report)"
echo "  - *_seq.bmp, *_omp.bmp, *_cuda.bmp, *_hybrid.bmp (processed images)"
