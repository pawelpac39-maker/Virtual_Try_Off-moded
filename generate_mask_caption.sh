#!/bin/bash

# Script to process all input images with segmentation and captioning

# Set the input directory
INPUT_DIR="INPUT"

# Check if INPUT directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: INPUT directory not found!"
    exit 1
fi

# Check if virtual environment exists
if [ ! -f ".venv/bin/python" ]; then
    echo "Error: Virtual environment not found at .venv/bin/python"
    exit 1
fi

# Process each .jpg file in INPUT directory (excluding mask and caption files)
for img_file in "$INPUT_DIR"/*.jpg; do
    # Skip if no jpg files found
    [ -e "$img_file" ] || continue
    
    # Get the basename without extension
    filename=$(basename "$img_file" .jpg)
    
    # Skip files that are already masks or processed files
    if [[ "$filename" == *"_binary_mask"* ]] || [[ "$filename" == *"_fine_mask"* ]]; then
        continue
    fi
    
    echo "================================================"
    echo "Processing: $img_file"
    echo "================================================"
    
    # Step 1: Segmentation - generate binary and fine masks
    echo "Step 1: Running segmentation..."
    .venv/bin/python -c "
from PIL import Image
from SegCloth import segment_clothing
img = Image.open('$img_file')
binary_mask, fine_mask = segment_clothing(img, category='upper_body')
binary_mask.save('$INPUT_DIR/${filename}_binary_mask.jpg')
fine_mask.save('$INPUT_DIR/${filename}_fine_mask.jpg')
"
    
    if [ $? -eq 0 ]; then
        echo "✓ Segmentation completed for $filename"
    else
        echo "✗ Segmentation failed for $filename"
        continue
    fi
    
    # Step 2: Captioning - generate caption text
    echo "Step 2: Running captioning..."
    .venv/bin/python precompute_utils/captioning_qwen.py \
        --pretrained_model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
        --image_path "$img_file" \
        --output_path "$INPUT_DIR/${filename}_caption.txt" \
        --image_category upper_body
    
    if [ $? -eq 0 ]; then
        echo "✓ Captioning completed for $filename"
    else
        echo "✗ Captioning failed for $filename"
    fi
    
    echo ""
done

echo "================================================"
echo "Processing completed!"
echo "================================================"
