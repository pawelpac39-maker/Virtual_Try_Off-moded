.PHONY: install masks captions clean inference help

# Default target
help:
	@echo "Available targets:"
	@echo "  make install    - Create virtual environment and install dependencies"
	@echo "  make masks      - Generate clothing masks for all images in INPUT/"
	@echo "  make captions   - Generate captions for all images in INPUT/"
	@echo "  make clean      - Remove all generated masks and captions from INPUT/"
	@echo "  make inference  - Run inference on all processed images in INPUT/"
	@echo "  make all        - Run complete pipeline (masks -> captions -> inference)"

# Install dependencies
install:
	@echo "Creating virtual environment..."
	uv venv
	@echo "Installing dependencies..."
	uv pip install -r requirements.txt
	@echo "✓ Installation completed!"

# Generate masks using the recognition + segmentation script
masks:
	@echo "Generating clothing masks..."
	@bash generate_mask_caption.sh
	@echo "✓ Masks generation completed!"

# Generate captions only (assumes masks already exist)
captions:
	@echo "Generating captions for INPUT images..."
	@for img in INPUT/*_U.jpg INPUT/*_L.jpg; do \
		if [ -f "$$img" ]; then \
			filename=$$(basename "$$img" .jpg); \
			category="upper_body"; \
			if [[ "$$filename" == *"_L" ]]; then \
				category="lower_body"; \
			fi; \
			echo "Processing $$img with category: $$category"; \
			.venv/bin/python precompute_utils/captioning_qwen.py \
				--pretrained_model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
				--image_path "$$img" \
				--output_path "INPUT/$${filename}_caption.txt" \
				--image_category "$$category"; \
		fi; \
	done
	@echo "✓ Captions generation completed!"

# Clean generated files
clean:
	@echo "Cleaning generated files from INPUT/..."
	@rm -f INPUT/*_mask.jpg INPUT/*_binary_mask.jpg INPUT/*_fine_mask.jpg INPUT/*.txt
	@echo "✓ Cleanup completed!"

# Run inference on all processed images
inference:
	@echo "Running inference on all INPUT images..."
	@mkdir -p outputs
	@for img in INPUT/*_U.jpg INPUT/*_L.jpg; do \
		if [ -f "$$img" ]; then \
			filename=$$(basename "$$img" .jpg); \
			category="upper_body"; \
			if [[ "$$filename" == *"_L" ]]; then \
				category="lower_body"; \
			fi; \
			echo "================================================"; \
			echo "Running inference for: $$img"; \
			echo "Category: $$category"; \
			echo "================================================"; \
			.venv/bin/python inference.py \
				--pretrained_model_name_or_path "stabilityai/stable-diffusion-3-medium-diffusers" \
				--pretrained_model_name_or_path_sd3_tryoff "davidelobba/TEMU-VTOFF" \
				--example_image "$$img" \
				--output_dir outputs \
				--width 768 \
				--height 1024 \
				--guidance_scale 2.0 \
				--num_inference_steps 28 \
				--category "$$category"; \
			if [ $$? -eq 0 ]; then \
				echo "✓ Inference completed for $$filename"; \
			else \
				echo "✗ Inference failed for $$filename"; \
			fi; \
			echo ""; \
		fi; \
	done
	@echo "================================================"
	@echo "✓ All inference tasks completed!"
	@echo "================================================"

# Complete pipeline
all: masks captions inference
	@echo "✓ Complete pipeline finished!"
