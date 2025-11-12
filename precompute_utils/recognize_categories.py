import mediapipe as mp
import cv2
import os
import sys
import argparse

def classify_clothing(image_path):
    """Classify if image shows upper_body or lower_body clothing."""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}", file=sys.stderr)
        return None
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image: {image_path}", file=sys.stderr)
        return None
    
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pose.close()
    
    if not results.pose_landmarks:
        print(f"Warning: No pose landmarks detected in {image_path}", file=sys.stderr)
        return "upper_body"  # Default to upper_body if no landmarks
    
    landmarks = results.pose_landmarks.landmark
    
    # Count visible upper body landmarks (11-16: shoulders, elbows, wrists)
    upper_body_visible = sum(1 for i in range(11, 17)
                            if landmarks[i].visibility > 0.5)
    
    # Count visible lower body landmarks (23-32: hips, knees, ankles, feet)
    lower_body_visible = sum(1 for i in range(23, 33)
                            if landmarks[i].visibility > 0.5)
    
    classification = "upper_body" if upper_body_visible > lower_body_visible else "lower_body"
    
    return classification

def rename_with_category(image_path):
    """Rename image file with category postfix (_U or _L)."""
    category = classify_clothing(image_path)
    
    if category is None:
        return None
    
    dir_name = os.path.dirname(image_path)
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    
    # Skip if already has postfix
    if name.endswith('_U') or name.endswith('_L'):
        return image_path
    
    postfix = '_U' if category == 'upper_body' else '_L'
    new_name = f"{name}{postfix}{ext}"
    new_path = os.path.join(dir_name, new_name)
    
    try:
        os.rename(image_path, new_path)
        print(f"Renamed: {base_name} -> {new_name} ({category})")
        return new_path
    except Exception as e:
        print(f"Error renaming {image_path}: {e}", file=sys.stderr)
        return None

def main():
    parser = argparse.ArgumentParser(description='Classify clothing images as upper or lower body')
    parser.add_argument('--input_dir', type=str, default='INPUT', 
                       help='Directory containing images to classify')
    parser.add_argument('--image_path', type=str, 
                       help='Single image to classify (optional)')
    
    args = parser.parse_args()
    
    if args.image_path:
        # Process single image
        result = rename_with_category(args.image_path)
        if result:
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        # Process all images in directory
        if not os.path.exists(args.input_dir):
            print(f"Error: Directory not found: {args.input_dir}", file=sys.stderr)
            sys.exit(1)
        
        image_files = [f for f in os.listdir(args.input_dir) 
                      if f.lower().endswith('.jpg') and 
                      not any(x in f for x in ['_binary_mask', '_fine_mask', '_U', '_L'])]
        
        if not image_files:
            print(f"No images to process in {args.input_dir}")
            sys.exit(0)
        
        print(f"Found {len(image_files)} images to classify\n")
        
        success_count = 0
        for img_file in image_files:
            img_path = os.path.join(args.input_dir, img_file)
            result = rename_with_category(img_path)
            if result:
                success_count += 1
        
        print(f"\nClassification completed: {success_count}/{len(image_files)} successful")

if __name__ == '__main__':
    main()
