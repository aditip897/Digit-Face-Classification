
def load_images(file_path, image_type='digit'):
    
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except Exception as e:
        print(f"file parsing err {file_path}: {e}")
        return []

    images = []
    if image_type == 'digit':
        height, width = 28, 28
    else:  
        height, width = 70, 60  
    
    total_lines = len(lines)
    print(f"lines read number: {file_path}: {total_lines}")
    
    expected_images = total_lines // height
    print(f"expected number: {image_type} imgs: {expected_images}")
    
    for i in range(0, total_lines, height):
        if i + height > total_lines:
            break
            
        current_image = []
        valid_image = True
        
        for j in range(height):
            if i + j >= total_lines:
                valid_image = False
                break
                
            line = lines[i + j].rstrip('\n')
            
            if image_type == 'digit':
                row = [2 if c == '#' else 1 if c == '+' else 0 for c in line[:width]]
            else:  
                if not line:
                    line = ' ' * width
                row = [2 if c == '#' else 0 for c in line[:width]]
            
            while len(row) < width:
                row.append(0)
            
            current_image.append(row)
        
        if valid_image and len(current_image) == height:
            images.append(current_image)
    
    print(f"number of image loaded: {len(images)} {image_type} from {file_path}")
    
    if len(images) != expected_images:
        print(f"error statement: {expected_images} images, expected {len(images)}")  
    return images

def load_labels(file_path):
    try:
        with open(file_path, 'r') as file:
            labels = [int(line.strip()) for line in file if line.strip().isdigit()]
        print(f"loaded {len(labels)} labels from {file_path}")
        return labels
    except Exception as e:
        print(f"err statement: with reading labels from {file_path}: {e}")
        return []