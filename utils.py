def rescale(value, original_range, target_range):
    original_min, original_max = original_range
    target_min, target_max = target_range
    
    # Handle the special case of 0
    if value == 0 and original_min == 0:
        return target_min
    
    # Perform the rescaling
    normalized_value = (value - original_min) / (original_max - original_min)
    rescaled_value = target_min + normalized_value * (target_max - target_min)
    
    return rescaled_value

def rescale_points(points, x_min, x_max, y_min, y_max):
    # Extract x and y values from the points
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]

    # Find the minimum and maximum x and y values
    x_min_val = min(x_values)
    x_max_val = max(x_values)
    y_min_val = min(y_values)
    y_max_val = max(y_values)

    # Scale x and y values to fit within the desired range
    scaled_points = [
        [
            ((point[0] - x_min_val) / (x_max_val - x_min_val)) * (x_max - x_min) + x_min,
            ((point[1] - y_min_val) / (y_max_val - y_min_val)) * (y_max - y_min) + y_min
        ]
        for point in points
    ]

    return scaled_points