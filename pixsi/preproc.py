def process_measurements(measurements,threshold=200):
    # Step 1: Shift all times by +1.6 µs
    measurements = [(t + 16, value) for t, value in measurements]

    blocks = []  # To hold the blocks
    current_block = []  # Temporary storage for the current block

    for i, (t, value) in enumerate(measurements):
        if not current_block:
            # Start a new block
            current_block.append((t, value))
        else:
            # Check if the measurement belongs to the current block
            if t - current_block[-1][0] <= 29:
                current_block.append((t, value))
            else:
                # Finalize the current block
                t_first = current_block[0][0]
                t_last = current_block[-1][0]
                current_block.insert(0, (t_first - 16, threshold))  # Add the pre-block measurement
                current_block.append((t_last + 28, 0))  # Add the post-block measurement
                blocks.append(current_block)  # Save the block
                current_block = [(t, value)]  # Start a new block

    # Add the last block if it exists
    if current_block:
        t_first = current_block[0][0]
        t_last = current_block[-1][0]
        current_block.insert(0, (t_first - 16, threshold))  # Add the pre-block measurement
        current_block.append((t_last + 28, 0))  # Add the post-block measurement
        blocks.append(current_block)

    return blocks
