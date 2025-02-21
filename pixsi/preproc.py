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
            if t - current_block[-1][0] <= 28 and current_block[-1][1]!=0:
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


def extend_measurements(measurements,threshold=200):
    from collections import defaultdict
    pixel_data = defaultdict(list)
    
    for pixelID, time, value in measurements:
        pixel_data[pixelID].append((time, value))

    new_measurements = []

    for pixelID, data in pixel_data.items():
        prev_time = None  # Track previous shifted time

        for i, (time, value) in enumerate(data):
            new_time = time + 16

            # Check if we need to add a threshold measurement 16 prior
            if prev_time is None or abs(new_time  - prev_time) > 28:
                new_measurements.append((pixelID, new_time - 16, threshold))

            # Add the actual measurement
            new_measurements.append((pixelID, new_time, value))

            # Check if we need to add a zero measurement 28 after
            if i == len(data) - 1 or data[i + 1][0] + 16 - new_time > 28:
                new_measurements.append((pixelID, min(new_time + 11,1599), 0))

            prev_time = new_time  # Update previous time

    return new_measurements




def define_signals(measurements,kernel_len,threshold=200):
    from collections import defaultdict
    pixel_data = defaultdict(list)
    maxtime=1600
    for pixelID, time, value in measurements:
        pixel_data[pixelID].append((time, value))
    signals = []
    ID=0
    for pixelID, data in pixel_data.items():
        t_st=0
        delta_t=0
        block=0
        tr=True
        for i, (time, value) in enumerate(data):
            if value==0:
                continue
            if value==threshold:
                tr=True
            else:
                tr=False
            signals.append((ID,pixelID,1,max(t_st,time),time+kernel_len-max(t_st,time),tr))
            ID+=1
            if value==threshold:
                t_st=time+kernel_len
                delta_t=16
            if value!=threshold and value!=0:
                t_st+=delta_t
                delta_t=28
    return signals
                

def define_signals_simple(measurements,kernel_len,threshold=200):
    from collections import defaultdict
    pixel_data = defaultdict(list)
    maxtime=1600
    for pixelID, time, value in measurements:
        pixel_data[pixelID].append((time, value))
    signals = []
    ID=0
    for pixelID, data in pixel_data.items():
        delta_t=16
        tr=True
        for i, (time, value) in enumerate(data):
            if value==0:
                continue
            if value==threshold:
                tr=True
                delta_t=16
                continue
            else:
                tr=False
            signals.append((ID,pixelID,1,time-delta_t,delta_t,tr))
            ID+=1
            delta_t=27
    return signals
                
            
