from .config import SHORT_HIT , LONG_HIT , INTERMEDIATE_HIT , SHORT_HIT_tick, LONG_HIT_tick

def process_measurements(measurements,threshold=200,time_tick=0.05):
    short_hit = SHORT_HIT_tick #int(SHORT_HIT/time_tick)
    long_hit = LONG_HIT_tick #int(LONG_HIT/time_tick)
    # Step 1: Shift all times by +1.6 µs
    measurements = [(t + short_hit, value) for t, value in measurements if t>=0]

    blocks = []  # To hold the blocks
    current_block = []  # Temporary storage for the current block

    for i, (t, value) in enumerate(measurements):
        if not current_block:
            # Start a new block
            current_block.append((t, value))
        else:
            # Check if the measurement belongs to the current block
            if t - current_block[-1][0] <= long_hit and current_block[-1][1]!=0:
                current_block.append((t, value))
            else:
                # Finalize the current block
                t_first = current_block[0][0]
                t_last = current_block[-1][0]
                current_block.insert(0, (t_first - short_hit, threshold))  # Add the pre-block measurement
                current_block.append((t_last + long_hit, 0))  # Add the post-block measurement
                blocks.append(current_block)  # Save the block
                current_block = [(t, value)]  # Start a new block

    # Add the last block if it exists
    if current_block:
        t_first = current_block[0][0]
        t_last = current_block[-1][0]
        current_block.insert(0, (t_first - short_hit, threshold))  # Add the pre-block measurement
        current_block.append((t_last + long_hit, 0))  # Add the post-block measurement
        blocks.append(current_block)

    return blocks


def extend_measurements(measurements,threshold=200,time_tick=0.05):
    short_hit = SHORT_HIT_tick #int(SHORT_HIT/time_tick)
    long_hit = LONG_HIT_tick #int(LONG_HIT/time_tick)
    from collections import defaultdict
    pixel_data = defaultdict(list)
    
    for pixelID, time, value in measurements:
        if time<0: continue
        pixel_data[pixelID].append((time, value))

    new_measurements = []

    for pixelID, data in pixel_data.items():
        prev_time = None  # Track previous shifted time

        for i, (time, value) in enumerate(data):
            new_time = time + short_hit

            # Check if we need to add a threshold measurement 16 prior
            #if prev_time is None or abs(new_time  - prev_time) > long_hit:
            #    new_measurements.append((pixelID, new_time - short_hit, threshold))

            # Add the actual measurement
            new_measurements.append((pixelID, new_time, value))

            # Check if we need to add a zero measurement 28 after NOTE: Note really neede right now, but can be in a future
            #if i == len(data) - 1 or data[i + 1][0] + short_hit - new_time > long_hit:
            #    new_measurements.append((pixelID, min(new_time + int(1.1/time_tick),12000), 0))

            prev_time = new_time  # Update previous time

    return new_measurements




def define_signals(measurements,kernel_len,threshold=200,time_tick=0.05):
    short_hit = SHORT_HIT_tick #int(SHORT_HIT/time_tick)
    long_hit = LONG_HIT_tick #int(LONG_HIT/time_tick)
    from collections import defaultdict
    pixel_data = defaultdict(list)
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
                delta_t=short_hit
            if value!=threshold and value!=0:
                t_st+=delta_t
                delta_t=long_hit
    return signals
                

def define_signals_simple(measurements,threshold=200,time_tick=0.05,dead_time=2):
    short_hit = SHORT_HIT_tick #int(SHORT_HIT/time_tick)
    long_hit = LONG_HIT_tick #int(LONG_HIT/time_tick)
    from collections import defaultdict
    pixel_data = defaultdict(list)
    for pixelID, time, value in sorted(measurements, key=lambda x: x[1]):
        if value==0 or value==threshold : continue
        pixel_data[pixelID].append((time, value))
    for pixel in pixel_data:
        pixel_data[pixel].sort()
    signals = []
    ID=0
    for pixelID, data in pixel_data.items():
        last_time = None
        for i, (time, value) in enumerate(data):
            if last_time is None or time-last_time>long_hit:
                delta_t=short_hit
                tr=False
            else:
                tr=False
                delta_t=long_hit
            if pixelID==(84,2):
                print((time,value),last_time,delta_t,long_hit)
            signals.append((ID,pixelID,1,time-delta_t,delta_t,tr))
            ID+=1
            last_time=time
    return signals
                
            
