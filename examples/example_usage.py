from signal_processing import kernel, current, trigger

# Example usage
k = kernel()
c = current(1, 20)
meas, actual = trigger(c)
print("Measurements:", meas)
