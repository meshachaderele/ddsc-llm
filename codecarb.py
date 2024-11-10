from codecarbon import EmissionsTracker

# Initialize the tracker
tracker = EmissionsTracker()

# Start tracking
tracker.start()

# Run your task
# Your code here (e.g., a function, loop, or entire script)
for i in range(1000000):
    x = i ** 2  # Example computation

# Stop tracking and get emissions
emissions = tracker.stop()

# Access energy consumption in kWh
#energy_consumption_kwh = emissions["energy_consumed"]  # in kWh
energy_consumption_kwh = tracker._total_energy.kWh  # Total energy in kWh


print(f"Energy consumed: {energy_consumption_kwh} kWh")
