"""
Generate HVAC data similar to the reference image
- Date range: Jan 13-19, 2026 (7 days)
- 3 HVAC units with oscillating temperature patterns
- Temperature range: ~40-65°F
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from hvac_data_gen import HVACDataGenerator, visualize_container

# Initialize generator with a seed for reproducibility
generator = HVACDataGenerator(seed=42)

# Generate data for one container with 3 HVAC units
# Matching the date range from the image: Jan 13-19, 2026
start_date = datetime(2026, 1, 13)
duration_days = 7

print("Generating HVAC data similar to the reference image...")
print(f"Date range: Jan 13-19, 2026 ({duration_days} days)")
print(f"HVAC units: 3 (units 0, 1, 2)")

# Generate normal data (no anomalies to match the image)
df = generator.generate_container_data(
    container_id=0,
    start_time=start_date,
    duration_days=duration_days,
    anomaly_config=None  # No anomalies
)

print(f"\nGenerated {len(df)} total records")
print(f"Records per unit: {len(df) // 3}")
print(f"Temperature range: {df['TmpRet'].min():.1f}°F - {df['TmpRet'].max():.1f}°F")

# Display sample data
print("\nFirst few records:")
print(df.head(15))

print("\nLast few records:")
print(df.tail(15))

# Create a visualization matching the reference image style
fig, ax = plt.subplots(figsize=(14, 6))

colors = ['#5470C6', '#EE6666', '#5DBCD2']  # Blue, Red, Cyan
labels = ['0', '1', '2']

for unit in range(3):
    unit_data = df[df['HVACNum'] == unit].sort_values('timestamp_et')
    ax.plot(unit_data['timestamp_et'], unit_data['TmpRet'],
           color=colors[unit], label=labels[unit], linewidth=1.5, alpha=0.9)

# Add threshold line at 35°F (matching the image)
ax.axhline(y=35, color='red', linestyle='--', linewidth=2, alpha=0.6)

# Formatting
ax.set_xlabel('timestamp_et', fontsize=12)
ax.set_ylabel('TmpRet', fontsize=12)
ax.set_title('HVAC Temperature Data', fontsize=13, fontweight='bold')
ax.legend(title='HVACNum', loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

# Format x-axis to show dates nicely
import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%Y'))
ax.xaxis.set_major_locator(mdates.DayLocator())

plt.tight_layout()

# Save the plot
save_path = '/Users/shelleygoel/Code/01_statistical_mod_blog/TSB-AD/explorations/hvac_generated_plot.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {save_path}")

# Also save the data
csv_path = '/Users/shelleygoel/Code/01_statistical_mod_blog/TSB-AD/explorations/hvac_generated_data.csv'
df.to_csv(csv_path, index=False)
print(f"Data saved to: {csv_path}")

plt.show()

print("\nGeneration complete!")
