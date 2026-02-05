"""
HVAC Synthetic Data Generator for BESS Containers
Generates realistic time-series data for multiple HVAC units with normal and anomalous behaviors
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
import random


class HVACDataGenerator:
    """Generate synthetic HVAC temperature data for BESS containers"""
    
    def __init__(self, seed: int = 42):
        """
        Initialize the generator
        
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        random.seed(seed)
        
        # Base parameters for normal operation
        self.base_temp = 50  # Base temperature
        self.temp_range = 15  # Temperature variation range
        self.charge_cycle_hours = 4  # Hours per charge cycle
        self.noise_std = 0.1  # Standard deviation of noise
        
    def _generate_cycle_timings(self,
                                duration_days: int,
                                cycles_per_day: int = 4) -> List[Tuple[int, int, float, float]]:
        """
        Generate cycle timing metadata (start, end, peak intensity, baseline offset)

        Args:
            duration_days: Number of days to simulate
            cycles_per_day: Number of charge cycles per day

        Returns:
            List of tuples: (cycle_start_minute, cycle_end_minute, peak_intensity, baseline_offset)
        """
        minutes_per_day = 24 * 60
        cycle_timings = []

        # Daily variability in cycle characteristics
        for day in range(duration_days):
            # Randomize cycles per day (3-5 cycles)
            day_cycles = cycles_per_day + np.random.randint(-1, 2)
            day_cycles = max(3, min(5, day_cycles))

            # Calculate cycle spacing for this day
            day_start = day * minutes_per_day
            day_end = (day + 1) * minutes_per_day

            # Randomize baseline temperature for the day
            day_baseline_offset = np.random.uniform(-0.15, 0.15)

            # Add some randomness to cycle start times
            for c in range(day_cycles):
                base_start = day_start + (c * minutes_per_day / day_cycles)
                jitter = np.random.randint(-20, 20)  # +/- 20 minutes
                cycle_start = int(base_start + jitter)

                # Ensure cycle starts within the day
                if cycle_start < day_start or cycle_start >= day_end - 20:
                    continue

                # Variable cycle duration (180-360 minutes)
                cycle_duration = np.random.randint(180, 3600)
                cycle_end = min(cycle_start + cycle_duration, day_end)

                if cycle_end <= cycle_start or (cycle_end - cycle_start) < 20:
                    continue

                # Variable peak intensity per cycle
                peak_intensity = np.random.uniform(0.85, 1.0)

                cycle_timings.append((cycle_start, cycle_end, peak_intensity, day_baseline_offset))

        return cycle_timings

    def generate_charge_cycles(self,
                               start_time: datetime,
                               duration_days: int,
                               cycles_per_day: int = 4,
                               unit_offset: int = 0,
                               cycle_timings: List[Tuple[int, int, float, float]] = None) -> np.ndarray:
        """
        Generate battery charge cycle pattern with sharp rise and gradual decline

        Args:
            start_time: Start datetime
            duration_days: Number of days to simulate
            cycles_per_day: Number of charge cycles per day
            unit_offset: Time offset in minutes to apply to each cycle (for unit-specific jitter)
            cycle_timings: Pre-computed cycle timings to use (if None, will generate new ones)

        Returns:
            Array of charge intensity values (0 to 1)
        """
        minutes_per_day = 24 * 60
        total_minutes = duration_days * minutes_per_day
        charge_pattern = np.zeros(total_minutes)

        # Generate or use provided cycle timings
        if cycle_timings is None:
            cycle_timings = self._generate_cycle_timings(duration_days, cycles_per_day)

        # Build charge pattern from cycle timings with unit offset
        for cycle_start, cycle_end, peak_intensity, day_baseline_offset in cycle_timings:
            # Apply unit-specific offset to this cycle
            offset_start = cycle_start + unit_offset
            offset_end = cycle_end + unit_offset

            # Skip cycles that start out of bounds
            if offset_start < 0 or offset_start >= total_minutes:
                continue

            # Clip cycles that extend past the end
            if offset_end > total_minutes:
                offset_end = total_minutes

            cycle_len = offset_end - offset_start
            if cycle_len < 20:
                continue

            # Very sharp rise: only 5-8% of cycle (like battery starts charging)
            # Use deterministic calculation based on cycle_len for consistent pattern
            rise_fraction = 0.065  # Middle of 0.05-0.08 range
            rise_len = int(cycle_len * rise_fraction)
            rise_len = max(5, min(rise_len, cycle_len - 5))  # At least 5 minutes rise, leave 5 for decline
            # Very gradual decline: rest of cycle
            decline_len = cycle_len - rise_len

            if decline_len <= 0:  # Safety check
                continue

            # Nearly vertical rise (very steep power curve)
            rise_curve = np.power(np.linspace(0, 1, rise_len), 0.2)
            charge_pattern[offset_start:offset_start+rise_len] = rise_curve * peak_intensity

            # Very gradual exponential decline (slow decay)
            decline_curve = np.power(np.linspace(1, 0, decline_len), 3.5)
            charge_pattern[offset_start+rise_len:offset_end] = decline_curve * peak_intensity

            # Add daily baseline offset
            charge_pattern[offset_start:offset_end] += day_baseline_offset
        
        # Ensure values stay in [0, 1]
        charge_pattern = np.clip(charge_pattern, 0, 1)
        
        return charge_pattern
    
    def generate_normal_unit(self,
                            start_time: datetime,
                            duration_days: int,
                            unit_id: int,
                            charge_pattern: np.ndarray,
                            daily_offsets: np.array,
                            unit_offset: float = 0.0) -> pd.DataFrame:
        """
        Generate data for a normally operating HVAC unit

        Args:
            start_time: Start datetime
            duration_days: Number of days
            unit_id: Unit identifier (0, 1, 2)
            charge_pattern: Charge cycle pattern
            daily_offsets: Daily temperature offsets
            unit_offset: Unit-specific temperature offset

        Returns:
            DataFrame with timestamp and temperature
        """
        total_minutes = len(charge_pattern)

        # Generate timestamps
        timestamps = [start_time + timedelta(minutes=i) for i in range(total_minutes)]

        # Base temperature follows charge pattern with unit-specific offset
        temperature = self.base_temp + unit_offset + charge_pattern * self.temp_range
        
        # Add day-to-day variation in minimum temperature
        minutes_per_day = 24 * 60
        for day in range(duration_days):
            day_start = day * minutes_per_day
            day_end = min((day + 1) * minutes_per_day, total_minutes)
            
            # Random offset for minimum temperature this day (-3 to +3 degrees)
            temperature[day_start:day_end] += daily_offsets[day]

        # Add realistic noise
        noise = np.random.normal(0, self.noise_std, total_minutes)
        temperature += noise
        
        df = pd.DataFrame({
            'timestamp_et': timestamps,
            'HVACNum': unit_id,
            'TmpRet': temperature,
            'anomaly': False,
            'anomaly_type': 'normal'
        })
        
        return df
    
    def inject_anomaly_lag(self,
                      df: pd.DataFrame,
                      unit_indices: pd.Index,
                      start_minutes: int,
                      duration_minutes: int,
                      lag_minutes: int) -> pd.DataFrame:
        """
        Inject lag anomaly for a specific unit

        Args:
            df: DataFrame with all units
            unit_indices: Indices for the specific unit being modified
            start_minutes: Minute offset from start of data
            duration_minutes: Duration of anomaly in minutes
            lag_minutes: How many minutes to lag

        Returns:
            Modified DataFrame
        """
        # Calculate end minute index
        end_minutes = min(start_minutes + duration_minutes, len(unit_indices))

        # Get the actual row indices for this unit's data
        anomaly_indices = unit_indices[start_minutes:end_minutes]

        # Copy lagged temperatures from earlier in the same unit's data
        if start_minutes >= lag_minutes:
            lag_start = start_minutes - lag_minutes
            lag_end = end_minutes - lag_minutes
            lag_indices = unit_indices[lag_start:lag_end]

            if len(lag_indices) == len(anomaly_indices):
                lagged_temps = df.loc[lag_indices, 'TmpRet'].values
                df.loc[anomaly_indices, 'TmpRet'] = lagged_temps

        df.loc[anomaly_indices, 'anomaly'] = True
        df.loc[anomaly_indices, 'anomaly_type'] = 'lag'
        return df

    def inject_anomaly_frequency(self,
                                 df: pd.DataFrame,
                                 unit_indices: pd.Index,
                                 start_minutes: int,
                                 duration_minutes: int,
                                 frequency_multiplier: float) -> pd.DataFrame:
        """
        Inject anomaly by changing the cycle frequency (compress or expand cycles)

        Args:
            df: DataFrame with all units
            unit_indices: Indices for the specific unit being modified
            start_minutes: Minute offset from start of data
            duration_minutes: Duration of anomaly in minutes
            frequency_multiplier: Factor to change frequency (>1 = faster cycles, <1 = slower cycles)

        Returns:
            Modified DataFrame
        """
        # Calculate end minute index
        end_minutes = min(start_minutes + duration_minutes, len(unit_indices))

        # Get the actual row indices for this unit's data
        anomaly_indices = unit_indices[start_minutes:end_minutes]

        if len(anomaly_indices) == 0:
            return df

        # Extract the temperature segment for this unit only
        temp_segment = df.loc[anomaly_indices, 'TmpRet'].values

        # Get mean temperature for the segment
        mean_temp = temp_segment.mean()

        # Detrend: subtract mean
        detrended = temp_segment - mean_temp

        # Compress or expand the signal by resampling
        original_length = len(detrended)

        # Create new time indices for resampling
        original_indices = np.arange(original_length)

        # For frequency_multiplier > 1: compress signal (more cycles)
        # For frequency_multiplier < 1: expand signal (fewer cycles)
        new_indices = np.linspace(0, original_length - 1, int(original_length / frequency_multiplier))

        # Interpolate to get resampled signal
        resampled = np.interp(new_indices, original_indices, detrended)

        # Tile or truncate to match original length
        if len(resampled) >= original_length:
            # Truncate
            modified_segment = resampled[:original_length]
        else:
            # Tile to fill the duration
            num_repeats = int(np.ceil(original_length / len(resampled)))
            tiled = np.tile(resampled, num_repeats)
            modified_segment = tiled[:original_length]

        # Add mean back
        modified_segment = modified_segment + mean_temp

        # Update the DataFrame for this unit only
        df.loc[anomaly_indices, 'TmpRet'] = modified_segment
        df.loc[anomaly_indices, 'anomaly'] = True
        df.loc[anomaly_indices, 'anomaly_type'] = 'frequency'

        return df
    
    
    
    def generate_container_data(self,
                               container_id: int,
                               start_time: datetime,
                               duration_days: int,
                               anomaly_config: List[Dict] = None) -> pd.DataFrame:
        """
        Generate complete data for one BESS container with 3 HVAC units
        
        Args:
            container_id: Container identifier
            start_time: Start datetime
            duration_days: Number of days to simulate
            anomaly_config: List of anomaly configurations
                           Format: [{'unit': 0, 'type': 'lag', 'start_day': 2, 
                                    'start_hour': 10, 'duration_hours': 3, 'params': {...}}]
        
        Returns:
            DataFrame with all units' data
        """
        # Generate daily baseline offsets once for all units
        daily_offsets = [np.random.uniform(-3, 3) for _ in range(duration_days)]

        # Generate cycle timings once (shared by all units)
        cycle_timings = self._generate_cycle_timings(duration_days)

        # Generate data for all 3 units with unit-specific time offsets
        all_data = []
        for unit_id in range(3):
            # Apply unit-specific time jitter (±2 hours) to each cycle
            if unit_id == 0:
                time_offset = 0
            else:
                time_offset = np.random.randint(-10, 10)  # ±2 hours for units 1 and 2

            # Apply unit-specific temperature offset [-2, 2] degrees
            temp_offset = np.random.uniform(-2, 2)

            # Generate charge pattern with unit-specific time offset applied to each cycle
            charge_pattern = self.generate_charge_cycles(
                start_time, duration_days, unit_offset=time_offset, cycle_timings=cycle_timings)

            df = self.generate_normal_unit(start_time, duration_days, unit_id, charge_pattern, daily_offsets, temp_offset)
            df['container_id'] = container_id
            all_data.append(df)
        
        # Combine all units
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Inject anomalies if specified
        if anomaly_config:
            for anomaly in anomaly_config:
                unit = anomaly['unit']
                anom_type = anomaly['type']
                start_day = anomaly.get('start_day', 0)
                start_hour = anomaly.get('start_hour', 0)
                duration_hours = anomaly.get('duration_hours', 1)
                params = anomaly.get('params', {})

                # Calculate start minute offset
                start_minutes = start_day * 24 * 60 + start_hour * 60
                duration_minutes = duration_hours * 60

                # Get indices for the specific unit
                unit_mask = (combined_df['HVACNum'] == unit)
                unit_indices = combined_df[unit_mask].index

                if start_minutes < len(unit_indices):
                    # Apply anomaly to this specific unit only
                    if anom_type == 'lag':
                        lag_min = params.get('lag_minutes', 30)
                        combined_df = self.inject_anomaly_lag(
                            combined_df, unit_indices, start_minutes, duration_minutes, lag_min)
                    elif anom_type == 'frequency':
                        freq_mult = params.get('frequency_multiplier', 2.0)
                        combined_df = self.inject_anomaly_frequency(
                            combined_df, unit_indices, start_minutes, duration_minutes, freq_mult)
        
        return combined_df.sort_values(['timestamp_et', 'HVACNum']).reset_index(drop=True)
    
    def generate_dataset(self,
                        num_containers: int,
                        start_date: str,
                        duration_days: int,
                        anomaly_probability: float = 0.2,
                        anomaly_configs: Dict[int, List[Dict]] = None) -> pd.DataFrame:
        """
        Generate complete dataset with multiple containers
        
        Args:
            num_containers: Number of containers to simulate
            start_date: Start date string (YYYY-MM-DD)
            duration_days: Number of days to simulate
            anomaly_probability: Probability of anomaly per container (if no config provided)
            anomaly_configs: Dictionary mapping container_id to anomaly configurations
            
        Returns:
            Complete DataFrame with all containers
        """
        start_time = datetime.strptime(start_date, '%Y-%m-%d')
        all_containers = []
        
        for container_id in range(num_containers):
            # Use provided config or generate random anomalies
            if anomaly_configs and container_id in anomaly_configs:
                config = anomaly_configs[container_id]
            elif random.random() < anomaly_probability:
                # Generate random anomaly
                config = self._generate_random_anomaly_config(duration_days)
            else:
                config = None
            
            df = self.generate_container_data(
                container_id, start_time, duration_days, config)
            all_containers.append(df)
        
        return pd.concat(all_containers, ignore_index=True)
    
    def _generate_random_anomaly_config(self, duration_days: int) -> List[Dict]:
        """Generate random anomaly configuration"""
        config = [{
            'unit': random.randint(0, 2),
            'type': 'lag',
            'start_day': random.randint(1, max(1, duration_days - 2)),
            'start_hour': random.randint(0, 23),
            'duration_hours': random.randint(2, 12),
            'params': {'lag_minutes': random.randint(20, 60)}
        }]

        return config


# Example usage
if __name__ == "__main__":
    # Initialize generator
    generator = HVACDataGenerator(seed=42)
    
    # Example 1: Generate data for single container with specific anomalies
    print("Generating data for single container with anomalies...")
    
    anomaly_config = [
        {
            'unit': 1,
            'type': 'lag',
            'start_day': 2,
            'start_hour': 8,
            'duration_hours': 6,
            'params': {'lag_minutes': 45}
        },
        {
            'unit': 2,
            'type': 'stuck',
            'start_day': 3,
            'start_hour': 14,
            'duration_hours': 4,
            'params': {'stuck_temp': 45}
        }
    ]
    
    single_container_df = generator.generate_container_data(
        container_id=0,
        start_time=datetime(2026, 1, 15),
        duration_days=5,
        anomaly_config=anomaly_config
    )
    
    print(f"\nGenerated {len(single_container_df)} records")
    print(f"Anomaly records: {single_container_df['anomaly'].sum()}")
    print("\nFirst few records:")
    print(single_container_df.head(10))
    
    # Example 2: Generate multi-container dataset
    print("\n" + "="*70)
    print("Generating multi-container dataset...")
    
    # Define specific anomalies for some containers
    multi_anomaly_configs = {
        0: [{'unit': 1, 'type': 'lag', 'start_day': 3,
             'start_hour': 10, 'duration_hours': 5,
             'params': {'lag_minutes': 45}}],
        2: [{'unit': 0, 'type': 'lag', 'start_day': 4,
             'start_hour': 6, 'duration_hours': 8,
             'params': {'lag_minutes': 60}}],
        4: [{'unit': 2, 'type': 'lag', 'start_day': 2,
             'start_hour': 12, 'duration_hours': 10,
             'params': {'lag_minutes': 30}}]
    }
    
    full_dataset = generator.generate_dataset(
        num_containers=10,
        start_date='2026-01-12',
        duration_days=7,
        anomaly_probability=0.3,
        anomaly_configs=multi_anomaly_configs
    )
    
    print(f"\nGenerated dataset with {len(full_dataset)} total records")
    print(f"Containers: {full_dataset['container_id'].nunique()}")
    print(f"Total anomaly records: {full_dataset['anomaly'].sum()}")
    print(f"Anomaly percentage: {100 * full_dataset['anomaly'].mean():.2f}%")
    
    # Summary by anomaly type
    print("\nAnomaly types distribution:")
    print(full_dataset[full_dataset['anomaly']]['anomaly_type'].value_counts())
    
    # Save dataset
    output_path = '/mnt/user-data/outputs/hvac_synthetic_dataset.csv'
    full_dataset.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")

    print("\nGeneration complete!")