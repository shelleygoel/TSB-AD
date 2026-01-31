"""
HVAC Synthetic Data Generator for BESS Containers
Generates realistic time-series data for multiple HVAC units with normal and anomalous behaviors
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
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
        
    def generate_charge_cycles(self, 
                               start_time: datetime, 
                               duration_days: int,
                               cycles_per_day: int = 4) -> np.ndarray:
        """
        Generate battery charge cycle pattern with sharp rise and gradual decline
        
        Args:
            start_time: Start datetime
            duration_days: Number of days to simulate
            cycles_per_day: Number of charge cycles per day
            
        Returns:
            Array of charge intensity values (0 to 1)
        """
        minutes_per_day = 24 * 60
        total_minutes = duration_days * minutes_per_day
        charge_pattern = np.zeros(total_minutes)
        
        # Daily variability in cycle characteristics
        for day in range(duration_days):
            # Randomize cycles per day (3-5 cycles)
            day_cycles = cycles_per_day + np.random.randint(-1, 2)
            day_cycles = max(3, min(5, day_cycles))
            
            # Randomize baseline temperature for the day
            day_baseline_offset = np.random.uniform(-0.15, 0.15)
            
            # Calculate cycle spacing for this day
            day_start = day * minutes_per_day
            day_end = (day + 1) * minutes_per_day
            
            # Add some randomness to cycle start times
            cycle_starts = []
            for c in range(day_cycles):
                base_start = day_start + (c * minutes_per_day / day_cycles)
                jitter = np.random.randint(-20, 20)  # +/- 20 minutes
                cycle_start = int(base_start + jitter)
                # Ensure cycle starts within the day
                if cycle_start >= day_start and cycle_start < day_end - 20:
                    cycle_starts.append(cycle_start)
            
            for cycle_idx, cycle_start in enumerate(cycle_starts):
                # Variable cycle duration (180-300 minutes)
                cycle_duration = np.random.randint(180, 3600)
                cycle_end = min(cycle_start + cycle_duration, day_end)
                
                if cycle_end <= cycle_start:
                    continue
                
                cycle_len = cycle_end - cycle_start
                
                if cycle_len < 20:  # Skip very short cycles
                    continue
                
                # Very sharp rise: only 5-8% of cycle (like battery starts charging)
                rise_len = int(cycle_len * np.random.uniform(0.05, 0.08))
                rise_len = max(5, min(rise_len, cycle_len - 5))  # At least 5 minutes rise, leave 5 for decline
                # Very gradual decline: rest of cycle
                decline_len = cycle_len - rise_len
                
                if decline_len <= 0:  # Safety check
                    continue
                
                # Variable peak intensity per cycle
                peak_intensity = np.random.uniform(0.85, 1.0)
                
                # Nearly vertical rise (very steep power curve)
                rise_curve = np.power(np.linspace(0, 1, rise_len), 0.2)
                charge_pattern[cycle_start:cycle_start+rise_len] = rise_curve * peak_intensity
                
                # Very gradual exponential decline (slow decay)
                decline_curve = np.power(np.linspace(1, 0, decline_len), 3.5)
                charge_pattern[cycle_start+rise_len:cycle_end] = decline_curve * peak_intensity
                
                # Add daily baseline offset
                charge_pattern[cycle_start:cycle_end] += day_baseline_offset
        
        # Ensure values stay in [0, 1]
        charge_pattern = np.clip(charge_pattern, 0, 1)
        
        return charge_pattern
    
    def generate_normal_unit(self,
                            start_time: datetime,
                            duration_days: int,
                            unit_id: int,
                            charge_pattern: np.ndarray,
                            daily_offsets: np.array) -> pd.DataFrame:
        """
        Generate data for a normally operating HVAC unit
        
        Args:
            start_time: Start datetime
            duration_days: Number of days
            unit_id: Unit identifier (0, 1, 2)
            charge_pattern: Charge cycle pattern
            
        Returns:
            DataFrame with timestamp and temperature
        """
        total_minutes = len(charge_pattern)
        
        # Generate timestamps
        timestamps = [start_time + timedelta(minutes=i) for i in range(total_minutes)]
        
        # Base temperature follows charge pattern
        temperature = self.base_temp + charge_pattern * self.temp_range
        
        # Add day-to-day variation in minimum temperature
        minutes_per_day = 24 * 60
        for day in range(duration_days):
            day_start = day * minutes_per_day
            day_end = min((day + 1) * minutes_per_day, total_minutes)
            
            # Random offset for minimum temperature this day (-3 to +3 degrees)
            temperature[day_start:day_end] += daily_offsets[day]
        
        # Add slight phase offset between units (they're not perfectly synchronized)
        phase_offset = unit_id * 2  # minutes
        if phase_offset > 0:
            temperature = np.roll(temperature, phase_offset)
        
        # Add realistic noise
        noise = np.random.normal(0, self.noise_std, total_minutes)
        temperature += noise
        
        # Add subtle daily ambient temperature variation (smaller than before)
        # hours = np.arange(total_minutes) / 60
        # daily_variation = 1.5 * np.sin(2 * np.pi * hours / 24)
        # temperature += daily_variation
        
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
                      start_idx: int,
                      duration_minutes: int,
                      lag_minutes: int) -> pd.DataFrame:
        end_idx = min(start_idx + duration_minutes, len(df))
        
        # Instead of rolling, copy from earlier in the same unit's data
        if start_idx >= lag_minutes:
            lagged_temps = df.loc[start_idx-lag_minutes:end_idx-lag_minutes-1, 'TmpRet'].values
            if len(lagged_temps) == (end_idx - start_idx):
                df.loc[start_idx:end_idx-1, 'TmpRet'] = lagged_temps
        
        df.loc[start_idx:end_idx-1, 'anomaly'] = True
        df.loc[start_idx:end_idx-1, 'anomaly_type'] = 'lag'
        return df
    
    
    def inject_anomaly_stuck(self,
                            df: pd.DataFrame,
                            start_idx: int,
                            duration_minutes: int,
                            stuck_temp: float = None) -> pd.DataFrame:
        """
        Inject stuck anomaly - unit temperature gets stuck at a value
        
        Args:
            df: DataFrame to modify
            start_idx: Starting index for anomaly
            duration_minutes: Duration of anomaly
            stuck_temp: Temperature to stick at (None for current temp)
            
        Returns:
            Modified DataFrame
        """
        end_idx = min(start_idx + duration_minutes, len(df))
        
        if stuck_temp is None:
            stuck_temp = df.loc[start_idx, 'TmpRet']
        
        # Add small noise to make it look like sensor readings
        noise = np.random.normal(0, 0.3, end_idx - start_idx)
        df.loc[start_idx:end_idx-1, 'TmpRet'] = stuck_temp + noise
        df.loc[start_idx:end_idx-1, 'anomaly'] = True
        df.loc[start_idx:end_idx-1, 'anomaly_type'] = 'stuck'
        
        return df
    
    def inject_anomaly_erratic(self,
                              df: pd.DataFrame,
                              start_idx: int,
                              duration_minutes: int) -> pd.DataFrame:
        """
        Inject erratic anomaly - unit has unstable temperature readings
        
        Args:
            df: DataFrame to modify
            start_idx: Starting index for anomaly
            duration_minutes: Duration of anomaly
            
        Returns:
            Modified DataFrame
        """
        end_idx = min(start_idx + duration_minutes, len(df))
        
        # Add large random fluctuations
        original_temps = df.loc[start_idx:end_idx-1, 'TmpRet'].values
        erratic_noise = np.random.normal(0, 5, end_idx - start_idx)
        
        df.loc[start_idx:end_idx-1, 'TmpRet'] = original_temps + erratic_noise
        df.loc[start_idx:end_idx-1, 'anomaly'] = True
        df.loc[start_idx:end_idx-1, 'anomaly_type'] = 'erratic'
        
        return df
    
    def inject_anomaly_drift(self,
                           df: pd.DataFrame,
                           start_idx: int,
                           duration_minutes: int,
                           drift_rate: float = 0.01) -> pd.DataFrame:
        """
        Inject drift anomaly - unit temperature gradually drifts from normal
        
        Args:
            df: DataFrame to modify
            start_idx: Starting index for anomaly
            duration_minutes: Duration of anomaly
            drift_rate: Rate of drift per minute
            
        Returns:
            Modified DataFrame
        """
        end_idx = min(start_idx + duration_minutes, len(df))
        
        # Add gradual drift
        drift = np.arange(end_idx - start_idx) * drift_rate
        df.loc[start_idx:end_idx-1, 'TmpRet'] += drift
        df.loc[start_idx:end_idx-1, 'anomaly'] = True
        df.loc[start_idx:end_idx-1, 'anomaly_type'] = 'drift'
        
        return df
    
    def inject_anomaly_reduced_cooling(self,
                                      df: pd.DataFrame,
                                      start_idx: int,
                                      duration_minutes: int,
                                      reduction_factor: float = 0.5) -> pd.DataFrame:
        """
        Inject reduced cooling anomaly - unit doesn't cool as effectively
        
        Args:
            df: DataFrame to modify
            start_idx: Starting index for anomaly
            duration_minutes: Duration of anomaly
            reduction_factor: Cooling effectiveness (0-1)
            
        Returns:
            Modified DataFrame
        """
        end_idx = min(start_idx + duration_minutes, len(df))
        
        # Reduce the amplitude of temperature changes
        mean_temp = df.loc[start_idx:end_idx-1, 'TmpRet'].mean()
        df.loc[start_idx:end_idx-1, 'TmpRet'] = \
            mean_temp + (df.loc[start_idx:end_idx-1, 'TmpRet'] - mean_temp) * reduction_factor
        df.loc[start_idx:end_idx-1, 'anomaly'] = True
        df.loc[start_idx:end_idx-1, 'anomaly_type'] = 'reduced_cooling'
        
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
        # Generate charge pattern
        charge_pattern = self.generate_charge_cycles(start_time, duration_days)
           # Generate daily baseline offsets once for all units
        daily_offsets = [np.random.uniform(-3, 3) for _ in range(duration_days)]
        
        # Generate data for all 3 units
        all_data = []
        for unit_id in range(3):
            df = self.generate_normal_unit(start_time, duration_days, unit_id, charge_pattern, daily_offsets)
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
                
                # Calculate start index
                start_minutes = start_day * 24 * 60 + start_hour * 60
                duration_minutes = duration_hours * 60
                
                # Get unit data
                unit_mask = (combined_df['HVACNum'] == unit)
                unit_indices = combined_df[unit_mask].index
                
                if start_minutes < len(unit_indices):
                    start_idx = unit_indices[start_minutes]
                    
                    # Apply anomaly
                    if anom_type == 'lag':
                        lag_min = params.get('lag_minutes', 30)
                        combined_df = self.inject_anomaly_lag(
                            combined_df, start_idx, duration_minutes, lag_min)
                    elif anom_type == 'stuck':
                        stuck_temp = params.get('stuck_temp', None)
                        combined_df = self.inject_anomaly_stuck(
                            combined_df, start_idx, duration_minutes, stuck_temp)
                    elif anom_type == 'erratic':
                        combined_df = self.inject_anomaly_erratic(
                            combined_df, start_idx, duration_minutes)
                    elif anom_type == 'drift':
                        drift_rate = params.get('drift_rate', 0.01)
                        combined_df = self.inject_anomaly_drift(
                            combined_df, start_idx, duration_minutes, drift_rate)
                    elif anom_type == 'reduced_cooling':
                        reduction = params.get('reduction_factor', 0.5)
                        combined_df = self.inject_anomaly_reduced_cooling(
                            combined_df, start_idx, duration_minutes, reduction)
        
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
        anomaly_types = ['lag', 'stuck', 'erratic', 'drift', 'reduced_cooling']
        
        config = [{
            'unit': random.randint(0, 2),
            'type': random.choice(anomaly_types),
            'start_day': random.randint(1, max(1, duration_days - 2)),
            'start_hour': random.randint(0, 23),
            'duration_hours': random.randint(2, 12),
            'params': self._generate_anomaly_params(random.choice(anomaly_types))
        }]
        
        return config
    
    def _generate_anomaly_params(self, anom_type: str) -> Dict:
        """Generate parameters for anomaly type"""
        if anom_type == 'lag':
            return {'lag_minutes': random.randint(20, 60)}
        elif anom_type == 'drift':
            return {'drift_rate': random.uniform(0.005, 0.02)}
        elif anom_type == 'reduced_cooling':
            return {'reduction_factor': random.uniform(0.3, 0.7)}
        return {}


def visualize_container(df: pd.DataFrame, 
                       container_id: int,
                       save_path: str = None):
    """
    Visualize HVAC data for a specific container
    
    Args:
        df: DataFrame with HVAC data
        container_id: Container to visualize
        save_path: Path to save figure (optional)
    """
    container_data = df[df['container_id'] == container_id]
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    colors = ['#5470C6', '#EE6666', '#5DBCD2']
    
    for unit in range(3):
        unit_data = container_data[container_data['HVACNum'] == unit]
        
        # Plot normal data
        normal_data = unit_data[~unit_data['anomaly']]
        if len(normal_data) > 0:
            ax.plot(normal_data['timestamp_et'], normal_data['TmpRet'], 
                   color=colors[unit], label=f'HVAC {unit}', linewidth=1.5)
        
        # Highlight anomalies
        anomaly_data = unit_data[unit_data['anomaly']]
        if len(anomaly_data) > 0:
            ax.plot(anomaly_data['timestamp_et'], anomaly_data['TmpRet'],
                   color='red', linewidth=2, alpha=0.7)
            
            # Add anomaly markers
            anomaly_types = anomaly_data['anomaly_type'].unique()
            for anom_type in anomaly_types:
                if anom_type != 'normal':
                    anom_subset = anomaly_data[anomaly_data['anomaly_type'] == anom_type]
                    mid_idx = len(anom_subset) // 2
                    if len(anom_subset) > 0:
                        ax.annotate(anom_type, 
                                  xy=(anom_subset.iloc[mid_idx]['timestamp_et'],
                                      anom_subset.iloc[mid_idx]['TmpRet']),
                                  xytext=(10, 10), textcoords='offset points',
                                  bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                                  fontsize=9)
    
    # Add threshold line
    ax.axhline(y=35, color='red', linestyle='--', linewidth=2, label='Threshold', alpha=0.5)
    
    ax.set_xlabel('Timestamp', fontsize=11)
    ax.set_ylabel('TmpRet (Temperature)', fontsize=11)
    ax.set_title(f'Container {container_id} - HVAC Units Temperature', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


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
    
    # Visualize
    visualize_container(single_container_df, container_id=0, 
                       save_path='/home/claude/hvac_example_container.png')
    
    # Example 2: Generate multi-container dataset
    print("\n" + "="*70)
    print("Generating multi-container dataset...")
    
    # Define specific anomalies for some containers
    multi_anomaly_configs = {
        0: [{'unit': 1, 'type': 'erratic', 'start_day': 3, 
             'start_hour': 10, 'duration_hours': 5}],
        2: [{'unit': 0, 'type': 'drift', 'start_day': 4, 
             'start_hour': 6, 'duration_hours': 8, 
             'params': {'drift_rate': 0.015}}],
        4: [{'unit': 2, 'type': 'reduced_cooling', 'start_day': 2,
             'start_hour': 12, 'duration_hours': 10,
             'params': {'reduction_factor': 0.4}}]
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
    
    # Visualize a few containers
    for container_id in [0, 2, 4]:
        visualize_container(full_dataset, container_id=container_id,
                          save_path=f'/home/claude/hvac_container_{container_id}.png')
    
    print("\nGeneration complete!")