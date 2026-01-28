import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional

def Calibration_DDF(
        precip_df: pd.DataFrame,
        precip_col: str,
        temp_df: pd.DataFrame,
        temp_col: str,
        swe_df: pd.DataFrame,
        swe_col: str,
        tm_lower: float,
        tm_upper: float,
        tm_interval: float,
        ddf_lower: float,
        ddf_upper: float,
        ddf_interval: float,
        plot: bool = True,
        date_df: Optional[pd.DataFrame] = None,
        date_col: Optional[str] = None
) -> Tuple[float, float, float, pd.DataFrame]:
    """
    Calibrate degree-day snowmelt model by finding optimal DDF and Tm parameters.

    Parameters
    ----------
    precip_df : pd.DataFrame
        DataFrame containing precipitation data
    precip_col : str
        Column name for precipitation values (mm)
    temp_df : pd.DataFrame
        DataFrame containing temperature data
    temp_col : str
        Column name for mean temperature values (°C)
    swe_df : pd.DataFrame
        DataFrame containing observed snow water equivalent data
    swe_col : str
        Column name for SWE values (mm)
    tm_lower : float
        Lower limit for threshold temperature Tm (°C)
    tm_upper : float
        Upper limit for threshold temperature Tm (°C)
    tm_interval : float
        Interval for testing Tm values
    ddf_lower : float
        Lower limit for degree-day factor (mm/°C/day)
    ddf_upper : float
        Upper limit for degree-day factor (mm/°C/day)
    ddf_interval : float
        Interval for testing DDF values
    plot : bool, default=True
        Whether to generate calibration and comparison plot
    date_df : pd.DataFrame, optional
        DataFrame containing date information for time-series plot
    date_col : str, optional
        Column name for dates (will be converted to datetime if not already)

    Returns
    -------
    best_tm : float
        Optimal threshold temperature (°C)
    best_ddf : float
        Optimal degree-day factor (mm/°C/day)
    best_nse : float
        Best Nash-Sutcliffe Efficiency achieved
    results_df : pd.DataFrame
        DataFrame with NSE values for all parameter combinations
    """

    # Extract data arrays
    precip = precip_df[precip_col].values
    temp = temp_df[temp_col].values
    swe_obs = swe_df[swe_col].values

    # Check that all arrays have the same length
    ndays = len(precip)
    if len(temp) != ndays or len(swe_obs) != ndays:
        raise ValueError("Precipitation, temperature, and SWE data must have the same length")

    # Generate parameter ranges
    n_tm = int(round((tm_upper - tm_lower) / tm_interval)) + 1
    n_ddf = int(round((ddf_upper - ddf_lower) / ddf_interval)) + 1

    tm_values = np.linspace(tm_lower, tm_upper, n_tm)
    ddf_values = np.linspace(ddf_lower, ddf_upper, n_ddf)

    print(f"Testing {len(tm_values)} Tm values from {tm_lower} to {tm_upper}°C")
    print(f"Testing {len(ddf_values)} DDF values from {ddf_lower} to {ddf_upper} mm/°C/day")
    print(f"Total parameter combinations: {len(tm_values) * len(ddf_values)}")

    # Initialize results storage
    nse_results = {}
    best_nse = -np.inf
    best_tm = None
    best_ddf = None

    # Loop through all Tm values
    for tm in tm_values:
        nse_for_tm = []

        # Loop through all DDF values
        for ddf in ddf_values:
            # Initialize arrays
            snow = np.zeros(ndays)
            melt = np.zeros(ndays)

            # Calculate snow storage for each day
            for i in range(ndays - 1):
                if temp[i + 1] <= tm:
                    # Too cold for melting
                    melt[i + 1] = 0
                    snow[i + 1] = snow[i] + precip[i + 1]
                else:
                    # Warm enough for melting
                    melt[i + 1] = ddf * (temp[i + 1] - tm)
                    snow[i + 1] = max(snow[i] - melt[i + 1], 0)

            # Calculate NSE
            nse = calculate_nse(snow, swe_obs)
            nse_for_tm.append(nse)

            # Track best parameters
            if nse > best_nse:
                best_nse = nse
                best_tm = tm
                best_ddf = ddf

        nse_results[tm] = nse_for_tm

    # Create results DataFrame
    results_df = pd.DataFrame(nse_results, index=ddf_values)
    results_df.index.name = 'DDF'

    # Print results
    print(f"\n{'=' * 60}")
    print(f"CALIBRATION RESULTS")
    print(f"{'=' * 60}")
    print(f"Best Tm:  {best_tm:.2f}°C")
    print(f"Best DDF: {best_ddf:.2f} mm/°C/day")
    print(f"Best NSE: {best_nse:.4f}")
    print(f"{'=' * 60}\n")

    # Generate plots if requested
    if plot:
        plot_calibration_results(results_df, tm_values, best_tm, best_ddf, best_nse)

        # Prepare dates for time-series plot
        dates = None
        if date_df is not None and date_col is not None:
            dates = pd.to_datetime(date_df[date_col])

        plot_timeseries_comparison(precip, temp, swe_obs, tm_values, results_df, ddf_values, dates)

    return best_tm, best_ddf, best_nse, results_df


def calculate_nse(simulated: np.ndarray, observed: np.ndarray) -> float:
    """
    Calculate Nash-Sutcliffe Efficiency.

    Parameters
    ----------
    simulated : np.ndarray
        Simulated values
    observed : np.ndarray
        Observed values

    Returns
    -------
    float
        Nash-Sutcliffe Efficiency (1 = perfect, 0 = as good as mean, <0 = worse than mean)
    """
    numerator = np.sum((simulated - observed) ** 2)
    denominator = np.sum((observed - np.mean(observed)) ** 2)

    if denominator == 0:
        return np.nan

    nse = 1 - (numerator / denominator)
    return nse


def plot_calibration_results(
        results_df: pd.DataFrame,
        tm_values: np.ndarray,
        best_tm: float,
        best_ddf: float,
        best_nse: float
) -> None:
    """
    Create calibration plot showing NSE vs DDF for different Tm values.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with NSE values for all parameter combinations
    tm_values : np.ndarray
        Array of tested Tm values
    best_tm : float
        Optimal threshold temperature
    best_ddf : float
        Optimal degree-day factor
    best_nse : float
        Best NSE achieved
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define color map for different Tm values
    colors = plt.cm.viridis(np.linspace(0, 1, len(tm_values)))

    # Plot NSE curves for each Tm
    for i, tm in enumerate(tm_values):
        ax.plot(results_df.index, results_df[tm],
                label=f'Tm = {tm:.2f}°C',
                color=colors[i], linewidth=2)

        # Mark maximum NSE for this Tm
        max_nse_idx = results_df[tm].idxmax()
        max_nse_val = results_df[tm].max()
        ax.scatter(max_nse_idx, max_nse_val,
                   color=colors[i], s=100, zorder=5, edgecolors='black', linewidths=1.5)

    # Highlight the overall best combination
    ax.scatter(best_ddf, best_nse,
               color='red', s=200, marker='*', zorder=10,
               edgecolors='black', linewidths=2,
               label=f'Best: Tm={best_tm:.2f}°C, DDF={best_ddf:.1f}')

    # Formatting
    ax.set_xlabel('Degree-Day Factor (mm/°C/day)', fontsize=12)
    ax.set_ylabel('Nash-Sutcliffe Efficiency (NSE)', fontsize=12)
    ax.set_title('Degree-Day Model Calibration Results', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_timeseries_comparison(
        precip: np.ndarray,
        temp: np.ndarray,
        swe_obs: np.ndarray,
        tm_values: np.ndarray,
        results_df: pd.DataFrame,
        ddf_values: np.ndarray,
        dates: Optional[pd.Series] = None
) -> None:
    """
    Create time-series plot comparing simulated snow with observed SWE
    for the best DDF of each Tm value.

    Parameters
    ----------
    precip : np.ndarray
        Precipitation data (mm)
    temp : np.ndarray
        Temperature data (°C)
    swe_obs : np.ndarray
        Observed SWE data (mm)
    tm_values : np.ndarray
        Array of tested Tm values
    results_df : pd.DataFrame
        DataFrame with NSE values for all parameter combinations
    ddf_values : np.ndarray
        Array of tested DDF values
    dates : pd.Series, optional
        Series of datetime objects for x-axis. If None, uses day numbers.
    """
    ndays = len(precip)

    # Determine x-axis values
    if dates is not None:
        x_values = dates
        xlabel = 'Date'
    else:
        x_values = range(ndays)
        xlabel = 'Day'

    # Find best DDF for each Tm and simulate snow
    fig, ax = plt.subplots(figsize=(14, 7))

    # Define colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(tm_values)))

    # Plot observed SWE first (as baseline)
    ax.plot(x_values, swe_obs, 'k-', linewidth=2.5,
            label='Observed SWE', alpha=0.7, zorder=10)

    # For each Tm, find best DDF and simulate
    for i, tm in enumerate(tm_values):
        # Find best DDF for this Tm
        best_ddf_idx = results_df[tm].idxmax()
        best_ddf = best_ddf_idx
        best_nse = results_df[tm].max()

        # Simulate snow with best parameters
        snow = np.zeros(ndays)
        melt = np.zeros(ndays)

        for j in range(ndays - 1):
            if temp[j + 1] <= tm:
                melt[j + 1] = 0
                snow[j + 1] = snow[j] + precip[j + 1]
            else:
                melt[j + 1] = best_ddf * (temp[j + 1] - tm)
                snow[j + 1] = max(snow[j] - melt[j + 1], 0)

        # Plot simulated snow
        ax.plot(x_values, snow, color=colors[i], linewidth=2,
                label=f'Tm={tm:.2f}°C, DDF={best_ddf:.2f} (NSE={best_nse:.3f})',
                alpha=0.8)

    # Formatting
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Snow Water Equivalent (mm)', fontsize=12)
    ax.set_title('Time-Series Comparison: Simulated Snow vs Observed SWE',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Format x-axis for dates
    if dates is not None:
        fig.autofmt_xdate()  # Rotate and align date labels

    plt.tight_layout()
    plt.show()

"""
# Example usage:
if __name__ == "__main__":
    # Create sample data
    ndays = 1000
    dates = pd.date_range('2020-01-01', periods=ndays, freq='D')

    # Sample precipitation (mm)
    precip_df = pd.DataFrame({
        'date': dates,
        'precip': np.random.gamma(2, 2, ndays)
    })

    # Sample temperature (°C) with seasonal variation
    temp_df = pd.DataFrame({
        'date': dates,
        'temp': 10 * np.sin(np.arange(ndays) * 2 * np.pi / 365) + np.random.normal(0, 2, ndays)
    })

    # Sample observed SWE (mm) - simplified
    swe_df = pd.DataFrame({
        'date': dates,
        'swe_obs': np.maximum(0, 50 * (1 - np.cos(np.arange(ndays) * 2 * np.pi / 365)) +
                              np.random.normal(0, 10, ndays))
    })

    # Run calibration
    best_tm, best_ddf, best_nse, results = calibrate_degree_day_model(
        precip_df=precip_df,
        precip_col='precip',
        temp_df=temp_df,
        temp_col='temp',
        swe_df=swe_df,
        swe_col='swe_obs',
        tm_lower=-1,
        tm_upper=1,
        tm_interval=1,
        ddf_lower=0.1,
        ddf_upper=10,
        ddf_interval=0.1,
        plot=True,
        date_df=precip_df,  # Using precip_df since it has dates
        date_col='date'
    )
"""

