import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Tuple


def calculate_snowmelt(
        best_tm: float,
        best_ddf: float,
        precip_df: pd.DataFrame,
        precip_col: str,
        temp_df: pd.DataFrame,
        temp_col: str,
        date_df: pd.DataFrame,
        date_col: str,
        output_folder: str,
        output_filename: str,
        plot: bool = True
) -> pd.DataFrame:
    """
    Calculate daily snow storage, snowmelt, and liquid water using the degree-day method.

    Parameters
    ----------
    best_tm : float
        Calibrated threshold temperature (°C) at which snowmelt begins
    best_ddf : float
        Calibrated degree-day factor (mm/°C/day)
    precip_df : pd.DataFrame
        DataFrame containing precipitation data
    precip_col : str
        Column name for precipitation values (mm)
    temp_df : pd.DataFrame
        DataFrame containing temperature data
    temp_col : str
        Column name for mean temperature values (°C)
    date_df : pd.DataFrame
        DataFrame containing date information
    date_col : str
        Column name for dates
    output_folder : str
        Path to folder where CSV file should be saved
    output_filename : str
        Name of the CSV file (e.g., 'snowmelt_results.csv')
    plot : bool, default=True
        Whether to generate visualization plot

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Date, Snow (mm), Melt (mm), Liquid_Water (mm)
    """

    # Extract data arrays
    precip = precip_df[precip_col].values
    temp = temp_df[temp_col].values
    dates = pd.to_datetime(date_df[date_col])

    # Check that all arrays have the same length
    ndays = len(precip)
    if len(temp) != ndays or len(dates) != ndays:
        raise ValueError("Precipitation, temperature, and date data must have the same length")

    print(f"Calculating snowmelt for {ndays} days...")
    print(f"Using parameters: Tm = {best_tm:.2f}°C, DDF = {best_ddf:.2f} mm/°C/day")

    # Initialize arrays
    snow = np.zeros(ndays)
    melt = np.zeros(ndays)
    liquid_water = np.zeros(ndays)

    # Calculate for each day
    for i in range(ndays - 1):
        if temp[i + 1] <= best_tm:
            # Too cold for melting
            melt[i + 1] = 0
            snow[i + 1] = snow[i] + precip[i + 1]
            liquid_water[i + 1] = 0
        else:
            # Warm enough for melting
            melt[i + 1] = best_ddf * (temp[i + 1] - best_tm)
            snow[i + 1] = max(snow[i] - melt[i + 1], 0)
            # Liquid water = precipitation + actual melt from existing snow
            liquid_water[i + 1] = precip[i + 1] + min(snow[i], melt[i + 1])

    # Create results DataFrame
    results_df = pd.DataFrame({
        'Date': dates,
        'Snow [mm]': snow,
        'Melt [mm]': melt,
        'Liquid_Water [mm]': liquid_water
    })

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Ensure filename ends with .csv
    if not output_filename.endswith('.csv'):
        output_filename += '.csv'

    # Create full output path
    output_path = os.path.join(output_folder, output_filename)

    # Save to CSV
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Print summary statistics
    print(f"\n{'=' * 60}")
    print(f"SUMMARY STATISTICS")
    print(f"{'=' * 60}")
    print(f"Total Snow Accumulation:    {snow.sum():.2f} mm")
    print(f"Total Snowmelt:             {melt.sum():.2f} mm")
    print(f"Total Liquid Water:         {liquid_water.sum():.2f} mm")
    print(f"Maximum Snow Storage:       {snow.max():.2f} mm")
    print(f"Days with Snow:             {np.sum(snow > 0)} days ({100 * np.sum(snow > 0) / ndays:.1f}%)")
    print(f"{'=' * 60}\n")

    # Generate plot if requested
    if plot:
        plot_snowmelt_results(results_df, best_tm, best_ddf)

    return results_df


def plot_snowmelt_results(
        results_df: pd.DataFrame,
        tm: float,
        ddf: float
) -> None:
    """
    Create visualization of snow storage, melt, and liquid water over time.
    Generates two plots: one combined and one with three subplots.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with Date, Snow [mm], Melt [mm], Liquid_Water [mm] columns
    tm : float
        Threshold temperature used
    ddf : float
        Degree-day factor used
    """

    dates = results_df['Date']

    # PLOT 1: Combined plot with creative layout
    fig1, ax1 = plt.subplots(figsize=(14, 8))

    # Create second y-axis for snow (right side)
    ax2 = ax1.twinx()

    # Plot Liquid Water on left axis (normal)
    line_lw = ax1.plot(dates, results_df['Liquid_Water [mm]'], color='darkblue',
                       linewidth=1.2, label='Liquid Water', alpha=0.8, zorder=3)

    # Plot Melt below x-axis on left axis (as negative values for plotting, but label as positive)
    bar_melt = ax1.bar(dates, -results_df['Melt [mm]'], color='darkred',
                       alpha=0.7, label='Potential Melt', width=1, zorder=2)

    # Plot Snow from top on right axis (inverted)
    bar_snow = ax2.bar(dates, results_df['Snow [mm]'], color='darkgray',
                       alpha=0.6, label='Snow Storage', width=1, zorder=1)

    # Configure left y-axis (Liquid Water and Melt)
    max_lw = results_df['Liquid_Water [mm]'].max()
    max_melt = results_df['Melt [mm]'].max()
    max_left = max(max_lw, max_melt)

    ax1.set_ylim(-max_left * 1.1, max_left * 1.1)
    ax1.axhline(y=0, color='black', linewidth=1.5, linestyle='-', zorder=4)

    # Get y-ticks and set custom labels to show positive values
    y_ticks = ax1.get_yticks()
    ax1.set_yticklabels([f'{abs(tick):.0f}' for tick in y_ticks])

    # Color the left y-axis ticks: green above x-axis, coral below
    # We need to iterate through the tick objects and their original positions
    for i, (tick_label, tick_pos) in enumerate(zip(ax1.get_yticklabels(), y_ticks)):
        if tick_pos > 0:  # Above x-axis
            tick_label.set_color('darkblue')
        elif tick_pos < 0:  # Below x-axis
            tick_label.set_color('darkred')
        else:  # At x-axis (0)
            tick_label.set_color('black')

    # Add split y-axis labels for left axis
    # Upper label for Liquid Water
    ax1.text(-0.04, 0.7, 'Liquid Water\n(mm/day)', transform=ax1.transAxes,
             fontsize=12, color='darkblue', fontweight='bold',
             ha='center', va='center', rotation=90)
    # Lower label for Melt
    ax1.text(-0.04, 0.25, 'Potential Melt\n(mm/day)', transform=ax1.transAxes,
             fontsize=12, color='darkred', fontweight='bold',
             ha='center', va='center', rotation=90)

    # Configure right y-axis (Snow - inverted, only in upper part)
    max_snow = results_df['Snow [mm]'].max()
    ax2.set_ylim(400, 0)  # Inverted: max at top, 0 at x-axis
    ax2.set_ylabel('Snow Storage (mm)', fontsize=12, color='darkgray', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='darkgray')

    # X-axis
    ax1.set_xlabel('Date', fontsize=12)
    ax1.grid(True, alpha=0.3, zorder=0)

    # Title
    ax1.set_title(f'DDF Snowmelt Model Results - (Tm={tm:.2f}°C, DDF={ddf:.2f} mm/°C/day)',
                  fontsize=14, fontweight='bold')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower center', ncol=3, fontsize=11, framealpha=0.9)

    fig1.autofmt_xdate()
    plt.tight_layout()
    plt.show()

    # PLOT 2: Three subplots (unchanged)
    fig2, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Subplot 1: Snow Storage
    axes[0].fill_between(dates, results_df['Snow [mm]'], alpha=0.6, color='gainsboro', label='Snow Storage')
    axes[0].plot(dates, results_df['Snow [mm]'], color='darkgray', linewidth=1)
    axes[0].set_ylabel('Snow Storage [mm]', fontsize=12)
    axes[0].set_title(f'DDF Snowmelt Model Results - (Tm={tm:.2f}°C, DDF={ddf:.2f} mm/°C/day)',
                      fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')

    # Subplot 2: Snowmelt
    axes[1].fill_between(dates, results_df['Melt [mm]'], alpha=0.6, color='coral', label='Potential Snowmelt')
    axes[1].plot(dates, results_df['Melt [mm]'], color='darkred', linewidth=1)
    axes[1].set_ylabel('Potential Snowmelt [mm/day]', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')

    # Subplot 3: Liquid Water
    axes[2].fill_between(dates, results_df['Liquid_Water [mm]'], alpha=0.6, color='steelblue', label='Liquid Water')
    axes[2].plot(dates, results_df['Liquid_Water [mm]'], color='darkblue', linewidth=1)
    axes[2].set_ylabel('Liquid Water [mm/day]', fontsize=12)
    axes[2].set_xlabel('Date', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right')

    fig2.autofmt_xdate()
    plt.tight_layout()
    plt.show()

'''
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

    # Example: use calibrated parameters from calibration function
    best_tm = 0.0  # Example calibrated value
    best_ddf = 2.5  # Example calibrated value

    # Calculate snowmelt
    results = calculate_snowmelt(
        best_tm=best_tm,
        best_ddf=best_ddf,
        precip_df=precip_df,
        precip_col='precip',
        temp_df=temp_df,
        temp_col='temp',
        date_df=precip_df,
        date_col='date',
        output_folder='./snowmelt_results',
        output_filename='snowmelt_results.csv',
        plot=True
    )

    print("\nFirst 10 rows of results:")
    print(results.head(10))
'''