"""Custom widget for matplotlib plots."""

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import numpy as np
import time

from PySide6.QtWidgets import QWidget, QVBoxLayout

from hotpixels.app import App
from hotpixels.profile import HotPixelProfile


class PlotWidget(QWidget):
    """Custom widget for matplotlib plots"""

    def __init__(self, app: App, parent=None):
        super().__init__(parent)
        self.app = app  # Shared App instance
        # Smaller figure size to fit without scrolling
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot_unified_hot_pixel_analysis(self, profile: HotPixelProfile):
        """Plot unified analysis with pie charts on top and bar charts below"""
        startTime = time.time()
        self.figure.clear()

        if not profile.common_statistics:
            return

        # Create a 3x2 grid layout with optimized spacing for legends
        gs = gridspec.GridSpec(3, 2, figure=self.figure,
                               height_ratios=[1.2, 1, 1],
                               hspace=0.6, wspace=0.25,
                               top=0.94, bottom=0.08)

        # Top row: Pie charts
        ax_pie1 = self.figure.add_subplot(gs[0, 0])
        ax_pie2 = self.figure.add_subplot(gs[0, 1])

        # Calculate fraction of hot pixels
        hot_pixel_fraction = profile.common_statistics.fraction_hot_pixels
        print("Debug:", "hot pixel fraction:", hot_pixel_fraction)
        normal_pixel_fraction = 1.0 - hot_pixel_fraction

        # Pie chart 1: Mean fraction of hot pixels
        labels1 = ['Normal Pixels', 'Hot Pixels']
        sizes1 = [normal_pixel_fraction * 100, hot_pixel_fraction * 100]
        colors1 = ['lightblue', 'red']
        explode1 = (0, 0.1)  # explode hot pixels slice

        wedges1, texts1, autotexts1 = ax_pie1.pie(sizes1, explode=explode1, colors=colors1, autopct='%1.4f%%',
                                                  shadow=True, startangle=90)
        ax_pie1.set_title('Mean Hot Pixel Fraction')
        ax_pie1.legend(wedges1, labels1, loc="lower center", bbox_to_anchor=(0.5, -0.25))

        # Pie chart 2: Hot pixel fraction by frame
        cs = profile.common_statistics
        common_fraction = cs.fraction_common_hot_pixels
        random_fraction = 1.0 - common_fraction

        labels2 = ['Random Hot Pixels', 'Common Hot Pixels']
        sizes2 = [random_fraction * 100, common_fraction * 100]
        colors2 = ['orange', 'darkred']
        explode2 = (0, 0.1)

        # Only show slices with meaningful values
        filtered_data = [(label, size, color, exp) for label, size, color, exp in
                         zip(labels2, sizes2, colors2, explode2) if size > 0.001]

        if filtered_data:
            labels2_f, sizes2_f, colors2_f, explode2_f = zip(*filtered_data)
            wedges2, texts2, autotexts2 = ax_pie2.pie(sizes2_f, explode=explode2_f, colors=colors2_f,
                                                      autopct='%1.4f%%', shadow=True, startangle=90)
            ax_pie2.legend(wedges2, labels2_f, loc="lower center", bbox_to_anchor=(0.5, -0.25))

        ax_pie2.set_title('Hot Pixel Composition')

        # Extract data for bar charts
        mean_values = profile.mean_values
        hot_pixel_counts = profile.hot_pixel_counts
        frame_numbers = list(range(1, len(mean_values) + 1))

        # Bottom row: Hot pixel statistics
        ax_bar1 = self.figure.add_subplot(gs[1, 0])
        ax_bar2 = self.figure.add_subplot(gs[1, 1])

        # Bar chart 1: Hot pixel count per frame
        ax_bar1.bar(frame_numbers, hot_pixel_counts, alpha=0.7, color='red')
        ax_bar1.set_title('Hot Pixels Count per Frame')
        ax_bar1.set_xlabel('Frame Number')
        ax_bar1.set_ylabel('Hot Pixel Count')
        ax_bar1.grid(True, alpha=0.3)

        # Bar chart 2: Mean values per frame (by color channel if available)
        if profile.frame_channel_means and len(profile.frame_channel_means) > 0 and profile.frame_channel_means[0]:
            # We have per-channel data - create grouped bars
            bar_width = 0.25
            
            # Extract channel data
            r_values = [frame_means.get('R', 0) for frame_means in profile.frame_channel_means]
            g_values = [frame_means.get('G', 0) for frame_means in profile.frame_channel_means]
            b_values = [frame_means.get('B', 0) for frame_means in profile.frame_channel_means]
            
            # Calculate bar positions
            r_positions = [x - bar_width for x in frame_numbers]
            g_positions = frame_numbers
            b_positions = [x + bar_width for x in frame_numbers]
            
            # Plot bars for each channel
            if any(r_values):
                ax_bar2.bar(r_positions, r_values, width=bar_width, alpha=0.7, color='red', label='Red')
            if any(g_values):
                ax_bar2.bar(g_positions, g_values, width=bar_width, alpha=0.7, color='green', label='Green')
            if any(b_values):
                ax_bar2.bar(b_positions, b_values, width=bar_width, alpha=0.7, color='blue', label='Blue')
            
            ax_bar2.set_title('Mean Pixel Value per Frame (By Channel)')
            ax_bar2.legend()
        else:
            # Fallback to overall mean values for older profiles
            ax_bar2.bar(frame_numbers, mean_values, alpha=0.7, color='blue')
            ax_bar2.set_title('Mean Pixel Value per Frame')
        
        ax_bar2.set_xlabel('Frame Number')
        ax_bar2.set_ylabel('Mean Value')
        ax_bar2.grid(True, alpha=0.3)

        # Third row: Temperature plot (if available)
        if profile.frame_temperatures and any(t is not None for t in profile.frame_temperatures):
            ax_temp = self.figure.add_subplot(gs[2, :])
            
            # Filter out None values
            valid_temps = [(i+1, temp) for i, temp in enumerate(profile.frame_temperatures) if temp is not None]
            if valid_temps:
                temp_frame_numbers, temperatures = zip(*valid_temps)
                ax_temp.plot(temp_frame_numbers, temperatures, 'o-', color='purple', linewidth=2, markersize=6)
                ax_temp.set_title('Sensor Temperature per Frame')
                ax_temp.set_xlabel('Frame Number')
                ax_temp.set_ylabel('Temperature (°C)')
                ax_temp.grid(True, alpha=0.3)
                
                # Add temperature statistics
                mean_temp = np.mean(temperatures)
                std_temp = np.std(temperatures)
                min_temp = min(temperatures)
                max_temp = max(temperatures)
                temp_range = max_temp - min_temp
                
                stats_text = f'Mean: {mean_temp:.1f}°C\nStd: {std_temp:.2f}°C\nRange: {temp_range:.1f}°C'
                ax_temp.text(0.02, 0.98, stats_text, transform=ax_temp.transAxes, fontsize=9,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        self.canvas.draw()
        print("Debug: Unified hot pixel analysis plotted in", time.time() - startTime, "seconds")

    def plot_hot_pixel_statistics(self, profile: HotPixelProfile):
        """Legacy method - redirect to unified plot"""
        self.plot_unified_hot_pixel_analysis(profile)

    def plot_hot_pixel_map(self, profile: HotPixelProfile):
        """Plot hot pixel locations colored by Bayer pattern"""
        startTime = time.time()
        self.figure.clear()

        ax = self.figure.add_subplot(111)

        # Extract hot pixel coordinates.
        hot_pixels = self.app.get_hot_pixels(profile)
        if hot_pixels is None:
            return

        # Get Bayer pattern if available
        bayer_pattern = None
        if profile.camera_metadata and profile.camera_metadata.bayer_pattern:
            bayer_pattern = profile.camera_metadata.bayer_pattern

        if bayer_pattern and len(bayer_pattern) == 4:
            # Separate pixels by Bayer color
            red_pixels = []
            green_pixels = []
            blue_pixels = []

            for y, x, _ in hot_pixels:
                # Determine Bayer color using the pattern
                bayer_index = (y % 2) * 2 + (x % 2)
                color = bayer_pattern[bayer_index]

                if color == 'R':
                    red_pixels.append((x, y))
                elif color == 'G':
                    green_pixels.append((x, y))
                elif color == 'B':
                    blue_pixels.append((x, y))

            # Plot each color separately
            if red_pixels:
                x_coords = [pixel[0] for pixel in red_pixels]
                y_coords = [pixel[1] for pixel in red_pixels]
                ax.scatter(x_coords, y_coords, c='red', s=1, alpha=0.7, label=f'Red ({len(red_pixels)})')

            if green_pixels:
                x_coords = [pixel[0] for pixel in green_pixels]
                y_coords = [pixel[1] for pixel in green_pixels]
                ax.scatter(x_coords, y_coords, c='green', s=1, alpha=0.7, label=f'Green ({len(green_pixels)})')

            if blue_pixels:
                x_coords = [pixel[0] for pixel in blue_pixels]
                y_coords = [pixel[1] for pixel in blue_pixels]
                ax.scatter(x_coords, y_coords, c='blue', s=1, alpha=0.7, label=f'Blue ({len(blue_pixels)})')

            ax.legend()
            title = f'Common Hot Pixels by Bayer Color ({len(hot_pixels)} total, Pattern: {bayer_pattern})'
        else:
            # Fallback to single color if no Bayer pattern available
            y_coords = [pixel[0] for pixel in hot_pixels]
            x_coords = [pixel[1] for pixel in hot_pixels]
            ax.scatter(x_coords, y_coords, c='red', s=1, alpha=0.6)
            title = f'Common Hot Pixel Locations ({len(hot_pixels)} pixels)'

        ax.set_title(title)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, alpha=0.3)

        # Invert Y axis to match image coordinates
        ax.invert_yaxis()

        self.figure.tight_layout()
        self.canvas.draw()
        print("Debug: Hot pixel map plotted in", time.time() - startTime, "seconds")

    def plot_dark_frame_histogram(self, profile: HotPixelProfile):
        """Plot histogram of dark frame profile values"""
        startTime = time.time()
        self.figure.clear()

        # Try to load the noise profile - first check for temporary data, then sidecar file
        noise_data = None

        # Get noise profile data (median noise frame)
        noise_data = None
        if hasattr(profile, 'get_median_noise_frame'):
            noise_data = profile.get_median_noise_frame()

        if noise_data is None:
            # Show message if no noise profile is available
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No dark frame profile available',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title('Dark Frame Profile Histogram')
            self.canvas.draw()
            return

        ax = self.figure.add_subplot(111)

        # Determine dynamic range for 16-bit data
        data_min = np.min(noise_data)
        data_max = np.max(noise_data)
        data_range = data_max - data_min

        # Use appropriate number of bins based on data range
        if data_range <= 256:
            num_bins = int(data_range) + 1
        else:
            num_bins = min(1024, int(data_range // 4))  # Use fewer bins for very large ranges

        # Calculate histogram with dynamic range
        hist, bins = np.histogram(noise_data.flatten(), bins=num_bins, range=(data_min, data_max))

        # Calculate bin width for proper bar display
        bin_width = (data_max - data_min) / num_bins if num_bins > 1 else 1

        # Plot histogram as bar chart
        ax.bar(bins[:-1], hist, width=bin_width * 0.8, alpha=0.7, color='darkblue', edgecolor='none')

        ax.set_title(f'Dark Frame Profile Histogram\n({noise_data.shape[0]}×{noise_data.shape[1]} pixels, 16-bit)')
        ax.set_xlabel(f'Pixel Value ({data_min} - {data_max})')
        ax.set_ylabel('Pixel Count (log scale)')
        ax.set_xlim(data_min, data_max)

        # Use logarithmic scale for Y-axis to better show the wide range of counts
        ax.set_yscale('log', base=10)

        # Set Y-axis limits - start from 1 to avoid log(0) issues
        max_count = np.max(hist)
        min_count = np.min(hist[hist > 0])  # Find minimum non-zero count
        ax.set_ylim(max(1, min_count), max_count * 2)  # Use 2x padding for log scale

        # Format Y-axis to show readable numbers on log scale
        from matplotlib.ticker import FuncFormatter

        def log_formatter(x, pos):
            if x >= 1_000_000:
                return f'{x/1_000_000:.1f}M'
            elif x >= 1_000:
                return f'{x/1_000:.0f}K'
            else:
                return f'{x:.0f}'
        ax.yaxis.set_major_formatter(FuncFormatter(log_formatter))

        ax.grid(True, alpha=0.3)

        # Add statistics text
        mean_val = np.mean(noise_data)
        std_val = np.std(noise_data)
        min_val = np.min(noise_data)
        max_val = np.max(noise_data)

        stats_text = f'Mean: {mean_val:.1f}\nStd: {std_val:.1f}\nMin: {min_val}\nMax: {max_val}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        self.figure.tight_layout()
        self.canvas.draw()

    def plot_deviation_threshold_comparison(self, deviation_data: dict):
        """Plot deviation threshold comparison as a line chart with derivatives"""
        self.figure.clear()

        if not deviation_data:
            # Show message if no data is available
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No deviation threshold comparison data available',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title('Hot Pixel Deviation Threshold Analysis')
            self.canvas.draw()
            return

        # Create subplots for main data and derivatives
        ax1 = self.figure.add_subplot(3, 1, 1)  # Main plot
        ax2 = self.figure.add_subplot(3, 1, 2)  # First derivative
        ax3 = self.figure.add_subplot(3, 1, 3)  # Second derivative

        # Extract threshold values and results
        thresholds = sorted(deviation_data.keys())
        values = [deviation_data[threshold] for threshold in thresholds]

        # Plot main data
        ax1.plot(thresholds, values, 'b-o', linewidth=2, markersize=6, label='Common Hot Pixel Fraction')
        ax1.set_title('Hot Pixel Deviation Threshold Analysis')
        ax1.set_ylabel('Mean Common Hot Pixel\nFraction by Frame')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Calculate and plot first derivatives if we have enough data points
        if len(values) > 1:
            first_derivatives = []
            derivative_thresholds = []
            for i in range(1, len(values)):
                derivative = values[i] - values[i-1]
                first_derivatives.append(derivative)
                derivative_thresholds.append(thresholds[i])

            ax2.plot(derivative_thresholds, first_derivatives, 'g-o',
                     linewidth=2, markersize=4, label='First Derivative')
            ax2.set_ylabel('First Derivative\n(Rate of Change)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)

        # Calculate and plot second derivatives if we have enough data points
        if len(values) > 2:
            second_derivatives = []
            second_derivative_thresholds = []
            for i in range(1, len(first_derivatives)):
                second_derivative = first_derivatives[i] - first_derivatives[i-1]
                second_derivatives.append(second_derivative)
                second_derivative_thresholds.append(derivative_thresholds[i])

            ax3.plot(second_derivative_thresholds, second_derivatives, 'r-o',
                     linewidth=2, markersize=4, label='Second Derivative')
            ax3.set_xlabel('Deviation Threshold (σ)')
            ax3.set_ylabel('Second Derivative\n(Curvature)')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)

            # Highlight the elbow point (most negative second derivative)
            if second_derivatives:
                min_idx = second_derivatives.index(min(second_derivatives))
                elbow_threshold = second_derivative_thresholds[min_idx]
                elbow_value = second_derivatives[min_idx]

                # Mark elbow point on all plots
                # Find corresponding values for main plot
                elbow_main_value = deviation_data.get(elbow_threshold + 2, None)  # Account for derivative offset
                if elbow_main_value:
                    ax1.axvline(x=elbow_threshold + 2, color='red', linestyle=':', alpha=0.7, label='Elbow Point')
                    ax1.plot(elbow_threshold + 2, elbow_main_value, 'ro',
                             markersize=8, markerfacecolor='none', markeredgewidth=2)

                ax2.axvline(x=elbow_threshold, color='red', linestyle=':', alpha=0.7)
                ax3.axvline(x=elbow_threshold, color='red', linestyle=':', alpha=0.7)
                ax3.plot(elbow_threshold, elbow_value, 'ro', markersize=8, markerfacecolor='none', markeredgewidth=2)

        # Format the axes
        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(min(thresholds) - 0.1, max(thresholds) + 0.1)

        # Add some statistics as text on the main plot
        best_threshold = max(deviation_data, key=deviation_data.get)
        best_value = deviation_data[best_threshold]

        stats_text = f'Best Threshold: {best_threshold}σ\nBest Value: {best_value:.4f}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        self.figure.tight_layout()
        self.canvas.draw()
