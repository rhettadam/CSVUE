# CSVUE - Advanced Data Visualization Tool

CSVUE is a powerful and user-friendly data visualization tool that supports a wide range of file formats and creates interactive plots. It provides an intuitive interface for exploring and visualizing your data with various plot types and customization options.

## Features

- **Multiple File Format Support**: CSV, Excel (xlsx/xls), JSON, Parquet
- **Interactive Plots**: Using Plotly for dynamic, interactive visualizations
- **Wide Range of Plot Types**:
  - Histogram
  - Scatterplot
  - Bar Plot
  - Line Plot
  - Box Plot
  - Violin Plot
  - Heatmap
  - 3D Scatter
  - Bubble Plot
  - Time Series
  - And more!
- **Advanced Customization**:
  - Color schemes
  - Plot themes
  - Custom labels and titles
  - Interactive legends
- **Data Preview**: Built-in data table viewer with search functionality
- **Statistical Summary**: Quick access to basic statistics and data information
- **Export Options**: Save plots as images or interactive HTML files

## Installation

1. Clone this repository or download the source code
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python csvue.py
   ```

2. Load your data:
   - Click the "Browse" button to select your data file
   - Supported formats: CSV, Excel (xlsx/xls), JSON, Parquet

3. Create visualizations:
   - Select a plot type from the dropdown menu
   - Choose variables for X and Y axes (when applicable)
   - Add color and size variables for additional dimensions
   - Customize the plot using available options
   - Click "Plot" to generate the visualization

4. Customize your plots:
   - Select different color schemes
   - Choose plot themes
   - Add custom titles and labels
   - Interact with the plot using Plotly's built-in tools

5. Export your work:
   - Save plots as PNG/JPG/PDF files
   - Export as interactive HTML files
   - Generate statistical summaries

## Tips for Best Results

- Use appropriate plot types for your data:
  - Categorical vs Numerical data
  - Time series data
  - Distributions
- Experiment with different color schemes and themes
- Use the data preview tab to understand your data structure
- Check the statistics tab for quick insights

## Requirements

- Python 3.7+
- See requirements.txt for full list of dependencies

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

---

## **Features**

- **Supported File Types**:
  - CSV
  - Excel (`.xlsx`, `.xls`)
  - JSON
  - Parquet

- **Plot Types**:
  - Histogram
  - Scatterplot
  - Bar Plot
  - Line Plot
  - Box Plot
  - Pie Chart

- **Dynamic Data Handling**:
  - Automatically detects categorical and continuous variables from your dataset.
  - Adjusts variable selection options based on the chosen plot type.

- **Customizable Plot Labels**:
  - Add titles, X-axis labels, and Y-axis labels for better clarity.

- **Save Plots**:
  - Export your visualizations as image files (`.png`, `.jpg`, or `.pdf`).
 
![Screenshot from 2024-12-14 16-17-33](https://github.com/user-attachments/assets/e2a15995-8a71-4ed3-ad57-3f85ff33ab88)

---
