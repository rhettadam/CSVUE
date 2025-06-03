![logo](https://github.com/user-attachments/assets/73a6a479-da06-4fe9-be14-0aadfd91a331)

CSVUE is a desktop application built with Python and Tkinter for loading, browsing, analyzing, and visualizing tabular data from various file formats. CSVUE streamlines exploratory data analysis (EDA) with an intuitive GUI and rich plotting capabilities.

# Key Features

## Multi-Format Data Loading
- Supported formats:
  - Tabular Data: **.csv, .tsv**
  - Spreadsheets: **.xlsx, .xls, .xlsm**
  - Data Exchange: **.json, .jsonl, .xml**
  - Big Data: **.parquet, .feather, .arrow**
  - Databases: **.sqlite, .db, .sqlite3**
  - Scientific: **.hdf, .h5**
  - Statistical Software:
    - R: **.rds**
    - SAS: **.sas7bdat**
    - SPSS: **.sav**
    - Stata: **.dta**
  - Python: **.pickle, .pkl**

- Files can be loaded and managed concurrently in a sidebar tree view

- Each dataset is parsed into a Pandas DataFrame for in-memory operations

## Data Structure Browser
- Sidebar tree structure to navigate loaded datasets and their columns

- Instant switching between datasets for focused analysis

## Data Browser Tab
### Interactive tabular view using pandastable with support for:

- Sorting

- Filtering by custom criteria

- Searching across all columns

- Exporting data to various formats

- Quick actions like removing duplicates or handling missing values

![image](https://github.com/user-attachments/assets/522dd4a1-856e-4cc8-9a94-fe688647096d)

## Visualization Engine
### Supports rich interactive plots via Plotly, including:

- Histogram, Scatterplot, Bar, Line, Box, Violin, Heatmap

- 3D Scatter, Bubble, Pie, Donut, Sunburst, Treemap

- Time Series, Parallel Coordinates, Radar, Area, KDE, Density

### Fully configurable plot options:

- X, Y, color, and size variables

- Themes (plotly, ggplot2, etc.) and color schemes (viridis, cividis, etc.)

- Rendering and exporting of plots as PNG, SVG, PDF, or HTML

- Embedded webview to render interactive Plotly plots directly in-app

![image](https://github.com/user-attachments/assets/2155bbed-d818-4f95-94af-c33639c1645b)

## Statistics Tab
Four sub-tabs for analysis:

- Overview: Dataset metadata (row count, memory usage, column types)

- Detailed Statistics: Descriptive stats for numerical and categorical columns

- Data Quality: Missing values, duplicate rows, constant columns, high correlations

- Distributions: Interactive Matplotlib plots (Histogram, Box, Violin, KDE)

- Export capability to generate full HTML or plain-text reports

![image](https://github.com/user-attachments/assets/1ef3b9f5-227a-4a27-a965-884e9472e6f7)

## Data Operations & Utilities
Quick Insights: Auto-generates descriptive summaries and correlations

Missing Data Handler: Options for filling with mean/median/mode, forward/backward fill, or drop

Duplicate Remover: Intelligent deduplication with column selection and keep-first/last logic

Data Type Converter: Bulk convert column types (e.g., string → datetime)

Dynamic Filtering: Build row filters with a GUI (e.g., col > 100 or col contains "abc")



