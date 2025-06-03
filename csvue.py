import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from ttkbootstrap import Style
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
from pandastable import Table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from PIL import Image, ImageTk
import io
import os
import json
import yaml
import sqlite3
import duckdb
import polars as pl
from datetime import datetime
import webview

class DataPlotterApp:
    def __init__(self, root):
        self.root = root
        self.style = Style(theme="darkly")
        self.root.title("CSVUE - Advanced Data Browser & Visualization")
        self.root.geometry("1400x900")

        # Initialize variables
        self.loaded_files = {}  # Dictionary to store all loaded dataframes
        self.current_file = None  # Currently selected file
        self.data = None  # Current dataframe
        self.categorical_columns = []
        self.continuous_columns = []
        self.datetime_columns = []
        self.current_fig = None
        self.preview_window = None
        self.last_directory = os.path.expanduser("~")
        self.temp_html = "temp_plot.html"
        
        # Create main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Setup components in the correct order
        self.setup_status_bar()  # Create status bar first
        self.setup_left_sidebar()
        self.setup_main_content()

    def setup_left_sidebar(self):
        # Left sidebar frame
        self.sidebar = ttk.Frame(self.main_container, style='Secondary.TFrame', relief='raised')
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)
        
        # File Operations Section
        file_ops = ttk.LabelFrame(self.sidebar, text="File", padding=5)
        file_ops.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(file_ops, text="Open File", bootstyle="info",
                  command=self.load_file).pack(fill=tk.X, pady=2)
        
        # Structure Browser Section
        structure = ttk.LabelFrame(self.sidebar, text="Structure Browser", padding=5)
        structure.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add Treeview for structure with scrollbar
        tree_frame = ttk.Frame(structure)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        tree_scroll = ttk.Scrollbar(tree_frame)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.structure_tree = ttk.Treeview(tree_frame, show='tree', 
                                         yscrollcommand=tree_scroll.set)
        self.structure_tree.pack(fill=tk.BOTH, expand=True)
        tree_scroll.config(command=self.structure_tree.yview)
        
        # Bind selection event
        self.structure_tree.bind('<<TreeviewSelect>>', self.on_file_select)
        
        # Initialize tree structure
        self.files_node = self.structure_tree.insert("", "end", text="Loaded Files", open=True)

    def on_file_select(self, event):
        selected_items = self.structure_tree.selection()
        if not selected_items:
            return
            
        selected_item = selected_items[0]
        parent = self.structure_tree.parent(selected_item)
        
        # Check if a file node is selected (its parent should be the files_node)
        if parent == self.files_node:
            filename = self.structure_tree.item(selected_item)['text']
            if filename in self.loaded_files:
                # Update current file and data
                self.current_file = filename
                self.data = self.loaded_files[filename]
                
                # Update column types
                self.detect_column_types()
                
                # Update variable selectors
                self.update_variable_selectors(None)
                
                # Update data preview
                self.display_preview()
                
                # Update statistics if on stats tab
                if self.notebook.select() == str(self.stats_tab):
                    self.generate_summary()
                
                # Update status bar
                self.status_msg.config(text=f"Current file: {filename}")
                self.row_count.config(text=f"Rows: {len(self.data)}")

    def load_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("CSV Files", "*.csv"),
                ("Excel Files", "*.xlsx *.xls"),
                ("JSON Files", "*.json"),
                ("Parquet Files", "*.parquet"),
            ]
        )
        if not file_path:
            return

        try:
            # Load the data
            filename = os.path.basename(file_path)
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
            elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
                df = pd.read_excel(file_path)
            elif file_path.endswith(".json"):
                df = pd.read_json(file_path)
            elif file_path.endswith(".parquet"):
                df = pd.read_parquet(file_path)
            else:
                raise ValueError("Unsupported file format")

            # Store the dataframe
            self.loaded_files[filename] = df
            
            # Update current file and data
            self.current_file = filename
            self.data = df
            
            # Add to structure tree
            file_node = self.structure_tree.insert(self.files_node, "end", text=filename)
            
            # Add columns under file node
            for col in df.columns:
                self.structure_tree.insert(file_node, "end", text=col)
            
            # Detect column types
            self.detect_column_types()
            
            # Update variable selectors
            self.update_variable_selectors(None)
            
            # Display the data
            self.display_preview()
            
            # Update status bar
            self.status_msg.config(text=f"Loaded: {filename}")
            self.row_count.config(text=f"Rows: {len(df)}")
            
            # Switch to Data Browser tab
            self.notebook.select(self.data_tab)
            
            messagebox.showinfo("Success", "File loaded successfully!")
        except Exception as e:
            self.status_msg.config(text="Error loading file")
            messagebox.showerror("Error", f"Failed to load file: {e}")

    def detect_column_types(self):
        if self.data is None:
            return
            
        self.categorical_columns = self.data.select_dtypes(include=["object", "category"]).columns.tolist()
        self.continuous_columns = self.data.select_dtypes(include=["number"]).columns.tolist()
        self.datetime_columns = self.data.select_dtypes(include=["datetime"]).columns.tolist()

    def update_variable_selectors(self, event=None):
        plot_type = self.plot_type.get()
        all_columns = self.data.columns.tolist() if self.data is not None else []

        # Reset all selectors
        for selector in [self.x_var, self.y_var, self.color_var, self.size_var]:
            selector.set("")
            selector["state"] = "disabled"

        if plot_type == "Histogram":
            self.x_var["values"] = self.continuous_columns
            self.x_var["state"] = "readonly"
            self.color_var["values"] = self.categorical_columns
            self.color_var["state"] = "readonly"

        elif plot_type == "Scatterplot":
            self.x_var["values"] = self.continuous_columns
            self.y_var["values"] = self.continuous_columns
            self.color_var["values"] = all_columns
            self.size_var["values"] = self.continuous_columns
            for selector in [self.x_var, self.y_var, self.color_var, self.size_var]:
                selector["state"] = "readonly"

        elif plot_type == "Bar Plot":
            self.x_var["values"] = self.categorical_columns
            self.y_var["values"] = self.continuous_columns
            self.color_var["values"] = self.categorical_columns
            for selector in [self.x_var, self.y_var, self.color_var]:
                selector["state"] = "readonly"

        elif plot_type == "Time Series":
            self.x_var["values"] = self.datetime_columns
            self.y_var["values"] = self.continuous_columns
            self.color_var["values"] = self.categorical_columns
            for selector in [self.x_var, self.y_var, self.color_var]:
                selector["state"] = "readonly"

        elif plot_type == "Heatmap":
            self.x_var["values"] = all_columns
            self.y_var["values"] = all_columns
            for selector in [self.x_var, self.y_var]:
                selector["state"] = "readonly"

        elif plot_type in ["Box Plot", "Violin Plot"]:
            self.x_var["values"] = self.categorical_columns
            self.y_var["values"] = self.continuous_columns
            self.color_var["values"] = self.categorical_columns
            for selector in [self.x_var, self.y_var, self.color_var]:
                selector["state"] = "readonly"

    def generate_summary(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return
            
        # Update dataset selector
        self.stats_dataset['values'] = list(self.loaded_files.keys())
        if self.current_file:
            self.stats_dataset.set(self.current_file)
        
        # Clear all text widgets
        self.overview_text.delete(1.0, tk.END)
        self.detailed_text.delete(1.0, tk.END)
        self.quality_text.delete(1.0, tk.END)
        
        # Generate Database Overview
        self.overview_text.insert(tk.END, "=== Database Overview ===\n\n")
        self.overview_text.insert(tk.END, "Loaded Datasets:\n")
        for name, df in self.loaded_files.items():
            self.overview_text.insert(tk.END, f"\n{name}:\n")
            self.overview_text.insert(tk.END, f"  - Rows: {len(df)}\n")
            self.overview_text.insert(tk.END, f"  - Columns: {len(df.columns)}\n")
            self.overview_text.insert(tk.END, f"  - Memory Usage: {df.memory_usage().sum() / 1024**2:.2f} MB\n")
        
        self.overview_text.insert(tk.END, "\n=== Current Dataset Summary ===\n\n")
        self.overview_text.insert(tk.END, f"Dataset: {self.current_file}\n")
        self.overview_text.insert(tk.END, f"Total Rows: {len(self.data)}\n")
        self.overview_text.insert(tk.END, f"Total Columns: {len(self.data.columns)}\n")
        self.overview_text.insert(tk.END, f"Memory Usage: {self.data.memory_usage().sum() / 1024**2:.2f} MB\n\n")
        
        # Column types summary
        self.overview_text.insert(tk.END, "Column Types:\n")
        type_counts = self.data.dtypes.value_counts()
        for dtype, count in type_counts.items():
            self.overview_text.insert(tk.END, f"  - {dtype}: {count} columns\n")
        
        # Generate Detailed Statistics
        self.detailed_text.insert(tk.END, "=== Numerical Statistics ===\n\n")
        numeric_data = self.data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            stats = numeric_data.agg(['count', 'mean', 'std', 'min', 'max', 
                                    'skew', 'kurtosis', 
                                    lambda x: x.quantile(0.25),
                                    lambda x: x.quantile(0.75),
                                    lambda x: x.quantile(0.5)]).round(2)
            stats.index = ['Count', 'Mean', 'Std Dev', 'Min', 'Max', 
                         'Skewness', 'Kurtosis', '25th Percentile',
                         '75th Percentile', 'Median']
            self.detailed_text.insert(tk.END, stats.to_string())
        
        self.detailed_text.insert(tk.END, "\n\n=== Categorical Statistics ===\n\n")
        categorical_data = self.data.select_dtypes(include=['object', 'category'])
        for col in categorical_data.columns:
            self.detailed_text.insert(tk.END, f"\n{col}:\n")
            value_counts = self.data[col].value_counts()
            self.detailed_text.insert(tk.END, f"  - Unique Values: {len(value_counts)}\n")
            self.detailed_text.insert(tk.END, "  - Top 5 Values:\n")
            for val, count in value_counts.head().items():
                self.detailed_text.insert(tk.END, f"    {val}: {count} ({count/len(self.data)*100:.1f}%)\n")
        
        # Generate Data Quality Metrics
        self.quality_text.insert(tk.END, "=== Data Quality Metrics ===\n\n")
        
        # Missing Values
        missing = self.data.isnull().sum()
        missing_pct = (missing / len(self.data) * 100).round(2)
        self.quality_text.insert(tk.END, "Missing Values:\n")
        for col, count in missing.items():
            if count > 0:
                self.quality_text.insert(tk.END, 
                    f"  - {col}: {count} values ({missing_pct[col]}%)\n")
        
        # Duplicate Rows
        duplicates = self.data.duplicated().sum()
        self.quality_text.insert(tk.END, f"\nDuplicate Rows: {duplicates} ")
        self.quality_text.insert(tk.END, f"({duplicates/len(self.data)*100:.1f}%)\n")
        
        # Constant Columns
        constant_cols = [col for col in self.data.columns 
                        if self.data[col].nunique() == 1]
        if constant_cols:
            self.quality_text.insert(tk.END, "\nConstant Columns:\n")
            for col in constant_cols:
                self.quality_text.insert(tk.END, f"  - {col}\n")
        
        # Correlation Analysis
        self.quality_text.insert(tk.END, "\n=== Correlation Analysis ===\n")
        numeric_data = self.data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            corr = numeric_data.corr()
            # Find highly correlated pairs
            high_corr = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    if abs(corr.iloc[i,j]) > 0.7:
                        high_corr.append((corr.index[i], corr.columns[j], 
                                        corr.iloc[i,j]))
            
            if high_corr:
                self.quality_text.insert(tk.END, "\nHighly Correlated Features:\n")
                for col1, col2, corr_val in high_corr:
                    self.quality_text.insert(tk.END, 
                        f"  - {col1} & {col2}: {corr_val:.2f}\n")
        
        # Generate Distribution Plots
        self.plot_distributions()

    def plot_distributions(self):
        if not self.data.empty:
            # Clear previous figure
            fig = self.dist_canvas.figure
            fig.clear()
            
            # Select numeric columns
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            n_cols = len(numeric_cols)
            
            if n_cols > 0:
                # Calculate grid dimensions
                n_rows = (n_cols - 1) // 3 + 1
                n_cols = min(n_cols, 3)
                
                # Create subplots
                for i, col in enumerate(numeric_cols):
                    ax = fig.add_subplot(n_rows, n_cols, i+1)
                    sns.histplot(data=self.data, x=col, ax=ax)
                    ax.set_title(f'{col} Distribution')
                
                fig.tight_layout()
                self.dist_canvas.draw()

    def search_data(self, *args):
        if hasattr(self, 'pt'):
            search_text = self.search_var.get()
            if search_text:
                # Filter the data based on search text
                filtered_data = self.data[
                    self.data.astype(str).apply(
                        lambda x: x.str.contains(search_text, case=False, na=False)
                    ).any(axis=1)
                ]
                self.pt.model.df = filtered_data
                self.pt.redraw()
            else:
                self.pt.model.df = self.data
                self.pt.redraw()

    def refresh_preview(self):
        if self.data is not None:
            self.display_preview()

    def display_preview(self):
        # Clear existing table
        for widget in self.table_frame.winfo_children():
            widget.destroy()
            
        # Create new table
        self.pt = Table(self.table_frame, dataframe=self.data, 
                       showtoolbar=True, showstatusbar=True)
        self.pt.show()
        
        # Enable sorting
        self.pt.sortTable = True
        
        # Bind right-click menu
        self.pt.bind('<Button-3>', self.pt.popupMenu)
        
        # Automatically adjust column widths
        self.pt.autoResizeColumns()
        self.pt.redraw()

    def save_plot(self):
        if not hasattr(self, "current_fig") or self.current_fig is None:
            messagebox.showerror("Error", "No plot to save.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG Files", "*.png"),
                ("JPEG Files", "*.jpg"),
                ("PDF Files", "*.pdf"),
                ("HTML Files", "*.html"),
                ("SVG Files", "*.svg")
            ]
        )
        if not file_path:
            return

        try:
            # Save based on file extension
            extension = os.path.splitext(file_path)[1].lower()
            if extension == '.html':
                self.current_fig.write_html(file_path)
            else:
                self.current_fig.write_image(file_path)
            
            # Update status
            self.status_msg.config(text=f"Plot saved to: {os.path.basename(file_path)}")
            messagebox.showinfo("Success", "Plot saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save plot: {e}")

    def export_to_html(self):
        if not hasattr(self, "current_fig") or self.current_fig is None:
            messagebox.showerror("Error", "No plot to export.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML Files", "*.html")]
        )
        if not file_path:
            return

        try:
            self.current_fig.write_html(
                file_path,
                include_plotlyjs='cdn',
                full_html=True
            )
            # Update status
            self.status_msg.config(text=f"Plot exported to: {os.path.basename(file_path)}")
            messagebox.showinfo("Success", "Plot exported successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export plot: {e}")

    def display_plotly_plot(self, fig):
        # Update layout to fit the frame size
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            height=600,  # Fixed height
            width=self.plot_frame.winfo_width()
        )
        
        # Save to temporary HTML file
        fig.write_html(
            self.temp_html,
            include_plotlyjs='cdn',
            full_html=True,
            config={'displayModeBar': True}
        )
        
        try:
            # If window exists, update it
            if hasattr(self, 'browser') and self.browser:
                self.browser.load_url('file://' + os.path.abspath(self.temp_html))
            else:
                # Create new window
                self.browser = webview.create_window(
                    'Plot',
                    'file://' + os.path.abspath(self.temp_html),
                    width=self.plot_frame.winfo_width(),
                    height=600
                )
                webview.start(func=self.setup_webview_thread, debug=True)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display plot: {e}")

    def setup_webview_thread(self):
        import time
        # Wait for window to be created
        while not webview.windows:
            time.sleep(0.1)
        
        # Keep window alive
        while webview.windows:
            try:
                webview.windows[0].evaluate_js('')
                time.sleep(0.1)
            except:
                break

    def plot_data(self, event=None):
        if self.data is None:
            messagebox.showerror("Error", "Please load data first!")
            return

        plot_type = self.plot_type.get()
        x_var = self.x_var.get()
        y_var = self.y_var.get()
        color_var = self.color_var.get()
        size_var = self.size_var.get()
        
        # Set theme
        pio.templates.default = self.plot_theme.get()

        try:
            if plot_type == "Histogram":
                fig = px.histogram(
                    self.data, x=x_var, 
                    color=color_var if color_var else None,
                    color_discrete_sequence=px.colors.sequential.Viridis
                )

            elif plot_type == "Scatterplot":
                fig = px.scatter(
                    self.data, x=x_var, y=y_var,
                    color=color_var if color_var else None,
                    size=size_var if size_var else None,
                    color_continuous_scale=self.color_scheme.get()
                )

            elif plot_type == "Bar Plot":
                fig = px.bar(
                    self.data, x=x_var, y=y_var,
                    color=color_var if color_var else None,
                    color_discrete_sequence=px.colors.sequential.Viridis
                )

            elif plot_type == "Time Series":
                fig = px.line(
                    self.data, x=x_var, y=y_var,
                    color=color_var if color_var else None,
                    color_discrete_sequence=px.colors.sequential.Viridis
                )

            elif plot_type == "Heatmap":
                pivot_data = pd.pivot_table(
                    self.data, values=self.continuous_columns[0],
                    index=x_var, columns=y_var, aggfunc='mean'
                )
                fig = px.imshow(
                    pivot_data,
                    color_continuous_scale=self.color_scheme.get()
                )

            elif plot_type == "Box Plot":
                fig = px.box(
                    self.data, x=x_var, y=y_var,
                    color=color_var if color_var else None,
                    color_discrete_sequence=px.colors.sequential.Viridis
                )

            elif plot_type == "Violin Plot":
                fig = px.violin(
                    self.data, x=x_var, y=y_var,
                    color=color_var if color_var else None,
                    color_discrete_sequence=px.colors.sequential.Viridis
                )

            # Store current figure
            self.current_fig = fig
            
            # Display the plot
            self.display_plotly_plot(fig)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot data: {e}")

    def __del__(self):
        # Clean up temporary files
        if hasattr(self, 'temp_html') and os.path.exists(self.temp_html):
            try:
                os.remove(self.temp_html)
            except:
                pass

    def setup_main_content(self):
        # Main content area
        self.main_content = ttk.Frame(self.main_container)
        self.main_content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create notebook for different views
        self.notebook = ttk.Notebook(self.main_content)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.data_tab = ttk.Frame(self.notebook)
        self.plot_tab = ttk.Frame(self.notebook)
        self.stats_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.data_tab, text="Data Browser")
        self.notebook.add(self.plot_tab, text="Visualization")
        self.notebook.add(self.stats_tab, text="Statistics")
        
        self.setup_data_browser()
        self.setup_visualization_tab()
        self.setup_stats_view()

    def setup_data_browser(self):
        # Toolbar frame
        toolbar = ttk.Frame(self.data_tab)
        toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        # Add toolbar buttons
        ttk.Button(toolbar, text="Refresh", bootstyle="info",
                  command=self.refresh_preview).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Export", bootstyle="success",
                  command=self.export_data).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Filter", bootstyle="warning",
                  command=self.show_filter_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Remove File", bootstyle="danger",
                  command=self.remove_current_file).pack(side=tk.LEFT, padx=2)
        
        # Add Data Analysis buttons
        analysis_frame = ttk.LabelFrame(toolbar, text="Data Analysis")
        analysis_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(analysis_frame, text="Quick Insights",
                  bootstyle="info",
                  command=self.show_quick_insights).pack(side=tk.LEFT, padx=2)
        ttk.Button(analysis_frame, text="Handle Missing",
                  bootstyle="warning",
                  command=self.handle_missing_values).pack(side=tk.LEFT, padx=2)
        ttk.Button(analysis_frame, text="Remove Duplicates",
                  bootstyle="danger",
                  command=self.remove_duplicates).pack(side=tk.LEFT, padx=2)
        ttk.Button(analysis_frame, text="Convert Types",
                  bootstyle="success",
                  command=self.show_type_conversion).pack(side=tk.LEFT, padx=2)
        
        # Search frame
        search_frame = ttk.Frame(toolbar)
        search_frame.pack(side=tk.RIGHT, padx=5)
        
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self.search_data)
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(side=tk.LEFT, padx=5)
        
        # Table frame
        self.table_frame = ttk.Frame(self.data_tab)
        self.table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def export_data(self):
        if self.data is None:
            messagebox.showerror("Error", "No data to export!")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[
                ("CSV Files", "*.csv"),
                ("Excel Files", "*.xlsx"),
                ("JSON Files", "*.json"),
                ("Parquet Files", "*.parquet")
            ]
        )
        
        if not file_path:
            return
            
        try:
            extension = os.path.splitext(file_path)[1].lower()
            
            if extension == '.csv':
                self.data.to_csv(file_path, index=False)
            elif extension == '.xlsx':
                self.data.to_excel(file_path, index=False)
            elif extension == '.json':
                self.data.to_json(file_path, orient='records')
            elif extension == '.parquet':
                self.data.to_parquet(file_path, index=False)
                
            self.status_msg.config(text=f"Data exported to: {os.path.basename(file_path)}")
            messagebox.showinfo("Success", "Data exported successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export data: {e}")

    def show_filter_dialog(self):
        if self.data is None:
            messagebox.showerror("Error", "No data to filter!")
            return
            
        # Create filter dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Filter Data")
        dialog.geometry("600x400")
        
        # Column selection
        column_frame = ttk.Frame(dialog)
        column_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(column_frame, text="Column:").pack(side=tk.LEFT)
        column_var = ttk.Combobox(column_frame, values=list(self.data.columns), state="readonly")
        column_var.pack(side=tk.LEFT, padx=5)
        
        # Operator selection
        operator_frame = ttk.Frame(dialog)
        operator_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(operator_frame, text="Operator:").pack(side=tk.LEFT)
        operator_var = ttk.Combobox(operator_frame, 
                                  values=["equals", "not equals", "contains", "greater than", 
                                         "less than", "starts with", "ends with"],
                                  state="readonly")
        operator_var.pack(side=tk.LEFT, padx=5)
        
        # Value entry
        value_frame = ttk.Frame(dialog)
        value_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(value_frame, text="Value:").pack(side=tk.LEFT)
        value_var = ttk.Entry(value_frame)
        value_var.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        def apply_filter():
            column = column_var.get()
            operator = operator_var.get()
            value = value_var.get()
            
            if not all([column, operator, value]):
                messagebox.showerror("Error", "Please fill in all filter criteria!")
            return

            try:
                # Convert value to numeric if column is numeric
                if self.data[column].dtype in ['int64', 'float64']:
                    value = float(value)
                
                # Apply filter based on operator
                if operator == "equals":
                    filtered_data = self.data[self.data[column] == value]
                elif operator == "not equals":
                    filtered_data = self.data[self.data[column] != value]
                elif operator == "contains":
                    filtered_data = self.data[self.data[column].astype(str).str.contains(value, case=False)]
                elif operator == "greater than":
                    filtered_data = self.data[self.data[column] > value]
                elif operator == "less than":
                    filtered_data = self.data[self.data[column] < value]
                elif operator == "starts with":
                    filtered_data = self.data[self.data[column].astype(str).str.startswith(value)]
                elif operator == "ends with":
                    filtered_data = self.data[self.data[column].astype(str).str.endswith(value)]
                
                # Update table with filtered data
                self.pt.model.df = filtered_data
                self.pt.redraw()
                
                # Update status
                self.status_msg.config(text=f"Filtered: {len(filtered_data)} rows")
                dialog.destroy()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to apply filter: {e}")
        
        def reset_filter():
            self.pt.model.df = self.data
            self.pt.redraw()
            self.status_msg.config(text=f"Filter reset")
            dialog.destroy()
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="Apply", bootstyle="success",
                  command=apply_filter).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset", bootstyle="warning",
                  command=reset_filter).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", bootstyle="danger",
                  command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def remove_current_file(self):
        if self.current_file is None:
            messagebox.showerror("Error", "No file selected!")
            return
            
        if messagebox.askyesno("Confirm", f"Are you sure you want to remove {self.current_file}?"):
            # Remove from loaded files
            del self.loaded_files[self.current_file]
            
            # Remove from tree view
            for item in self.structure_tree.get_children(self.files_node):
                if self.structure_tree.item(item)['text'] == self.current_file:
                    self.structure_tree.delete(item)
                    break
            
            # Clear current file if it was the removed one
            if self.current_file == self.current_file:
                self.current_file = None
                self.data = None
                
                # Clear the table
                if hasattr(self, 'pt'):
                    self.pt.model.df = pd.DataFrame()
                    self.pt.redraw()
                
                # Update status
                self.status_msg.config(text="File removed")
                self.row_count.config(text="Rows: 0")
            
            # If there are other files, switch to the first one
            remaining_files = list(self.loaded_files.keys())
            if remaining_files:
                self.current_file = remaining_files[0]
                self.data = self.loaded_files[self.current_file]
                self.display_preview()
                self.status_msg.config(text=f"Switched to: {self.current_file}")
                self.row_count.config(text=f"Rows: {len(self.data)}")

    def setup_visualization_tab(self):
        # Plot options frame
        options_frame = ttk.LabelFrame(self.plot_tab, text="Plot Configuration", padding=10)
        options_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Plot type selection
        plot_type_frame = ttk.Frame(options_frame)
        plot_type_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(plot_type_frame, text="Plot Type:").pack(side=tk.LEFT, padx=5)
        self.plot_type = ttk.Combobox(
            plot_type_frame,
            values=[
                "Histogram", "Scatterplot", "Bar Plot", "Line Plot", 
                "Box Plot", "Violin Plot", "Heatmap", "3D Scatter",
                "Bubble Plot", "Area Plot", "Density Plot", 
                "Radar Chart", "Parallel Coordinates", "Time Series",
                "Pie Chart", "Donut Chart", "Sunburst", "Treemap"
            ],
            state="readonly",
            width=30
        )
        self.plot_type.pack(side=tk.LEFT, padx=5)
        self.plot_type.bind("<<ComboboxSelected>>", self.update_variable_selectors)
        
        # Plot title and labels
        title_frame = ttk.Frame(options_frame)
        title_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(title_frame, text="Title:").pack(side=tk.LEFT, padx=5)
        self.plot_title = ttk.Entry(title_frame, width=40)
        self.plot_title.pack(side=tk.LEFT, padx=5)
        
        # Variables frame
        vars_frame = ttk.Frame(options_frame)
        vars_frame.pack(fill=tk.X, pady=5)
        
        # X Variable
        ttk.Label(vars_frame, text="X:").pack(side=tk.LEFT, padx=5)
        self.x_var = ttk.Combobox(vars_frame, state="disabled", width=20)
        self.x_var.pack(side=tk.LEFT, padx=5)
        
        # Y Variable
        ttk.Label(vars_frame, text="Y:").pack(side=tk.LEFT, padx=5)
        self.y_var = ttk.Combobox(vars_frame, state="disabled", width=20)
        self.y_var.pack(side=tk.LEFT, padx=5)
        
        # Color Variable
        ttk.Label(vars_frame, text="Color:").pack(side=tk.LEFT, padx=5)
        self.color_var = ttk.Combobox(vars_frame, state="disabled", width=20)
        self.color_var.pack(side=tk.LEFT, padx=5)
        
        # Size Variable
        ttk.Label(vars_frame, text="Size:").pack(side=tk.LEFT, padx=5)
        self.size_var = ttk.Combobox(vars_frame, state="disabled", width=20)
        self.size_var.pack(side=tk.LEFT, padx=5)
        
        # Style options frame
        style_frame = ttk.LabelFrame(self.plot_tab, text="Style Options", padding=10)
        style_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Color scheme
        ttk.Label(style_frame, text="Color Scheme:").pack(side=tk.LEFT, padx=5)
        self.color_scheme = ttk.Combobox(
            style_frame,
            values=[
                "viridis", "plasma", "inferno", "magma", "cividis",
                "Plotly", "Plotly3", "D3", "G10", "T10", "Alphabet",
                "Dark24", "Light24", "Set1", "Set2", "Set3", "Pastel",
                "Bold", "Safe", "Vivid", "Prism"
            ],
            state="readonly",
            width=20
        )
        self.color_scheme.set("viridis")
        self.color_scheme.pack(side=tk.LEFT, padx=5)
        
        # Theme
        ttk.Label(style_frame, text="Theme:").pack(side=tk.LEFT, padx=5)
        self.plot_theme = ttk.Combobox(
            style_frame,
            values=[
                "plotly", "plotly_white", "plotly_dark", 
                "seaborn", "ggplot2", "simple_white"
            ],
            state="readonly",
            width=20
        )
        self.plot_theme.set("plotly")
        self.plot_theme.pack(side=tk.LEFT, padx=5)
        
        # Legend position
        ttk.Label(style_frame, text="Legend:").pack(side=tk.LEFT, padx=5)
        self.legend_pos = ttk.Combobox(
            style_frame,
            values=["top left", "top right", "bottom left", "bottom right"],
            state="readonly",
            width=20
        )
        self.legend_pos.set("top right")
        self.legend_pos.pack(side=tk.LEFT, padx=5)
        
        # Buttons frame
        button_frame = ttk.Frame(self.plot_tab)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Plot button
        ttk.Button(button_frame, text="Plot", 
                  bootstyle="success",
                  command=lambda: self.plot_data(None)).pack(side=tk.LEFT, padx=5)
        
        # Save button
        ttk.Button(button_frame, text="Save Plot",
                  bootstyle="info",
                  command=self.save_plot).pack(side=tk.LEFT, padx=5)
        
        # Export button
        ttk.Button(button_frame, text="Export to HTML",
                  bootstyle="warning",
                  command=self.export_to_html).pack(side=tk.LEFT, padx=5)
        
        # Plot display area
        self.plot_frame = ttk.Frame(self.plot_tab)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def setup_stats_view(self):
        # Statistics controls
        controls_frame = ttk.Frame(self.stats_tab)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add database selector with label frame
        selector_frame = ttk.LabelFrame(controls_frame, text="Dataset Selection", padding=5)
        selector_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.stats_dataset = ttk.Combobox(selector_frame, state="readonly", width=50)
        self.stats_dataset.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Add refresh button next to combobox
        ttk.Button(selector_frame, text="↻", width=3,
                  bootstyle="info",
                  command=self.refresh_dataset_list).pack(side=tk.LEFT, padx=2)
        
        # Bind selection event
        self.stats_dataset.bind("<<ComboboxSelected>>", self.update_statistics)
        
        # Controls frame for additional options
        options_frame = ttk.LabelFrame(controls_frame, text="Analysis Options", padding=5)
        options_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(options_frame, text="Generate Summary", 
                  bootstyle="info",
                  command=self.generate_summary).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(options_frame, text="Export Report",
                  bootstyle="success",
                  command=self.export_statistics).pack(side=tk.LEFT, padx=5)
        
        # Create notebook for different statistics views
        self.stats_notebook = ttk.Notebook(self.stats_tab)
        self.stats_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Overview tab
        self.overview_tab = ttk.Frame(self.stats_notebook)
        self.stats_notebook.add(self.overview_tab, text="Overview")
        
        # Create text widget for overview with scrollbar
        overview_frame = ttk.Frame(self.overview_tab)
        overview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        overview_scroll = ttk.Scrollbar(overview_frame)
        overview_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.overview_text = tk.Text(overview_frame, wrap=tk.WORD,
                                   yscrollcommand=overview_scroll.set)
        self.overview_text.pack(fill=tk.BOTH, expand=True)
        overview_scroll.config(command=self.overview_text.yview)
        
        # Detailed Statistics tab
        self.detailed_tab = ttk.Frame(self.stats_notebook)
        self.stats_notebook.add(self.detailed_tab, text="Detailed Statistics")
        
        # Create text widget for detailed stats with scrollbar
        detailed_frame = ttk.Frame(self.detailed_tab)
        detailed_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        detailed_scroll = ttk.Scrollbar(detailed_frame)
        detailed_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.detailed_text = tk.Text(detailed_frame, wrap=tk.WORD,
                                   yscrollcommand=detailed_scroll.set)
        self.detailed_text.pack(fill=tk.BOTH, expand=True)
        detailed_scroll.config(command=self.detailed_text.yview)
        
        # Data Quality tab
        self.quality_tab = ttk.Frame(self.stats_notebook)
        self.stats_notebook.add(self.quality_tab, text="Data Quality")
        
        # Create text widget for quality with scrollbar
        quality_frame = ttk.Frame(self.quality_tab)
        quality_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        quality_scroll = ttk.Scrollbar(quality_frame)
        quality_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.quality_text = tk.Text(quality_frame, wrap=tk.WORD,
                                  yscrollcommand=quality_scroll.set)
        self.quality_text.pack(fill=tk.BOTH, expand=True)
        quality_scroll.config(command=self.quality_text.yview)
        
        # Distribution Analysis tab
        self.dist_tab = ttk.Frame(self.stats_notebook)
        self.stats_notebook.add(self.dist_tab, text="Distributions")
        
        # Add controls for distribution plots
        dist_controls = ttk.Frame(self.dist_tab)
        dist_controls.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(dist_controls, text="Plot Type:").pack(side=tk.LEFT, padx=5)
        self.dist_plot_type = ttk.Combobox(dist_controls, 
            values=["Histogram", "Box Plot", "Violin Plot", "KDE"],
            state="readonly", width=20)
        self.dist_plot_type.set("Histogram")
        self.dist_plot_type.pack(side=tk.LEFT, padx=5)
        self.dist_plot_type.bind("<<ComboboxSelected>>", 
                               lambda e: self.plot_distributions())
        
        # Create canvas for distribution plots with toolbar
        self.dist_figure = plt.Figure(figsize=(10, 6))
        self.dist_canvas = FigureCanvasTkAgg(self.dist_figure, self.dist_tab)
        self.dist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add matplotlib toolbar
        toolbar_frame = ttk.Frame(self.dist_tab)
        toolbar_frame.pack(fill=tk.X)
        NavigationToolbar2Tk(self.dist_canvas, toolbar_frame)
        
        # Initialize the statistics view
        self.refresh_dataset_list()

    def refresh_dataset_list(self):
        """Update the dataset selector with currently loaded files."""
        current = self.stats_dataset.get()  # Store current selection
        
        # Update values
        files = list(self.loaded_files.keys())
        self.stats_dataset['values'] = files
        
        # Restore selection or select first item
        if current and current in files:
            self.stats_dataset.set(current)
        elif files:
            self.stats_dataset.set(files[0])
            self.update_statistics()
        
        # Update status
        self.status_msg.config(text=f"Found {len(files)} dataset(s)")

    def update_statistics(self, event=None):
        """Update statistics when a different dataset is selected."""
        selected = self.stats_dataset.get()
        if not selected:
            return
            
        try:
            # Update current data
            self.data = self.loaded_files[selected]
            self.current_file = selected
            
            # Generate new summary
            self.generate_summary()
            
            # Update status
            self.status_msg.config(text=f"Showing statistics for: {selected}")
            self.row_count.config(text=f"Rows: {len(self.data)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update statistics: {str(e)}")

    def export_statistics(self):
        """Export current statistics to a report file."""
        if not self.data is not None:
            messagebox.showerror("Error", "No data to export statistics for!")
            return
        
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".html",
                filetypes=[("HTML Report", "*.html"),
                          ("Text Report", "*.txt")]
            )
            
            if not file_path:
                return

            if file_path.endswith('.html'):
                self.export_html_report(file_path)
            else:
                self.export_text_report(file_path)
                
            messagebox.showinfo("Success", "Statistics exported successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export statistics: {str(e)}")

    def export_html_report(self, file_path):
        """Export statistics as an HTML report."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                # Write HTML header
                f.write("""
                <html>
                <head>
                    <title>Data Analysis Report</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; }
                        h1, h2 { color: #2c3e50; }
                        .section { margin: 20px 0; padding: 10px; border: 1px solid #eee; }
                        table { border-collapse: collapse; width: 100%; }
                        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                        th { background-color: #f5f5f5; }
                    </style>
                </head>
                <body>
                """)
                
                # Overview section
                f.write(f"<h1>Data Analysis Report - {self.current_file}</h1>")
                f.write("<div class='section'>")
                f.write(self.overview_text.get(1.0, tk.END).replace('\n', '<br>'))
                f.write("</div>")
                
                # Detailed Statistics
                f.write("<div class='section'>")
                f.write("<h2>Detailed Statistics</h2>")
                f.write(self.detailed_text.get(1.0, tk.END).replace('\n', '<br>'))
                f.write("</div>")
                
                # Data Quality
                f.write("<div class='section'>")
                f.write("<h2>Data Quality</h2>")
                f.write(self.quality_text.get(1.0, tk.END).replace('\n', '<br>'))
                f.write("</div>")
                
                # Save current plot
                if hasattr(self, 'dist_figure'):
                    plot_path = os.path.splitext(file_path)[0] + "_distributions.png"
                    self.dist_figure.savefig(plot_path)
                    f.write("<div class='section'>")
                    f.write("<h2>Distributions</h2>")
                    f.write(f"<img src='{os.path.basename(plot_path)}' alt='Distributions'>")
                    f.write("</div>")
                
                f.write("</body></html>")
                
        except Exception as e:
            raise Exception(f"Failed to export HTML report: {str(e)}")

    def export_text_report(self, file_path):
        """Export statistics as a text report."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"Data Analysis Report - {self.current_file}\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("Overview\n")
                f.write("-" * 20 + "\n")
                f.write(self.overview_text.get(1.0, tk.END))
                f.write("\n")
                
                f.write("Detailed Statistics\n")
                f.write("-" * 20 + "\n")
                f.write(self.detailed_text.get(1.0, tk.END))
                
                f.write("Data Quality\n")
                f.write("-" * 20 + "\n")
                f.write(self.quality_text.get(1.0, tk.END))
                
        except Exception as e:
            raise Exception(f"Failed to export text report: {str(e)}")

    def plot_distributions(self):
        """Generate distribution plots for numerical columns."""
        if not hasattr(self, 'data') or self.data is None or self.data.empty:
            return
            
        try:
            # Clear previous figure
            self.dist_figure.clear()
            
            # Select numeric columns
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            n_cols = len(numeric_cols)
            
            if n_cols > 0:
                # Calculate grid dimensions
                n_rows = (n_cols - 1) // 3 + 1
                n_cols = min(n_cols, 3)
                
                plot_type = self.dist_plot_type.get()
                
                # Create subplots
                for i, col in enumerate(numeric_cols):
                    ax = self.dist_figure.add_subplot(n_rows, n_cols, i+1)
                    
                    if plot_type == "Histogram":
                        sns.histplot(data=self.data, x=col, ax=ax)
                    elif plot_type == "Box Plot":
                        sns.boxplot(data=self.data, y=col, ax=ax)
                    elif plot_type == "Violin Plot":
                        sns.violinplot(data=self.data, y=col, ax=ax)
                    elif plot_type == "KDE":
                        sns.kdeplot(data=self.data[col], ax=ax)
                    
                    ax.set_title(f'{col} Distribution')
                
                self.dist_figure.tight_layout()
                self.dist_canvas.draw()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot distributions: {str(e)}")

    def show_quick_insights(self):
        if self.data is None:
            messagebox.showerror("Error", "No data to analyze!")
            return
            
        # Create insights window
        insights = tk.Toplevel(self.root)
        insights.title("Quick Insights")
        insights.geometry("800x600")
        
        # Create notebook for different insights
        notebook = ttk.Notebook(insights)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Summary Statistics
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="Summary Stats")
        
        stats_text = tk.Text(stats_frame, wrap=tk.WORD)
        stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add summary statistics
        stats_text.insert(tk.END, "=== Numerical Summary ===\n\n")
        stats_text.insert(tk.END, self.data.describe().to_string())
        stats_text.insert(tk.END, "\n\n=== Data Types ===\n\n")
        stats_text.insert(tk.END, self.data.dtypes.to_string())
        
        # Correlation Analysis
        corr_frame = ttk.Frame(notebook)
        notebook.add(corr_frame, text="Correlations")
        
        # Only calculate correlations for numeric columns
        numeric_data = self.data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            fig = plt.Figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            sns.heatmap(numeric_data.corr(), annot=True, ax=ax, cmap='coolwarm')
            
            canvas = FigureCanvasTkAgg(fig, corr_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Missing Values Analysis
        missing_frame = ttk.Frame(notebook)
        notebook.add(missing_frame, text="Missing Values")
        
        missing_text = tk.Text(missing_frame, wrap=tk.WORD)
        missing_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        missing_text.insert(tk.END, "=== Missing Values Summary ===\n\n")
        missing_info = pd.DataFrame({
            'Missing Values': self.data.isnull().sum(),
            'Percentage': (self.data.isnull().sum() / len(self.data) * 100).round(2)
        })
        missing_text.insert(tk.END, missing_info.to_string())

    def handle_missing_values(self):
        if self.data is None:
            messagebox.showerror("Error", "No data to process!")
            return
            
        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Handle Missing Values")
        dialog.geometry("500x400")
        
        # Column selection
        column_frame = ttk.LabelFrame(dialog, text="Select Columns")
        column_frame.pack(fill=tk.X, padx=5, pady=5)
        
        columns_with_missing = [col for col in self.data.columns 
                              if self.data[col].isnull().any()]
        
        column_vars = {}
        for col in columns_with_missing:
            var = tk.BooleanVar(value=True)
            column_vars[col] = var
            ttk.Checkbutton(column_frame, text=col, variable=var).pack(anchor=tk.W)
        
        # Strategy selection
        strategy_frame = ttk.LabelFrame(dialog, text="Fill Strategy")
        strategy_frame.pack(fill=tk.X, padx=5, pady=5)
        
        strategy_var = tk.StringVar(value="mean")
        strategies = {
            "mean": "Mean (numeric only)",
            "median": "Median (numeric only)",
            "mode": "Mode (any type)",
            "ffill": "Forward Fill",
            "bfill": "Backward Fill",
            "drop": "Drop Rows",
            "value": "Custom Value"
        }
        
        for val, text in strategies.items():
            ttk.Radiobutton(strategy_frame, text=text, 
                          value=val, variable=strategy_var).pack(anchor=tk.W)
        
        # Custom value entry
        value_frame = ttk.Frame(dialog)
        value_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(value_frame, text="Custom Value:").pack(side=tk.LEFT)
        custom_value = ttk.Entry(value_frame)
        custom_value.pack(side=tk.LEFT, padx=5)
        
        def apply_strategy():
            selected_cols = [col for col, var in column_vars.items() if var.get()]
            if not selected_cols:
                messagebox.showerror("Error", "Please select at least one column!")
                return
                
            strategy = strategy_var.get()
            try:
                for col in selected_cols:
                    if strategy == "mean":
                        if np.issubdtype(self.data[col].dtype, np.number):
                            self.data[col].fillna(self.data[col].mean(), inplace=True)
                    elif strategy == "median":
                        if np.issubdtype(self.data[col].dtype, np.number):
                            self.data[col].fillna(self.data[col].median(), inplace=True)
                    elif strategy == "mode":
                        self.data[col].fillna(self.data[col].mode()[0], inplace=True)
                    elif strategy == "ffill":
                        self.data[col].fillna(method='ffill', inplace=True)
                    elif strategy == "bfill":
                        self.data[col].fillna(method='bfill', inplace=True)
                    elif strategy == "drop":
                        self.data.dropna(subset=[col], inplace=True)
                    elif strategy == "value":
                        value = custom_value.get()
                        if np.issubdtype(self.data[col].dtype, np.number):
                            try:
                                value = float(value)
                            except:
                                pass
                        self.data[col].fillna(value, inplace=True)
                
                self.display_preview()
                self.status_msg.config(text="Missing values handled successfully")
                dialog.destroy()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to handle missing values: {e}")
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Apply", 
                  bootstyle="success",
                  command=apply_strategy).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel",
                  bootstyle="danger", 
                  command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def remove_duplicates(self):
        if self.data is None:
            messagebox.showerror("Error", "No data to process!")
            return
            
        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Remove Duplicates")
        dialog.geometry("400x500")
        
        # Column selection
        column_frame = ttk.LabelFrame(dialog, text="Select Columns to Consider")
        column_frame.pack(fill=tk.X, padx=5, pady=5)
        
        column_vars = {}
        for col in self.data.columns:
            var = tk.BooleanVar(value=True)
            column_vars[col] = var
            ttk.Checkbutton(column_frame, text=col, variable=var).pack(anchor=tk.W)
        
        # Options
        options_frame = ttk.LabelFrame(dialog, text="Options")
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        keep_var = tk.StringVar(value="first")
        ttk.Radiobutton(options_frame, text="Keep First Occurrence", 
                       value="first", variable=keep_var).pack(anchor=tk.W)
        ttk.Radiobutton(options_frame, text="Keep Last Occurrence", 
                       value="last", variable=keep_var).pack(anchor=tk.W)
        
        def remove_dups():
            selected_cols = [col for col, var in column_vars.items() if var.get()]
            if not selected_cols:
                messagebox.showerror("Error", "Please select at least one column!")
                return
                
            try:
                original_len = len(self.data)
                self.data.drop_duplicates(subset=selected_cols, 
                                       keep=keep_var.get(), 
                                       inplace=True)
                removed = original_len - len(self.data)
                
                self.display_preview()
                self.status_msg.config(text=f"Removed {removed} duplicate rows")
                dialog.destroy()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to remove duplicates: {e}")
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Remove", 
                  bootstyle="success",
                  command=remove_dups).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel",
                  bootstyle="danger", 
                  command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def show_type_conversion(self):
        if self.data is None:
            messagebox.showerror("Error", "No data to process!")
            return
            
        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Convert Data Types")
        dialog.geometry("500x600")
        
        # Create scrollable frame
        canvas = tk.Canvas(dialog)
        scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Column type selection
        type_vars = {}
        current_types = self.data.dtypes
        
        for col in self.data.columns:
            frame = ttk.LabelFrame(scrollable_frame, text=col)
            frame.pack(fill=tk.X, padx=5, pady=2)
            
            type_var = tk.StringVar(value=str(current_types[col]))
            type_vars[col] = type_var
            
            types = ["int64", "float64", "str", "datetime64[ns]", "category", "bool"]
            combo = ttk.Combobox(frame, values=types, 
                               textvariable=type_var, state="readonly")
            combo.pack(side=tk.LEFT, padx=5, pady=2)
            
            ttk.Label(frame, text=f"Current: {current_types[col]}").pack(
                side=tk.LEFT, padx=5)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        def apply_conversions():
            try:
                for col, type_var in type_vars.items():
                    new_type = type_var.get()
                    if str(self.data[col].dtype) != new_type:
                        try:
                            if new_type == "datetime64[ns]":
                                self.data[col] = pd.to_datetime(self.data[col])
                            else:
                                self.data[col] = self.data[col].astype(new_type)
                        except Exception as e:
                            messagebox.showerror("Error", 
                                f"Failed to convert {col} to {new_type}: {e}")
                            return
                
                self.display_preview()
                self.status_msg.config(text="Data types converted successfully")
                dialog.destroy()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to convert data types: {e}")
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Apply", 
                  bootstyle="success",
                  command=apply_conversions).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel",
                  bootstyle="danger", 
                  command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def setup_status_bar(self):
        # Status bar
        self.status_bar = ttk.Frame(self.root, relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Status message
        self.status_msg = ttk.Label(self.status_bar, text="Ready")
        self.status_msg.pack(side=tk.LEFT, padx=5)
        
        # Row count
        self.row_count = ttk.Label(self.status_bar, text="Rows: 0")
        self.row_count.pack(side=tk.RIGHT, padx=5)

if __name__ == "__main__":
    root = tk.Tk()
    plotter = DataPlotterApp(root)
    root.mainloop()
