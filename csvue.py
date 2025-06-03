import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from ttkbootstrap import Style
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
from pandastable import Table
from PIL import Image, ImageTk
import io
import os
import json
import yaml
import sqlite3
import duckdb
import polars as pl
from datetime import datetime
from scipy.interpolate import griddata
import psutil
import threading
import pyreadr
from scipy.stats import gaussian_kde

class SQLiteManager:
    def __init__(self):
        self.connections = {}
        self.locks = {}
        
    def connect(self, db_path):
        """Create or return an existing connection with proper settings."""
        if db_path not in self.connections:
            try:
                # Enable proper threading support
                conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
                # Enable foreign key support
                conn.execute("PRAGMA foreign_keys = ON")
                # Enable extended error codes for better error handling
                conn.execute("PRAGMA full_column_names = ON")
                # Use write-ahead logging for better concurrency
                conn.execute("PRAGMA journal_mode = WAL")
                # Create lock for this connection
                self.locks[db_path] = threading.Lock()
                self.connections[db_path] = conn
            except sqlite3.Error as e:
                raise Exception(f"Failed to connect to database: {str(e)}")
        return self.connections[db_path]
    
    def execute_query(self, db_path, query, params=None):
        """Execute a query with proper locking and error handling."""
        if db_path not in self.connections:
            raise Exception("Database not connected")
            
        conn = self.connections[db_path]
        lock = self.locks[db_path]
        
        with lock:
            try:
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                    
                if query.lower().strip().startswith(('select', 'pragma', 'explain')):
                    return cursor.fetchall()
                else:
                    conn.commit()
                    return cursor.rowcount
            except sqlite3.Error as e:
                conn.rollback()
                raise Exception(f"SQL error: {str(e)}")
            finally:
                cursor.close()
    
    def close_all(self):
        """Safely close all database connections."""
        for conn in self.connections.values():
            try:
                conn.close()
            except:
                pass
        self.connections.clear()
        self.locks.clear()
    
    def close(self, db_path):
        """Safely close a specific database connection."""
        if db_path in self.connections:
            try:
                self.connections[db_path].close()
                del self.connections[db_path]
                del self.locks[db_path]
            except:
                pass

class DataPlotterApp:
    def __init__(self, root):
        self.root = root
        self.style = Style(theme="darkly")
        self.root.title("CSVUE")
        self.root.geometry("1400x900")

        # Initialize variables
        self.loaded_files = {}  # Dictionary to store all loaded dataframes
        self.db_connections = {}  # Dictionary to store database connections
        self.current_file = None  # Currently selected file
        self.data = None  # Current dataframe
        self.filtered_data = None  # Filtered view of current dataframe
        self.categorical_columns = []
        self.continuous_columns = []
        self.datetime_columns = []
        self.current_fig = None
        self.preview_window = None
        self.last_directory = os.path.expanduser("~")
        self.temp_html = "temp_plot.html"
        
        # Initialize SQLite manager
        self.sqlite_manager = SQLiteManager()
        
        # Setup status bar first
        self.setup_status_bar()
        
        # Create main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Setup components
        self.setup_left_sidebar()
        self.setup_main_content()
        
        # Start memory monitoring
        self.update_memory_usage()
        
        # Bind cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def on_closing(self):
        """Handle application cleanup on closing."""
        try:
            # Close all database connections
            self.sqlite_manager.close_all()
            
            # Remove temporary files
            if hasattr(self, 'temp_html') and os.path.exists(self.temp_html):
                try:
                    os.remove(self.temp_html)
                except:
                    pass
                    
            # Destroy the window
            self.root.destroy()
        except:
            # If cleanup fails, force close
            self.root.destroy()

    def setup_status_bar(self):
        """Setup enhanced status bar with memory monitoring."""
        # Status bar frame with border and padding
        self.status_bar = ttk.Frame(self.root, style='Secondary.TFrame', relief='sunken', border=1)
    def on_window_resize(self, event):
        """Handle window resize events"""
        if hasattr(self, 'current_fig') and self.current_fig:
            self.fig.set_size_inches(
                self.plot_frame.winfo_width() / self.fig.get_dpi(),
                self.plot_frame.winfo_height() / self.fig.get_dpi()
            )
            self.canvas.draw()

    def setup_left_sidebar(self):
        # Left sidebar frame
        self.sidebar = ttk.Frame(self.main_container, style='Secondary.TFrame', relief='raised')
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)
        
        # Logo frame at the top
        logo_frame = ttk.Frame(self.sidebar)
        logo_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Load and display logo
        logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
        if os.path.exists(logo_path):
            # Load the image
            logo_image = Image.open(logo_path)
            # Resize to fit sidebar (150px width while maintaining aspect ratio)
            aspect_ratio = logo_image.height / logo_image.width
            new_width = 100
            new_height = int(new_width * aspect_ratio)
            logo_image = logo_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            # Convert to PhotoImage
            logo_photo = ImageTk.PhotoImage(logo_image)
            # Create and pack label
            logo_label = ttk.Label(logo_frame, image=logo_photo)
            logo_label.image = logo_photo  # Keep a reference!
            logo_label.pack(pady=10)
        
        # Add separator
        ttk.Separator(self.sidebar, orient='horizontal').pack(fill=tk.X, padx=5, pady=10)
        
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
        
        # Get the full path to the selected item
        path = []
        current = selected_item
        while current:
            path.insert(0, self.structure_tree.item(current)['text'])
            current = self.structure_tree.parent(current)
        
        # Remove "Loaded Files" from path
        if path and path[0] == "Loaded Files":
            path.pop(0)
        
        if not path:
            return
            
        # Handle database tables
        if len(path) >= 2:  # Database/table path
            full_name = f"{path[0]}/{path[1]}"
            if full_name in self.loaded_files:
                self.current_file = full_name
                self.data = self.loaded_files[full_name]
                self.detect_column_types()
                self.update_variable_selectors(None)
                self.display_preview()
                
                # Update status bar
                self.status_msg.config(text=f"Current table: {path[1]} from {path[0]}")
                self.row_count.config(text=f"Rows: {len(self.data)}")
        # Handle single files
        elif path[0] in self.loaded_files:
            self.current_file = path[0]
            self.data = self.loaded_files[path[0]]
            self.detect_column_types()
            self.update_variable_selectors(None)
            self.display_preview()
            
            # Update status bar
            self.status_msg.config(text=f"Current file: {path[0]}")
            self.row_count.config(text=f"Rows: {len(self.data)}")

    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Open Data File",
            filetypes=[
                ("All Supported Files", "*.csv *.tsv *.xlsx *.xls *.xlsm *.json *.parquet *.feather *.arrow *.pickle *.pkl *.sqlite *.db *.sqlite3 *.hdf *.h5 *.sas7bdat *.sav *.dta *.xml *.rds"),
                ("CSV/TSV Files", "*.csv *.tsv"),
                ("Excel Files", "*.xlsx *.xls *.xlsm"),
                ("JSON Files", "*.json *.jsonl"),
                ("Parquet/Arrow Files", "*.parquet *.feather *.arrow"),
                ("SQLite Database", "*.sqlite *.db *.sqlite3"),
                ("HDF5 Files", "*.hdf *.h5"),
                ("Statistical Files", "*.sas7bdat *.sav *.dta *.rds"),  # Added RDS
                ("Pickle Files", "*.pickle *.pkl"),
                ("XML Files", "*.xml"),
                ("All Files", "*.*")
            ],
            initialdir=self.last_directory
        )
        if not file_path:
            return

        try:
            # Load the data
            filename = os.path.basename(file_path)
            extension = os.path.splitext(file_path)[1].lower()
            
            if extension in ['.csv', '.tsv']:
                separator = ',' if extension == '.csv' else '\t'
                df = pd.read_csv(file_path, sep=separator)
                self.status_type.config(text="🗎")  # File icon
            elif extension in ['.xlsx', '.xls', '.xlsm']:
                df = pd.read_excel(file_path)
                self.status_type.config(text="🗎")  # File icon
            elif extension in ['.json', '.jsonl']:
                df = pd.read_json(file_path)
                self.status_type.config(text="🗎")  # File icon
            elif extension == '.parquet':
                df = pd.read_parquet(file_path)
                self.status_type.config(text="🗎")  # File icon
            elif extension == '.feather':
                df = pd.read_feather(file_path)
                self.status_type.config(text="🗎")  # File icon
            elif extension == '.arrow':
                import pyarrow.feather as feather
                df = feather.read_feather(file_path)
                self.status_type.config(text="🗎")  # File icon
            elif extension in ['.pickle', '.pkl']:
                df = pd.read_pickle(file_path)
                self.status_type.config(text="🗎")  # File icon
            elif extension in ['.hdf', '.h5']:
                df = pd.read_hdf(file_path)
                self.status_type.config(text="🗎")  # File icon
            elif extension == '.sas7bdat':
                df = pd.read_sas(file_path)
                self.status_type.config(text="🗎")  # File icon
            elif extension == '.sav':
                df = pd.read_spss(file_path)
                self.status_type.config(text="🗎")  # File icon
            elif extension == '.dta':
                df = pd.read_stata(file_path)
                self.status_type.config(text="🗎")  # File icon
            elif extension == '.xml':
                df = pd.read_xml(file_path)
                self.status_type.config(text="🗎")  # File icon
            elif extension == '.rds':
                result = pyreadr.read_r(file_path)
                # RDS files can contain multiple objects, but typically contain one
                # Get the first (and usually only) dataframe from the result
                df = next(iter(result.values()))
                self.status_type.config(text="🗎")  # File icon
            elif extension in ['.sqlite', '.db', '.sqlite3']:
                # Connect to SQLite database
                conn = sqlite3.connect(file_path)
                self.status_type.config(text="🗃")  # Database icon
                
                # Get list of tables
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                if not tables:
                    messagebox.showinfo("Info", "No tables found in database!")
                    return
                
                # Create a node for the database
                db_node = self.structure_tree.insert(self.files_node, "end", text=filename)
                
                # Load each table
                for table_name in [t[0] for t in tables]:
                    try:
                        # Load table into dataframe
                        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
                        
                        # Store with database prefix to avoid name conflicts
                        full_name = f"{filename}/{table_name}"
                        self.loaded_files[full_name] = df
                        
                        # Add table node under database node
                        table_node = self.structure_tree.insert(db_node, "end", text=table_name)
                        
                        # Add columns under table node
                        for col in df.columns:
                            self.structure_tree.insert(table_node, "end", text=col)
                        
                        # If this is the first table, select it
                        if not self.current_file:
                            self.current_file = full_name
                            self.data = df
                            
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to load table {table_name}: {e}")
                
                # Store the connection
                if not hasattr(self, 'db_connections'):
                    self.db_connections = {}
                self.db_connections[filename] = conn
                
                # Update display for first table
                if self.data is not None:
                    self.detect_column_types()
                    self.update_variable_selectors(None)
                    self.display_preview()
                    self.status_msg.config(text=f"Loaded database: {filename}")
                    self.row_count.config(text=f"Rows: {len(self.data)}")
                return
            else:
                raise ValueError("Unsupported file format")

            # For non-SQLite files, continue with normal loading
            self.loaded_files[filename] = df
            self.current_file = filename
            self.data = df
            self.filtered_data = df.copy()  # Initialize filtered data
            
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

    def update_variable_selectors(self, event):
        plot_type = self.plot_type.get()
        all_columns = self.data.columns.tolist() if self.data is not None else []

        # Reset all selectors
        for selector in [self.x_var, self.y_var, self.color_var]:
            selector.set("")
            selector["state"] = "disabled"

        if plot_type == "Histogram":
            self.x_var["values"] = self.continuous_columns
            self.x_var["state"] = "readonly"
            self.color_var["values"] = self.categorical_columns
            self.color_var["state"] = "readonly"

        elif plot_type == "Scatter Plot":
            self.x_var["values"] = self.continuous_columns
            self.y_var["values"] = self.continuous_columns
            self.color_var["values"] = all_columns
            for selector in [self.x_var, self.y_var, self.color_var]:
                selector["state"] = "readonly"

        elif plot_type == "Bar Plot":
            self.x_var["values"] = self.categorical_columns
            self.y_var["values"] = self.continuous_columns
            self.color_var["values"] = self.categorical_columns
            for selector in [self.x_var, self.y_var, self.color_var]:
                selector["state"] = "readonly"

        elif plot_type == "Line Plot":
            self.x_var["values"] = self.continuous_columns + self.datetime_columns
            self.y_var["values"] = self.continuous_columns
            self.color_var["values"] = self.categorical_columns
            for selector in [self.x_var, self.y_var, self.color_var]:
                selector["state"] = "readonly"

        elif plot_type == "Heatmap":
            self.x_var["values"] = ["None"] + all_columns
            self.y_var["values"] = ["None"] + all_columns
            self.color_var["values"] = ["None"] + self.continuous_columns
            for selector in [self.x_var, self.y_var, self.color_var]:
                selector["state"] = "readonly"
            
            # Add heatmap specific options
            heatmap_opts = ttk.Frame(self.plot_specific_frame)
            heatmap_opts.pack(fill=tk.X, pady=5)
            
            # Annotations
            self.plot_specific_options['annot'] = tk.BooleanVar(value=True)
            ttk.Checkbutton(heatmap_opts, text="Show Values", 
                          variable=self.plot_specific_options['annot']).pack(side=tk.LEFT, padx=5)
            
            # Format string for annotations
            ttk.Label(heatmap_opts, text="Format:").pack(side=tk.LEFT, padx=5)
            self.plot_specific_options['fmt'] = ttk.Entry(heatmap_opts, width=8)
            self.plot_specific_options['fmt'].insert(0, '.2g')
            self.plot_specific_options['fmt'].pack(side=tk.LEFT, padx=5)
            
            # Center value
            ttk.Label(heatmap_opts, text="Center:").pack(side=tk.LEFT, padx=5)
            self.plot_specific_options['center'] = ttk.Entry(heatmap_opts, width=8)
            self.plot_specific_options['center'].pack(side=tk.LEFT, padx=5)
            
            # Square cells
            self.plot_specific_options['square'] = tk.BooleanVar(value=True)
            ttk.Checkbutton(heatmap_opts, text="Square Cells", 
                          variable=self.plot_specific_options['square']).pack(side=tk.LEFT, padx=5)
            
            # Line widths
            ttk.Label(heatmap_opts, text="Line Width:").pack(side=tk.LEFT, padx=5)
            self.plot_specific_options['linewidths'] = ttk.Spinbox(
                heatmap_opts, from_=0, to=2, increment=0.1, width=5)
            self.plot_specific_options['linewidths'].set(0.5)
            self.plot_specific_options['linewidths'].pack(side=tk.LEFT, padx=5)
            
            # Robust scaling
            self.plot_specific_options['robust'] = tk.BooleanVar(value=False)
            ttk.Checkbutton(heatmap_opts, text="Robust Scaling", 
                          variable=self.plot_specific_options['robust']).pack(side=tk.LEFT, padx=5)

        elif plot_type in ["Box Plot", "Violin Plot"]:
            self.x_var["values"] = self.categorical_columns
            self.y_var["values"] = self.continuous_columns
            self.color_var["values"] = self.categorical_columns
            for selector in [self.x_var, self.y_var, self.color_var]:
                selector["state"] = "readonly"

        elif plot_type == "Pie Chart":
            self.x_var["values"] = ["None"] + self.continuous_columns
            self.x_var["state"] = "readonly"
            self.color_var["values"] = ["None"] + self.categorical_columns
            self.color_var["state"] = "readonly"
            self.y_var["state"] = "disabled"
            
            # Add pie chart specific options
            pie_opts = ttk.Frame(self.plot_specific_frame)
            pie_opts.pack(fill=tk.X, pady=5)
            
            # Explode
            ttk.Label(pie_opts, text="Explode:").pack(side=tk.LEFT, padx=5)
            self.plot_specific_options['explode'] = ttk.Entry(pie_opts, width=15)
            self.plot_specific_options['explode'].insert(0, "0.0")
            self.plot_specific_options['explode'].pack(side=tk.LEFT, padx=5)
            ttk.Label(pie_opts, text="(comma-separated values)").pack(side=tk.LEFT)
            
            # Start angle
            ttk.Label(pie_opts, text="Start Angle:").pack(side=tk.LEFT, padx=5)
            self.plot_specific_options['startangle'] = ttk.Spinbox(
                pie_opts, from_=0, to=360, increment=45, width=5)
            self.plot_specific_options['startangle'].set(0)
            self.plot_specific_options['startangle'].pack(side=tk.LEFT, padx=5)
            
            # Shadow
            self.plot_specific_options['shadow'] = tk.BooleanVar(value=False)
            ttk.Checkbutton(pie_opts, text="Shadow", 
                          variable=self.plot_specific_options['shadow']).pack(side=tk.LEFT, padx=5)
            
            # Auto percent
            self.plot_specific_options['autopct'] = tk.BooleanVar(value=True)
            ttk.Checkbutton(pie_opts, text="Show Percentages", 
                          variable=self.plot_specific_options['autopct']).pack(side=tk.LEFT, padx=5)
            
            # Radius
            ttk.Label(pie_opts, text="Radius:").pack(side=tk.LEFT, padx=5)
            self.plot_specific_options['radius'] = ttk.Spinbox(
                pie_opts, from_=0.1, to=2.0, increment=0.1, width=5)
            self.plot_specific_options['radius'].set(1.0)
            self.plot_specific_options['radius'].pack(side=tk.LEFT, padx=5)

        elif plot_type == "Density Plot":
            self.x_var["values"] = ["None"] + self.continuous_columns
            self.x_var["state"] = "readonly"
            self.color_var["values"] = ["None"] + self.categorical_columns
            self.color_var["state"] = "readonly"
            self.y_var["state"] = "disabled"
            
            # Add density plot specific options
            density_opts = ttk.Frame(self.plot_specific_frame)
            density_opts.pack(fill=tk.X, pady=5)
            
            # Bandwidth adjustment
            ttk.Label(density_opts, text="Bandwidth Factor:").pack(side=tk.LEFT, padx=5)
            self.plot_specific_options['bw_factor'] = ttk.Spinbox(
                density_opts, from_=0.1, to=2.0, increment=0.05, width=5)
            self.plot_specific_options['bw_factor'].set(0.25)
            self.plot_specific_options['bw_factor'].pack(side=tk.LEFT, padx=5)
            
            # Fill
            self.plot_specific_options['fill'] = tk.BooleanVar(value=True)
            ttk.Checkbutton(density_opts, text="Fill", 
                          variable=self.plot_specific_options['fill']).pack(side=tk.LEFT, padx=5)
            
            # Alpha (transparency)
            ttk.Label(density_opts, text="Transparency:").pack(side=tk.LEFT, padx=5)
            self.plot_specific_options['alpha'] = ttk.Spinbox(
                density_opts, from_=0.0, to=1.0, increment=0.1, width=5)
            self.plot_specific_options['alpha'].set(0.4)
            self.plot_specific_options['alpha'].pack(side=tk.LEFT, padx=5)
            
            # Number of points
            ttk.Label(density_opts, text="Points:").pack(side=tk.LEFT, padx=5)
            self.plot_specific_options['points'] = ttk.Spinbox(
                density_opts, from_=50, to=500, increment=50, width=5)
            self.plot_specific_options['points'].set(200)
            self.plot_specific_options['points'].pack(side=tk.LEFT, padx=5)

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
                ("SVG Files", "*.svg")
            ]
        )
        if not file_path:
            return

        try:
            self.current_fig.savefig(file_path, bbox_inches='tight', dpi=300)
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

    def plot_data(self, event=None):
        if self.data is None:
            messagebox.showerror("Error", "Please load data first!")
            return
            
        if self.filtered_data is None:
            self.filtered_data = self.data.copy()

        try:
            plot_type = self.plot_type.get()
            x_var = self.x_var.get()
            y_var = self.y_var.get()
            color_var = self.color_var.get()
            
            # Handle None values in variable selections
            if x_var == "None": x_var = None
            if y_var == "None": y_var = None
            if color_var == "None": color_var = None
            
            # Safely initialize plot_opts from plot_specific_options
            plot_opts = {}
            if hasattr(self, 'plot_specific_options'):
                for key, var in self.plot_specific_options.items():
                    try:
                        if isinstance(var, (tk.BooleanVar, tk.StringVar, tk.IntVar, tk.DoubleVar)):
                            plot_opts[key] = var.get()
                        elif hasattr(var, 'get') and hasattr(var, 'winfo_exists'):
                            # For Tkinter widgets, check if they still exist
                            if var.winfo_exists():
                                plot_opts[key] = var.get()
                            else:
                                plot_opts[key] = None
                        else:
                            plot_opts[key] = var
                    except (tk.TclError, AttributeError, ValueError):
                        # If any error occurs while getting the value, skip it
                        plot_opts[key] = None
            
            # Validate required variables based on plot type
            if plot_type in ["Scatter Plot", "Line Plot"] and not (x_var and y_var):
                messagebox.showerror("Error", "Please select both X and Y variables!")
                return
            elif plot_type in ["Bar Plot", "Box Plot", "Violin Plot"] and not (x_var and y_var):
                messagebox.showerror("Error", "Please select both X and Y variables!")
                return
            elif plot_type == "Histogram" and not x_var:
                messagebox.showerror("Error", "Please select X variable!")
                return

            # Clear previous plot
            self.fig.clear()
            
            # Get style settings
            try:
                plt.style.use(self.plot_style.get())
            except:
                plt.style.use('default')
            
            # Apply font settings safely
            try:
                plt.rcParams['font.family'] = self.font_family.get()
                plt.rcParams['font.size'] = int(self.font_size.get())
            except:
                plt.rcParams['font.family'] = 'Arial'
                plt.rcParams['font.size'] = 12
            
            # Create subplot
            ax = self.fig.add_subplot(111)
            
            # Generate automatic labels and title if not manually set
            if not self.x_label.get() and x_var:
                ax.set_xlabel(x_var)
            else:
                ax.set_xlabel(self.x_label.get())
                
            if not self.y_label.get() and y_var:
                ax.set_ylabel(y_var)
            else:
                ax.set_ylabel(self.y_label.get())
            
            # Generate automatic title if not manually set
            if not self.plot_title.get():
                if plot_type == "Scatter Plot":
                    title = f"Scatter Plot of {y_var} vs {x_var}"
                    if color_var:
                        title += f" (colored by {color_var})"
                elif plot_type == "Line Plot":
                    title = f"Line Plot of {y_var} vs {x_var}"
                    if color_var:
                        title += f" (grouped by {color_var})"
                elif plot_type == "Bar Plot":
                    title = f"Bar Plot of {y_var} by {x_var}"
                    if color_var:
                        title += f" (grouped by {color_var})"
                elif plot_type == "Histogram":
                    title = f"Histogram of {x_var}"
                    if color_var:
                        title += f" (grouped by {color_var})"
                elif plot_type in ["Box Plot", "Violin Plot"]:
                    title = f"{plot_type} of {y_var} by {x_var}"
                    if color_var:
                        title += f" (grouped by {color_var})"
                elif plot_type == "Heatmap":
                    title = f"Heatmap of {y_var} vs {x_var}"
                    if color_var:
                        title += f" (values: {color_var})"
                else:
                    # Default title for any other plot type
                    title = f"{plot_type}"
                    if x_var and y_var:
                        title += f" of {y_var} vs {x_var}"
                    elif x_var:
                        title += f" of {x_var}"
                    if color_var:
                        title += f" (grouped by {color_var})"
                ax.set_title(title, pad=float(self.title_pad.get()))
            else:
                ax.set_title(self.plot_title.get(), pad=float(self.title_pad.get()))
            
            # Apply axis limits
            if self.x_min.get() and self.x_max.get():
                try:
                    ax.set_xlim(float(self.x_min.get()), float(self.x_max.get()))
                except ValueError:
                    pass
            if self.y_min.get() and self.y_max.get():
                try:
                    ax.set_ylim(float(self.y_min.get()), float(self.y_max.get()))
                except ValueError:
                    pass
            
            # Apply tick locators
            from matplotlib import ticker
            if self.x_locator.get() != "Auto":
                if self.x_locator.get() == "MultipleLocator":
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
                elif self.x_locator.get() == "FixedLocator":
                    ax.xaxis.set_major_locator(ticker.FixedLocator([0, 1, 5]))
                elif self.x_locator.get() == "LinearLocator":
                    ax.xaxis.set_major_locator(ticker.LinearLocator(numticks=5))
                elif self.x_locator.get() == "LogLocator":
                    ax.xaxis.set_major_locator(ticker.LogLocator(base=10))
                elif self.x_locator.get() == "MaxNLocator":
                    ax.xaxis.set_major_locator(ticker.MaxNLocator(n=5))
                elif self.x_locator.get() == "NullLocator":
                    ax.xaxis.set_major_locator(ticker.NullLocator())
            
            if self.y_locator.get() != "Auto":
                if self.y_locator.get() == "MultipleLocator":
                    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
                elif self.y_locator.get() == "FixedLocator":
                    ax.yaxis.set_major_locator(ticker.FixedLocator([0, 1, 5]))
                elif self.y_locator.get() == "LinearLocator":
                    ax.yaxis.set_major_locator(ticker.LinearLocator(numticks=5))
                elif self.y_locator.get() == "LogLocator":
                    ax.yaxis.set_major_locator(ticker.LogLocator(base=10))
                elif self.y_locator.get() == "MaxNLocator":
                    ax.yaxis.set_major_locator(ticker.MaxNLocator(n=5))
                elif self.y_locator.get() == "NullLocator":
                    ax.yaxis.set_major_locator(ticker.NullLocator())
            
            # Apply tick formatters
            if self.x_formatter.get() != "Auto":
                if self.x_formatter.get() == "ScalarFormatter":
                    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
                elif self.x_formatter.get() == "PercentFormatter":
                    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=5))
                elif self.x_formatter.get() == "StrMethodFormatter":
                    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x}'))
                elif self.x_formatter.get() == "FuncFormatter":
                    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"[{x:.2f}]"))
                elif self.x_formatter.get() == "FormatStrFormatter":
                    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
                elif self.x_formatter.get() == "NullFormatter":
                    ax.xaxis.set_major_formatter(ticker.NullFormatter())
            
            if self.y_formatter.get() != "Auto":
                if self.y_formatter.get() == "ScalarFormatter":
                    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
                elif self.y_formatter.get() == "PercentFormatter":
                    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=5))
                elif self.y_formatter.get() == "StrMethodFormatter":
                    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x}'))
                elif self.y_formatter.get() == "FuncFormatter":
                    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"[{x:.2f}]"))
                elif self.y_formatter.get() == "FormatStrFormatter":
                    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
                elif self.y_formatter.get() == "NullFormatter":
                    ax.yaxis.set_major_formatter(ticker.NullFormatter())
            
            # Apply grid settings
            ax.grid(self.show_grid.get(), which='major', 
                   linestyle=self.grid_style.get(),
                   alpha=float(self.grid_alpha.get()))
            if self.show_minor_grid.get():
                ax.grid(True, which='minor', 
                       linestyle=self.grid_style.get(),
                       alpha=float(self.grid_alpha.get()) * 0.5)
            
            # Rest of your plotting code remains the same
            if plot_type == "Line Plot":
                if color_var:
                    for category in self.filtered_data[color_var].unique():
                        mask = self.filtered_data[color_var] == category
                        ax.plot(self.filtered_data[x_var][mask], 
                               self.filtered_data[y_var][mask],
                               label=category,
                               linestyle=plot_opts.get('linestyle', '-'),
                               marker=plot_opts.get('marker', 'o'),
                               linewidth=plot_opts.get('linewidth', 2),
                               markerfacecolor='white',
                               markeredgewidth=1.5)
                else:
                    ax.plot(self.filtered_data[x_var], 
                           self.filtered_data[y_var],
                           linestyle=plot_opts.get('linestyle', '-'),
                           marker=plot_opts.get('marker', 'o'),
                           linewidth=plot_opts.get('linewidth', 2),
                           markerfacecolor='white',
                           markeredgewidth=1.5)
                    
            elif plot_type == "Scatter Plot":
                scatter_kwargs = {
                    'marker': plot_opts.get('marker', 'o') if plot_opts.get('marker') != 'None' else 'o',
                    'edgecolor': 'none',  # Changed from 'white' to 'none' to remove borders
                    'linewidth': 1
                }
                
                # Handle alpha parameter properly
                try:
                    alpha = float(plot_opts.get('alpha', 0.6))
                    scatter_kwargs['alpha'] = alpha if alpha > 0 else None
                except (ValueError, TypeError):
                    scatter_kwargs['alpha'] = 0.6
                
                # Handle size parameter properly
                try:
                    size = float(plot_opts.get('size', 50))
                    if size > 0:
                        scatter_kwargs['s'] = size
                except (ValueError, TypeError):
                    scatter_kwargs['s'] = 50
                
                if color_var:
                    # Handle categorical color variables
                    if self.filtered_data[color_var].dtype == 'object' or pd.api.types.is_categorical_dtype(self.filtered_data[color_var]):
                        # Get unique categories and map them to numbers
                        categories = self.filtered_data[color_var].unique()
                        color_map = {cat: i for i, cat in enumerate(categories)}
                        c = [color_map[val] for val in self.filtered_data[color_var]]
                        scatter = ax.scatter(self.filtered_data[x_var], 
                                          self.filtered_data[y_var],
                                          c=c,
                                          cmap=self.color_scheme.get(),
                                          **scatter_kwargs)
                        
                        # Create custom colorbar with category labels
                        cbar = self.fig.colorbar(scatter, ax=ax)
                        cbar.set_ticks(range(len(categories)))
                        cbar.set_ticklabels(categories)
                        cbar.set_label(color_var)
                    else:
                        # For numeric color variables, use them directly
                        scatter = ax.scatter(self.filtered_data[x_var], 
                                          self.filtered_data[y_var],
                                          c=self.filtered_data[color_var],
                                          cmap=self.color_scheme.get(),
                                          **scatter_kwargs)
                        self.fig.colorbar(scatter, label=color_var)
                else:
                    ax.scatter(self.filtered_data[x_var], 
                             self.filtered_data[y_var],
                             **scatter_kwargs)
                    
            elif plot_type == "Bar Plot":
                orientation = plot_opts.get('orientation', 'vertical')
                # Convert width to float
                width = float(plot_opts.get('width', 0.8))
                
                if orientation == 'vertical':
                    if color_var:
                        grouped = self.filtered_data.groupby([x_var, color_var])[y_var].mean().unstack()
                        grouped.plot(kind='bar', 
                                  ax=ax, 
                                  width=width,
                                  cmap=self.color_scheme.get())
                    else:
                        self.filtered_data.groupby(x_var)[y_var].mean().plot(
                            kind='bar',
                            ax=ax,
                            color=plt.cm.get_cmap(self.color_scheme.get())(0.6),
                            width=width)
                else:  # horizontal
                    if color_var:
                        grouped = self.filtered_data.groupby([x_var, color_var])[y_var].mean().unstack()
                        grouped.plot(kind='barh', 
                                  ax=ax, 
                                  height=width,  # Use height instead of width for horizontal bars
                                  cmap=self.color_scheme.get())
                    else:
                        self.filtered_data.groupby(x_var)[y_var].mean().plot(
                            kind='barh',
                            ax=ax,
                            color=plt.cm.get_cmap(self.color_scheme.get())(0.6),
                            height=width)  # Use height instead of width for horizontal bars
                    
            elif plot_type == "Histogram":
                if color_var:
                    for category in self.filtered_data[color_var].unique():
                        mask = self.filtered_data[color_var] == category
                        ax.hist(self.filtered_data[x_var][mask],
                               bins=int(plot_opts.get('bins', 30)),
                               alpha=0.5,
                               density=plot_opts.get('density', False),
                               cumulative=plot_opts.get('cumulative', False),
                               label=category,
                               edgecolor='black')
                else:
                    ax.hist(self.filtered_data[x_var],
                           bins=int(plot_opts.get('bins', 30)),
                           density=plot_opts.get('density', False),
                           cumulative=plot_opts.get('cumulative', False),
                           color=plt.cm.get_cmap(self.color_scheme.get())(0.6),
                           edgecolor='black')

            elif plot_type == "Box Plot":
                try:
                    width = float(plot_opts.get('width', 0.8))
                    if width <= 0:
                        width = 0.8
                except (ValueError, TypeError):
                    width = 0.8
                
                # Only pass hue if it has a value
                plot_kwargs = {
                    'data': self.filtered_data,
                    'x': x_var,
                    'y': y_var,
                    'ax': ax,
                    'width': width,
                    'showfliers': plot_opts.get('showfliers', True),
                    'notch': plot_opts.get('notch', False)
                }
                
                if color_var:  # Only add hue and palette if color_var is set
                    plot_kwargs['hue'] = color_var
                    plot_kwargs['palette'] = self.color_scheme.get()
                
                sns.boxplot(**plot_kwargs)
                    
            elif plot_type == "Violin Plot":
                try:
                    width = float(plot_opts.get('width', 0.8))
                    if width <= 0:
                        width = 0.8
                except (ValueError, TypeError):
                    width = 0.8
                
                inner = plot_opts.get('inner')
                if inner == 'None':
                    inner = None
                
                # Only pass hue if it has a value
                plot_kwargs = {
                    'data': self.filtered_data,
                    'x': x_var,
                    'y': y_var,
                    'ax': ax,
                    'width': width,
                    'inner': inner
                }
                
                if color_var:  # Only add hue and palette if color_var is set
                    plot_kwargs['hue'] = color_var
                    plot_kwargs['palette'] = self.color_scheme.get()
                
                sns.violinplot(**plot_kwargs)

            elif plot_type == "Heatmap":
                if not (x_var and y_var):
                    messagebox.showerror("Error", "Please select both X and Y variables!")
                    return
                
                # Prepare data for heatmap
                if color_var:
                    # If color_var is specified, use it for values
                    pivot_data = self.filtered_data.pivot_table(
                        index=y_var, 
                        columns=x_var, 
                        values=color_var,
                        aggfunc='mean'
                    )
                else:
                    # If no color_var, create correlation matrix
                    numeric_data = self.filtered_data[[x_var, y_var]].select_dtypes(include=[np.number])
                    pivot_data = numeric_data.corr()
                
                # Get heatmap options
                heatmap_kwargs = {
                    'data': pivot_data,
                    'ax': ax,
                    'cmap': self.color_scheme.get(),
                    'square': self.plot_specific_options['square'].get(),
                    'linewidths': float(self.plot_specific_options['linewidths'].get()),
                    'robust': self.plot_specific_options['robust'].get(),
                    'annot': self.plot_specific_options['annot'].get(),
                    'fmt': self.plot_specific_options['fmt'].get()
                }
                
                # Add center if specified
                center_val = self.plot_specific_options['center'].get()
                if center_val:
                    try:
                        heatmap_kwargs['center'] = float(center_val)
                    except ValueError:
                        pass
                
                # Create heatmap
                sns.heatmap(**heatmap_kwargs)
                
                # Rotate x-axis labels for better readability
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)

            elif plot_type == "Pie Chart":
                if not x_var:
                    messagebox.showerror("Error", "Please select a value variable!")
                    return
                
                # Prepare data for pie chart
                if color_var:
                    # Group data by color variable and calculate sum/mean of x_var
                    pie_data = self.filtered_data.groupby(color_var)[x_var].sum()
                    labels = pie_data.index
                else:
                    # Use value counts if no color variable
                    pie_data = self.filtered_data[x_var].value_counts()
                    labels = pie_data.index
                
                # Process explode values
                try:
                    explode_str = plot_opts.get('explode', '0.0')
                    explode = [float(x.strip()) for x in explode_str.split(',')]
                    # Pad with zeros if not enough values
                    explode.extend([0.0] * (len(pie_data) - len(explode)))
                    # Trim if too many values
                    explode = explode[:len(pie_data)]
                except (ValueError, AttributeError):
                    explode = None
                
                # Get other pie chart options
                pie_kwargs = {
                    'labels': labels,
                    'explode': explode,
                    'autopct': '%1.1f%%' if plot_opts.get('autopct', True) else None,
                    'shadow': plot_opts.get('shadow', False),
                    'radius': float(plot_opts.get('radius', 1.0)),
                    'startangle': float(plot_opts.get('startangle', 0))
                }
                
                # Create pie chart
                wedges, texts, autotexts = ax.pie(pie_data, **pie_kwargs)
                
                # Customize appearance
                if pie_kwargs['autopct']:
                    plt.setp(autotexts, size=8, weight="bold")
                plt.setp(texts, size=10)
                
                # Equal aspect ratio ensures circular pie
                ax.axis('equal')

            # Add legend with new options
            if color_var or plot_type in ["Line Plot", "Scatter Plot"]:
                legend_kwargs = {
                    'loc': self.legend_pos.get(),
                    'frameon': self.legend_frame.get(),
                    'title': self.legend_title.get() if self.legend_title.get() else None,
                    'fontsize': int(self.font_size.get()) * 0.8
                }
                ax.legend(**legend_kwargs)
            
            # Adjust layout
            self.fig.tight_layout()
            self.canvas.draw()
            
            # Store current figure
            self.current_fig = self.fig

        except Exception as e:
            # Get detailed error information
            import traceback
            error_details = traceback.format_exc()
            print(f"Error details:\n{error_details}")  # For debugging
            messagebox.showerror("Error", f"Failed to create plot: {str(e)}\nPlease check your selections and try again.")

    def __del__(self):
        # Close all webview windows
        if hasattr(self, 'browser') and self.browser:
            try:
                for window in webview.windows:
                    window.destroy()
            except:
                pass
            
        # Close all database connections
        for conn in self.db_connections.values():
            try:
                conn.close()
            except:
                pass
                
        # Clean up temporary files
        if hasattr(self, 'temp_html') and os.path.exists(self.temp_html):
            try:
                os.remove(self.temp_html)
            except:
                pass

    def setup_main_content(self):
        # Main content area
        self.main_container = ttk.Frame(self.main_container)
        self.main_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create notebook for different views
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.data_tab = ttk.Frame(self.notebook)
        self.plot_tab = ttk.Frame(self.notebook)
        self.db_tab = ttk.Frame(self.notebook)  # Database tab
        
        self.notebook.add(self.data_tab, text="Data Browser")
        self.notebook.add(self.plot_tab, text="Visualization")
        self.notebook.add(self.db_tab, text="Database")
        
        self.setup_data_browser()
        self.setup_visualization_tab()
        self.setup_database_view()

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
        dialog.transient(self.root)  # Make dialog modal
        
        # Column selection
        column_frame = ttk.Frame(dialog)
        column_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(column_frame, text="Column:").pack(side=tk.LEFT)
        column_var = ttk.Combobox(column_frame, values=list(self.data.columns), state="readonly")
        column_var.pack(side=tk.LEFT, padx=5)
        if len(self.data.columns) > 0:
            column_var.set(self.data.columns[0])
        
        # Operator selection
        operator_frame = ttk.Frame(dialog)
        operator_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(operator_frame, text="Operator:").pack(side=tk.LEFT)
        operators = ["equals", "not equals", "contains", "greater than", "less than", 
                    "greater than or equal", "less than or equal",
                    "starts with", "ends with", "is null", "is not null"]
        operator_var = ttk.Combobox(operator_frame, values=operators, state="readonly")
        operator_var.pack(side=tk.LEFT, padx=5)
        operator_var.set(operators[0])
        
        # Value frame
        value_frame = ttk.Frame(dialog)
        value_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(value_frame, text="Value:").pack(side=tk.LEFT)
        
        # Create a container frame for value widgets
        value_container = ttk.Frame(value_frame)
        value_container.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Create both widgets but only show entry initially
        value_entry = ttk.Entry(value_container)
        value_combo = ttk.Combobox(value_container, state="readonly")
        value_entry.pack(fill=tk.X, expand=True)
        
        # Keep track of current widget
        current_widget = {"widget": value_entry}

        def update_value_widget(*args):
            selected_col = column_var.get()
            if selected_col in self.data.columns:
                unique_values = self.data[selected_col].unique()
                
                # Remove current widget
                current_widget["widget"].pack_forget()
                
                if len(unique_values) < 50:  # Show dropdown for reasonable number of unique values
                    value_combo["values"] = sorted(unique_values.tolist())
                    value_combo.set("")  # Clear previous value
                    value_combo.pack(fill=tk.X, expand=True)
                    current_widget["widget"] = value_combo
                else:
                    value_entry.delete(0, tk.END)  # Clear previous value
                    value_entry.pack(fill=tk.X, expand=True)
                    current_widget["widget"] = value_entry

        # Bind column selection to update value widget
        column_var.bind('<<ComboboxSelected>>', update_value_widget)
        
        def apply_filter():
            column = column_var.get()
            operator = operator_var.get()
            value = current_widget["widget"].get()
            
            if not column:
                messagebox.showerror("Error", "Please select a column!")
                return
                
            try:
                # Create a copy of the original data
                self.filtered_data = self.data.copy()
                
                # Convert value to appropriate type if column is numeric
                if self.data[column].dtype in ['int64', 'float64'] and value and operator not in ['is null', 'is not null']:
                    try:
                        value = float(value)
                    except ValueError:
                        messagebox.showerror("Error", "Please enter a valid number for numeric column!")
                        return
                
                # Apply filter based on operator
                if operator == "equals":
                    self.filtered_data = self.filtered_data[self.filtered_data[column] == value]
                elif operator == "not equals":
                    self.filtered_data = self.filtered_data[self.filtered_data[column] != value]
                elif operator == "contains":
                    self.filtered_data = self.filtered_data[self.filtered_data[column].astype(str).str.contains(str(value), case=False, na=False)]
                elif operator == "greater than":
                    self.filtered_data = self.filtered_data[self.filtered_data[column] > value]
                elif operator == "less than":
                    self.filtered_data = self.filtered_data[self.filtered_data[column] < value]
                elif operator == "greater than or equal":
                    self.filtered_data = self.filtered_data[self.filtered_data[column] >= value]
                elif operator == "less than or equal":
                    self.filtered_data = self.filtered_data[self.filtered_data[column] <= value]
                elif operator == "starts with":
                    self.filtered_data = self.filtered_data[self.filtered_data[column].astype(str).str.startswith(str(value), na=False)]
                elif operator == "ends with":
                    self.filtered_data = self.filtered_data[self.filtered_data[column].astype(str).str.endswith(str(value), na=False)]
                elif operator == "is null":
                    self.filtered_data = self.filtered_data[self.filtered_data[column].isna()]
                elif operator == "is not null":
                    self.filtered_data = self.filtered_data[self.filtered_data[column].notna()]
                
                # Update table with filtered data
                self.pt.model.df = self.filtered_data
                self.pt.redraw()
                
                # Update status
                self.status_msg.config(text=f"Filtered: {len(self.filtered_data)} of {len(self.data)} rows")
                self.row_count.config(text=f"Rows: {len(self.filtered_data)}")
                
                # If there's an active plot, update it with filtered data
                if hasattr(self, 'current_fig') and self.current_fig:
                    self.plot_data(None)
                    
                dialog.destroy()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to apply filter: {str(e)}")
        
        def reset_filter():
            self.filtered_data = self.data.copy()
            self.pt.model.df = self.filtered_data
            self.pt.redraw()
            self.status_msg.config(text="Filter reset")
            self.row_count.config(text=f"Rows: {len(self.filtered_data)}")
            
            # If there's an active plot, update it with unfiltered data
            if hasattr(self, 'current_fig') and self.current_fig:
                self.plot_data(None)
                
            dialog.destroy()
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Apply", 
                  bootstyle="success",
                  command=apply_filter).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset", 
                  bootstyle="warning",
                  command=reset_filter).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", 
                  bootstyle="danger",
                  command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        # Center the dialog on screen
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f'{width}x{height}+{x}+{y}')

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
        
        # Create notebook for plot options
        self.plot_options_notebook = ttk.Notebook(options_frame)
        self.plot_options_notebook.pack(fill=tk.X, padx=5, pady=5)
        
        # Base Options tab
        self.base_options_tab = ttk.Frame(self.plot_options_notebook)
        self.plot_options_notebook.add(self.base_options_tab, text="Base Options")
        
        # Plot type selection
        plot_type_frame = ttk.Frame(self.base_options_tab)
        plot_type_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(plot_type_frame, text="Plot Type:").pack(side=tk.LEFT, padx=5)
        self.plot_type = ttk.Combobox(
            plot_type_frame,
            values=[
                # Basic plots
                "Line Plot", "Scatter Plot", "Bar Plot", "Histogram",
                "Box Plot", "Violin Plot", "Heatmap", "Area Plot",
                "Pie Chart", "Density Plot",
                # Statistical plots
                "Error Bar",
            ],
            state="readonly",
            width=30
        )
        self.plot_type.pack(side=tk.LEFT, padx=5)
        self.plot_type.bind("<<ComboboxSelected>>", self.update_plot_options)
        
        # Variables frame
        vars_frame = ttk.Frame(self.base_options_tab)
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

        # Plot-specific options frame
        self.plot_specific_frame = ttk.LabelFrame(self.base_options_tab, text="Plot Options", padding=5)
        self.plot_specific_frame.pack(fill=tk.X, pady=5)
        
        # Style & Font tab (merged)
        style_tab = ttk.Frame(self.plot_options_notebook)
        self.plot_options_notebook.add(style_tab, text="Style & Font")
        
        # Create a notebook for style sub-options
        style_notebook = ttk.Notebook(style_tab)
        style_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Basic Style tab
        basic_style = ttk.Frame(style_notebook)
        style_notebook.add(basic_style, text="Basic")
        
        # Color scheme
        ttk.Label(basic_style, text="Color Scheme:").pack(anchor=tk.W, padx=5, pady=2)
        self.color_scheme = ttk.Combobox(
            basic_style,
            values=[
                "viridis", "plasma", "inferno", "magma", "cividis",
                "Spectral", "RdYlBu", "coolwarm", "bwr", "seismic"
            ],
            state="readonly",
            width=20
        )
        self.color_scheme.set("viridis")
        self.color_scheme.pack(anchor=tk.W, padx=5, pady=2)
        
        # Plot style
        ttk.Label(basic_style, text="Plot Style:").pack(anchor=tk.W, padx=5, pady=2)
        self.plot_style = ttk.Combobox(
            basic_style,
            values=[
                "default", "seaborn-v0_8", "seaborn-v0_8-darkgrid", 
                "seaborn-v0_8-whitegrid", "bmh", "classic", "dark_background",
                "fast", "grayscale", "Solarize_Light2", "tableau-colorblind10"
            ],
            state="readonly",
            width=20
        )
        self.plot_style.set("default")
        self.plot_style.pack(anchor=tk.W, padx=5, pady=2)
        
        # Font settings
        ttk.Label(basic_style, text="Font Family:").pack(anchor=tk.W, padx=5, pady=2)
        self.font_family = ttk.Combobox(
            basic_style,
            values=["Arial", "Times New Roman", "Helvetica", "Courier", "Verdana"],
            width=20,
            state="readonly"
        )
        self.font_family.set("Arial")
        self.font_family.pack(anchor=tk.W, padx=5, pady=2)
        
        ttk.Label(basic_style, text="Font Size:").pack(anchor=tk.W, padx=5, pady=2)
        self.font_size = ttk.Spinbox(basic_style, from_=8, to=24, width=10)
        self.font_size.set(12)
        self.font_size.pack(anchor=tk.W, padx=5, pady=2)
        
        # Ticks & Labels tab
        ticks_tab = ttk.Frame(style_notebook)
        style_notebook.add(ticks_tab, text="Ticks & Labels")
        
        # X-axis settings
        x_frame = ttk.LabelFrame(ticks_tab, text="X-Axis")
        x_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # X-axis label
        ttk.Label(x_frame, text="Label:").pack(side=tk.LEFT, padx=5)
        self.x_label = ttk.Entry(x_frame, width=20)
        self.x_label.pack(side=tk.LEFT, padx=5)
        
        # X-axis limits
        ttk.Label(x_frame, text="Limits:").pack(side=tk.LEFT, padx=5)
        self.x_min = ttk.Entry(x_frame, width=8)
        self.x_min.pack(side=tk.LEFT, padx=2)
        ttk.Label(x_frame, text="to").pack(side=tk.LEFT)
        self.x_max = ttk.Entry(x_frame, width=8)
        self.x_max.pack(side=tk.LEFT, padx=2)
        
        # X-axis tick settings
        x_tick_frame = ttk.Frame(x_frame)
        x_tick_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(x_tick_frame, text="Tick Locator:").pack(side=tk.LEFT, padx=5)
        self.x_locator = ttk.Combobox(
            x_tick_frame,
            values=[
                "Auto", "MultipleLocator", "FixedLocator", "LinearLocator",
                "LogLocator", "MaxNLocator", "NullLocator"
            ],
            state="readonly",
            width=15
        )
        self.x_locator.set("Auto")
        self.x_locator.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(x_tick_frame, text="Formatter:").pack(side=tk.LEFT, padx=5)
        self.x_formatter = ttk.Combobox(
            x_tick_frame,
            values=[
                "Auto", "ScalarFormatter", "PercentFormatter", "StrMethodFormatter",
                "FuncFormatter", "FormatStrFormatter", "NullFormatter"
            ],
            state="readonly",
            width=15
        )
        self.x_formatter.set("Auto")
        self.x_formatter.pack(side=tk.LEFT, padx=5)
        
        # Y-axis settings (similar to X-axis)
        y_frame = ttk.LabelFrame(ticks_tab, text="Y-Axis")
        y_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(y_frame, text="Label:").pack(side=tk.LEFT, padx=5)
        self.y_label = ttk.Entry(y_frame, width=20)
        self.y_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(y_frame, text="Limits:").pack(side=tk.LEFT, padx=5)
        self.y_min = ttk.Entry(y_frame, width=8)
        self.y_min.pack(side=tk.LEFT, padx=2)
        ttk.Label(y_frame, text="to").pack(side=tk.LEFT)
        self.y_max = ttk.Entry(y_frame, width=8)
        self.y_max.pack(side=tk.LEFT, padx=2)
        
        y_tick_frame = ttk.Frame(y_frame)
        y_tick_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(y_tick_frame, text="Tick Locator:").pack(side=tk.LEFT, padx=5)
        self.y_locator = ttk.Combobox(
            y_tick_frame,
            values=[
                "Auto", "MultipleLocator", "FixedLocator", "LinearLocator",
                "LogLocator", "MaxNLocator", "NullLocator"
            ],
            state="readonly",
            width=15
        )
        self.y_locator.set("Auto")
        self.y_locator.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(y_tick_frame, text="Formatter:").pack(side=tk.LEFT, padx=5)
        self.y_formatter = ttk.Combobox(
            y_tick_frame,
            values=[
                "Auto", "ScalarFormatter", "PercentFormatter", "StrMethodFormatter",
                "FuncFormatter", "FormatStrFormatter", "NullFormatter"
            ],
            state="readonly",
            width=15
        )
        self.y_formatter.set("Auto")
        self.y_formatter.pack(side=tk.LEFT, padx=5)
        
        # Legend & Title tab
        legend_tab = ttk.Frame(style_notebook)
        style_notebook.add(legend_tab, text="Legend & Title")
        
        # Title settings
        title_frame = ttk.LabelFrame(legend_tab, text="Title")
        title_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(title_frame, text="Title:").pack(side=tk.LEFT, padx=5)
        self.plot_title = ttk.Entry(title_frame, width=30)
        self.plot_title.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(title_frame, text="Pad:").pack(side=tk.LEFT, padx=5)
        self.title_pad = ttk.Spinbox(title_frame, from_=0, to=50, width=5)
        self.title_pad.set(10)
        self.title_pad.pack(side=tk.LEFT, padx=5)
        
        # Legend settings
        legend_frame = ttk.LabelFrame(legend_tab, text="Legend")
        legend_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Legend position
        ttk.Label(legend_frame, text="Position:").pack(side=tk.LEFT, padx=5)
        self.legend_pos = ttk.Combobox(
            legend_frame,
            values=["best", "upper right", "upper left", "lower right", "lower left",
                   "center left", "center right", "lower center", "upper center", "center"],
            state="readonly",
            width=15
        )
        self.legend_pos.set("best")
        self.legend_pos.pack(side=tk.LEFT, padx=5)
        
        # Legend title
        ttk.Label(legend_frame, text="Title:").pack(side=tk.LEFT, padx=5)
        self.legend_title = ttk.Entry(legend_frame, width=20)
        self.legend_title.pack(side=tk.LEFT, padx=5)
        
        # Legend frame
        self.legend_frame = tk.BooleanVar(value=True)
        ttk.Checkbutton(legend_frame, text="Show Frame", 
                       variable=self.legend_frame).pack(side=tk.LEFT, padx=5)
        
        # Grid tab
        grid_tab = ttk.Frame(self.plot_options_notebook)
        self.plot_options_notebook.add(grid_tab, text="Grid")
        
        # Major grid
        self.show_grid = tk.BooleanVar(value=True)
        ttk.Checkbutton(grid_tab, text="Show Major Grid", 
                       variable=self.show_grid).pack(anchor=tk.W, padx=5, pady=2)
        
        # Minor grid
        self.show_minor_grid = tk.BooleanVar(value=False)
        ttk.Checkbutton(grid_tab, text="Show Minor Grid", 
                       variable=self.show_minor_grid).pack(anchor=tk.W, padx=5, pady=2)
        
        # Grid style
        ttk.Label(grid_tab, text="Grid Style:").pack(anchor=tk.W, padx=5, pady=2)
        self.grid_style = ttk.Combobox(
            grid_tab,
            values=["-", "--", "-.", ":"],
            state="readonly",
            width=10
        )
        self.grid_style.set("-")
        self.grid_style.pack(anchor=tk.W, padx=5, pady=2)
        
        # Grid alpha
        ttk.Label(grid_tab, text="Grid Alpha:").pack(anchor=tk.W, padx=5, pady=2)
        self.grid_alpha = ttk.Spinbox(grid_tab, from_=0, to=1, increment=0.1, width=5)
        self.grid_alpha.set(0.5)
        self.grid_alpha.pack(anchor=tk.W, padx=5, pady=2)
        
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
        
        # Plot display area
        self.plot_frame = ttk.Frame(self.plot_tab)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create matplotlib figure and canvas
        self.fig = plt.Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add matplotlib toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        
        # Initialize plot-specific options dictionary
        self.plot_specific_options = {}

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
        if self.data is None:
            messagebox.showerror("Error", "No data to export statistics for!")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML Report", "*.html"),
                      ("Text Report", "*.txt")]
        )
        
        if not file_path:
            return

        try:
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
        if not self.data.empty:
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
        """Setup enhanced status bar with memory monitoring."""
        # Import psutil at the top of the file
        import psutil
        
        # Status bar frame
        self.status_bar = ttk.Frame(self.root, relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Left section - File/DB info
        self.status_left = ttk.Frame(self.status_bar)
        self.status_left.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # File/DB indicator with icon
        self.status_type = ttk.Label(self.status_left, text="🗎", font=('Segoe UI', 9))
        self.status_type.pack(side=tk.LEFT, padx=(5,0))
        
        # Status message
        self.status_msg = ttk.Label(self.status_left, text="Ready")
        self.status_msg.pack(side=tk.LEFT, padx=5)
        
        # Center section - Memory usage
        self.status_center = ttk.Frame(self.status_bar)
        self.status_center.pack(side=tk.LEFT, padx=10)
        
        # Memory icon and label
        ttk.Label(self.status_center, text="🗘", font=('Segoe UI', 9)).pack(side=tk.LEFT)
        self.memory_label = ttk.Label(self.status_center, text="Memory: 0 MB")
        self.memory_label.pack(side=tk.LEFT, padx=5)
        
        # Right section - Row count
        self.status_right = ttk.Frame(self.status_bar)
        self.status_right.pack(side=tk.RIGHT)
        
        self.row_count = ttk.Label(self.status_right, text="Rows: 0")
        self.row_count.pack(side=tk.RIGHT, padx=5)
        
        # Start memory monitoring
        self.update_memory_usage()
        
    def update_memory_usage(self):
        """Update memory usage display."""
        try:
            # Get memory info
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024  # Convert to MB
            
            # Update label with formatted memory usage
            self.memory_label.config(text=f"Memory: {memory_mb:.1f} MB")
            
            # Update status type icon based on current view
            if hasattr(self, 'db_conn') and self.db_conn:
                self.status_type.config(text="🗃")  # Database icon
            elif hasattr(self, 'data') and self.data is not None:
                self.status_type.config(text="🗎")  # File icon
            else:
                self.status_type.config(text="")  # No icon
                
        except Exception as e:
            self.memory_label.config(text="Memory: N/A")
            
        # Schedule next update
        self.root.after(1000, self.update_memory_usage)  # Update every second

    def update_status(self, message):
        """Update status bar with message."""
        self.status_msg.config(text=message)
        
    def setup_database_view(self):
        """Setup the database view tab with enhanced functionality."""
        # Main container with split view
        main_paned = ttk.PanedWindow(self.db_tab, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)

        # Left panel for database structure
        left_frame = ttk.LabelFrame(main_paned, text="Database Explorer", padding=5)
        main_paned.add(left_frame, weight=1)

        # Database connection controls
        conn_frame = ttk.Frame(left_frame)
        conn_frame.pack(fill=tk.X, padx=5, pady=5)

        # Connection type selector
        ttk.Label(conn_frame, text="Type:").pack(side=tk.LEFT, padx=5)
        self.db_type = ttk.Combobox(conn_frame, values=["SQLite", "DuckDB"], state="readonly", width=10)
        self.db_type.set("SQLite")
        self.db_type.pack(side=tk.LEFT, padx=5)

        # Database path
        path_frame = ttk.Frame(left_frame)
        path_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(path_frame, text="Database:").pack(side=tk.LEFT, padx=5)
        self.db_path = ttk.Entry(path_frame)
        self.db_path.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Button frame
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(btn_frame, text="Browse", 
                  bootstyle="info",
                  command=self.select_database).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Connect", 
                  bootstyle="success",
                  command=self.connect_database).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Disconnect",
                  bootstyle="danger",
                  command=self.disconnect_database).pack(side=tk.LEFT, padx=2)

        # Database structure tree
        tree_frame = ttk.Frame(left_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        tree_scroll = ttk.Scrollbar(tree_frame)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.db_tree = ttk.Treeview(tree_frame, show='tree',
                                   yscrollcommand=tree_scroll.set)
        self.db_tree.pack(fill=tk.BOTH, expand=True)
        tree_scroll.config(command=self.db_tree.yview)
        
        # Bind double-click event
        self.db_tree.bind('<Double-1>', self.on_tree_double_click)

        # Right panel with notebook
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=3)

        # Create notebook for different views
        self.db_notebook = ttk.Notebook(right_frame)
        self.db_notebook.pack(fill=tk.BOTH, expand=True)

        # SQL Editor tab
        self.sql_tab = ttk.Frame(self.db_notebook)
        self.db_notebook.add(self.sql_tab, text="SQL Editor")

        # Query toolbar
        query_toolbar = ttk.Frame(self.sql_tab)
        query_toolbar.pack(fill=tk.X, padx=5, pady=5)

        # Add query template dropdown
        ttk.Label(query_toolbar, text="Templates:").pack(side=tk.LEFT, padx=5)
        self.query_template = ttk.Combobox(query_toolbar, values=[
            "SELECT * FROM table_name",
            "SELECT column1, column2 FROM table_name WHERE condition",
            "INSERT INTO table_name (column1, column2) VALUES (value1, value2)",
            "UPDATE table_name SET column1 = value1 WHERE condition",
            "DELETE FROM table_name WHERE condition",
            "CREATE INDEX index_name ON table_name (column_name)",
            "EXPLAIN QUERY PLAN SELECT * FROM table_name"
        ], width=50)
        self.query_template.pack(side=tk.LEFT, padx=5)
        self.query_template.bind('<<ComboboxSelected>>', self.insert_query_template)

        # Add execution controls
        ttk.Button(query_toolbar, text="Execute (F5)", 
                  bootstyle="success",
                  command=self.execute_query).pack(side=tk.RIGHT, padx=5)
        ttk.Button(query_toolbar, text="Clear", 
                  bootstyle="danger",
                  command=self.clear_query).pack(side=tk.RIGHT, padx=5)

        # SQL Text widget with line numbers
        sql_frame = ttk.Frame(self.sql_tab)
        sql_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add line numbers
        self.line_numbers = tk.Text(sql_frame, width=4, padx=3, takefocus=0,
                                  border=0, background='lightgray',
                                  state='disabled')
        self.line_numbers.pack(side=tk.LEFT, fill=tk.Y)

        # Add SQL editor with scrollbar
        sql_scroll = ttk.Scrollbar(sql_frame)
        sql_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.sql_text = tk.Text(sql_frame, wrap=tk.NONE,
                               yscrollcommand=sql_scroll.set)
        self.sql_text.pack(fill=tk.BOTH, expand=True)
        sql_scroll.config(command=self.sql_text.yview)

        # Bind text changes to update line numbers
        self.sql_text.bind('<KeyPress>', self.update_line_numbers)
        self.sql_text.bind('<KeyRelease>', self.update_line_numbers)

        # Add horizontal scrollbar
        sql_hscroll = ttk.Scrollbar(self.sql_tab, orient=tk.HORIZONTAL,
                                   command=self.sql_text.xview)
        sql_hscroll.pack(fill=tk.X)
        self.sql_text.configure(xscrollcommand=sql_hscroll.set)

        # Results tab
        self.results_tab = ttk.Frame(self.db_notebook)
        self.db_notebook.add(self.results_tab, text="Results")

        # Add results toolbar
        results_toolbar = ttk.Frame(self.results_tab)
        results_toolbar.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(results_toolbar, text="Export Results",
                  bootstyle="info",
                  command=self.export_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(results_toolbar, text="Copy to Clipboard",
                  bootstyle="info",
                  command=self.copy_results).pack(side=tk.LEFT, padx=5)

        self.result_status = ttk.Label(results_toolbar, text="")
        self.result_status.pack(side=tk.RIGHT, padx=5)

        # Results table frame
        self.results_frame = ttk.Frame(self.results_tab)
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Initialize variables
        self.current_results = None
        self.execution_time = 0

    def disconnect_database(self):
        """Safely disconnect from the current database."""
        try:
            if hasattr(self, 'db_conn') and self.db_conn:
                self.db_conn.close()
                self.db_conn = None
                self.db_tree.delete(*self.db_tree.get_children())
                self.status_msg.config(text="Database disconnected")
                messagebox.showinfo("Success", "Database disconnected successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to disconnect: {str(e)}")

    def update_line_numbers(self, event=None):
        """Update line numbers in the SQL editor."""
        def count_lines(text_widget):
            final_index = str(text_widget.index('end-1c'))
            return int(final_index.split('.')[0])

        line_count = count_lines(self.sql_text)
        line_numbers_text = '\n'.join(str(i) for i in range(1, line_count + 1))
        self.line_numbers.config(state='normal')
        self.line_numbers.delete('1.0', tk.END)
        self.line_numbers.insert('1.0', line_numbers_text)
        self.line_numbers.config(state='disabled')

    def insert_query_template(self, event=None):
        """Insert selected query template into SQL editor."""
        template = self.query_template.get()
        if template:
            self.sql_text.insert(tk.INSERT, template + "\n")
            self.query_template.set('')  # Reset selection

    def clear_query(self):
        """Clear the SQL editor."""
        self.sql_text.delete('1.0', tk.END)
        self.update_line_numbers()

    def on_tree_double_click(self, event):
        """Handle double-click on database tree items."""
        item = self.db_tree.selection()[0]
        item_text = self.db_tree.item(item, "text")
        parent = self.db_tree.parent(item)
        
        if parent:  # This is a table or column
            parent_text = self.db_tree.item(parent, "text")
            if parent_text == "Tables":
                # Insert SELECT statement for the table
                query = f"SELECT * FROM {item_text} LIMIT 100;"
                self.sql_text.delete('1.0', tk.END)
                self.sql_text.insert('1.0', query)
                self.execute_query()

    def export_results(self):
        """Export query results to a file."""
        if self.current_results is None:
            messagebox.showerror("Error", "No results to export!")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[
                ("CSV Files", "*.csv"),
                ("Excel Files", "*.xlsx"),
                ("JSON Files", "*.json")
            ]
        )
        
        if not file_path:
            return

        try:
            extension = os.path.splitext(file_path)[1].lower()
            
            if extension == '.csv':
                self.current_results.to_csv(file_path, index=False)
            elif extension == '.xlsx':
                self.current_results.to_excel(file_path, index=False)
            elif extension == '.json':
                self.current_results.to_json(file_path, orient='records')
                
            self.status_msg.config(text=f"Results exported to: {os.path.basename(file_path)}")
            messagebox.showinfo("Success", "Results exported successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results: {str(e)}")

    def copy_results(self):
        """Copy query results to clipboard."""
        if self.current_results is None:
            messagebox.showerror("Error", "No results to copy!")
            return

        try:
            self.current_results.to_clipboard(index=False)
            self.status_msg.config(text="Results copied to clipboard")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy results: {str(e)}")

    def execute_query(self, query, params=None):
        """Execute a query using the SQLite manager."""
        if not hasattr(self, 'db_conn') or not self.db_conn:
            raise Exception("No database connection")
            
        return self.sqlite_manager.execute_query(self.db_path.get(), query, params)

    def select_database(self):
        """Open file dialog to select existing database or create new one."""
        db_type = self.db_type.get()
        if db_type == "SQLite":
            file_types = [
                ("SQLite Database", "*.db;*.sqlite;*.sqlite3"),
                ("All Files", "*.*")
            ]
            default_ext = ".db"
        else:  # DuckDB
            file_types = [
                ("DuckDB Database", "*.duckdb"),
                ("All Files", "*.*")
            ]
            default_ext = ".duckdb"

        # First try to open existing database
        file_path = filedialog.askopenfilename(
            filetypes=file_types,
            initialdir=self.last_directory,
            title="Select Existing Database"
        )
        
        # If user cancels opening existing, ask if they want to create new
        if not file_path and messagebox.askyesno(
            "Create New Database",
            "No database selected. Would you like to create a new database?"
        ):
            file_path = filedialog.asksaveasfilename(
                defaultextension=default_ext,
                filetypes=file_types,
                initialdir=self.last_directory,
                title="Create New Database"
            )
        
        if file_path:
            self.last_directory = os.path.dirname(file_path)
            self.db_path.delete(0, tk.END)
            self.db_path.insert(0, file_path)
            
            # If this is a new database being created, initialize it
            if not os.path.exists(file_path):
                try:
                    if db_type == "SQLite":
                        conn = sqlite3.connect(file_path)
                        conn.close()
                    else:  # DuckDB
                        conn = duckdb.connect(file_path)
                        conn.close()
                    messagebox.showinfo("Success", "New database created successfully!")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to create new database: {str(e)}")
                    return

        try:
            # Close existing connection if any
            if hasattr(self, 'db_conn') and self.db_conn:
                self.db_conn.close()

            db_type = self.db_type.get()
            
            if db_type == "SQLite":
                self.db_conn = sqlite3.connect(file_path)
                # Enable foreign key support
                self.db_conn.execute("PRAGMA foreign_keys = ON")
                # Get list of tables
                cursor = self.db_conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
            else:  # DuckDB
                self.db_conn = duckdb.connect(file_path)
                # Get list of tables
                cursor = self.db_conn.cursor()
                cursor.execute("SHOW TABLES")
                tables = cursor.fetchall()

            # Clear existing tree items
            self.db_tree.delete(*self.db_tree.get_children())
            
            # Add tables node
            tables_node = self.db_tree.insert("", "end", text="Tables", open=True)
            
            # Add each table and its columns
            for table in tables:
                table_name = table[0]
                table_node = self.db_tree.insert(tables_node, "end", text=table_name)
                
                # Get column information
                if db_type == "SQLite":
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = cursor.fetchall()
                    for col in columns:
                        # col[1] is column name, col[2] is data type
                        self.db_tree.insert(table_node, "end", 
                                          text=f"{col[1]} ({col[2]})")
                else:  # DuckDB
                    cursor.execute(f"DESCRIBE {table_name}")
                    columns = cursor.fetchall()
                    for col in columns:
                        # col[0] is column name, col[1] is data type
                        self.db_tree.insert(table_node, "end", 
                                          text=f"{col[0]} ({col[1]})")

            # Update status
            self.status_msg.config(
                text=f"Connected to {db_type} database: {os.path.basename(file_path)}"
            )
            messagebox.showinfo("Success", 
                f"Connected to {db_type} database successfully!\nFound {len(tables)} tables.")

        except sqlite3.Error as e:
            messagebox.showerror("SQLite Error", f"Database error: {str(e)}")
        except duckdb.Error as e:
            messagebox.showerror("DuckDB Error", f"Database error: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to connect to database: {str(e)}")

    def connect_database(self):
        """Connect to the selected database with enhanced error handling."""
        if not self.db_path.get():
            messagebox.showerror("Error", "Please select a database file first!")
            return
            
        try:
            file_path = self.db_path.get()
            
            # Close existing connection if any
            if hasattr(self, 'db_conn') and self.db_conn:
                self.sqlite_manager.close(file_path)

            # Connect using SQLite manager
            self.db_conn = self.sqlite_manager.connect(file_path)
            
            # Get list of tables
            tables = self.sqlite_manager.execute_query(file_path, 
                "SELECT name FROM sqlite_master WHERE type='table'")

            # Clear existing tree items
            self.db_tree.delete(*self.db_tree.get_children())
            
            # Add tables node
            tables_node = self.db_tree.insert("", "end", text="Tables", open=True)
            
            # Add each table and its columns
            for table in tables:
                table_name = table[0]
                table_node = self.db_tree.insert(tables_node, "end", text=table_name)
                
                # Get column information
                columns = self.sqlite_manager.execute_query(file_path, 
                    f"PRAGMA table_info({table_name})")
                
                for col in columns:
                    # col[1] is column name, col[2] is data type
                    self.db_tree.insert(table_node, "end", 
                                      text=f"{col[1]} ({col[2]})")

            # Update status
            self.status_msg.config(
                text=f"Connected to SQLite database: {os.path.basename(file_path)}"
            )
            messagebox.showinfo("Success", 
                f"Connected to database successfully!\nFound {len(tables)} tables.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to connect to database: {str(e)}")

    def update_plot_options(self, event=None):
        """Update plot options based on selected plot type."""
        plot_type = self.plot_type.get()
        all_columns = self.data.columns.tolist() if self.data is not None else []

        # Reset all selectors
        for selector in [self.x_var, self.y_var, self.color_var]:
            selector.set("")
            selector["state"] = "disabled"

        # Clear existing plot-specific options
        for widget in self.plot_specific_frame.winfo_children():
            widget.destroy()

        # Initialize plot_specific_options if it doesn't exist
        if not hasattr(self, 'plot_specific_options'):
            self.plot_specific_options = {}

        # Update variable selectors based on plot type
        if plot_type == "Line Plot":
            self.x_var["values"] = ["None"] + self.continuous_columns + self.datetime_columns
            self.y_var["values"] = ["None"] + self.continuous_columns
            self.color_var["values"] = ["None"] + self.categorical_columns
            for selector in [self.x_var, self.y_var, self.color_var]:
                selector["state"] = "readonly"
            
            # Add line plot specific options
            line_opts = ttk.Frame(self.plot_specific_frame)
            line_opts.pack(fill=tk.X, pady=5)
            
            # Line style
            ttk.Label(line_opts, text="Line Style:").pack(side=tk.LEFT, padx=5)
            self.plot_specific_options['linestyle'] = ttk.Combobox(
                line_opts, values=['None', '-', '--', '-.', ':'], state="readonly", width=10)
            self.plot_specific_options['linestyle'].set('-')
            self.plot_specific_options['linestyle'].pack(side=tk.LEFT, padx=5)
            
            # Marker
            ttk.Label(line_opts, text="Marker:").pack(side=tk.LEFT, padx=5)
            self.plot_specific_options['marker'] = ttk.Combobox(
                line_opts, values=['None', 'o', 's', '^', 'v', 'D', '*', '+', 'x'], 
                state="readonly", width=10)
            self.plot_specific_options['marker'].set('o')
            self.plot_specific_options['marker'].pack(side=tk.LEFT, padx=5)
            
            # Line width
            ttk.Label(line_opts, text="Line Width:").pack(side=tk.LEFT, padx=5)
            self.plot_specific_options['linewidth'] = ttk.Spinbox(
                line_opts, from_=0, to=10, increment=0.5, width=5)
            self.plot_specific_options['linewidth'].set(2)
            self.plot_specific_options['linewidth'].pack(side=tk.LEFT, padx=5)

        elif plot_type == "Scatter Plot":
            self.x_var["values"] = ["None"] + self.continuous_columns
            self.y_var["values"] = ["None"] + self.continuous_columns
            self.color_var["values"] = ["None"] + all_columns
            for selector in [self.x_var, self.y_var, self.color_var]:
                selector["state"] = "readonly"
            
            # Add scatter plot specific options
            scatter_opts = ttk.Frame(self.plot_specific_frame)
            scatter_opts.pack(fill=tk.X, pady=5)
            
            # Marker
            ttk.Label(scatter_opts, text="Marker:").pack(side=tk.LEFT, padx=5)
            self.plot_specific_options['marker'] = ttk.Combobox(
                scatter_opts, values=['None', 'o', 's', '^', 'v', 'D', '*', '+', 'x'], 
                state="readonly", width=10)
            self.plot_specific_options['marker'].set('o')
            self.plot_specific_options['marker'].pack(side=tk.LEFT, padx=5)
            
            # Size
            ttk.Label(scatter_opts, text="Size:").pack(side=tk.LEFT, padx=5)
            self.plot_specific_options['size'] = ttk.Spinbox(
                scatter_opts, from_=0, to=200, increment=10, width=5)
            self.plot_specific_options['size'].set(50)
            self.plot_specific_options['size'].pack(side=tk.LEFT, padx=5)
            
            # Alpha
            ttk.Label(scatter_opts, text="Transparency:").pack(side=tk.LEFT, padx=5)
            self.plot_specific_options['alpha'] = ttk.Spinbox(
                scatter_opts, from_=0, to=1.0, increment=0.1, width=5)
            self.plot_specific_options['alpha'].set(0.6)
            self.plot_specific_options['alpha'].pack(side=tk.LEFT, padx=5)

        elif plot_type == "Bar Plot":
            self.x_var["values"] = ["None"] + self.categorical_columns
            self.y_var["values"] = ["None"] + self.continuous_columns
            self.color_var["values"] = ["None"] + self.categorical_columns
            for selector in [self.x_var, self.y_var, self.color_var]:
                selector["state"] = "readonly"
            
            # Add bar plot specific options
            bar_opts = ttk.Frame(self.plot_specific_frame)
            bar_opts.pack(fill=tk.X, pady=5)
            
            # Bar width
            ttk.Label(bar_opts, text="Bar Width:").pack(side=tk.LEFT, padx=5)
            self.plot_specific_options['width'] = ttk.Spinbox(
                bar_opts, from_=0, to=1.0, increment=0.1, width=5)
            self.plot_specific_options['width'].set(0.8)
            self.plot_specific_options['width'].pack(side=tk.LEFT, padx=5)
            
            # Bar orientation
            ttk.Label(bar_opts, text="Orientation:").pack(side=tk.LEFT, padx=5)
            self.plot_specific_options['orientation'] = ttk.Combobox(
                bar_opts, values=['None', 'vertical', 'horizontal'], state="readonly", width=10)
            self.plot_specific_options['orientation'].set('vertical')
            self.plot_specific_options['orientation'].pack(side=tk.LEFT, padx=5)

        elif plot_type == "Histogram":
            self.x_var["values"] = ["None"] + self.continuous_columns
            self.x_var["state"] = "readonly"
            self.color_var["values"] = ["None"] + self.categorical_columns
            self.color_var["state"] = "readonly"
            
            # Add histogram specific options
            hist_opts = ttk.Frame(self.plot_specific_frame)
            hist_opts.pack(fill=tk.X, pady=5)
            
            # Number of bins
            ttk.Label(hist_opts, text="Bins:").pack(side=tk.LEFT, padx=5)
            self.plot_specific_options['bins'] = ttk.Spinbox(
                hist_opts, from_=0, to=100, increment=5, width=5)
            self.plot_specific_options['bins'].set(30)
            self.plot_specific_options['bins'].pack(side=tk.LEFT, padx=5)
            
            # Density
            self.plot_specific_options['density'] = tk.BooleanVar(value=False)
            ttk.Checkbutton(hist_opts, text="Normalize", 
                          variable=self.plot_specific_options['density']).pack(side=tk.LEFT, padx=5)
            
            # Cumulative
            self.plot_specific_options['cumulative'] = tk.BooleanVar(value=False)
            ttk.Checkbutton(hist_opts, text="Cumulative", 
                          variable=self.plot_specific_options['cumulative']).pack(side=tk.LEFT, padx=5)

        elif plot_type in ["Box Plot", "Violin Plot"]:
            self.x_var["values"] = self.categorical_columns
            self.y_var["values"] = self.continuous_columns
            self.color_var["values"] = self.categorical_columns
            for selector in [self.x_var, self.y_var, self.color_var]:
                selector["state"] = "readonly"
            
            # Add box/violin plot specific options
            box_opts = ttk.Frame(self.plot_specific_frame)
            box_opts.pack(fill=tk.X, pady=5)
            
            # Width
            ttk.Label(box_opts, text="Width:").pack(side=tk.LEFT, padx=5)
            self.plot_specific_options['width'] = ttk.Spinbox(
                box_opts, from_=0, to=1.0, increment=0.1, width=5)
            self.plot_specific_options['width'].set(0.8)
            self.plot_specific_options['width'].pack(side=tk.LEFT, padx=5)
            
            if plot_type == "Box Plot":
                # Show outliers
                self.plot_specific_options['showfliers'] = tk.BooleanVar(value=True)
                ttk.Checkbutton(box_opts, text="Show Outliers", 
                              variable=self.plot_specific_options['showfliers']).pack(side=tk.LEFT, padx=5)
                
                # Notch
                self.plot_specific_options['notch'] = tk.BooleanVar(value=False)
                ttk.Checkbutton(box_opts, text="Show Notch", 
                              variable=self.plot_specific_options['notch']).pack(side=tk.LEFT, padx=5)
            else:  # Violin Plot
                # Show inner points
                ttk.Label(box_opts, text="Inner:").pack(side=tk.LEFT, padx=5)
                self.plot_specific_options['inner'] = ttk.Combobox(
                    box_opts, values=['None', 'box', 'stick', 'point'], 
                    state="readonly", width=10)
                self.plot_specific_options['inner'].set('box')
                self.plot_specific_options['inner'].pack(side=tk.LEFT, padx=5)

        elif plot_type == "Heatmap":
            self.x_var["values"] = ["None"] + all_columns
            self.y_var["values"] = ["None"] + all_columns
            self.color_var["values"] = ["None"] + self.continuous_columns
            for selector in [self.x_var, self.y_var, self.color_var]:
                selector["state"] = "readonly"
            
            # Add heatmap specific options
            heatmap_opts = ttk.Frame(self.plot_specific_frame)
            heatmap_opts.pack(fill=tk.X, pady=5)
            
            # Annotations
            self.plot_specific_options['annot'] = tk.BooleanVar(value=True)
            ttk.Checkbutton(heatmap_opts, text="Show Values", 
                          variable=self.plot_specific_options['annot']).pack(side=tk.LEFT, padx=5)
            
            # Format string for annotations
            ttk.Label(heatmap_opts, text="Format:").pack(side=tk.LEFT, padx=5)
            self.plot_specific_options['fmt'] = ttk.Entry(heatmap_opts, width=8)
            self.plot_specific_options['fmt'].insert(0, '.2g')
            self.plot_specific_options['fmt'].pack(side=tk.LEFT, padx=5)
            
            # Center value
            ttk.Label(heatmap_opts, text="Center:").pack(side=tk.LEFT, padx=5)
            self.plot_specific_options['center'] = ttk.Entry(heatmap_opts, width=8)
            self.plot_specific_options['center'].pack(side=tk.LEFT, padx=5)
            
            # Square cells
            self.plot_specific_options['square'] = tk.BooleanVar(value=True)
            ttk.Checkbutton(heatmap_opts, text="Square Cells", 
                          variable=self.plot_specific_options['square']).pack(side=tk.LEFT, padx=5)
            
            # Line widths
            ttk.Label(heatmap_opts, text="Line Width:").pack(side=tk.LEFT, padx=5)
            self.plot_specific_options['linewidths'] = ttk.Spinbox(
                heatmap_opts, from_=0, to=2, increment=0.1, width=5)
            self.plot_specific_options['linewidths'].set(0.5)
            self.plot_specific_options['linewidths'].pack(side=tk.LEFT, padx=5)
            
            # Robust scaling
            self.plot_specific_options['robust'] = tk.BooleanVar(value=False)
            ttk.Checkbutton(heatmap_opts, text="Robust Scaling", 
                          variable=self.plot_specific_options['robust']).pack(side=tk.LEFT, padx=5)

        elif plot_type == "Density Plot":
            self.x_var["values"] = ["None"] + self.continuous_columns
            self.x_var["state"] = "readonly"
            self.color_var["values"] = ["None"] + self.categorical_columns
            self.color_var["state"] = "readonly"
            self.y_var["state"] = "disabled"
            
            # Add density plot specific options
            density_opts = ttk.Frame(self.plot_specific_frame)
            density_opts.pack(fill=tk.X, pady=5)
            
            # Bandwidth adjustment
            ttk.Label(density_opts, text="Bandwidth Factor:").pack(side=tk.LEFT, padx=5)
            self.plot_specific_options['bw_factor'] = ttk.Spinbox(
                density_opts, from_=0.1, to=2.0, increment=0.05, width=5)
            self.plot_specific_options['bw_factor'].set(0.25)
            self.plot_specific_options['bw_factor'].pack(side=tk.LEFT, padx=5)
            
            # Fill
            self.plot_specific_options['fill'] = tk.BooleanVar(value=True)
            ttk.Checkbutton(density_opts, text="Fill", 
                          variable=self.plot_specific_options['fill']).pack(side=tk.LEFT, padx=5)
            
            # Alpha (transparency)
            ttk.Label(density_opts, text="Transparency:").pack(side=tk.LEFT, padx=5)
            self.plot_specific_options['alpha'] = ttk.Spinbox(
                density_opts, from_=0.0, to=1.0, increment=0.1, width=5)
            self.plot_specific_options['alpha'].set(0.4)
            self.plot_specific_options['alpha'].pack(side=tk.LEFT, padx=5)
            
            # Number of points
            ttk.Label(density_opts, text="Points:").pack(side=tk.LEFT, padx=5)
            self.plot_specific_options['points'] = ttk.Spinbox(
                density_opts, from_=50, to=500, increment=50, width=5)
            self.plot_specific_options['points'].set(200)
            self.plot_specific_options['points'].pack(side=tk.LEFT, padx=5)

if __name__ == "__main__":
    root = tk.Tk()
    # Set icon from assets directory
    icon_path = os.path.join(os.path.dirname(__file__), "assets", "logo.ico")
    png_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
    
    # Windows-specific taskbar icon setup
    try:
        import ctypes
        myappid = 'company.csvue.1.0'  # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        if os.path.exists(icon_path):
            root.iconbitmap(default=icon_path)
    except Exception:
        pass  # Not on Windows or other error
    
    # Set window icon - try ICO first, then PNG as fallback
    if os.path.exists(icon_path):
        root.iconbitmap(icon_path)
    elif os.path.exists(png_path):
        icon_image = tk.PhotoImage(file=png_path)
        root.iconphoto(True, icon_image)
    
    plotter = DataPlotterApp(root)
    root.mainloop()
