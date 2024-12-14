import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from ttkbootstrap import Style
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class DataPlotterApp:
    def __init__(self, root):
        self.root = root
        self.style = Style(theme="darkly")
        self.root.title("CSVUE")
        self.root.geometry("900x300")

        self.data = None
        self.categorical_columns = []
        self.continuous_columns = []

        self.setup_ui()

    def setup_ui(self):
        # File selection frame
        file_frame = ttk.Frame(self.root, padding=10)
        file_frame.pack(fill=tk.X)

        ttk.Label(file_frame, text="Data File:").pack(side=tk.LEFT, padx=5)
        self.file_entry = ttk.Entry(file_frame, width=60, bootstyle = 'light')
        self.file_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Browse", bootstyle="warning", command=self.load_file).pack(side=tk.LEFT, padx=5)

        # Plot options frame
        options_frame = ttk.Labelframe(self.root, text="Plot Options", padding=10,bootstyle='info')
        options_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(options_frame, text="Plot Type:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.plot_type = ttk.Combobox(
            options_frame,
            values=["Histogram", "Scatterplot", "Bar Plot", "Line Plot", "Box Plot", "Pie Chart"],
            state="readonly",
        )
        self.plot_type.bind("<<ComboboxSelected>>", self.update_variable_selectors)
        self.plot_type.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        ttk.Label(options_frame, text="X Variable:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.x_var = ttk.Combobox(options_frame, state="disabled", style = 'light')
        self.x_var.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)

        ttk.Label(options_frame, text="Y Variable:").grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)
        self.y_var = ttk.Combobox(options_frame, state="disabled", style = 'light')
        self.y_var.grid(row=0, column=5, padx=5, pady=5, sticky=tk.W)

        # Additional options
        add_labels_frame = ttk.Labelframe(self.root, text="Labels", padding=10, bootstyle="info")
        add_labels_frame.pack(fill=tk.X,padx=10, pady=10)

        ttk.Label(add_labels_frame, text="Title:").pack(side=tk.LEFT, padx=5)
        self.title_entry = ttk.Entry(add_labels_frame, width=20, style = 'light')
        self.title_entry.pack(side=tk.LEFT, padx=5)

        ttk.Label(add_labels_frame, text="X Label:").pack(side=tk.LEFT, padx=5)
        self.xlabel_entry = ttk.Entry(add_labels_frame, width=20, style = 'light')
        self.xlabel_entry.pack(side=tk.LEFT, padx=5)

        ttk.Label(add_labels_frame, text="Y Label:").pack(side=tk.LEFT, padx=5)
        self.ylabel_entry = ttk.Entry(add_labels_frame, width=20, style = 'light')
        self.ylabel_entry.pack(side=tk.LEFT, padx=5)

        # Buttons
        button_frame = ttk.Frame(self.root, padding=10)
        button_frame.pack(fill=tk.X)

        ttk.Button(button_frame, text="Plot", bootstyle="warning", command=self.plot_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Plot", bootstyle="warning", command=self.save_plot).pack(side=tk.LEFT, padx=5)

        # Plot display area
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

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

        self.file_entry.delete(0, tk.END)
        self.file_entry.insert(0, file_path)

        try:
            if file_path.endswith(".csv"):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
                self.data = pd.read_excel(file_path)
            elif file_path.endswith(".json"):
                self.data = pd.read_json(file_path)
            elif file_path.endswith(".parquet"):
                self.data = pd.read_parquet(file_path)
            else:
                raise ValueError("Unsupported file format")

            self.detect_column_types()
            messagebox.showinfo("Success", "File loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

    def detect_column_types(self):
        self.categorical_columns = self.data.select_dtypes(include=["object", "category"]).columns.tolist()
        self.continuous_columns = self.data.select_dtypes(include=["number"]).columns.tolist()

    def update_variable_selectors(self, event=None):
        plot_type = self.plot_type.get()

        if plot_type == "Histogram":
            self.x_var["values"] = self.continuous_columns
            self.x_var["state"] = "readonly"
            self.y_var.set("")
            self.y_var["state"] = "disabled"

        elif plot_type == "Scatterplot":
            self.x_var["values"] = self.continuous_columns
            self.y_var["values"] = self.continuous_columns
            self.x_var["state"] = "readonly"
            self.y_var["state"] = "readonly"

        elif plot_type == "Bar Plot":
            self.x_var["values"] = self.categorical_columns
            self.y_var["values"] = self.continuous_columns
            self.x_var["state"] = "readonly"
            self.y_var["state"] = "readonly"

        elif plot_type == "Line Plot":
            self.x_var["values"] = self.continuous_columns
            self.y_var["values"] = self.continuous_columns
            self.x_var["state"] = "readonly"
            self.y_var["state"] = "readonly"

        elif plot_type == "Box Plot":
            self.x_var["values"] = self.categorical_columns
            self.y_var["values"] = self.continuous_columns
            self.x_var["state"] = "readonly"
            self.y_var["state"] = "readonly"

        elif plot_type == "Pie Chart":
            self.x_var["values"] = self.categorical_columns
            self.x_var["state"] = "readonly"
            self.y_var.set("")
            self.y_var["state"] = "disabled"

    def plot_data(self):
        plot_type = self.plot_type.get()
        x_var = self.x_var.get()
        y_var = self.y_var.get()

        if not plot_type or not x_var or (plot_type not in ["Histogram", "Pie Chart"] and not y_var):
            messagebox.showerror("Error", "Please select all required options.")
            return

        fig, ax = plt.subplots()

        try:
            if plot_type == "Histogram":
                self.data[x_var].plot(kind="hist", ax=ax, title=f"Histogram of {x_var}")

            elif plot_type == "Scatterplot":
                self.data.plot.scatter(x=x_var, y=y_var, ax=ax, title=f"Scatterplot of {x_var} vs {y_var}")

            elif plot_type == "Bar Plot":
                self.data.groupby(x_var)[y_var].mean().plot(kind="bar", ax=ax, title=f"Bar Plot of {y_var} by {x_var}")

            elif plot_type == "Line Plot":
                self.data.plot(x=x_var, y=y_var, ax=ax, title=f"Line Plot of {y_var} vs {x_var}")

            elif plot_type == "Box Plot":
                self.data.boxplot(column=y_var, by=x_var, ax=ax)
                ax.set_title(f"Box Plot of {y_var} by {x_var}")
                ax.set_ylabel(y_var)

            elif plot_type == "Pie Chart":
                self.data[x_var].value_counts().plot(kind="pie", ax=ax, title=f"Pie Chart of {x_var}")
                ax.set_ylabel('')

            # Set title and labels
            title = self.title_entry.get()
            xlabel = self.xlabel_entry.get()
            ylabel = self.ylabel_entry.get()

            if title:
                ax.set_title(title)
            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)
                
            self.root.geometry("900x800")

            self.display_plot(fig)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot data: {e}")

    def display_plot(self, fig):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def save_plot(self):
        if not hasattr(self, "current_fig") or self.current_fig is None:
            messagebox.showerror("Error", "No plot to save.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg"), ("PDF Files", "*.pdf")]
        )
        if not file_path:
            return

        try:
            self.current_fig.savefig(file_path)
            messagebox.showinfo("Success", "Plot saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save plot: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DataPlotterApp(root)
    root.mainloop()
