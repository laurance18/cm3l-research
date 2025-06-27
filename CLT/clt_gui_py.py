"""
Enhanced Classical Lamination Theory (CLT) Analysis Tool v2.0

This application provides comprehensive composite laminate analysis using Classical Lamination Theory.
Features:
- Material property input for each ply (E1, E2, G12, ν12)
- Ply orientation angles (θ) and thickness specification
- Computation of [Q] matrices in local and global coordinate systems
- Assembly of laminate stiffness matrices [A], [B], and [D]
- Mechanical load input (Nx, Ny, Nxy, Mx, My, Mxy)
- Calculation of midplane strains and curvatures
- Ply-by-ply stress and strain analysis
- Visualization of stress/strain distributions through thickness

Dependencies:
- numpy: Matrix operations and numerical calculations
- matplotlib: Plotting and visualization
- tkinter: GUI framework

Author: CLT Analysis Tool v2.0
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
from typing import List, Tuple, Dict, Optional
import math


class CLTCore:
    """Core Classical Lamination Theory calculations and matrix operations."""
    
    @staticmethod
    def compute_q_matrix(E1: float, E2: float, G12: float, v12: float) -> np.ndarray:
        """
        Compute the reduced stiffness matrix [Q] for an orthotropic lamina in principal axes.
        
        Args:
            E1: Longitudinal modulus (Pa)
            E2: Transverse modulus (Pa) 
            G12: In-plane shear modulus (Pa)
            v12: Major Poisson's ratio
            
        Returns:
            3x3 reduced stiffness matrix [Q]
        """
        # Calculate minor Poisson's ratio using reciprocal relationship
        v21 = v12 * E2 / E1
        
        # Check physical constraints
        if v12 * v21 >= 1.0:
            raise ValueError("Invalid Poisson's ratios: v12 * v21 must be < 1")
        
        # Calculate matrix components
        denominator = 1 - v12 * v21
        Q11 = E1 / denominator
        Q22 = E2 / denominator
        Q12 = v12 * E2 / denominator
        Q66 = G12
        
        Q = np.array([
            [Q11, Q12, 0],
            [Q12, Q22, 0],
            [0, 0, Q66]
        ], dtype=float)
        
        return Q
    
    @staticmethod
    def transform_q_matrix(Q: np.ndarray, theta_deg: float) -> np.ndarray:
        """
        Transform reduced stiffness matrix [Q] to global coordinates [Q̄].
        
        Args:
            Q: 3x3 reduced stiffness matrix in local coordinates
            theta_deg: Ply orientation angle in degrees
            
        Returns:
            3x3 transformed reduced stiffness matrix [Q̄]
        """
        # Convert angle to radians and compute trigonometric functions
        theta_rad = np.radians(theta_deg)
        m = np.cos(theta_rad)
        n = np.sin(theta_rad)
        
        # Powers for transformation
        m2, n2 = m**2, n**2
        m4, n4 = m**4, n**4
        
        # Extract Q matrix components
        Q11, Q12, Q22, Q66 = Q[0,0], Q[0,1], Q[1,1], Q[2,2]
        
        # Transformed stiffness matrix components
        Q_bar_11 = Q11*m4 + 2*(Q12 + 2*Q66)*m2*n2 + Q22*n4
        Q_bar_12 = (Q11 + Q22 - 4*Q66)*m2*n2 + Q12*(m4 + n4)
        Q_bar_22 = Q11*n4 + 2*(Q12 + 2*Q66)*m2*n2 + Q22*m4
        Q_bar_16 = (Q11 - Q12 - 2*Q66)*m*n*(m2 - n2) + (Q12 - Q22 + 2*Q66)*m*n*(m2 - n2)
        Q_bar_26 = (Q11 - Q12 - 2*Q66)*m*n*(n2 - m2) + (Q12 - Q22 + 2*Q66)*m*n*(n2 - m2)
        Q_bar_66 = (Q11 + Q22 - 2*Q12 - 2*Q66)*m2*n2 + Q66*(m4 + n4)
        
        Q_bar = np.array([
            [Q_bar_11, Q_bar_12, Q_bar_16],
            [Q_bar_12, Q_bar_22, Q_bar_26],
            [Q_bar_16, Q_bar_26, Q_bar_66]
        ], dtype=float)
        
        return Q_bar
    
    @staticmethod
    def compute_abd_matrices(plies: List[Tuple[float, float, float, float, float, float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
        """
        Compute laminate stiffness matrices [A], [B], and [D].
        
        Args:
            plies: List of tuples (E1, E2, G12, v12, thickness, theta_deg)
            
        Returns:
            Tuple of (A_matrix, B_matrix, D_matrix, z_coordinates)
        """
        if not plies:
            raise ValueError("No plies defined")
        
        # Calculate total thickness and z-coordinates
        total_thickness = sum(ply[4] for ply in plies)
        z_coords = [-total_thickness / 2]
        
        for ply in plies:
            z_coords.append(z_coords[-1] + ply[4])
        
        # Initialize matrices
        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        D = np.zeros((3, 3))
        
        # Calculate contributions from each ply
        for k, ply in enumerate(plies):
            E1, E2, G12, v12, thickness, theta = ply
            
            # Compute local and transformed stiffness matrices
            Q = CLTCore.compute_q_matrix(E1, E2, G12, v12)
            Q_bar = CLTCore.transform_q_matrix(Q, theta)
            
            # Z-coordinates for current ply
            z_k = z_coords[k + 1]
            z_k_minus_1 = z_coords[k]
            
            # Add contributions to A, B, D matrices
            A += Q_bar * (z_k - z_k_minus_1)
            B += 0.5 * Q_bar * (z_k**2 - z_k_minus_1**2)
            D += (1/3) * Q_bar * (z_k**3 - z_k_minus_1**3)
        
        return A, B, D, z_coords
    
    @staticmethod
    def solve_midplane_response(A: np.ndarray, B: np.ndarray, D: np.ndarray, 
                               loads: np.ndarray, moments: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve for midplane strains and curvatures.
        
        Args:
            A, B, D: Laminate stiffness matrices
            loads: Force resultants [Nx, Ny, Nxy]
            moments: Moment resultants [Mx, My, Mxy]
            
        Returns:
            Tuple of (midplane_strains, curvatures)
        """
        # Assemble ABD matrix
        ABD_upper = np.hstack([A, B])
        ABD_lower = np.hstack([B, D])
        ABD = np.vstack([ABD_upper, ABD_lower])
        
        # Assemble load vector
        load_vector = np.hstack([loads, moments])
        
        # Solve linear system
        try:
            response = np.linalg.solve(ABD, load_vector)
        except np.linalg.LinAlgError:
            raise ValueError("Singular ABD matrix - check laminate configuration")
        
        midplane_strains = response[:3]
        curvatures = response[3:]
        
        return midplane_strains, curvatures
    
    @staticmethod
    def compute_ply_stresses_strains(plies: List[Tuple[float, float, float, float, float, float]],
                                   z_coords: List[float], midplane_strains: np.ndarray, 
                                   curvatures: np.ndarray) -> List[Dict]:
        """
        Compute stresses and strains for each ply.
        
        Args:
            plies: List of ply definitions
            z_coords: Z-coordinates through thickness
            midplane_strains: Midplane strain components
            curvatures: Curvature components
            
        Returns:
            List of dictionaries containing ply results
        """
        results = []
        
        for k, ply in enumerate(plies):
            E1, E2, G12, v12, thickness, theta = ply
            
            # Z-coordinates for current ply
            z_bottom = z_coords[k]
            z_top = z_coords[k + 1]
            z_mid = (z_bottom + z_top) / 2
            
            # Global strains at mid-ply
            global_strains = midplane_strains + curvatures * z_mid
            
            # Compute stiffness matrices
            Q = CLTCore.compute_q_matrix(E1, E2, G12, v12)
            Q_bar = CLTCore.transform_q_matrix(Q, theta)
            
            # Global stresses
            global_stresses = Q_bar @ global_strains
            
            # Transform to local coordinates for local stresses/strains
            local_strains = CLTCore.transform_strain_to_local(global_strains, theta)
            local_stresses = Q @ local_strains
            
            ply_result = {
                'ply_number': k + 1,
                'theta': theta,
                'z_bottom': z_bottom,
                'z_top': z_top,
                'z_mid': z_mid,
                'global_strains': global_strains,
                'global_stresses': global_stresses,
                'local_strains': local_strains,
                'local_stresses': local_stresses,
                'thickness': thickness
            }
            
            results.append(ply_result)
        
        return results
    
    @staticmethod
    def transform_strain_to_local(global_strain: np.ndarray, theta_deg: float) -> np.ndarray:
        """Transform strain from global to local coordinate system."""
        theta_rad = np.radians(theta_deg)
        m, n = np.cos(theta_rad), np.sin(theta_rad)
        
        # Strain transformation matrix
        T = np.array([
            [m**2, n**2, 2*m*n],
            [n**2, m**2, -2*m*n],
            [-m*n, m*n, m**2 - n**2]
        ])
        
        return T @ global_strain


class MaterialDatabase:
    """Database of common composite material properties."""
    
    MATERIALS = {
        "Carbon/Epoxy (AS4/3501-6)": {
            "E1": 126e9,  # Pa
            "E2": 11e9,   # Pa
            "G12": 6.6e9, # Pa
            "v12": 0.28,
            "density": 1600,  # kg/m³
        },
        "Glass/Epoxy (E-Glass/LY556)": {
            "E1": 53.48e9,
            "E2": 17.7e9,
            "G12": 5.83e9,
            "v12": 0.278,
            "density": 2100,
        },
        "Carbon/Epoxy (T300/5208)": {
            "E1": 181e9,
            "E2": 10.3e9,
            "G12": 7.17e9,
            "v12": 0.28,
            "density": 1600,
        },
        "Aramid/Epoxy (Kevlar 49/Epoxy)": {
            "E1": 76e9,
            "E2": 5.5e9,
            "G12": 2.3e9,
            "v12": 0.34,
            "density": 1380,
        },
        "Custom": {
            "E1": 150e9,
            "E2": 10e9,
            "G12": 5e9,
            "v12": 0.3,
            "density": 1500,
        }
    }
    
    @classmethod
    def get_material_names(cls) -> List[str]:
        """Get list of available material names."""
        return list(cls.MATERIALS.keys())
    
    @classmethod
    def get_material_properties(cls, name: str) -> Dict:
        """Get material properties by name."""
        return cls.MATERIALS.get(name, cls.MATERIALS["Custom"])


class CLTAnalysisApp:
    """Enhanced CLT Analysis Application with improved GUI and visualization."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Enhanced Classical Lamination Theory Analysis v2.0")
        self.root.geometry("1400x900")
        
        # Application state
        self.plies = []
        self.results = None
        
        # Style configuration
        self.setup_styles()
        
        # Create GUI elements
        self.create_widgets()
        
        # Initialize with default laminate
        self.add_default_plies()
    
    def setup_styles(self):
        """Configure GUI styles and themes."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure custom styles
        style.configure('Title.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Header.TLabel', font=('Arial', 10, 'bold'))
        style.configure('Results.TLabel', font=('Courier', 9))
    
    def create_widgets(self):
        """Create and arrange GUI widgets."""
        # Main container with notebook for tabbed interface
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_input_tab()
        self.create_results_tab()
        self.create_visualization_tab()
    
    def create_input_tab(self):
        """Create the input tab for laminate definition and loads."""
        self.input_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.input_frame, text="Laminate Definition")
        
        # Main paned window for input layout
        paned = ttk.PanedWindow(self.input_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Ply definition
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=2)
        
        # Ply definition section
        ply_frame = ttk.LabelFrame(left_frame, text="Ply Definition", padding=10)
        ply_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Ply controls
        controls_frame = ttk.Frame(ply_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(controls_frame, text="Add Ply", command=self.add_ply).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls_frame, text="Remove Ply", command=self.remove_ply).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls_frame, text="Clear All", command=self.clear_plies).pack(side=tk.LEFT, padx=(0, 5))
        
        # Ply table with scrollbars
        table_frame = ttk.Frame(ply_frame)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview for ply data
        columns = ('Ply', 'Material', 'E1 (GPa)', 'E2 (GPa)', 'G12 (GPa)', 'ν12', 'Thickness (mm)', 'θ (°)')
        self.ply_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=10)
        
        # Configure columns
        for col in columns:
            self.ply_tree.heading(col, text=col)
            if col == 'Ply':
                self.ply_tree.column(col, width=50)
            elif col == 'Material':
                self.ply_tree.column(col, width=120)
            else:
                self.ply_tree.column(col, width=80)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.ply_tree.yview)
        h_scrollbar = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.ply_tree.xview)
        self.ply_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack treeview and scrollbars
        self.ply_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Bind double-click to edit ply
        self.ply_tree.bind('<Double-1>', self.edit_ply)
        
        # Right panel - Loads and analysis
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)
        
        # Load definition section
        load_frame = ttk.LabelFrame(right_frame, text="Applied Loads", padding=10)
        load_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Force resultants
        force_frame = ttk.LabelFrame(load_frame, text="Force Resultants (N/m)")
        force_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.load_vars = {}
        force_labels = ['Nx', 'Ny', 'Nxy']
        for i, label in enumerate(force_labels):
            ttk.Label(force_frame, text=f"{label}:").grid(row=i//3, column=(i%3)*2, sticky=tk.W, padx=5, pady=2)
            var = tk.DoubleVar(value=0.0)
            self.load_vars[label] = var
            ttk.Entry(force_frame, textvariable=var, width=12).grid(row=i//3, column=(i%3)*2+1, padx=5, pady=2)
        
        # Moment resultants
        moment_frame = ttk.LabelFrame(load_frame, text="Moment Resultants (N·m/m)")
        moment_frame.pack(fill=tk.X, pady=(5, 0))
        
        moment_labels = ['Mx', 'My', 'Mxy']
        for i, label in enumerate(moment_labels):
            ttk.Label(moment_frame, text=f"{label}:").grid(row=i//3, column=(i%3)*2, sticky=tk.W, padx=5, pady=2)
            var = tk.DoubleVar(value=0.0)
            self.load_vars[label] = var
            ttk.Entry(moment_frame, textvariable=var, width=12).grid(row=i//3, column=(i%3)*2+1, padx=5, pady=2)
        
        # Analysis controls
        analysis_frame = ttk.LabelFrame(right_frame, text="Analysis", padding=10)
        analysis_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(analysis_frame, text="Analyze Laminate", 
                  command=self.analyze_laminate, style='Title.TLabel').pack(pady=10)
        
        # File operations
        file_frame = ttk.LabelFrame(right_frame, text="File Operations", padding=10)
        file_frame.pack(fill=tk.X)
        
        ttk.Button(file_frame, text="Save Configuration", command=self.save_configuration).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Load Configuration", command=self.load_configuration).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Export Results", command=self.export_results).pack(fill=tk.X, pady=2)
    
    def create_results_tab(self):
        """Create the results tab for displaying analysis results."""
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Analysis Results")
        
        # Create scrolled text widget for results
        text_frame = ttk.Frame(self.results_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.results_text = tk.Text(text_frame, wrap=tk.NONE, font=('Courier', 10))
        
        # Scrollbars for results text
        v_scroll = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        h_scroll = ttk.Scrollbar(text_frame, orient=tk.HORIZONTAL, command=self.results_text.xview)
        self.results_text.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        # Pack text and scrollbars
        self.results_text.grid(row=0, column=0, sticky='nsew')
        v_scroll.grid(row=0, column=1, sticky='ns')
        h_scroll.grid(row=1, column=0, sticky='ew')
        
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)
    
    def create_visualization_tab(self):
        """Create the visualization tab for plotting results."""
        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="Visualization")
        
        # Control panel for visualization options
        control_frame = ttk.Frame(self.viz_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(control_frame, text="Plot Type:").pack(side=tk.LEFT)
        self.plot_type = tk.StringVar(value="Stress Distribution")
        plot_combo = ttk.Combobox(control_frame, textvariable=self.plot_type, 
                                 values=["Stress Distribution", "Strain Distribution", 
                                        "Material Properties"], state="readonly")
        plot_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Update Plot", command=self.update_plot).pack(side=tk.LEFT, padx=5)
        
        # Matplotlib figure for visualization
        self.setup_matplotlib()
    
    def setup_matplotlib(self):
        """Setup matplotlib figure and canvas."""
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Laminate Analysis Visualization')
        
        self.canvas = FigureCanvasTkAgg(self.fig, self.viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def add_default_plies(self):
        """Add default ply configuration for demonstration."""
        default_config = [
            ("Carbon/Epoxy (AS4/3501-6)", 0.125, 0),
            ("Carbon/Epoxy (AS4/3501-6)", 0.125, 45),
            ("Carbon/Epoxy (AS4/3501-6)", 0.125, -45),
            ("Carbon/Epoxy (AS4/3501-6)", 0.125, 90)
        ]
        
        for material, thickness, angle in default_config:
            self.add_ply_with_params(material, thickness, angle)
    
    def add_ply(self):
        """Add a new ply with default properties."""
        self.add_ply_with_params("Carbon/Epoxy (AS4/3501-6)", 0.125, 0)
    
    def add_ply_with_params(self, material_name="Custom", thickness=0.125, angle=0):
        """Add a ply with specified parameters."""
        material_props = MaterialDatabase.get_material_properties(material_name)
        
        ply = {
            'material': material_name,
            'E1': material_props['E1'],
            'E2': material_props['E2'],
            'G12': material_props['G12'],
            'v12': material_props['v12'],
            'thickness': thickness / 1000,  # Convert mm to m
            'angle': angle
        }
        
        self.plies.append(ply)
        self.update_ply_display()
    
    def remove_ply(self):
        """Remove selected ply."""
        selection = self.ply_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a ply to remove.")
            return
        
        # Get index of selected item
        item = selection[0]
        index = self.ply_tree.index(item)
        
        # Remove from plies list
        if 0 <= index < len(self.plies):
            del self.plies[index]
            self.update_ply_display()
    
    def clear_plies(self):
        """Clear all plies."""
        if messagebox.askyesno("Confirm", "Clear all plies?"):
            self.plies.clear()
            self.update_ply_display()
    
    def edit_ply(self, event):
        """Edit selected ply properties."""
        selection = self.ply_tree.selection()
        if not selection:
            return
        
        item = selection[0]
        index = self.ply_tree.index(item)
        
        if 0 <= index < len(self.plies):
            self.show_ply_editor(index)
    
    def show_ply_editor(self, index):
        """Show ply property editor dialog."""
        ply = self.plies[index]
        
        # Create editor window
        editor = tk.Toplevel(self.root)
        editor.title(f"Edit Ply {index + 1}")
        editor.geometry("400x350")
        editor.transient(self.root)
        editor.grab_set()
        
        # Material selection
        ttk.Label(editor, text="Material:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        material_var = tk.StringVar(value=ply['material'])
        material_combo = ttk.Combobox(editor, textvariable=material_var, 
                                     values=MaterialDatabase.get_material_names(), state="readonly")
        material_combo.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        
        # Property entries
        props = ['E1', 'E2', 'G12', 'v12']
        prop_vars = {}
        
        for i, prop in enumerate(props, 1):
            ttk.Label(editor, text=f"{prop}:").grid(row=i, column=0, sticky=tk.W, padx=5, pady=5)
            var = tk.DoubleVar(value=ply[prop] / 1e9 if prop != 'v12' else ply[prop])
            prop_vars[prop] = var
            ttk.Entry(editor, textvariable=var).grid(row=i, column=1, sticky=tk.EW, padx=5, pady=5)
        
        # Thickness
        ttk.Label(editor, text="Thickness (mm):").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        thickness_var = tk.DoubleVar(value=ply['thickness'] * 1000)
        ttk.Entry(editor, textvariable=thickness_var).grid(row=5, column=1, sticky=tk.EW, padx=5, pady=5)
        
        # Angle
        ttk.Label(editor, text="Angle (°):").grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)
        angle_var = tk.DoubleVar(value=ply['angle'])
        ttk.Entry(editor, textvariable=angle_var).grid(row=6, column=1, sticky=tk.EW, padx=5, pady=5)
        
        # Update material properties when material changes
        def update_material_props():
            material_props = MaterialDatabase.get_material_properties(material_var.get())
            for prop in props:
                prop_vars[prop].set(material_props[prop] / 1e9 if prop != 'v12' else material_props[prop])
        
        material_combo.bind('<<ComboboxSelected>>', lambda e: update_material_props())
        
        # Buttons
        button_frame = ttk.Frame(editor)
        button_frame.grid(row=7, column=0, columnspan=2, pady=20)
        
        def save_changes():
            try:
                # Update ply properties
                ply['material'] = material_var.get()
                for prop in props:
                    ply[prop] = prop_vars[prop].get() * 1e9 if prop != 'v12' else prop_vars[prop].get()
                ply['thickness'] = thickness_var.get() / 1000  # Convert mm to m
                ply['angle'] = angle_var.get()
                
                self.update_ply_display()
                editor.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Invalid input: {e}")
        
        ttk.Button(button_frame, text="Save", command=save_changes).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=editor.destroy).pack(side=tk.LEFT, padx=5)
        
        editor.grid_columnconfigure(1, weight=1)
    
    def update_ply_display(self):
        """Update the ply display table."""
        # Clear existing items
        for item in self.ply_tree.get_children():
            self.ply_tree.delete(item)
        
        # Add plies to display
        for i, ply in enumerate(self.plies):
            values = (
                i + 1,
                ply['material'],
                f"{ply['E1']/1e9:.1f}",
                f"{ply['E2']/1e9:.1f}",
                f"{ply['G12']/1e9:.1f}",
                f"{ply['v12']:.3f}",
                f"{ply['thickness']*1000:.3f}",
                f"{ply['angle']:.1f}"
            )
            self.ply_tree.insert('', tk.END, values=values)
    
    def analyze_laminate(self):
        """Perform CLT analysis of the laminate."""
        if not self.plies:
            messagebox.showerror("Error", "No plies defined. Please add plies first.")
            return
        
        try:
            # Convert plies to analysis format
            ply_data = []
            for ply in self.plies:
                ply_data.append((
                    ply['E1'], ply['E2'], ply['G12'], ply['v12'], 
                    ply['thickness'], ply['angle']
                ))
            
            # Get load inputs
            loads = np.array([
                self.load_vars['Nx'].get(),
                self.load_vars['Ny'].get(),
                self.load_vars['Nxy'].get()
            ])
            
            moments = np.array([
                self.load_vars['Mx'].get(),
                self.load_vars['My'].get(),
                self.load_vars['Mxy'].get()
            ])
            
            # Perform analysis
            A, B, D, z_coords = CLTCore.compute_abd_matrices(ply_data)
            midplane_strains, curvatures = CLTCore.solve_midplane_response(A, B, D, loads, moments)
            ply_results = CLTCore.compute_ply_stresses_strains(ply_data, z_coords, midplane_strains, curvatures)
            
            # Store results
            self.results = {
                'A': A, 'B': B, 'D': D, 'z_coords': z_coords,
                'midplane_strains': midplane_strains, 'curvatures': curvatures,
                'ply_results': ply_results, 'loads': loads, 'moments': moments
            }
            
            # Display results
            self.display_results()
            self.update_plot()
            
            # Switch to results tab
            self.notebook.select(1)
            
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Analysis failed: {str(e)}")
    
    def display_results(self):
        """Display analysis results in the results text widget."""
        if not self.results:
            return
        
        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        
        # Format and display results
        output = []
        
        # Header
        output.append("="*80)
        output.append("CLASSICAL LAMINATION THEORY ANALYSIS RESULTS")
        output.append("="*80)
        
        # Laminate configuration
        output.append(f"\nLAMINATE CONFIGURATION ({len(self.plies)} plies):")
        output.append("-" * 40)
        total_thickness = sum(ply['thickness'] for ply in self.plies) * 1000  # mm
        output.append(f"Total thickness: {total_thickness:.3f} mm")
        
        for i, ply in enumerate(self.plies):
            output.append(f"Ply {i+1:2d}: {ply['material']:<25} t={ply['thickness']*1000:.3f}mm θ={ply['angle']:6.1f}°")
        
        # Applied loads
        output.append(f"\nAPPLIED LOADS:")
        output.append("-" * 40)
        loads = self.results['loads']
        moments = self.results['moments']
        output.append(f"Nx = {loads[0]:12.3e} N/m     Mx = {moments[0]:12.3e} N⋅m/m")
        output.append(f"Ny = {loads[1]:12.3e} N/m     My = {moments[1]:12.3e} N⋅m/m")
        output.append(f"Nxy= {loads[2]:12.3e} N/m     Mxy= {moments[2]:12.3e} N⋅m/m")
        
        # ABD matrices
        output.append(f"\nSTIFFNESS MATRICES:")
        output.append("-" * 40)
        
        A, B, D = self.results['A'], self.results['B'], self.results['D']
        
        output.append("A Matrix (Extensional stiffness, N/m):")
        for i in range(3):
            output.append(f"[{A[i,0]:12.3e} {A[i,1]:12.3e} {A[i,2]:12.3e}]")
        
        output.append("\nB Matrix (Coupling stiffness, N):")
        for i in range(3):
            output.append(f"[{B[i,0]:12.3e} {B[i,1]:12.3e} {B[i,2]:12.3e}]")
        
        output.append("\nD Matrix (Bending stiffness, N⋅m):")
        for i in range(3):
            output.append(f"[{D[i,0]:12.3e} {D[i,1]:12.3e} {D[i,2]:12.3e}]")
        
        # Midplane response
        eps0 = self.results['midplane_strains']
        kappa = self.results['curvatures']
        
        output.append(f"\nMIDPLANE RESPONSE:")
        output.append("-" * 40)
        output.append(f"Midplane strains:")
        output.append(f"  ε₀ₓ  = {eps0[0]:12.6e}")
        output.append(f"  ε₀ᵧ  = {eps0[1]:12.6e}")
        output.append(f"  γ₀ₓᵧ = {eps0[2]:12.6e}")
        
        output.append(f"\nCurvatures:")
        output.append(f"  κₓ   = {kappa[0]:12.6e} m⁻¹")
        output.append(f"  κᵧ   = {kappa[1]:12.6e} m⁻¹")
        output.append(f"  κₓᵧ  = {kappa[2]:12.6e} m⁻¹")
        
        # Ply-by-ply results
        output.append(f"\nPLY-BY-PLY RESULTS:")
        output.append("-" * 40)
        
        for ply_result in self.results['ply_results']:
            ply_num = ply_result['ply_number']
            theta = ply_result['theta']
            z_mid = ply_result['z_mid'] * 1000  # Convert to mm
            
            output.append(f"\nPly {ply_num} (θ = {theta}°, z = {z_mid:.3f} mm):")
            
            # Global stresses and strains
            global_strains = ply_result['global_strains']
            global_stresses = ply_result['global_stresses']
            
            output.append(f"  Global strains:   εₓ={global_strains[0]:10.6e}  εᵧ={global_strains[1]:10.6e}  γₓᵧ={global_strains[2]:10.6e}")
            output.append(f"  Global stresses:  σₓ={global_stresses[0]/1e6:10.3f}  σᵧ={global_stresses[1]/1e6:10.3f}  τₓᵧ={global_stresses[2]/1e6:10.3f} MPa")
            
            # Local stresses and strains
            local_strains = ply_result['local_strains']
            local_stresses = ply_result['local_stresses']
            
            output.append(f"  Local strains:    ε₁={local_strains[0]:10.6e}  ε₂={local_strains[1]:10.6e}  γ₁₂={local_strains[2]:10.6e}")
            output.append(f"  Local stresses:   σ₁={local_stresses[0]/1e6:10.3f}  σ₂={local_stresses[1]/1e6:10.3f}  τ₁₂={local_stresses[2]/1e6:10.3f} MPa")
        
        # Insert text
        self.results_text.insert(tk.END, '\n'.join(output))
    
    def update_plot(self):
        """Update visualization plots."""
        if not self.results:
            return
        
        # Clear previous plots
        for ax in self.axes.flat:
            ax.clear()
        
        plot_type = self.plot_type.get()
        
        if plot_type == "Stress Distribution":
            self.plot_stress_distribution()
        elif plot_type == "Strain Distribution":
            self.plot_strain_distribution()
        elif plot_type == "Material Properties":
            self.plot_material_properties()
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def plot_stress_distribution(self):
        """Plot stress distribution through thickness."""
        ply_results = self.results['ply_results']
        
        # Extract data for plotting
        z_positions = []
        sigma_x = []
        sigma_y = []
        tau_xy = []
        
        for result in ply_results:
            z_positions.append(result['z_mid'] * 1000)  # Convert to mm
            sigma_x.append(result['global_stresses'][0] / 1e6)  # Convert to MPa
            sigma_y.append(result['global_stresses'][1] / 1e6)
            tau_xy.append(result['global_stresses'][2] / 1e6)
        
        # Plot stress distributions
        self.axes[0,0].plot(sigma_x, z_positions, 'ro-', linewidth=2, markersize=6)
        self.axes[0,0].set_xlabel('σₓ (MPa)')
        self.axes[0,0].set_ylabel('Z position (mm)')
        self.axes[0,0].set_title('Normal Stress σₓ')
        self.axes[0,0].grid(True, alpha=0.3)
        
        self.axes[0,1].plot(sigma_y, z_positions, 'go-', linewidth=2, markersize=6)
        self.axes[0,1].set_xlabel('σᵧ (MPa)')
        self.axes[0,1].set_ylabel('Z position (mm)')
        self.axes[0,1].set_title('Normal Stress σᵧ')
        self.axes[0,1].grid(True, alpha=0.3)
        
        self.axes[1,0].plot(tau_xy, z_positions, 'bo-', linewidth=2, markersize=6)
        self.axes[1,0].set_xlabel('τₓᵧ (MPa)')
        self.axes[1,0].set_ylabel('Z position (mm)')
        self.axes[1,0].set_title('Shear Stress τₓᵧ')
        self.axes[1,0].grid(True, alpha=0.3)
        
        # Ply boundaries
        z_coords = np.array(self.results['z_coords']) * 1000  # Convert to mm
        ply_angles = [ply['angle'] for ply in self.plies]
        
        y_pos = np.arange(len(ply_angles))
        self.axes[1,1].barh(y_pos, ply_angles, color='lightblue', edgecolor='black')
        self.axes[1,1].set_xlabel('Ply Angle (°)')
        self.axes[1,1].set_ylabel('Ply Number')
        self.axes[1,1].set_title('Ply Orientation')
        self.axes[1,1].set_yticks(y_pos)
        self.axes[1,1].set_yticklabels([f'Ply {i+1}' for i in range(len(ply_angles))])
        self.axes[1,1].grid(True, alpha=0.3)
    
    def plot_strain_distribution(self):
        """Plot strain distribution through thickness."""
        ply_results = self.results['ply_results']
        
        # Extract data for plotting
        z_positions = []
        epsilon_x = []
        epsilon_y = []
        gamma_xy = []
        
        for result in ply_results:
            z_positions.append(result['z_mid'] * 1000)  # Convert to mm
            epsilon_x.append(result['global_strains'][0] * 1e6)  # Convert to microstrain
            epsilon_y.append(result['global_strains'][1] * 1e6)
            gamma_xy.append(result['global_strains'][2] * 1e6)
        
        # Plot strain distributions
        self.axes[0,0].plot(epsilon_x, z_positions, 'ro-', linewidth=2, markersize=6)
        self.axes[0,0].set_xlabel('εₓ (μɛ)')
        self.axes[0,0].set_ylabel('Z position (mm)')
        self.axes[0,0].set_title('Normal Strain εₓ')
        self.axes[0,0].grid(True, alpha=0.3)
        
        self.axes[0,1].plot(epsilon_y, z_positions, 'go-', linewidth=2, markersize=6)
        self.axes[0,1].set_xlabel('εᵧ (μɛ)')
        self.axes[0,1].set_ylabel('Z position (mm)')
        self.axes[0,1].set_title('Normal Strain εᵧ')
        self.axes[0,1].grid(True, alpha=0.3)
        
        self.axes[1,0].plot(gamma_xy, z_positions, 'bo-', linewidth=2, markersize=6)
        self.axes[1,0].set_xlabel('γₓᵧ (μɛ)')
        self.axes[1,0].set_ylabel('Z position (mm)')
        self.axes[1,0].set_title('Shear Strain γₓᵧ')
        self.axes[1,0].grid(True, alpha=0.3)
        
        # Total laminate thickness
        total_thickness = sum(ply['thickness'] for ply in self.plies) * 1000
        self.axes[1,1].text(0.5, 0.5, f'Total Thickness:\n{total_thickness:.3f} mm\n\n'
                           f'Number of Plies:\n{len(self.plies)}', 
                           transform=self.axes[1,1].transAxes, ha='center', va='center',
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        self.axes[1,1].set_title('Laminate Summary')
        self.axes[1,1].axis('off')
    
    def plot_material_properties(self):
        """Plot material properties of each ply."""
        if not self.plies:
            return
        
        ply_numbers = range(1, len(self.plies) + 1)
        
        # Extract material properties
        E1_values = [ply['E1']/1e9 for ply in self.plies]  # Convert to GPa
        E2_values = [ply['E2']/1e9 for ply in self.plies]
        G12_values = [ply['G12']/1e9 for ply in self.plies]
        v12_values = [ply['v12'] for ply in self.plies]
        
        # Plot material properties
        width = 0.35
        x = np.arange(len(ply_numbers))
        
        self.axes[0,0].bar(x - width/2, E1_values, width, label='E₁', color='skyblue')
        self.axes[0,0].bar(x + width/2, E2_values, width, label='E₂', color='lightcoral')
        self.axes[0,0].set_xlabel('Ply Number')
        self.axes[0,0].set_ylabel('Modulus (GPa)')
        self.axes[0,0].set_title('Elastic Moduli')
        self.axes[0,0].set_xticks(x)
        self.axes[0,0].set_xticklabels(ply_numbers)
        self.axes[0,0].legend()
        self.axes[0,0].grid(True, alpha=0.3)
        
        self.axes[0,1].bar(ply_numbers, G12_values, color='lightgreen')
        self.axes[0,1].set_xlabel('Ply Number')
        self.axes[0,1].set_ylabel('Shear Modulus (GPa)')
        self.axes[0,1].set_title('In-plane Shear Modulus G₁₂')
        self.axes[0,1].grid(True, alpha=0.3)
        
        self.axes[1,0].bar(ply_numbers, v12_values, color='gold')
        self.axes[1,0].set_xlabel('Ply Number')
        self.axes[1,0].set_ylabel('Poisson\'s Ratio')
        self.axes[1,0].set_title('Major Poisson\'s Ratio ν₁₂')
        self.axes[1,0].grid(True, alpha=0.3)
        
        # Thickness distribution
        thicknesses = [ply['thickness']*1000 for ply in self.plies]  # Convert to mm
        self.axes[1,1].bar(ply_numbers, thicknesses, color='plum')
        self.axes[1,1].set_xlabel('Ply Number')
        self.axes[1,1].set_ylabel('Thickness (mm)')
        self.axes[1,1].set_title('Ply Thickness Distribution')
        self.axes[1,1].grid(True, alpha=0.3)
    
    def save_configuration(self):
        """Save current laminate configuration to file."""
        filename = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                config = {
                    'plies': self.plies,
                    'loads': {name: var.get() for name, var in self.load_vars.items()}
                }
                
                # Use UTF-8 encoding to handle Unicode characters
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                
                messagebox.showinfo("Success", "Configuration saved successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {e}")
    
    def load_configuration(self):
        """Load laminate configuration from file."""
        filename = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Use UTF-8 encoding to handle Unicode characters
                with open(filename, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Load plies
                self.plies = config.get('plies', [])
                
                # Load loads
                loads = config.get('loads', {})
                for name, value in loads.items():
                    if name in self.load_vars:
                        self.load_vars[name].set(value)
                
                self.update_ply_display()
                messagebox.showinfo("Success", "Configuration loaded successfully.")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {e}")
    
    def export_results(self):
        """Export analysis results to file."""
        if not self.results:
            messagebox.showwarning("Warning", "No results to export. Please run analysis first.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Get results text from the widget
                results_text = self.results_text.get(1.0, tk.END)
                
                # Use UTF-8 encoding to handle Unicode characters
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(results_text)
                
                messagebox.showinfo("Success", "Results exported successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {e}")
    
    def run(self):
        """Start the application."""
        self.root.mainloop()


def main():
    """Main function to start the CLT Analysis application."""
    try:
        app = CLTAnalysisApp()
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        messagebox.showerror("Startup Error", f"Failed to start application: {e}")


if __name__ == "__main__":
    main()
