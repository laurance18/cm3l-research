"""
Quad to Double-Double Laminate Converter
========================================

This application converts legacy quad laminate configurations to optimized 
double-double laminate designs based on Stanford's Double-Double methodology.

Author: CM3L Research
Date: 2025
"""

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import openpyxl


@dataclass
class Ply:
    """Represents a single ply in a laminate"""
    material: str
    thickness: float  # mm
    orientation: float  # degrees
    stiffness_properties: Dict[str, float]


@dataclass
class LaminateConfig:
    """Represents a complete laminate configuration"""
    name: str
    plies: List[Ply]
    total_thickness: float
    stacking_sequence: List[int]
    design_requirements: Dict[str, float]


class MaterialDatabase:
    """Database of common composite materials"""
    
    def __init__(self):
        self.materials = {
            "AS4/8552": {
                "E11": 147.0,  # GPa - Longitudinal modulus
                "E22": 10.5,   # GPa - Transverse modulus  
                "G12": 4.5,    # GPa - Shear modulus
                "v12": 0.3,    # Poisson's ratio
                "density": 1.58,  # g/cm³
                "ply_thickness": 0.125,  # mm typical
                "strength_tension_11": 2280,  # MPa
                "strength_compression_11": 1725,  # MPa
                "strength_tension_22": 57,     # MPa
                "strength_compression_22": 228, # MPa
                "strength_shear_12": 76,       # MPa
            },
            "T700/M21": {
                "E11": 130.0,
                "E22": 7.7,
                "G12": 4.8,
                "v12": 0.3,
                "density": 1.58,
                "ply_thickness": 0.125,
                "strength_tension_11": 2500,
                "strength_compression_11": 1200,
                "strength_tension_22": 50,
                "strength_compression_22": 200,
                "strength_shear_12": 80,
            },
            "IM7/8552": {
                "E11": 165.0,
                "E22": 9.0,
                "G12": 5.2,
                "v12": 0.32,
                "density": 1.60,
                "ply_thickness": 0.125,
                "strength_tension_11": 2723,
                "strength_compression_11": 1100,
                "strength_tension_22": 49,
                "strength_compression_22": 200,
                "strength_shear_12": 90,
            }
        }
    
    def get_material(self, name: str) -> Dict[str, float]:
        """Get material properties by name"""
        return self.materials.get(name, self.materials["AS4/8552"])
    
    def list_materials(self) -> List[str]:
        """List available materials"""
        return list(self.materials.keys())


class LaminateAnalyzer:
    """Analyzes laminate mechanical properties using Classical Laminate Theory"""
    
    @staticmethod
    def calculate_q_matrix(E11: float, E22: float, G12: float, v12: float) -> np.ndarray:
        """Calculate reduced stiffness matrix Q"""
        v21 = v12 * E22 / E11
        Q11 = E11 / (1 - v12 * v21)
        Q12 = v12 * E22 / (1 - v12 * v21)
        Q22 = E22 / (1 - v12 * v21)
        Q66 = G12
        
        Q = np.array([
            [Q11, Q12, 0],
            [Q12, Q22, 0],
            [0, 0, Q66]
        ])
        return Q
    
    @staticmethod
    def transform_q_matrix(Q: np.ndarray, theta: float) -> np.ndarray:
        """Transform Q matrix for angle theta (in degrees)"""
        theta_rad = np.radians(theta)
        c = np.cos(theta_rad)
        s = np.sin(theta_rad)
        
        T = np.array([
            [c**2, s**2, 2*s*c],
            [s**2, c**2, -2*s*c],
            [-s*c, s*c, c**2 - s**2]
        ])
        
        T_inv = np.array([
            [c**2, s**2, -2*s*c],
            [s**2, c**2, 2*s*c],
            [s*c, -s*c, c**2 - s**2]
        ])
        
        return T_inv @ Q @ T
    
    def calculate_abd_matrix(self, laminate: LaminateConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate A, B, D matrices using Classical Laminate Theory"""
        n_plies = len(laminate.plies)
        h = laminate.total_thickness
        
        # Initialize matrices
        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        D = np.zeros((3, 3))
        
        # Calculate z-coordinates for each ply
        z = np.linspace(-h/2, h/2, n_plies + 1)
        
        for i, ply in enumerate(laminate.plies):
            # Get material properties
            props = ply.stiffness_properties
            Q = self.calculate_q_matrix(
                props["E11"], props["E22"], 
                props["G12"], props["v12"]
            )
            
            # Transform for ply orientation
            Q_bar = self.transform_q_matrix(Q, ply.orientation)
            
            # Ply boundaries
            z_bottom = z[i]
            z_top = z[i + 1]
            
            # Add contributions to A, B, D matrices
            A += Q_bar * (z_top - z_bottom)
            B += Q_bar * (z_top**2 - z_bottom**2) / 2
            D += Q_bar * (z_top**3 - z_bottom**3) / 3
        
        return A, B, D
    
    def calculate_equivalent_properties(self, laminate: LaminateConfig) -> Dict[str, float]:
        """Calculate equivalent laminate properties"""
        A, B, D = self.calculate_abd_matrix(laminate)
        h = laminate.total_thickness
        
        # Equivalent moduli
        Ex = A[0, 0] / h
        Ey = A[1, 1] / h
        Gxy = A[2, 2] / h
        vxy = A[0, 1] / A[1, 1]
        
        # Flexural properties
        Dx = D[0, 0]
        Dy = D[1, 1]
        Dxy = D[2, 2]
        
        return {
            "Ex": Ex,
            "Ey": Ey,
            "Gxy": Gxy,
            "vxy": vxy,
            "Dx": Dx,
            "Dy": Dy,
            "Dxy": Dxy,
            "thickness": h,
            "areal_weight": sum(ply.stiffness_properties["density"] * ply.thickness 
                               for ply in laminate.plies)
        }


class DoubleDoubleOptimizer:
    """Optimizes double-double laminate configurations"""
    
    def __init__(self, material_db: MaterialDatabase, analyzer: LaminateAnalyzer):
        self.material_db = material_db
        self.analyzer = analyzer
    
    def generate_dd_candidates(self, target_properties: Dict[str, float], 
                             material_name: str) -> List[LaminateConfig]:
        """Generate double-double laminate candidates"""
        material = self.material_db.get_material(material_name)
        candidates = []
        
        # Double-double basic patterns with two angle sets
        # Common patterns: [±θ₁/±θ₂]ₙ where θ₁, θ₂ are optimized
        angle_pairs = [
            (0, 45),    # Standard
            (0, 60),    # High shear
            (15, 75),   # Balanced
            (22.5, 67.5), # Quasi-isotropic variant
            (30, 90),   # High transverse
        ]
        
        for i, (theta1, theta2) in enumerate(angle_pairs):
            # Determine required thickness to match target properties
            target_thickness = target_properties.get("thickness", 2.0)
            ply_thickness = material["ply_thickness"]
            
            # Calculate number of double-double units needed
            n_units = max(1, int(target_thickness / (4 * ply_thickness)))
            
            # Create double-double stacking sequence
            plies = []
            for unit in range(n_units):
                # Double-double unit: [+θ₁/-θ₁/+θ₂/-θ₂]
                orientations = [theta1, -theta1, theta2, -theta2]
                for angle in orientations:
                    ply = Ply(
                        material=material_name,
                        thickness=ply_thickness,
                        orientation=angle,
                        stiffness_properties=material
                    )
                    plies.append(ply)
            
            laminate = LaminateConfig(
                name=f"DD_Pattern_{i+1}_{theta1}_{theta2}",
                plies=plies,
                total_thickness=len(plies) * ply_thickness,
                stacking_sequence=list(range(len(plies))),
                design_requirements=target_properties
            )
            
            candidates.append(laminate)
        
        return candidates
    
    def optimize_dd_laminate(self, quad_laminate: LaminateConfig) -> LaminateConfig:
        """Convert quad laminate to optimized double-double"""
        # Analyze quad laminate properties
        quad_props = self.analyzer.calculate_equivalent_properties(quad_laminate)
        
        # Determine primary material from quad laminate
        primary_material = quad_laminate.plies[0].material if quad_laminate.plies else "AS4/8552"
        
        # Generate double-double candidates
        candidates = self.generate_dd_candidates(quad_props, primary_material)
        
        # Evaluate candidates and select best match
        best_candidate = None
        best_score = float('inf')
        
        for candidate in candidates:
            dd_props = self.analyzer.calculate_equivalent_properties(candidate)
            
            # Calculate matching score (lower is better)
            score = self._calculate_matching_score(quad_props, dd_props)
            
            if score < best_score:
                best_score = score
                best_candidate = candidate
        
        return best_candidate
    
    def _calculate_matching_score(self, target_props: Dict[str, float], 
                                candidate_props: Dict[str, float]) -> float:
        """Calculate how well candidate matches target properties"""
        # Weight factors for different properties
        weights = {
            "Ex": 0.3,
            "Ey": 0.3,
            "Gxy": 0.2,
            "Dx": 0.1,
            "Dy": 0.1
        }
        
        score = 0.0
        for prop, weight in weights.items():
            if prop in target_props and prop in candidate_props:
                if target_props[prop] != 0:
                    error = abs(candidate_props[prop] - target_props[prop]) / abs(target_props[prop])
                    score += weight * error
        
        return score


class QuadToDDConverterGUI:
    """Main GUI application for quad to double-double conversion"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Quad to Double-Double Laminate Converter")
        self.root.geometry("1200x800")
        
        # Initialize components
        self.material_db = MaterialDatabase()
        self.analyzer = LaminateAnalyzer()
        self.optimizer = DoubleDoubleOptimizer(self.material_db, self.analyzer)
        
        # Current laminates
        self.quad_laminate = None
        self.dd_laminate = None
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the GUI interface"""
        # Create main notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Input tab
        self.setup_input_tab(notebook)
        
        # Analysis tab
        self.setup_analysis_tab(notebook)
        
        # Results tab
        self.setup_results_tab(notebook)
        
        # Menu bar
        self.setup_menu()
    
    def setup_menu(self):
        """Setup menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Quad Laminate...", command=self.load_quad_laminate)
        file_menu.add_command(label="Save DD Laminate...", command=self.save_dd_laminate)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def setup_input_tab(self, notebook):
        """Setup input tab for quad laminate definition"""
        input_frame = ttk.Frame(notebook)
        notebook.add(input_frame, text="Input Quad Laminate")
        
        # Left panel - Laminate definition
        left_panel = ttk.LabelFrame(input_frame, text="Quad Laminate Definition")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Laminate name
        ttk.Label(left_panel, text="Laminate Name:").pack(anchor=tk.W, padx=5, pady=2)
        self.name_entry = ttk.Entry(left_panel, width=30)
        self.name_entry.pack(padx=5, pady=2)
        
        # Ply table
        ttk.Label(left_panel, text="Ply Stacking Sequence:").pack(anchor=tk.W, padx=5, pady=(10,2))
        
        # Treeview for ply data
        columns = ("Ply#", "Material", "Thickness", "Orientation")
        self.ply_tree = ttk.Treeview(left_panel, columns=columns, show="headings", height=10)
        
        for col in columns:
            self.ply_tree.heading(col, text=col)
            self.ply_tree.column(col, width=100)
        
        self.ply_tree.pack(padx=5, pady=2, fill=tk.BOTH, expand=True)
        
        # Ply entry controls
        entry_frame = ttk.Frame(left_panel)
        entry_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(entry_frame, text="Material:").grid(row=0, column=0, sticky=tk.W, padx=2)
        self.material_combo = ttk.Combobox(entry_frame, values=self.material_db.list_materials(), width=12)
        self.material_combo.grid(row=0, column=1, padx=2)
        self.material_combo.set("AS4/8552")
        
        ttk.Label(entry_frame, text="Thickness:").grid(row=0, column=2, sticky=tk.W, padx=2)
        self.thickness_entry = ttk.Entry(entry_frame, width=10)
        self.thickness_entry.grid(row=0, column=3, padx=2)
        self.thickness_entry.insert(0, "0.125")
        
        ttk.Label(entry_frame, text="Angle:").grid(row=1, column=0, sticky=tk.W, padx=2)
        self.angle_entry = ttk.Entry(entry_frame, width=10)
        self.angle_entry.grid(row=1, column=1, padx=2)
        self.angle_entry.insert(0, "0")
        
        ttk.Button(entry_frame, text="Add Ply", command=self.add_ply).grid(row=1, column=2, padx=5)
        ttk.Button(entry_frame, text="Remove Ply", command=self.remove_ply).grid(row=1, column=3, padx=5)
        
        # Convert button
        ttk.Button(left_panel, text="Convert to Double-Double", 
                  command=self.convert_laminate).pack(pady=10)
        
        # Right panel - Quick input templates
        right_panel = ttk.LabelFrame(input_frame, text="Quick Templates")
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        ttk.Label(right_panel, text="Common Quad Patterns:").pack(anchor=tk.W, padx=5, pady=5)
        
        templates = [
            ("Quasi-isotropic [0/45/90/-45]s", [0, 45, 90, -45, -45, 90, 45, 0]),
            ("Cross-ply [0/90]s", [0, 90, 90, 0]),
            ("Angle-ply [±45]s", [45, -45, -45, 45]),
            ("Standard Quad [0/±45/90]s", [0, 45, -45, 90, 90, -45, 45, 0])
        ]
        
        for name, sequence in templates:
            ttk.Button(right_panel, text=name, 
                      command=lambda seq=sequence, n=name: self.load_template(n, seq)).pack(
                      fill=tk.X, padx=5, pady=2)
    
    def setup_analysis_tab(self, notebook):
        """Setup analysis tab for comparison"""
        analysis_frame = ttk.Frame(notebook)
        notebook.add(analysis_frame, text="Analysis & Comparison")
        
        # Create figure for plots
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle("Laminate Properties Comparison")
        
        canvas = FigureCanvasTkAgg(self.fig, analysis_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Properties comparison table
        props_frame = ttk.LabelFrame(analysis_frame, text="Properties Comparison")
        props_frame.pack(fill=tk.X, padx=5, pady=5)
        
        columns = ("Property", "Quad Laminate", "Double-Double", "Difference (%)")
        self.props_tree = ttk.Treeview(props_frame, columns=columns, show="headings", height=6)
        
        for col in columns:
            self.props_tree.heading(col, text=col)
            self.props_tree.column(col, width=150)
        
        self.props_tree.pack(padx=5, pady=5, fill=tk.X)
    
    def setup_results_tab(self, notebook):
        """Setup results tab for output"""
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="Results & Export")
        
        # Double-double laminate display
        dd_frame = ttk.LabelFrame(results_frame, text="Optimized Double-Double Laminate")
        dd_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Results text area
        self.results_text = tk.Text(dd_frame, wrap=tk.WORD, height=15)
        scrollbar = ttk.Scrollbar(dd_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Export controls
        export_frame = ttk.LabelFrame(results_frame, text="Export Options")
        export_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(export_frame, text="Export to Excel", 
                  command=self.export_to_excel).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(export_frame, text="Export to JSON", 
                  command=self.export_to_json).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(export_frame, text="Generate Report", 
                  command=self.generate_report).pack(side=tk.LEFT, padx=5, pady=5)
    
    def add_ply(self):
        """Add a ply to the laminate definition"""
        try:
            material = self.material_combo.get()
            thickness = float(self.thickness_entry.get())
            angle = float(self.angle_entry.get())
            
            ply_num = len(self.ply_tree.get_children()) + 1
            self.ply_tree.insert("", tk.END, values=(ply_num, material, thickness, angle))
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values for thickness and angle.")
    
    def remove_ply(self):
        """Remove selected ply from laminate definition"""
        selected = self.ply_tree.selection()
        if selected:
            self.ply_tree.delete(selected[0])
            # Renumber remaining plies
            for i, item in enumerate(self.ply_tree.get_children()):
                values = list(self.ply_tree.item(item)["values"])
                values[0] = i + 1
                self.ply_tree.item(item, values=values)
    
    def load_template(self, name: str, sequence: List[int]):
        """Load a predefined template"""
        # Clear existing plies
        for item in self.ply_tree.get_children():
            self.ply_tree.delete(item)
        
        # Set name
        self.name_entry.delete(0, tk.END)
        self.name_entry.insert(0, name)
        
        # Add plies from template
        for i, angle in enumerate(sequence):
            self.ply_tree.insert("", tk.END, values=(i+1, "AS4/8552", 0.125, angle))
    
    def convert_laminate(self):
        """Convert quad laminate to double-double"""
        try:
            # Build quad laminate from GUI
            self.quad_laminate = self._build_quad_laminate()
            
            if not self.quad_laminate.plies:
                messagebox.showwarning("Warning", "Please define at least one ply.")
                return
            
            # Perform conversion
            self.dd_laminate = self.optimizer.optimize_dd_laminate(self.quad_laminate)
            
            # Update analysis and results
            self.update_analysis()
            self.update_results()
            
            messagebox.showinfo("Success", "Conversion completed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Conversion failed: {str(e)}")
    
    def _build_quad_laminate(self) -> LaminateConfig:
        """Build quad laminate from GUI input"""
        plies = []
        total_thickness = 0.0
        
        for item in self.ply_tree.get_children():
            values = self.ply_tree.item(item)["values"]
            material_name = values[1]
            thickness = float(values[2])
            orientation = float(values[3])
            
            material_props = self.material_db.get_material(material_name)
            
            ply = Ply(
                material=material_name,
                thickness=thickness,
                orientation=orientation,
                stiffness_properties=material_props
            )
            plies.append(ply)
            total_thickness += thickness
        
        return LaminateConfig(
            name=self.name_entry.get() or "Quad Laminate",
            plies=plies,
            total_thickness=total_thickness,
            stacking_sequence=list(range(len(plies))),
            design_requirements={}
        )
    
    def update_analysis(self):
        """Update analysis plots and comparison table"""
        if not (self.quad_laminate and self.dd_laminate):
            return
        
        # Calculate properties
        quad_props = self.analyzer.calculate_equivalent_properties(self.quad_laminate)
        dd_props = self.analyzer.calculate_equivalent_properties(self.dd_laminate)
        
        # Clear plots
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
        
        # Plot 1: Stiffness comparison
        properties = ["Ex", "Ey", "Gxy"]
        quad_values = [quad_props[prop] for prop in properties]
        dd_values = [dd_props[prop] for prop in properties]
        
        x = np.arange(len(properties))
        width = 0.35
        
        self.ax1.bar(x - width/2, quad_values, width, label="Quad", alpha=0.8)
        self.ax1.bar(x + width/2, dd_values, width, label="Double-Double", alpha=0.8)
        self.ax1.set_title("In-Plane Stiffness (GPa)")
        self.ax1.set_xticks(x)
        self.ax1.set_xticklabels(properties)
        self.ax1.legend()
        
        # Plot 2: Flexural stiffness
        flex_props = ["Dx", "Dy", "Dxy"]
        quad_flex = [quad_props[prop] for prop in flex_props]
        dd_flex = [dd_props[prop] for prop in flex_props]
        
        self.ax2.bar(x - width/2, quad_flex, width, label="Quad", alpha=0.8)
        self.ax2.bar(x + width/2, dd_flex, width, label="Double-Double", alpha=0.8)
        self.ax2.set_title("Flexural Stiffness")
        self.ax2.set_xticks(x)
        self.ax2.set_xticklabels(flex_props)
        self.ax2.legend()
        
        # Plot 3: Ply orientation distribution (Quad)
        quad_angles = [ply.orientation for ply in self.quad_laminate.plies]
        self.ax3.hist(quad_angles, bins=20, alpha=0.7, edgecolor='black')
        self.ax3.set_title("Quad Laminate Ply Orientations")
        self.ax3.set_xlabel("Angle (degrees)")
        self.ax3.set_ylabel("Number of Plies")
        
        # Plot 4: Ply orientation distribution (DD)
        dd_angles = [ply.orientation for ply in self.dd_laminate.plies]
        self.ax4.hist(dd_angles, bins=20, alpha=0.7, edgecolor='black', color='orange')
        self.ax4.set_title("Double-Double Laminate Ply Orientations")
        self.ax4.set_xlabel("Angle (degrees)")
        self.ax4.set_ylabel("Number of Plies")
        
        self.fig.tight_layout()
        self.fig.canvas.draw()
        
        # Update comparison table
        for item in self.props_tree.get_children():
            self.props_tree.delete(item)
        
        comparison_props = ["Ex", "Ey", "Gxy", "thickness", "areal_weight"]
        for prop in comparison_props:
            quad_val = quad_props.get(prop, 0)
            dd_val = dd_props.get(prop, 0)
            
            if quad_val != 0:
                diff_pct = ((dd_val - quad_val) / quad_val) * 100
            else:
                diff_pct = 0
            
            self.props_tree.insert("", tk.END, values=(
                prop, f"{quad_val:.2f}", f"{dd_val:.2f}", f"{diff_pct:+.1f}%"
            ))
    
    def update_results(self):
        """Update results display"""
        if not self.dd_laminate:
            return
        
        # Clear results text
        self.results_text.delete(1.0, tk.END)
        
        # Generate detailed results
        results = self._generate_detailed_results()
        self.results_text.insert(tk.END, results)
    
    def _generate_detailed_results(self) -> str:
        """Generate detailed results text"""
        if not (self.quad_laminate and self.dd_laminate):
            return "No conversion results available."
        
        quad_props = self.analyzer.calculate_equivalent_properties(self.quad_laminate)
        dd_props = self.analyzer.calculate_equivalent_properties(self.dd_laminate)
        
        results = f"""
DOUBLE-DOUBLE LAMINATE CONVERSION RESULTS
========================================

Original Quad Laminate: {self.quad_laminate.name}
Converted Double-Double: {self.dd_laminate.name}

STACKING SEQUENCES:
------------------
Quad Laminate:
"""
        
        for i, ply in enumerate(self.quad_laminate.plies):
            results += f"  Ply {i+1}: {ply.material} @ {ply.orientation}° ({ply.thickness:.3f} mm)\n"
        
        results += f"\nDouble-Double Laminate:\n"
        for i, ply in enumerate(self.dd_laminate.plies):
            results += f"  Ply {i+1}: {ply.material} @ {ply.orientation}° ({ply.thickness:.3f} mm)\n"
        
        results += f"""
PROPERTIES COMPARISON:
---------------------
                     Quad        Double-Double    Change
Ex (GPa):           {quad_props['Ex']:8.2f}     {dd_props['Ex']:8.2f}    {((dd_props['Ex']-quad_props['Ex'])/quad_props['Ex']*100):+6.1f}%
Ey (GPa):           {quad_props['Ey']:8.2f}     {dd_props['Ey']:8.2f}    {((dd_props['Ey']-quad_props['Ey'])/quad_props['Ey']*100):+6.1f}%
Gxy (GPa):          {quad_props['Gxy']:8.2f}     {dd_props['Gxy']:8.2f}    {((dd_props['Gxy']-quad_props['Gxy'])/quad_props['Gxy']*100):+6.1f}%
Thickness (mm):     {quad_props['thickness']:8.3f}     {dd_props['thickness']:8.3f}    {((dd_props['thickness']-quad_props['thickness'])/quad_props['thickness']*100):+6.1f}%
Areal Weight:       {quad_props['areal_weight']:8.3f}     {dd_props['areal_weight']:8.3f}    {((dd_props['areal_weight']-quad_props['areal_weight'])/quad_props['areal_weight']*100):+6.1f}%

ADVANTAGES OF DOUBLE-DOUBLE DESIGN:
-----------------------------------
• Simplified layup with repeating double-double units
• Improved damage tolerance with thin plies
• Better optimization potential for weight savings
• Reduced minimum gage requirements
• Enhanced manufacturing efficiency
• Homogeneous laminate behavior similar to aluminum

MANUFACTURING NOTES:
-------------------
• Use thin plies ({self.dd_laminate.plies[0].thickness:.3f} mm) for optimal performance
• Maintain ±angle pairs within each double-double unit
• Consider autoclave or out-of-autoclave curing as appropriate
• Implement proper quality control for ply orientation accuracy
"""
        
        return results
    
    def load_quad_laminate(self):
        """Load quad laminate from file"""
        filename = filedialog.askopenfilename(
            title="Load Quad Laminate",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                if filename.endswith(('.xlsx', '.xls')):
                    self._load_from_excel(filename)
                elif filename.endswith('.json'):
                    self._load_from_json(filename)
                else:
                    messagebox.showerror("Error", "Unsupported file format.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def _load_from_excel(self, filename: str):
        """Load laminate from Excel file"""
        # This is a placeholder - implement based on your Excel file format
        messagebox.showinfo("Info", "Excel loading not yet implemented. Please use manual input or JSON files.")
    
    def _load_from_json(self, filename: str):
        """Load laminate from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Clear existing plies
        for item in self.ply_tree.get_children():
            self.ply_tree.delete(item)
        
        # Load data
        self.name_entry.delete(0, tk.END)
        self.name_entry.insert(0, data.get('name', 'Loaded Laminate'))
        
        for i, ply_data in enumerate(data.get('plies', [])):
            self.ply_tree.insert("", tk.END, values=(
                i+1, 
                ply_data.get('material', 'AS4/8552'),
                ply_data.get('thickness', 0.125),
                ply_data.get('orientation', 0)
            ))
    
    def save_dd_laminate(self):
        """Save double-double laminate to file"""
        if not self.dd_laminate:
            messagebox.showwarning("Warning", "No double-double laminate to save. Please convert first.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Double-Double Laminate",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                if filename.endswith('.json'):
                    self._save_to_json(filename)
                elif filename.endswith('.xlsx'):
                    self._save_to_excel(filename)
                else:
                    messagebox.showerror("Error", "Unsupported file format.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {str(e)}")
    
    def _save_to_json(self, filename: str):
        """Save laminate to JSON file"""
        data = {
            'name': self.dd_laminate.name,
            'type': 'double-double',
            'total_thickness': self.dd_laminate.total_thickness,
            'plies': [
                {
                    'material': ply.material,
                    'thickness': ply.thickness,
                    'orientation': ply.orientation
                }
                for ply in self.dd_laminate.plies
            ],
            'properties': self.analyzer.calculate_equivalent_properties(self.dd_laminate)
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        messagebox.showinfo("Success", f"Double-double laminate saved to {filename}")
    
    def _save_to_excel(self, filename: str):
        """Save laminate to Excel file"""
        # Create workbook with multiple sheets
        wb = openpyxl.Workbook()
        
        # Ply data sheet
        ws_plies = wb.active
        ws_plies.title = "Ply_Data"
        ws_plies.append(["Ply #", "Material", "Thickness (mm)", "Orientation (deg)"])
        
        for i, ply in enumerate(self.dd_laminate.plies):
            ws_plies.append([i+1, ply.material, ply.thickness, ply.orientation])
        
        # Properties sheet
        ws_props = wb.create_sheet("Properties")
        dd_props = self.analyzer.calculate_equivalent_properties(self.dd_laminate)
        
        ws_props.append(["Property", "Value", "Unit"])
        for prop, value in dd_props.items():
            unit = "GPa" if prop in ["Ex", "Ey", "Gxy"] else "mm" if prop == "thickness" else ""
            ws_props.append([prop, value, unit])
        
        wb.save(filename)
        messagebox.showinfo("Success", f"Double-double laminate saved to {filename}")
    
    def export_to_excel(self):
        """Export current results to Excel"""
        if not self.dd_laminate:
            messagebox.showwarning("Warning", "No results to export. Please convert first.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export Results to Excel",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if filename:
            self._save_to_excel(filename)
    
    def export_to_json(self):
        """Export current results to JSON"""
        if not self.dd_laminate:
            messagebox.showwarning("Warning", "No results to export. Please convert first.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export Results to JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            self._save_to_json(filename)
    
    def generate_report(self):
        """Generate detailed conversion report"""
        if not (self.quad_laminate and self.dd_laminate):
            messagebox.showwarning("Warning", "No conversion data available for report.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Conversion Report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(self._generate_detailed_results())
                messagebox.showinfo("Success", f"Conversion report saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save report: {str(e)}")
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
Quad to Double-Double Laminate Converter
Version 1.0

This application converts legacy quad laminate configurations to optimized 
double-double laminate designs based on Stanford's Double-Double methodology 
developed by Stephen Tsai.

Features:
• Classical Laminate Theory analysis
• Multiple double-double optimization patterns
• Properties comparison and visualization
• Export to Excel, JSON, and text formats

Developed for CM3L Research
© 2025
        """
        messagebox.showinfo("About", about_text)
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()


def main():
    """Main application entry point"""
    app = QuadToDDConverterGUI()
    app.run()


if __name__ == "__main__":
    main()
