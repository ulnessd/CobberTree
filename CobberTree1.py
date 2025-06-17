import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import py3Dmol

# Scikit-learn imports for machine learning
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split

from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QListWidget, QAbstractItemView, QListWidgetItem,
    QTabWidget, QPushButton, QSpinBox, QTextEdit, QFormLayout
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon, QFont, QPixmap

from dataclasses import dataclass
from typing import List


# --- Data Structures and Helper Functions ---
@dataclass
class Hydrocarbon:
    name: str;
    formula: str;
    num_carbons: int;
    num_branches: int;
    boiling_point: float;
    sdf_path: str


def calculate_weighted_variance(group1: List[Hydrocarbon], group2: List[Hydrocarbon]) -> float:
    n1, n2 = len(group1), len(group2)
    n_total = n1 + n2
    if n_total == 0: return 0.0
    var1 = np.var([hc.boiling_point for hc in group1]) if n1 > 1 else 0.0
    var2 = np.var([hc.boiling_point for hc in group2]) if n2 > 1 else 0.0
    return (n1 / n_total) * var1 + (n2 / n_total) * var2


# --- Custom Widgets for the GUI ---
class ViewerWindow(QWidget):
    def __init__(self):
        super().__init__();
        self.setWindowTitle("3D Molecule Viewer");
        self.setGeometry(200, 200, 500, 500)
        layout = QVBoxLayout();
        self.web_view = QWebEngineView();
        layout.addWidget(self.web_view);
        self.setLayout(layout)

    def update_view(self, sdf_path):
        try:
            with open(sdf_path, 'r') as f:
                sdf_data = f.read()
            view = py3Dmol.view(width=480, height=480);
            view.addModel(sdf_data, 'sdf');
            view.setStyle({'stick': {}});
            view.zoomTo()
            self.web_view.setHtml(view.write_html())
        except FileNotFoundError:
            self.web_view.setHtml(f"<html><body><h2>Error: File not found</h2><p>{sdf_path}</p></body></html>")


class MoleculeListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent);
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setAcceptDrops(True);
        self.setIconSize(QSize(80, 50))

    def dropEvent(self, event):
        source_widget = event.source()
        if isinstance(source_widget, QListWidget) and source_widget is not self:
            item = source_widget.takeItem(source_widget.row(source_widget.currentItem()));
            self.addItem(item)


# --- NEW: Widget for the Automated Tree Tab ---
class AutomatedTreeWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.tree_image_path = "decision_tree.png"

        main_layout = QHBoxLayout(self)
        controls_layout = QVBoxLayout()
        display_layout = QVBoxLayout()

        # Controls
        form_layout = QFormLayout()
        self.depth_spinner = QSpinBox()
        self.depth_spinner.setRange(2, 10)
        self.depth_spinner.setValue(3)
        self.generate_button = QPushButton("Generate Decision Tree")
        self.predict_button = QPushButton("Predict Unknowns")

        form_layout.addRow("Max Tree Depth:", self.depth_spinner)
        controls_layout.addLayout(form_layout)
        controls_layout.addWidget(self.generate_button)
        controls_layout.addWidget(self.predict_button)
        controls_layout.addStretch()

        # Display
        self.tree_image_label = QLabel("Click 'Generate' to create and display the decision tree.")
        self.tree_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)

        display_layout.addWidget(self.tree_image_label, 3)  # More stretch
        display_layout.addWidget(QLabel("<b>Analysis & Predictions:</b>"))
        display_layout.addWidget(self.results_text, 1)

        main_layout.addLayout(controls_layout, 1)
        main_layout.addLayout(display_layout, 4)

        # Connect signals
        self.generate_button.clicked.connect(self.generate_tree)
        self.predict_button.clicked.connect(self.predict_unknowns)
        self.predict_button.setEnabled(False)  # Disable until model is trained

    def generate_tree(self):
        try:
            df = pd.read_csv("hydrocarbon_data.csv")
        except FileNotFoundError:
            self.results_text.setText(
                "Error: hydrocarbon_data.csv not found.\nPlease make sure the data file is in the same directory.")
            return

        features = ['Num_Carbons', 'Num_Branches', 'Molar_Mass', 'Density']
        target = 'Boiling_Point'

        X = df[features]
        y = df[target]

        max_depth = self.depth_spinner.value()
        self.model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        self.model.fit(X, y)

        # Plot tree
        plt.figure(figsize=(20, 10))
        plot_tree(self.model, feature_names=features, filled=True, rounded=True, fontsize=10)
        plt.title(f"Decision Tree for Alkane Boiling Points (Max Depth = {max_depth})", fontsize=16)
        plt.savefig(self.tree_image_path, dpi=150)
        plt.close()

        # Display image
        pixmap = QPixmap(self.tree_image_path)
        self.tree_image_label.setPixmap(pixmap.scaled(self.tree_image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                      Qt.TransformationMode.SmoothTransformation))

        # Display results
        importances = self.model.feature_importances_
        feature_importance_text = "\n".join([f"  - {name}: {imp:.2%}" for name, imp in zip(features, importances)])

        results = (
            f"--- Model Training Results ---\n"
            f"Dataset size: {len(df)} molecules\n"
            f"Tree Depth: {max_depth}\n\n"
            f"Feature Importances:\n{feature_importance_text}"
        )
        self.results_text.setText(results)
        self.predict_button.setEnabled(True)

    def predict_unknowns(self):
        unknowns = {
            "2,2,4-Trimethylpentane": pd.DataFrame([[8, 3, 114.23, 0.692]],
                                                   columns=['Num_Carbons', 'Num_Branches', 'Molar_Mass', 'Density']),
            "3-Ethyl-2,2-dimethylpentane": pd.DataFrame([[9, 3, 128.26, 0.729]],
                                                        columns=['Num_Carbons', 'Num_Branches', 'Molar_Mass',
                                                                 'Density']),
        }

        prediction_text = "\n\n--- Predictions for Unknowns ---\n"
        for name, data in unknowns.items():
            prediction = self.model.predict(data)[0]
            prediction_text += f"Predicted Boiling Point for {name}:\n  -> {prediction:.2f} °C\n"

        self.results_text.append(prediction_text)


# --- Main Application Window ---
class MainWindow(QMainWindow):
    def __init__(self, manual_dataset):
        super().__init__()
        self.setWindowTitle("CobberTree -- Alkane Decision Tree Explorer")
        self.setGeometry(100, 100, 1200, 800)

        # Create Tab Widget
        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        # Create Tab 1: Manual Sorter
        manual_sorter_widget = self.create_manual_sorter_tab(manual_dataset)
        tabs.addTab(manual_sorter_widget, "Manual Sorter")

        # Create Tab 2: Automated Tree
        automated_tree_widget = AutomatedTreeWidget()
        tabs.addTab(automated_tree_widget, "Automated Tree")

    def create_manual_sorter_tab(self, dataset):
        # This method encapsulates all the logic for the first tab
        container = QWidget()
        self.manual_dataset = dataset
        self.hc_map = {hc.name: hc for hc in self.manual_dataset}
        self.best_score = float('inf')
        self.viewer_window = ViewerWindow()

        main_layout = QVBoxLayout(container)
        bins_layout = QHBoxLayout()
        scoreboard_layout = QHBoxLayout()
        self.deck_list_widget = MoleculeListWidget()
        self.bin1_list_widget = MoleculeListWidget()
        self.bin2_list_widget = MoleculeListWidget()
        self._connect_manual_signals()

        bin1_score_frame = self.create_scoreboard_frame("Bin 1")
        bin2_score_frame = self.create_scoreboard_frame("Bin 2")
        total_score_frame = self.create_total_score_section("Total")
        main_layout.addWidget(QLabel("<h2>Unsorted Deck</h2>"))
        main_layout.addWidget(self.deck_list_widget, 2)  # Adjusted stretch
        bins_layout.addWidget(self.create_bin_section("Bin 1", self.bin1_list_widget))
        bins_layout.addWidget(self.create_bin_section("Bin 2", self.bin2_list_widget))
        main_layout.addLayout(bins_layout, 3)  # Adjusted stretch
        scoreboard_layout.addWidget(bin1_score_frame)
        scoreboard_layout.addWidget(bin2_score_frame)
        scoreboard_layout.addWidget(total_score_frame)
        main_layout.addLayout(scoreboard_layout)

        self.populate_deck()
        return container

    # Renamed to avoid conflicts
    def _connect_manual_signals(self):
        self.bin1_list_widget.model().rowsInserted.connect(self.update_manual_calculations)
        self.bin2_list_widget.model().rowsInserted.connect(self.update_manual_calculations)
        self.deck_list_widget.model().rowsInserted.connect(self.update_manual_calculations)
        self.bin1_list_widget.model().rowsRemoved.connect(self.update_manual_calculations)
        self.bin2_list_widget.model().rowsRemoved.connect(self.update_manual_calculations)
        self.deck_list_widget.model().rowsRemoved.connect(self.update_manual_calculations)
        self.deck_list_widget.itemClicked.connect(self.show_3d_viewer)
        self.bin1_list_widget.itemClicked.connect(self.show_3d_viewer)
        self.bin2_list_widget.itemClicked.connect(self.show_3d_viewer)

    def show_3d_viewer(self, item):
        molecule_name = item.text().split('|')[0].strip()
        hc = self.hc_map.get(molecule_name)
        if hc:
            self.viewer_window.setWindowTitle(f"3D View: {hc.name}");
            self.viewer_window.update_view(hc.sdf_path)
            self.viewer_window.show();
            self.viewer_window.raise_()

    def update_manual_calculations(self):
        group1_hcs = [self.hc_map[self.bin1_list_widget.item(i).text().split('|')[0].strip()] for i in
                      range(self.bin1_list_widget.count())]
        group2_hcs = [self.hc_map[self.bin2_list_widget.item(i).text().split('|')[0].strip()] for i in
                      range(self.bin2_list_widget.count())]
        n1 = len(group1_hcs);
        self.bin1_count_label.setText(f"Count (n): {n1}")
        if n1 > 0:
            mean1 = np.mean([hc.boiling_point for hc in group1_hcs]);
            var1 = np.var([hc.boiling_point for hc in group1_hcs]) if n1 > 1 else 0.0
            self.bin1_mean_label.setText(f"Mean BP: {mean1:.2f}");
            self.bin1_var_label.setText(f"Variance: {var1:.2f}")
        else:
            self.bin1_mean_label.setText("Mean BP: N/A"); self.bin1_var_label.setText("Variance: N/A")
        n2 = len(group2_hcs);
        self.bin2_count_label.setText(f"Count (n): {n2}")
        if n2 > 0:
            mean2 = np.mean([hc.boiling_point for hc in group2_hcs]);
            var2 = np.var([hc.boiling_point for hc in group2_hcs]) if n2 > 1 else 0.0
            self.bin2_mean_label.setText(f"Mean BP: {mean2:.2f}");
            self.bin2_var_label.setText(f"Variance: {var2:.2f}")
        else:
            self.bin2_mean_label.setText("Mean BP: N/A"); self.bin2_var_label.setText("Variance: N/A")
        if n1 > 0 or n2 > 0:
            total_cost = calculate_weighted_variance(group1_hcs, group2_hcs)
            self.total_cost_label.setText(f"<h3>TOTAL COST: {total_cost:.2f}</h3>")
            if self.deck_list_widget.count() == 0:
                if total_cost < self.best_score: self.best_score = total_cost; self.best_score_label.setText(
                    f"Best Score: {self.best_score:.2f}")
        else:
            self.total_cost_label.setText("<h3>TOTAL COST: N/A</h3>")

    def populate_deck(self):
        for hc in self.manual_dataset:
            item_text = (
                f"{hc.name:<20s} | BP: {hc.boiling_point:>5.1f}°C | #C: {hc.num_carbons} | #Br: {hc.num_branches}")
            list_item = QListWidgetItem(item_text)
            font = QFont("Courier New");
            font.setPointSize(10);
            list_item.setFont(font)
            self.deck_list_widget.addItem(list_item)

    # --- Helper methods for Manual Sorter GUI ---
    def create_bin_section(self, title, list_widget):
        layout = QVBoxLayout();
        label = QLabel(f"<h3>{title}</h3>");
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label);
        layout.addWidget(list_widget);
        frame = QFrame();
        frame.setFrameShape(QFrame.Shape.StyledPanel);
        frame.setLayout(layout)
        return frame

    def create_scoreboard_frame(self, title):
        frame = QFrame();
        frame.setFrameShape(QFrame.Shape.StyledPanel);
        layout = QVBoxLayout();
        layout.addWidget(QLabel(f"<b>{title}</b>"))
        if title == "Bin 1":
            self.bin1_count_label = QLabel("Count (n): 0");
            self.bin1_mean_label = QLabel("Mean BP: N/A");
            self.bin1_var_label = QLabel("Variance: N/A")
            layout.addWidget(self.bin1_count_label);
            layout.addWidget(self.bin1_mean_label);
            layout.addWidget(self.bin1_var_label)
        else:
            self.bin2_count_label = QLabel("Count (n): 0");
            self.bin2_mean_label = QLabel("Mean BP: N/A");
            self.bin2_var_label = QLabel("Variance: N/A")
            layout.addWidget(self.bin2_count_label);
            layout.addWidget(self.bin2_mean_label);
            layout.addWidget(self.bin2_var_label)
        layout.addStretch();
        frame.setLayout(layout)
        return frame

    def create_total_score_section(self, title):
        layout = QVBoxLayout();
        label = QLabel(f"<b>{title}</b>");
        layout.addWidget(label)
        self.total_cost_label = QLabel("<h3>TOTAL COST: N/A</h3>");
        self.best_score_label = QLabel("Best Score: N/A")
        layout.addWidget(self.total_cost_label);
        layout.addWidget(self.best_score_label);
        layout.addStretch()
        frame = QFrame();
        frame.setFrameShape(QFrame.Shape.StyledPanel);
        frame.setLayout(layout)
        return frame


# --- Main execution block ---
if __name__ == "__main__":
    # The smaller dataset for the manual sorting task
    sdf = lambda name: f"AlkaneStructures/{name.lower().replace('-', '')}.sdf"
    manual_dataset = [
        Hydrocarbon("Butane", "C4H10", 4, 0, -0.5, sdf("Butane")),
        Hydrocarbon("Pentane", "C5H12", 5, 0, 36.1, sdf("Pentane")),
        Hydrocarbon("Hexane", "C6H14", 6, 0, 68.7, sdf("Hexane")),
        Hydrocarbon("Heptane", "C7H16", 7, 0, 98.4, sdf("Heptane")),
        Hydrocarbon("Octane", "C8H18", 8, 0, 125.7, sdf("Octane")),
        Hydrocarbon("Isobutane", "C4H10", 4, 1, -11.7, sdf("Isobutane")),
        Hydrocarbon("Isopentane", "C5H12", 5, 1, 27.8, sdf("Isopentane")),
        Hydrocarbon("3-Methylpentane", "C6H14", 6, 1, 63.3, sdf("3-Methylpentane")),
        Hydrocarbon("2-Methylheptane", "C8H18", 8, 1, 117.6, sdf("2-Methylheptane")),
        Hydrocarbon("Neopentane", "C5H12", 5, 2, 9.5, sdf("Neopentane")),
        Hydrocarbon("2,2-Dimethylhexane", "C8H18", 8, 2, 106.8, sdf("2,2-Dimethylhexane")),
    ]

    app = QApplication(sys.argv)
    window = MainWindow(sorted(manual_dataset, key=lambda hc: (hc.num_carbons, hc.num_branches)))
    window.show()
    sys.exit(app.exec())
