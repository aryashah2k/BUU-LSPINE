import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import seaborn as sns
from pathlib import Path

class BUUSpineDataset:
    def __init__(self, base_path):
        self.base_path = base_path
        self.ap_path = os.path.join(base_path, 'AP')
        self.la_path = os.path.join(base_path, 'LA')
        
    def load_case(self, case_id):
        ap_image = Image.open(os.path.join(self.ap_path, f'{case_id}Y0.jpg'))
        ap_coords = pd.read_csv(os.path.join(self.ap_path, f'{case_id}Y0.csv'), 
                              header=None, 
                              names=['x1', 'y1', 'x2', 'y2', 'class'])
        
        la_image = Image.open(os.path.join(self.la_path, f'{case_id}Y1.jpg'))
        la_coords = pd.read_csv(os.path.join(self.la_path, f'{case_id}Y1.csv'),
                              header=None,
                              names=['x1', 'y1', 'x2', 'y2', 'class'])
        
        return ap_image, la_image, ap_coords, la_coords

    def create_plot_structure(self, output_dir):
        plot_types = ['original_views', 'intensity_dist', 'class_dist', 
                     'coordinate_dist', 'vertebrae_analysis']
        paths = {}
        for plot_type in plot_types:
            path = os.path.join(output_dir, plot_type)
            os.makedirs(path, exist_ok=True)
            paths[plot_type] = path
        return paths

    def plot_original_views(self, case_id, ap_img, la_img, ap_coords, la_coords, save_dir):
        # AP View
        fig_ap, ax_ap = plt.subplots(figsize=(10, 10))
        ax_ap.imshow(ap_img, cmap='gray')
        for _, row in ap_coords.iterrows():
            ax_ap.plot([row.x1, row.x2], [row.y1, row.y2], 'r-', linewidth=1)
            ax_ap.scatter([row.x1, row.x2], [row.y1, row.y2], c='yellow', s=20)
        ax_ap.set_title('AP View')
        fig_ap.savefig(os.path.join(save_dir, f'{case_id}_ap.png'))
        plt.close(fig_ap)

        # LA View
        fig_la, ax_la = plt.subplots(figsize=(10, 10))
        ax_la.imshow(la_img, cmap='gray')
        for _, row in la_coords.iterrows():
            ax_la.plot([row.x1, row.x2], [row.y1, row.y2], 'r-', linewidth=1)
            ax_la.scatter([row.x1, row.x2], [row.y1, row.y2], c='yellow', s=20)
        ax_la.set_title('LA View')
        fig_la.savefig(os.path.join(save_dir, f'{case_id}_la.png'))
        plt.close(fig_la)

    def plot_intensity_distribution(self, case_id, ap_img, la_img, save_dir):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(np.array(ap_img).ravel(), bins=256, color='blue', alpha=0.5, label='AP')
        ax.hist(np.array(la_img).ravel(), bins=256, color='red', alpha=0.5, label='LA')
        ax.set_title('Intensity Distribution')
        ax.legend()
        fig.savefig(os.path.join(save_dir, f'{case_id}_intensity.png'))
        plt.close(fig)

    def plot_class_distribution(self, case_id, ap_coords, la_coords, save_dir):
        fig, ax = plt.subplots(figsize=(10, 6))
        classes = pd.concat([ap_coords['class'], la_coords['class']])
        sns.countplot(data=classes.to_frame(), x='class', ax=ax)
        ax.set_title('Class Distribution')
        fig.savefig(os.path.join(save_dir, f'{case_id}_classes.png'))
        plt.close(fig)

    def plot_coordinate_distribution(self, case_id, ap_coords, la_coords, save_dir):
        # AP coordinates
        fig_ap, ax_ap = plt.subplots(figsize=(10, 6))
        ax_ap.scatter(ap_coords['x1'], ap_coords['y1'], c='blue', label='Left')
        ax_ap.scatter(ap_coords['x2'], ap_coords['y2'], c='red', label='Right')
        ax_ap.set_title('AP Coordinates Distribution')
        ax_ap.legend()
        fig_ap.savefig(os.path.join(save_dir, f'{case_id}_ap_coords.png'))
        plt.close(fig_ap)

        # LA coordinates
        fig_la, ax_la = plt.subplots(figsize=(10, 6))
        ax_la.scatter(la_coords['x1'], la_coords['y1'], c='blue', label='Left')
        ax_la.scatter(la_coords['x2'], la_coords['y2'], c='red', label='Right')
        ax_la.set_title('LA Coordinates Distribution')
        ax_la.legend()
        fig_la.savefig(os.path.join(save_dir, f'{case_id}_la_coords.png'))
        plt.close(fig_la)

    def process_all_cases(self, output_dir):
        plot_dirs = self.create_plot_structure(output_dir)
        
        # Get all case IDs
        case_ids = [f.split('Y0.jpg')[0] for f in os.listdir(self.ap_path) 
                   if f.endswith('Y0.jpg')]
        
        for case_id in case_ids:
            print(f"Processing case: {case_id}")
            try:
                ap_img, la_img, ap_coords, la_coords = self.load_case(case_id)
                
                # Generate all plots
                self.plot_original_views(case_id, ap_img, la_img, ap_coords, la_coords, 
                                      plot_dirs['original_views'])
                self.plot_intensity_distribution(case_id, ap_img, la_img, 
                                              plot_dirs['intensity_dist'])
                self.plot_class_distribution(case_id, ap_coords, la_coords, 
                                          plot_dirs['class_dist'])
                self.plot_coordinate_distribution(case_id, ap_coords, la_coords, 
                                               plot_dirs['coordinate_dist'])
                
            except Exception as e:
                print(f"Error processing case {case_id}: {str(e)}")

# Usage example
dataset = BUUSpineDataset('E:\BUU-LSPINE\BUU-LSPINE_400\BUU-LSPINE_400')
dataset.process_all_cases('E:\BUU-LSPINE\BUU-LSPINE_400\output')
