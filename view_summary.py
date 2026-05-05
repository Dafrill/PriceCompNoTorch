
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import random

RESULTS = "./results"

pdf_path = "all_summaries.pdf"

print("Creating PDF with all summaries...")


with PdfPages(pdf_path) as pdf:
    for folder in sorted(os.listdir(RESULTS)):
        path = os.path.join(RESULTS, folder)
        if not os.path.isdir(path):
            continue
        
        png_files = [f for f in os.listdir(path) if f.endswith('.png') and not f.endswith('.png.png')]
        
        if len(png_files) == 0:
            continue
        
        n_show = min(10, len(png_files))
        selected = random.sample(png_files, n_show)
        
        cols = min(5, n_show)
        rows = (n_show + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        if n_show == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [ax for row in axes for ax in row]
        
        for i, f in enumerate(selected):
            img_path = os.path.join(path, f)
            img = plt.imread(img_path)
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(f.split('.')[0], fontsize=8)
        
        for i in range(n_show, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f"{folder} ({n_show}/{len(png_files)} stocks)")
        plt.tight_layout()
        pdf.savefig(fig, dpi=100)
        plt.close()
        
        print(f"Added: {folder}")

print(f"\nSaved: {pdf_path}")
