# Figure Extraction Guide

This guide explains how to extract figures from the Chinchilla paper PDF and save them as PNG files.

## Required Figures

The README.md references the following figures that need to be extracted from the paper:

1. **Figure 1: Scaling Predictions Comparison** → Save as `images/figure1_scaling_predictions.png`
   - Location: Page 2 of the paper
   - Shows comparison of Chinchilla vs. Kaplan scaling predictions

2. **Figure 3: IsoFLOP Curves** → Save as `images/figure3_isoflop_curves.png`
   - Location: Page 6 of the paper
   - Shows IsoFLOP profiles with loss contours

3. **Figure 4: Parametric Loss Function Fit** → Save as `images/figure4_parametric_fit.png`
   - Location: Page 7 of the paper
   - Shows fitted parametric loss function with contours

4. **Figure 6: MMLU Results** → Save as `images/figure6_mmlu_results.png`
   - Location: Page 12 of the paper
   - Shows MMLU benchmark comparison results

## Method 1: Using Preview (macOS)

1. Open `Training Compute-Optimal Large Language Models (Chinchilla).pdf` in Preview
2. Navigate to the page containing the figure
3. Use the selection tool to select just the figure
4. Copy the selection (⌘+C)
5. Create a new file (⌘+N)
6. Paste the figure (⌘+V)
7. Export as PNG:
   - File → Export...
   - Format: PNG
   - Save to the `images/` folder with the appropriate filename

## Method 2: Using Adobe Acrobat

1. Open the PDF in Adobe Acrobat
2. Use the "Snapshot Tool" or "Select" tool
3. Select the figure you want to extract
4. Right-click and choose "Copy Image"
5. Open an image editor (Preview, Photoshop, GIMP)
6. Paste and save as PNG to the `images/` folder

## Method 3: Using Python (pdfplumber or PyMuPDF)

If you prefer automation:

```python
import fitz  # PyMuPDF
from PIL import Image

# Open PDF
pdf_path = "Training Compute-Optimal Large Language Models (Chinchilla).pdf"
doc = fitz.open(pdf_path)

# Extract specific pages
figures_to_extract = {
    1: "images/figure1_scaling_predictions.png",  # Page 2 (0-indexed: page 1)
    5: "images/figure3_isoflop_curves.png",      # Page 6 (0-indexed: page 5)
    6: "images/figure4_parametric_fit.png",      # Page 7 (0-indexed: page 6)
    11: "images/figure6_mmlu_results.png",       # Page 12 (0-indexed: page 11)
}

for page_num, output_path in figures_to_extract.items():
    page = doc[page_num]
    pix = page.get_pixmap(dpi=300)  # High resolution
    pix.save(output_path)

print("Figures extracted successfully!")
```

## Recommended Settings

- **Resolution**: 300 DPI (for high-quality images)
- **Format**: PNG (for lossless compression)
- **Naming**: Use descriptive names matching the README references
- **Location**: Save all images to the `images/` folder

## After Extraction

Once you've extracted all figures:

1. Verify all 4 PNG files are in the `images/` folder
2. Check that filenames match exactly:
   - `figure1_scaling_predictions.png`
   - `figure3_isoflop_curves.png`
   - `figure4_parametric_fit.png`
   - `figure6_mmlu_results.png`
3. Commit the images to git:
   ```bash
   git add images/*.png
   git commit -m "Add paper figures for presentation"
   git push
   ```

## Troubleshooting

**Issue**: Figures appear blurry
- Solution: Increase DPI/resolution to 300 or higher

**Issue**: Text in figures is hard to read
- Solution: Crop tightly around the figure to exclude extra whitespace

**Issue**: Figure colors look different
- Solution: Ensure you're using PNG format (not JPEG) to preserve quality

## Note on Copyright

These figures are from the original Chinchilla paper by Hoffmann et al. (2022). They should only be used for educational purposes in your presentation with proper citation.
