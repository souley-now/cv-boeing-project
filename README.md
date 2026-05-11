# CV Boeing Project

Classical lane-boundary detection on highway dashcam video using color filtering, perspective warping, robust polynomial fitting, and temporal smoothing.

## Repository Layout

- `data/videos/`: source input videos
- `notebooks/proj_framework.ipynb`: notebook version of the project pipeline
- `scripts/lane_detection_pipeline.py`: runnable script for batch processing and metrics
- `docs/recommendation.md`: final written report
- `docs/CV-FINAL-PROJECT.pdf`: exported report PDF
- `results/`: generated metrics, snapshots, and processed videos

## Setup

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the pipeline:

```powershell
python scripts/lane_detection_pipeline.py
```

Generate the PDF report:

```powershell
python scripts/generate_recommendation_pdf.py
```

## Notes

- Source videos are kept in version control.
- Generated outputs are written to `results/` and ignored by git.
- Older root-level generated videos were removed in favor of the organized `results/` directory.
