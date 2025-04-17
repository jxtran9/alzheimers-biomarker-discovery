# Alzheimers-Biomarker-Discovery

Alzheimer’s Disease is a neurodegenerative disorder that affects millions worldwide. Current diagnosis methods rely on cognitive tests and brain imaging, which are expensive and often detect the disease too late.

This project aims to:

- Identify genetic biomarkers that distinguish Alzheimer’s patients from healthy individuals
- Train a machine learning model that can classify AD vs. Healthy based on gene expression
- Provide a reproducible pipeline that can be extended for biomarker discovery in other diseases

By leveraging statistical analysis and Random Forest models, this tool helps surface potential early indicators of Alzheimer’s Disease.

---

## Features

- **Preprocessing**: Cleans and normalizes gene expression data
- **Biomarker Identification**: Selects significant genes using statistical tests and feature importance
- **Machine Learning**: Trains a Random Forest classifier to distinguish between Healthy and AD samples
- **Evaluation**: Generates classification reports, confusion matrices, and feature importance plots

---

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/jxtran9/alzheimers-biomarker-discovery.git
   ```
2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate   # On Mac/Linux
   .\venv\Scripts\activate    # On Windows
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

---

## Dataset

This project uses gene expression data from the [GSE48350 - NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE48350) dataset.

To run the full pipeline:

1. Download the raw dataset from the link above.
2. Rename the file (if necessary) to: `GSE48350_series_matrix.txt`
3. Place the file inside the `data/` directory of this project.

> This file is not included in the repository due to GitHub's 100MB file size limit.

---

## Usage

After placing the dataset in the `data/` folder, you can run the entire pipeline by executing:

```
python scripts/main.py
```

This will:

- Preprocess the raw gene expression data
- Identify potential biomarkers using statistical tests
- Train a Random Forest classifier
- Save generated plots (e.g., confusion matrix, feature importance) and summary CSVs (e.g., top biomarkers) to the project directory

You can also run the scripts individually from the `scripts/` folder if you want to test specific steps in isolation.

---

## Dependencies

A selection of key packages used in this project:

- Python 3.13+
- pandas
- numpy
- matplotlib
- scikit-learn
- scipy
- imbalanced-learn
- joblib
- pyarrow

---

## References

- **Dataset**: [GSE48350 - NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE48350)
- **Machine Learning Model**: Random Forest Classifier
- **Statistical Analysis**: T-tests for biomarker selection

---

## Project Structure

- `scripts/` – Preprocessing, training, and analysis scripts
- `data/` – Sample data files and required dataset (not included due to size)
- Project directory – Output plots and summary CSVs (e.g., confusion matrix, top biomarkers)

---

## Credits

This project was completed as part of the CSS 483: Bioinformatics Algorithms course at the University of Washington Bothell.

Developed collaboratively by a two-person team.
