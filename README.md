# PII Data Detection

![Project Logo](PIID.png) <!-- Image added here -->

## Introduction
This project is designed to help detect and remove Personally Identifiable Information (PII) from educational data. It leverages advanced Natural Language Processing (NLP) techniques and pre-trained models to identify various types of PII, such as names, email addresses, phone numbers, and street addresses.

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/zulqarnainalipk/PII-Data-Detection.git  
   cd PII_Data_Detection
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Option 1: Run as a Python script

To run the PII detection pipeline using the main Python script:

```bash
python src/main.py
```

This will process the test data (as configured in `src/config.py`), perform PII detection, and generate a `submission.csv` file with the identified PII entities.

### Option 2: Use the Jupyter Notebook

You can also run the pipeline interactively using the Jupyter Notebook provided:

```text
PII_Data_Detection.ipynb
```

This is useful for exploration, debugging, or learning the step-by-step pipeline.

## Project Structure

```
PII_Data_Detection/
├── src/
│   ├── config.py
│   ├── main.py
│   └── utils.py
├── PII_Data_Detection.ipynb
├── README.md
├── requirements.txt
├── PIID  # Project image (can be .png or .jpg)
└── .gitignore
```

* `src/config.py`: Configuration settings for the model, file paths, and hyperparameters.
* `src/main.py`: Main pipeline script for PII detection and submission generation.
* `src/utils.py`: Helper functions for data loading, span detection, and output formatting.
* `PII_Data_Detection.ipynb`: Jupyter notebook version of the pipeline for interactive usage.
* `README.md`: Project overview, setup guide, and usage instructions.
* `requirements.txt`: Required Python packages.
* `.gitignore`: Files and directories excluded from version control.

## Credits and Acknowledgments

This project was originally created for a competition and later adapted for broader use cases in educational data privacy.  

Inspired by the work of Aleksandr Lavrikov on Kaggle.

## Contact

Created by [Zulqarnain Ali](https://www.linkedin.com/in/zulqarnainalipk/) – feel free to connect or reach out!
