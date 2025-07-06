# PII Data Detection

## Introduction
This project is designed to help detect and remove Personally Identifiable Information (PII) from educational data. It leverages advanced natural language processing (NLP) techniques and pre-trained models to identify various types of PII, such as names, email addresses, phone numbers, and street addresses.

## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/PII_Data_Detection.git
   cd PII_Data_Detection
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To run the PII detection process, execute the `main.py` script:

```bash
python src/main.py
```

This will process the test data (as configured in `src/config.py`), perform PII detection, and generate a `submission.csv` file with the identified PII entities.

## Project Structure
```
PII_Data_Detection/
├── src/
│   ├── config.py
│   ├── main.py
│   └── utils.py
├── README.md
├── requirements.txt
└── .gitignore
```

- `src/config.py`: Contains configuration settings for the PII detection model, including data paths, model parameters, and training hyperparameters.
- `src/main.py`: The main script to run the PII detection pipeline, from data loading and tokenization to prediction and submission file generation.
- `src/utils.py`: A collection of utility functions used across the project, including data processing, span finding, and prediction handling.
- `README.md`: This file, providing an overview of the project, installation, and usage instructions.
- `requirements.txt`: Lists all Python dependencies required to run the project.
- `.gitignore`: Specifies intentionally untracked files that Git should ignore.

## Credits and Acknowledgments
This project is inspired by the work of Aleksandr Lavrikov, available at [this Kaggle project](https://www.kaggle.com/code/lavrikovav/0-968-to-onnx-30-200-speedup-pii-inference).

