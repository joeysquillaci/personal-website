# LSTM Time-Series Workbench

A desktop Tkinter application for training and evaluating an LSTM-based stock time-series model, with built-in reporting, charting, and education content.

## Project Structure

- `app.py`  
  Main application entrypoint and GUI workflow controller (Training Dashboard, Evaluation, Education).

- `view_helpers.py`  
  UI helper routines for chart settings, chart rendering, education card population, and session detail dialogs.

- `modeling.py`  
  Core model/training utilities:
  - `LSTMModel` (PyTorch model)
  - `EarlyStopping` (validation-based stopping + best-weight restore)

- `technical_helpers.py`  
  Feature engineering and preprocessing helpers:
  - RSI/MACD
  - Support/resistance estimation
  - Sequence generation for LSTM (`create_sequences`)

- `education_content.py`  
  Text content used by Education tab cards.

## Dependencies

Install these Python packages using "pip install _______":

- `numpy`
- `pandas`
- `yfinance`
- `matplotlib`
- `torch`
- `scikit-learn`

Also required (may be auto-installed based on the system):

- Python 3.10+
- Tkinter support in your Python installation (for GUI)

Example:

```bash
pip install numpy pandas yfinance matplotlib torch scikit-learn
```

## Run the Project

From this directory:

```bash
cd project_location
python app.py 
```
*may need to use "python3" if on Mac

## Basic Usage

1. Open **Training Dashboard** and set ticker + horizon (or Custom settings).  
2. Click **Train and Evaluate**.  
3. Review results in **Evaluation** (model predictions, chart, scorecards, feature importance, etc).  
4. Use **Education** cards for interpretation help and workflow guidance.

## Notes

- Evaluation chart settings support independent toggles for supports and resistances.
- Summary sections in evaluation use model outputs plus technical context for quick interpretation.

## Contributions

- **Project author:** Joey Squillaci
- **AI-Utilization/Contribution**
  - Understanding and developing GUI frameworks
  - Understanding and developing features for the prototype
  - Debugging support
  - Formatting/cleaning up code blocks
  - Converting project report from a docx format to a latex format
  - Generating framework for README.md file
  * The core model/training/evaluation code was developed by Joey Squillaci and made to fit within the format of a human-accessible/readable GUI (Tkinter)

