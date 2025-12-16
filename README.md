# ml4641-project
My ML project for CS 4641. This project was done with the help of Diego Mateo.


# Stock Price Direction Prediction with Sentiment Analysis

This project investigates whether daily Apple headline sentiment across the full 2020 COVID year can improve one-day-ahead price direction predictions beyond technical indicators alone. The workflow trains a sentiment classifier on the Financial PhraseBank dataset, applies it to Apple news headlines, and combines sentiment features with technical indicators to predict stock price movements using multiple ensemble classifiers (Random Forest, Gradient Boosting, Extra Trees).

## Running the Model Training

To train the models and generate results:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the training pipeline (supervised baselines + sentiment):**
   ```bash
   python run_pipeline.py
   ```

3. **Optional: Rolling evaluation for more robust RF metrics:**
   ```bash
   python run_rolling_eval.py
   ```

4. **Generate comparison plots/tables across models:**
   ```bash
   python run_comparison.py
   ```

5. **View results:**
   - Model metrics are saved to `reports/metrics.json`
   - Comparison tables are saved to `reports/model_comparison.csv` and `reports/model_hyperparameters.csv`
   - Rolling RF metrics are saved to `reports/metrics_rolling.json` (if run)
   - Visualizations are saved to `reports/figures/`
   - Trained sentiment model is saved to `models/sentiment_model.joblib`

## Project Structure

### `/src/`: Source code directory containing all Python modules

**`/src/__init__.py`**: Package initialization file for the src module

**`/src/paths.py`**: Centralized path management for project directories (data, models, reports, figures)

**`/src/config.py`**: Project-wide configuration dataclass for data windows, seeds, feature toggles, and model hyperparameters

**`/src/data_utils.py`**: Data ingestion utilities for loading Financial PhraseBank, fetching stock price data from CSV or yfinance, and loading stock news from Hugging Face datasets

**`/src/features.py`**: Feature engineering functions including text cleaning, technical indicator computation (returns, volatility, SMA, RSI), daily sentiment aggregation, and model frame construction

**`/src/modeling.py`**: Model training functions for sentiment classification (TF-IDF + Logistic Regression) and stock direction prediction (Random Forest, Gradient Boosting, Extra Trees), including hyperparameter tuning and result dataclasses

**`/src/evaluation.py`**: Shared evaluation helpers for classification tasks including metrics computation, confusion matrices, and ROC curve data

**`/src/pipeline_utils.py`**: Reusable helpers for preparing modeling frames, loading processed data, and orchestrating ticker-specific data preparation workflows

**`/src/sequence_model.py`**: LSTM sequence model implementation for next-day price prediction and derived direction metrics (optional, not included in main pipeline)

**`/src/visualization.py`**: Plotting utilities for generating sentiment distribution charts, confusion matrices, sentiment vs. price comparisons, feature importance plots, ROC curves, and model comparison bar charts

### `/models/`: Directory for saved trained models

**`/models/sentiment_model.joblib`**: Serialized sentiment classifier pipeline (TF-IDF vectorizer + Logistic Regression) trained on Financial PhraseBank

**`/models/lstm_price_model.pt`**: Optional PyTorch LSTM model for price prediction (if sequence model training is enabled)

### `/reports/`: Directory for generated reports, metrics, and visualizations

**`/reports/metrics.json`**: JSON file containing evaluation metrics for the sentiment model and all direction prediction models (Random Forest with sentiment, technical-only baseline, Gradient Boosting, Extra Trees)

**`/reports/model_comparison.csv`**: Comparison table aggregating metrics across all direction prediction models

**`/reports/model_hyperparameters.csv`**: Table documenting hyperparameters and configuration for each model variant

**`/reports/metrics_rolling.json`**: Aggregated rolling evaluation metrics for Random Forest models (if rolling evaluation is run)

**`/reports/figures/`: Directory containing all generated visualization plots

**`/reports/figures/sentiment_distribution.png`**: Bar chart showing the distribution of sentiment labels in the Financial PhraseBank dataset

**`/reports/figures/sentiment_confusion.png`**: Confusion matrix heatmap for the sentiment classifier performance

**`/reports/figures/sentiment_vs_price.png`**: Dual-axis line chart comparing daily Apple stock price with aggregated sentiment scores over time

**`/reports/figures/feature_importance.png`**: Horizontal bar plot showing the top feature importances from the Random Forest direction prediction model

**`/reports/figures/roc_curve_rf.png`**: ROC curve visualization for the Random Forest direction prediction model

**`/reports/figures/roc_curve_gbdt.png`**: ROC curve visualization for the Gradient Boosting direction prediction model

**`/reports/figures/roc_curve_extratrees.png`**: ROC curve visualization for the Extra Trees direction prediction model

**`/reports/figures/direction_confusion_rf.png`**: Confusion matrix for the Random Forest direction prediction model

**`/reports/figures/direction_confusion_baseline.png`**: Confusion matrix for the technical-only Random Forest baseline model

**`/reports/figures/direction_confusion_gbdt.png`**: Confusion matrix for the Gradient Boosting direction prediction model

**`/reports/figures/direction_confusion_extratrees.png`**: Confusion matrix for the Extra Trees direction prediction model

### `/data/`: Directory for raw and processed data files (created automatically)

**`/data/raw/`: Directory for raw data files including stock price CSVs and downloaded datasets

**`/data/processed/`: Directory for processed intermediate data files such as daily sentiment aggregations and model-ready feature frames

### `/run_pipeline.py`: Main orchestration script that executes the complete modeling workflow including sentiment training, feature engineering, and training multiple direction prediction models (Random Forest, Gradient Boosting, Extra Trees, and technical-only baseline)

### `/run_rolling_eval.py`: Rolling evaluation script that performs time-series cross-validation to generate more robust metrics for Random Forest models with and without sentiment features

### `/run_comparison.py`: Aggregates model metrics and hyperparameters into comparison tables (CSV) and generates comparison bar charts for key metrics across all models

### `/requirements.txt`: Python package dependencies list
