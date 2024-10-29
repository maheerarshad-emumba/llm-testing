import pandas as pd

columns_needed = ["query", "response", "ground truth"]

df = pd.read_csv("C:/Users/Emumba/Documents/genie research/llm-testing-main/llm-testing-main/hallucination/llm_responses.csv", usecols=columns_needed)
# Rename columns to match function requirements
df.rename(columns={
    "query": "prompt",
    "response": "Model A",     # or any specific model names if there are multiple
    "ground truth": "reference"
}, inplace=True)
# Display the resulting DataFrame with only the required columns
print(df)

import pandas as pd
from sentence_transformers import CrossEncoder

class Reliability_evaluator:
    """
    This class is designed to evaluate hallucination scores for multiple
    model outputs against reference sentences using a CrossEncoder model.
    """
    def __init__(self, model_name='vectara/hallucination_evaluation_model'):
        """
        Initializes the SentenceSimilarityEvaluator with a specified model.

        Parameters:
        - model_name (str): The name of the model to be used for hallucination evaluation.
        """
        self.model = CrossEncoder(model_name, trust_remote_code=True)

    def predict_hallucination_score(self, dataframe:pd.DataFrame):
        """
        Predicts similarity scores for each model output in the DataFrame against the reference sentences.

        Parameters:
        - dataframe (pandas.DataFrame): A DataFrame containing 'reference' and multiple model output columns.

        Returns:
        - results (pandas.DataFrame): The original DataFrame with additional columns for hallucination scores.
        """
        results = dataframe.copy()
        for column in dataframe.columns:
            if column not in ["prompt", "reference"]:
                sentence_pairs = list(zip(dataframe["reference"], dataframe[column]))
                scores = self.model.predict(sentence_pairs)
                results[f"{column}-reliability-Score"] = [{'hallucination_score': round(score,2)} for score in scores]
        return results

Reliability_eval = Reliability_evaluator()

# Compute hallucination scores
results_df = Reliability_eval.predict_hallucination_score(df)

results_df.to_csv('hallucination_scores.csv')

# Print or further process the results
print(results_df)