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

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
model_name = 'vectara/hallucination_evaluation_model'
model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
def predict_hallucination_score(dataframe:pd.DataFrame):
        results = dataframe.copy()
        for column in dataframe.columns:
            if column not in ["prompt", "reference"]:
                sentence_pairs = list(zip(dataframe["reference"], dataframe[column]))
                scores = model.predict(sentence_pairs)
                results[f"{column}-reliability-Score"] = [{'hallucination_score': round(score.item(), 2)} for score in scores]
        return results

results_df =predict_hallucination_score(df)
print(results_df)

results_df.to_csv('hallucination_scores.csv')
