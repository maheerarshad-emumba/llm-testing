# RAG System Evaluation Framework

This repository provides a framework for evaluating Retrieval-Augmented Generation (RAG) systems across various critical evaluation areas. The framework is designed to help assess and benchmark the performance of RAG systems using specific tools, pre-defined queries, and evaluation metrics.

## Evaluation Areas

The evaluation framework covers the following areas to assess a RAG system's performance:

- **Response Validation**: Evaluates the factual accuracy of responses against ground truth data.
- **Hallucination**: Measures the generation of irrelevant or incorrect information.
- **Out-of-Context**: Tests how the system handles queries unrelated to the provided context.
- **Security**: Areas like biasness, harmful content and violence, jailbreak, prompt injection, and toxicity are also evaluated.

Each evaluation area is organized into its own directory, which includes:

1. **Tool-Specific Code**: The tool or implementation used for evaluation is stored in its respective subdirectory.
2. **Queries**: The queries used for evaluation are included in the main directory of each area.
3. **Evaluation Results**: Results for the evaluation (e.g., scores) are saved as CSV files.

## How to Use

1. Set Up the Environment
Clone the repository:
git clone <repository_url>
cd <repository_name>

2. Install the required dependencies:
pip install -r requirements.txt

3. Add OpenAI API Key
Before running the tools, ensure you add your OpenAI API key to the relevant code files. Look for a placeholder like OPENAI_API_KEY in the tool-specific code and set your API key.

4. Run Evaluation Tools
Navigate to the desired evaluation area directory.
Run the evaluation tool code.
This will execute the evaluation for the corresponding area and save the results as a CSV file in the evaluations/ directory.

4. Explore Results
The evaluation results are saved as CSV files. You can view these results to analyze the system's performance.
