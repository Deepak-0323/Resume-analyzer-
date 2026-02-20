# AI Resume Analyzer

This is a Streamlit web application that automatically analyzes and scores resumes against a job description using Natural Language Processing (NLP). It helps recruiters and hiring managers quickly identify the most relevant candidates from a large pool of applicants.

## Features

- **Resume Parsing**: Extracts text from PDF and DOCX files.
- **Semantic Matching**: Uses TF-IDF and Cosine Similarity to calculate a match score between a resume and a job description.
- **Candidate Ranking**: Ranks candidates from highest to lowest match score.
- **Key Insights**: Provides an overview of top skills, score distribution, and other metrics like average experience and word count.
- **Duplicate Detection**: Performs a basic check to flag potential duplicate resumes.
- **Downloadable Results**: Exports the analysis results to a CSV file.

## How to Run

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    (You'll need to create a `requirements.txt` file first. See the next section.)

4.  **Download the spaCy model**:
    ```bash
    python -m spacy download en_core_web_sm
    ```

5.  **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```

## Dependencies

The following libraries are required to run the application. You can create a `requirements.txt` file with these entries: