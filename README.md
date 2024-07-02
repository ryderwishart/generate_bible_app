# Bible Translation Assistant

## Purpose

The Bible Translation Assistant is a Streamlit-based application designed to facilitate the generation of Bible translations in specific registers and target languages. It leverages advanced language models to create accessible translations while maintaining fidelity to the original text.

## Key Features and Benefits

1. **Source File Generation**
   - Upload existing JSON files or generate new source files from eBible options
   - Benefit: Flexibility in starting with pre-existing data or creating new datasets
   - Over 1000 openly-licensed reference Bibles available for use

> Note: some of the Bibles in the eBible corpus are marked as ND (non-derivative), which means you cannot create a derivative or modified version of the Bible. This may constrain your activities, but generating a new Bible with reference to multiple Bibles may not be considered derivative work. 

> Note: every generated Bible in this app is a translation of the original Hebrew and Greek Bible, based on the Macula version of the N1904 Bible. The Hebrew source is available [here](https://github.com/Clear-Bible/macula-hebrew/raw/main/WLC/tsv/macula-hebrew.tsv) and the Greek source is available [here](https://github.com/Clear-Bible/macula-greek/raw/main/Nestle1904/tsv/macula-greek-Nestle1904.tsv).

1. **Multiple Language Model Support**
   - Choose from various LLM models (e.g., Claude, GPT-4, GPT-3.5)
   - Benefit: Optimize translation quality and cost based on specific needs

2. **Customizable Translation Register**
   - Set the desired reading level and style for translations
   - Benefit: Create translations tailored to specific audience needs (e.g., children, ESL learners)

3. **Target Language Selection**
   - Specify the desired output language for translations
   - Benefit: Generate translations for any target language, expanding accessibility

4. **Verse-by-Verse Translation**
   - Generate translations for individual verses or entire chapters
   - Benefit: Granular control over the translation process

5. **Translation Analytics**
   - View similarity scores (Levenshtein, Jaccard, BLEU) between generated and source texts
   - Benefit: Ensure translations maintain an appropriate balance between accessibility and fidelity
   - Ensure that the generated translation is no more similar to one reference Bible than to any other

6. **Cost Tracking**
   - Monitor the cost of generated translations in real-time
   - Benefit: Manage budget and resources effectively

7. **Bulk Translation Generation**
   - Generate translations for entire chapters at once
   - Benefit: Streamline the translation process for larger sections of text

8. **TSV Export**
   - Serialize translations to TSV format for further analysis or processing
   - Benefit: Easy integration with other tools and workflows

9.  **Progress Tracking**
    - Save and load translations, allowing work to be resumed across sessions
    - Benefit: Manage long-term translation projects efficiently

## How to Run the App

1. Ensure you have Python installed on your system.

2. Clone this repository and navigate to the project directory.

3. Install the required dependencies:

```
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your API keys:

```
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key
```

5. Run the Streamlit app:

```
python -m streamlit run generate_bible_app.py
```

6. Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`).

## Note

This app requires API keys for language models (Anthropic, OpenAI, etc.). Ensure you have the necessary permissions and credits to use these services.
