import streamlit as st
import json
import os
from dotenv import load_dotenv
from difflib import SequenceMatcher
from nltk.tokenize import word_tokenize
import numpy as np
from litellm import completion, completion_cost
import warnings
from collections import Counter
import pandas as pd
import re
import csv
import requests
from typing import List, Dict, TypedDict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add this new TypedDict for type hinting
class EBibleOption(TypedDict):
    languageCode: str
    translationId: str
    languageName: str
    languageNameInEnglish: str
    shortTitle: str
    # Add other fields as needed
def load_ebible_options() -> List[EBibleOption]:
    with open("./all_ebible_options.json", "r", encoding="utf-8") as f:
        return json.load(f)

def format_ebible_option(option: EBibleOption) -> str:
    return f"{option['languageCode']}-{option['translationId']} ({option['shortTitle']})"

load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Add this near the top of the file, after imports
AVAILABLE_MODELS = [
    "claude-3-5-sonnet-20240620",
    "gpt-4o",
    "gpt-3.5-turbo",
    "claude-3-haiku-20240307",
    "claude-instant-1.2"
]

# Update the load_bible_structure function
def load_bible_structure(json_data):
    structure = {}
    for vref in json_data.keys():
        book, chapter_verse = vref.split()
        chapter, _ = chapter_verse.split(':')
        if book not in structure:
            structure[book] = {}
        if int(chapter) not in structure[book]:
            structure[book][int(chapter)] = []
        structure[book][int(chapter)].append(int(chapter_verse.split(':')[1]))
    return structure

# Load or initialize translations
def load_translations():
    if os.path.exists("translations.json"):
        with open("translations.json", "r") as f:
            return json.load(f)
    return {}

# Save translations
def save_translations(translations):
    with open("translations.json", "w") as f:
        json.dump(translations, f)

def generate_translation(verse: str, source_text: str, available_translations: list, model: str, register: str, target_language: str) -> tuple[str, float]:
    prompt = f"""Translate the following Bible verse into {target_language}. Don't add additional commentary. Use the following register:    
{register}
    
Source: {source_text}

Available translations:
{chr(10).join(f"{i+1}. {translation}" for i, translation in enumerate(available_translations))}

Please provide a new translation based on the source by combining the available reference translations.
Again, it is critical for accessibility purposes to translate into {target_language}.
Make sure that your generated translation is NOT identical to any of the available translations. It cannot be the exact same, as that would not fulfill our accessibility requirements.
If you only have one sample in the target language, try to vary the vocabulary, grammar, and syntax to slightly paraphrase the original text.
Output your translation in the following format:

FINAL TRANSLATION:
[Your translation here]
"""

    response = completion(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024
    )

    content = response['choices'][0]['message']['content']
    cost = completion_cost(completion_response=response)
    
    # Extract the final translation using regex
    match = re.search(r'FINAL TRANSLATION:\s*(.*?)\s*$', content, re.DOTALL)
    if match:
        return match.group(1).strip(), float(cost)
    else:
        return "Translation not found in the expected format.", float(cost)

# Add this function to calculate similarity between two texts
def calculate_pairwise_similarity(text1: str, text2: str) -> dict:
    levenshtein = SequenceMatcher(None, text1, text2).ratio()
    jaccard = len(set(text1.split()) & set(text2.split())) / len(set(text1.split()) | set(text2.split()))
    bleu = calculate_bleu(text1, text2)
    average = np.mean([levenshtein, jaccard, bleu])
    return {
        'Levenshtein': round(levenshtein, 2),
        'Jaccard': round(jaccard, 2),
        'BLEU': round(bleu, 2),
        'Average': round(average, 2)
    }

# Add this function to calculate BLEU score
def calculate_bleu(text1: str, text2: str) -> float:
    def count_ngrams(text, n):
        return Counter([' '.join(text.split()[i:i+n]) for i in range(len(text.split())-n+1)])
    
    def modified_precision(reference, candidate, n):
        ref_counts = count_ngrams(reference, n)
        cand_counts = count_ngrams(candidate, n)
        total = sum(cand_counts.values())
        correct = sum(min(ref_counts[ngram], count) for ngram, count in cand_counts.items())
        return correct / total if total > 0 else 0
    
    weights = [0.25, 0.25, 0.25, 0.25]
    precisions = [modified_precision(text1, text2, n) for n in range(1, 5)]
    return sum(w * p for w, p in zip(weights, precisions) if p != 0)

def color_scale(val):
    if val > 0.99:
        return f'background-color: rgba(0, 255, 0, 0.05)'
    else:
        alpha = 1 - min(val, 1)  # Invert the alpha value
        return f'background-color: rgba(255, 0, 0, {alpha})'

# Add these functions after the existing functions

def serialize_translations_to_tsv(translations, source_data):
    data = []
    for verse_id, translation_info in translations.items():
        row = {
            'Verse': verse_id,
            'Generated Translation': translation_info['translation'],
            'Model': translation_info['model'],
            'Source': source_data[verse_id]['source']
        }
        
        # Add available translations
        for lang_key in source_data[verse_id]:
            if lang_key.startswith('lang_'):
                row[f'Translation {lang_key}'] = source_data[verse_id][lang_key]
        
        # Add similarity scores
        similarity = calculate_pairwise_similarity(translation_info['translation'], source_data[verse_id]['source'])
        for metric, score in similarity.items():
            row[f'Similarity {metric}'] = score
        
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv('translations_audit.tsv', sep='\t', index=False, quoting=csv.QUOTE_NONNUMERIC)

def calculate_average_similarity(translations, source_data):
    input_translation_similarities = {}
    overall_similarities = []

    for verse_id, translation_info in translations.items():
        if 'translation' in translation_info:
            verse_similarities = []
            for lang_key in source_data[verse_id]:
                if lang_key.startswith('lang_'):
                    similarity = calculate_pairwise_similarity(translation_info['translation'], source_data[verse_id][lang_key])
                    if lang_key not in input_translation_similarities:
                        input_translation_similarities[lang_key] = []
                    input_translation_similarities[lang_key].append(similarity['Average'])
                    verse_similarities.append(similarity['Average'])
            
            if verse_similarities:
                overall_similarities.append(sum(verse_similarities) / len(verse_similarities))

    overall_avg = sum(overall_similarities) / len(overall_similarities) if overall_similarities else 0
    input_avgs = {lang: sum(sims) / len(sims) for lang, sims in input_translation_similarities.items()}

    return overall_avg, input_avgs

# Add these new functions
def download_file(url: str, local_path: str) -> None:
    if not os.path.exists(local_path):
        response = requests.get(url)
        if response.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(response.content)
            st.success(f"Downloaded: {local_path}")
        else:
            st.error(f"Failed to download: {local_path}")
    else:
        st.info(f"File already exists: {local_path}")

def download_ebible_file(language_code: str, file_suffix: str = "") -> str:
    base_url = "https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/"
    filename = f"{language_code}{file_suffix}.txt"
    url = base_url + filename
    
    local_path = os.path.join("bible_sources", filename)
    os.makedirs("bible_sources", exist_ok=True)

def download_ebible_file(language_code: str, file_suffix: str = "") -> Optional[str]:
    base_url = "https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/"
    filename = f"{language_code}{file_suffix}.txt"
    url = base_url + filename
    
    local_path = os.path.join("bible_sources", filename)
    os.makedirs("bible_sources", exist_ok=True)
    
    if os.path.exists(local_path):
        st.info(f"File already exists: {local_path}")
    else:
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            with open(local_path, "wb") as f:
                f.write(response.content)
            st.success(f"Downloaded: {local_path}")
        except requests.RequestException as e:
            st.error(f"Failed to download {filename}: {str(e)}")
            return None
    
    return local_path

def download_vref_file() -> str:
    vref_url = "https://raw.githubusercontent.com/BibleNLP/ebible/main/metadata/vref.txt"
    vref_path = os.path.join("bible_sources", "vref.txt")
    
    if os.path.exists(vref_path):
        st.info(f"File already exists: {vref_path}")
    else:
        download_file(vref_url, vref_path)
    return vref_path

def download_macula_files():
    os.makedirs("bible_sources", exist_ok=True)
    
    hebrew_url = "https://github.com/Clear-Bible/macula-hebrew/raw/main/WLC/tsv/macula-hebrew.tsv"
    greek_url = "https://github.com/Clear-Bible/macula-greek/raw/main/Nestle1904/tsv/macula-greek-Nestle1904.tsv"
    
    hebrew_path = os.path.join("bible_sources", "macula-hebrew.tsv")
    greek_path = os.path.join("bible_sources", "macula-greek.tsv")
    
    download_file(hebrew_url, hebrew_path)
    download_file(greek_url, greek_path)

def load_bible_content(file_path: str, vref_path: str) -> Dict[str, str]:
    try:
        with open(vref_path, 'r', encoding='utf-8') as vref_file:
            vrefs = [line.strip() for line in vref_file]
        
        with open(file_path, 'r', encoding='utf-8') as bible_file:
            verses = [line.strip() for line in bible_file]
        
        return dict(zip(vrefs, verses))
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return {}

def load_vref_content(vref_path: str) -> List[str]:
    with open(vref_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def load_macula_content() -> Dict[str, str]:
    hebrew_path = os.path.join("bible_sources", "macula-hebrew.tsv")
    greek_path = os.path.join("bible_sources", "macula-greek.tsv")
    
    macula_content = {}
    
    for path in [hebrew_path, greek_path]:
        df = pd.read_csv(path, sep="\t", usecols=['ref', 'text'])
        for _, row in df.iterrows():
            vref = row['ref'].split('!')[0]
            macula_content[vref] = macula_content.get(vref, '') + ' ' + str(row['text'])
    
    return macula_content

def align_verses(bibles: List[Dict[str, str]], macula_content: Dict[str, str], vrefs: List[str]) -> Dict[str, Dict[str, str]]:
    aligned_verses = {}
    
    for vref in tqdm(vrefs, desc="Aligning verses"):
        verse_data = {"source": macula_content.get(vref, "")}
        
        for i, bible in enumerate(bibles):
            lang_code = f"lang_{i}"
            verse_data[lang_code] = bible.get(vref, "")
        
        if any(verse_data.values()):
            aligned_verses[vref] = verse_data
    
    return aligned_verses

def process_ebible_files(file_list: List[str]) -> Dict[str, Dict[str, str]]:
    download_macula_files()
    vref_path = download_vref_file()
    vrefs = load_vref_content(vref_path)
    macula_content = load_macula_content()
    
    with ThreadPoolExecutor(max_workers=len(file_list)) as executor:
        future_to_file = {executor.submit(download_ebible_file, file_info): file_info for file_info in file_list}
        bibles = []
        for future in as_completed(future_to_file):
            try:
                file_path = future.result()
                if file_path:
                    bibles.append(load_bible_content(file_path, vref_path))
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    if not bibles:
        st.error("No Bible files were successfully loaded.")
        return {}
    
    return align_verses(bibles, macula_content, vrefs)
def load_bible_content(file_path: str, vref_path: str) -> Dict[str, str]:
    try:
        with open(vref_path, 'r', encoding='utf-8') as vref_file:
            vrefs = [line.strip() for line in vref_file]
        
        with open(file_path, 'r', encoding='utf-8') as bible_file:
            verses = [line.strip() for line in bible_file]
        
        return dict(zip(vrefs, verses))
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return {}

def load_vref_content(vref_path: str) -> List[str]:
    with open(vref_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def load_macula_content() -> Dict[str, str]:
    hebrew_path = os.path.join("bible_sources", "macula-hebrew.tsv")
    greek_path = os.path.join("bible_sources", "macula-greek.tsv")
    
    macula_content = {}
    
    for path in [hebrew_path, greek_path]:
        df = pd.read_csv(path, sep="\t", usecols=['ref', 'text'])
        for _, row in df.iterrows():
            vref = row['ref'].split('!')[0]
            macula_content[vref] = macula_content.get(vref, '') + ' ' + str(row['text'])
    
    return macula_content

def align_verses(bibles: List[Dict[str, str]], macula_content: Dict[str, str], vrefs: List[str]) -> Dict[str, Dict[str, str]]:
    aligned_verses = {}
    
    for vref in tqdm(vrefs, desc="Aligning verses"):
        verse_data = {"source": macula_content.get(vref, "")}
        
        for i, bible in enumerate(bibles):
            lang_code = f"lang_{i}"
            verse_data[lang_code] = bible.get(vref, "")
        
        if any(verse_data.values()):
            aligned_verses[vref] = verse_data
    
    return aligned_verses

def process_ebible_files(file_list: List[str]) -> Dict[str, Dict[str, str]]:
    download_macula_files()
    vref_path = download_vref_file()
    vrefs = load_vref_content(vref_path)
    macula_content = load_macula_content()
    
    with ThreadPoolExecutor(max_workers=len(file_list)) as executor:
        future_to_file = {executor.submit(download_ebible_file, file_info): file_info for file_info in file_list}
        bibles = []
        for future in as_completed(future_to_file):
            file_path = future.result()
            if file_path:
                try:
                    bibles.append(load_bible_content(file_path, vref_path))
                except FileNotFoundError:
                    st.error(f"File not found: {file_path}")
    
    if not bibles:
        st.error("No Bible files were successfully loaded.")
        return {}
    
    return align_verses(bibles, macula_content, vrefs)

# Modify the source_file_section function
def source_file_section():
    st.header("Source File Generation")
    
    # Option to upload existing JSON
    uploaded_file = st.file_uploader("Upload existing JSON file", type="json")
    
    if uploaded_file:
        source_data = json.load(uploaded_file)
        st.success("JSON file loaded successfully!")
        return source_data
    
    # Option to generate new source file
    st.subheader("Or generate a new source file")
    
    # Load available eBible options
    ebible_options = load_ebible_options()
    
    # Create a list of formatted options for the multiselect
    formatted_options = [format_ebible_option(option) for option in ebible_options]
    
    selected_options = st.multiselect("Select languages to include", formatted_options)
    
    if st.button("Generate Source File") and selected_options:
        # Convert selected options back to language codes
        selected_languages = [option.split()[0] for option in selected_options]
        
        with st.spinner("Generating source file..."):
            source_data = process_ebible_files(selected_languages)
        
        if source_data:
            st.success("Source file generated successfully!")
            
            # Save the generated data
            output_file = f"aligned_verses_{'_'.join(selected_languages)}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(source_data, f, ensure_ascii=False, indent=2)
            
            st.download_button(
                label="Download JSON",
                data=json.dumps(source_data, ensure_ascii=False, indent=2),
                file_name=output_file,
                mime="application/json"
            )
            
            return source_data
    
    return None

# Streamlit app
def main():
    st.title("Bible Translation Assistant")
    
    # Add the new source file section
    source_data = source_file_section()
    
    if source_data:
        bible_structure = load_bible_structure(source_data)
        translations = load_translations()

        # Sidebar
        st.sidebar.title("Navigation")
        selected_model = st.sidebar.selectbox("Select LLM Model", AVAILABLE_MODELS)
        register = st.sidebar.text_input("Translation register", "Grade 3-4 reading level. Give a simplified semantic rendering, even if you have to add words or circumlocations.")
        target_language = st.sidebar.text_input("Target language", "French")
        selected_book = st.sidebar.selectbox("Select a book", list(bible_structure.keys()))
        selected_chapter = st.sidebar.selectbox("Select a chapter", list(bible_structure[selected_book].keys()))

        # Initialize session state for total cost if not exists
        if 'total_cost' not in st.session_state:
            st.session_state.total_cost = 0

        # Display total cost in sidebar
        st.sidebar.markdown("---")
        st.sidebar.subheader("Cost Calculator")
        st.sidebar.metric("Total Cost", f"${st.session_state.total_cost:.4f}")

        # Display average similarity in sidebar with explanation
        overall_avg, input_avgs = calculate_average_similarity(translations, source_data)
        
        st.sidebar.metric("Overall Average Similarity", f"{overall_avg:.2f}")
        
        st.sidebar.markdown("**Average Similarity to Input Translations:**")
        col1, col2 = st.sidebar.columns(2)
        for i, (lang, avg) in enumerate(input_avgs.items()):
            with col1 if i % 2 == 0 else col2:
                st.metric(f"Similarity to {lang}", f"{int(avg * 100)}%")
        st.sidebar.caption("""
        These metrics represent:
        1. The overall average similarity between our generated translations and all input translations.
        2. The average similarity between our generated translations and each specific input translation.

        Ideally, we want:
        - The overall average to be in a moderate range (e.g., 0.4 - 0.6).
        - The averages for each input translation to be similar to each other and close to the overall average.

        This balance indicates that our translations:
        - Are accessible for the target audience
        - Maintain fidelity to the original meaning
        - Are not biased towards any single existing translation

        If the overall average or any input translation average is too high (> 0.7), 
        our translations might be too literal or similar to existing ones.
        If too low (< 0.3), we might be diverging too much from the source meaning.

        Monitor these values to ensure a balance between accessibility, fidelity, and unbiased adaptation.
        """)

        # Main area
        st.header(f"{selected_book} Chapter {selected_chapter}")

        for verse in bible_structure[selected_book][selected_chapter]:
            verse_id = f"{selected_book} {selected_chapter}:{verse}"
            st.subheader(f"Verse {verse}")

            # Display source verse and available translations
            if verse_id in source_data:
                with st.expander("Source and Available Translations", expanded=True):
                    st.markdown("**Source verse:**")
                    st.code(source_data[verse_id]['source'], language="hebrew")
                    
                    for lang_key in source_data[verse_id]:
                        if lang_key.startswith('lang_'):
                            st.markdown(f"**Translation {lang_key}:**")
                            st.info(source_data[verse_id][lang_key])

            # Create a DataFrame for similarity scores with correct data types
            similarity_df = pd.DataFrame({
                'Translation': pd.Series(dtype='str'),
                'Levenshtein': pd.Series(dtype='float'),
                'Jaccard': pd.Series(dtype='float'),
                'BLEU': pd.Series(dtype='float'),
                'Average': pd.Series(dtype='float'),
                'Description': pd.Series(dtype='str')
            })

            # Display existing generated translation or generate new one
            if verse_id in translations:
                with st.expander("Generated Translation and Analytics", expanded=True):
                    st.markdown("**Current generated translation:**")
                    st.success(translations[verse_id]['translation'])
                    
                    # Display cost per verse if available
                    if 'cost' in translations[verse_id]:
                        st.markdown(f"**Cost:** ${translations[verse_id]['cost']:.4f}")
                    
                    # Calculate similarity scores for each input translation
                    similarity_df = pd.DataFrame(columns=['Translation', 'Levenshtein', 'Jaccard', 'BLEU', 'Average', 'Description'])
                    for lang_key in source_data[verse_id]:
                        if lang_key.startswith('lang_'):
                            input_translation = source_data[verse_id][lang_key]
                            generated_translation = translations[verse_id]['translation']
                            
                            if not input_translation or not generated_translation:
                                new_row = pd.DataFrame({
                                    'Translation': [lang_key],
                                    'Levenshtein': [np.nan],
                                    'Jaccard': [np.nan],
                                    'BLEU': [np.nan],
                                    'Average': [np.nan],
                                    'Description': ['NA - Empty translation']
                                })
                            else:
                                similarity = calculate_pairwise_similarity(generated_translation, input_translation)
                                similarity_percentage = round(similarity['Average'] * 100, 2)
                                description = f"{similarity_percentage}% similar to {lang_key}"
                                new_row = pd.DataFrame({
                                    'Translation': [lang_key],
                                    'Levenshtein': [similarity['Levenshtein']],
                                    'Jaccard': [similarity['Jaccard']],
                                    'BLEU': [similarity['BLEU']],
                                    'Average': [similarity['Average']],
                                    'Description': [description]
                                })
                            similarity_df = pd.concat([similarity_df, new_row], ignore_index=True)
                    
                    # Sort the DataFrame by Average score (most similar first), putting NaN values at the end
                    similarity_df = similarity_df.sort_values('Average', ascending=False, na_position='last')
                    
                    # Display the similarity scores table
                    st.markdown("**Similarity Scores:**")
                    st.dataframe(similarity_df.style
                                 .apply(lambda x: [color_scale(v) if pd.notnull(v) else '' for v in x], subset=['Levenshtein', 'Jaccard', 'BLEU', 'Average'])
                                 .apply(lambda x: ['background-color: #ccffcc; font-weight: bold' if v == x.max() and pd.notnull(v) else '' for v in x], subset=['Average'])
                                 .format({col: '{:.2f}' for col in ['Levenshtein', 'Jaccard', 'BLEU', 'Average']})
                    )
                    
                    # Update the legend
                    st.markdown("""
                    **Legend:**
                    - **Levenshtein:** Measures string similarity. Higher score indicates higher similarity.
                    - **Jaccard:** Measures word overlap. Higher score indicates higher similarity.
                    - **BLEU:** Measures translation quality. Higher score indicates higher similarity.
                    - **Average:** Average of the above scores. Higher score indicates higher overall similarity.
                    - **Description:** Provides a percentage similarity based on the Average score or indicates if a translation is empty.
                    
                    All scores range from 0 to 1, where 1 indicates perfect similarity and 0 indicates complete dissimilarity.
                    'NA' indicates that either the input or generated translation was empty.
                    The table is sorted by Average score, with the most similar translations at the top and empty translations at the bottom.
                    
                    Color intensity increases with similarity (greener = more similar).
                    The highest Average score is highlighted in light green.
                    """)
                    
                    if 'model' in translations[verse_id]:
                        st.markdown(f"**Model used:** {translations[verse_id]['model']}")
            else:
                st.warning("No generated translation available.")

            if st.button(f"Generate translation for verse {verse}", key=f"generate_{verse_id}"):
                with st.spinner("Generating translation..."):
                    source_text = source_data[verse_id]['source']
                    available_translations = [source_data[verse_id][k] for k in source_data[verse_id] if k.startswith('lang_')]
                    
                    new_translation, cost = generate_translation(verse_id, source_text, available_translations, selected_model, register, target_language)
                    similarity = calculate_pairwise_similarity(new_translation, source_text)
                    translations[verse_id] = {
                        'translation': new_translation,
                        'similarity': similarity,
                        'model': selected_model,
                        'cost': cost
                    }
                    save_translations(translations)
                    
                    # Update total cost
                    st.session_state.total_cost += cost
                
                st.success("Translation generated successfully!")
                st.markdown("**New translation:**")
                st.info(new_translation)
                st.markdown(f"**Cost:** ${cost:.4f}")
                st.markdown("**Overlap diagnostics:**")
                st.json(similarity)
                
                st.rerun()
            
            st.markdown("---")  # Add a horizontal line between verses

        if st.button(f"Generate all translations for {selected_book} {selected_chapter}"):
            total_cost = 0
            for verse in bible_structure[selected_book][selected_chapter]:
                verse_id = f"{selected_book} {selected_chapter}:{verse}"
                if verse_id not in translations:
                    source_text = source_data[verse_id]['source']
                    available_translations = [source_data[verse_id][k] for k in source_data[verse_id] if k.startswith('lang_')]
                    
                    new_translation, cost = generate_translation(verse_id, source_text, available_translations, selected_model, register, target_language)
                    similarity = calculate_pairwise_similarity(new_translation, source_text)
                    translations[verse_id] = {
                        'translation': new_translation,
                        'similarity': similarity,
                        'model': selected_model,
                        'cost': cost
                    }
                    total_cost += cost
            
            save_translations(translations)
            st.session_state.total_cost += total_cost
            st.success(f"All translations generated successfully! Total cost: ${total_cost:.4f}")
            st.rerun()

        if st.button("Serialize translations to TSV"):
            serialize_translations_to_tsv(translations, source_data)
            st.success("Translations serialized to 'translations_audit.tsv'")

    else:
        st.warning("Please upload a JSON file or generate a new source file to continue.")

if __name__ == "__main__":
    main()
