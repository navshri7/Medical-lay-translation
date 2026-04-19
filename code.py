from datasets import load_dataset


dataset = load_dataset("cbasu/Med-EASi")


data = dataset["train"]

l = len(data)
print("The length of the data is: ", l, "\n")


    
    
    


from transformers import pipeline
ner_pipeline = pipeline(
    "ner",
    model="d4data/biomedical-ner-all",
    aggregation_strategy="simple"  
)



import pandas as pd

file_path = "CHV_Concepts.tsv"

df = pd.read_csv(file_path, sep='\t', header=None, usecols=[0, 1, 2, 3, 4])
df.columns = ['id', 'term', 'prefnamea', 'prefnameb', 'explanation']

print(df.head())

from transformers import pipeline


biobert_qa = pipeline(
    "question-answering",
    model="dmis-lab/biobert-large-cased-v1.1-squad",
    tokenizer="dmis-lab/biobert-large-cased-v1.1-squad"
)

def biomedical_definition_lookup(term: str) -> str:
    """
    Query a BioBERT QA model for a short definition of a biomedical term.
    This uses the question: 'What is <term> in medicine?'

    Returns:
        - definition string (short phrase)
        - None if no usable definition
    """

    
    if not term or len(term.strip()) == 0:
        return None

    question = f"What is {term} in medicine?"

    
    
    
    context = (
        f"{term} is a medical term. "
        f"This passage contains general medical information to help answer questions "
        f"about diseases, procedures, symptoms, and clinical entities."
    )

    try:
        result = biobert_qa(question=question, context=context)

        answer = result.get("answer", "").strip()

        
        if (
            answer
            and len(answer.split()) > 1       
            and "[CLS]" not in answer
            and "[SEP]" not in answer
            and answer.lower() not in {"unknown", "no", "none"}
        ):
            return answer

    except Exception as e:
        print(f"[WARN] BioBERT lookup failed for '{term}': {e}")

    return None

import re
import uuid
from typing import List, Dict, Tuple






chv_lookup: Dict[str, Dict] = {}
for _, row in df.iterrows():
    explanation = row['explanation'] if pd.notna(row['explanation']) else None
    if explanation == "\\N":
      
      explanation = None

    candidates = []
    for col in ['term', 'prefnamea', 'prefnameb']:
        v = row.get(col)
        if pd.notna(v) and isinstance(v, str) and v.strip():
            candidates.append(v.strip().lower())
    for cand in candidates:
        
        chv_lookup[cand] = {
            "chv_id": row['id'],
            "canonical": row.get('term'),
            "explanation": explanation
        }


def normalize_key(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r'[\s\-–—]+', ' ', s)  
    s = re.sub(r'[^\w\s/]', '', s)   
    return s

def chv_find(span_text: str) -> Dict:
    """
    Try to find a CHV explanation for the given span text.
    Returns dict or None.
    """
    key = normalize_key(span_text)
    
    if key in chv_lookup:
        return chv_lookup[key]
    
    tokens = key.split()
    
    for L in range(len(tokens), 0, -1):
        for i in range(0, len(tokens)-L+1):
            sub = " ".join(tokens[i:i+L])
            if sub in chv_lookup:
                return chv_lookup[sub]
    return None
import re

def replace_with_parentheses(text: str) -> str:
    match = re.search(r'\((.*?)\)', text)
    if match:
        return match.group(1)  
    return text  

s = "a medical term (inflammation of the liver)"
print(replace_with_parentheses(s))



def process_report_and_entities(report_text: str,
                                entities: List[Dict]
                               ) -> Tuple[str, Dict[str, Dict]]:
    """
    report_text: original text
    entities: list of dicts with keys: {"start":int,"end":int,"text":str,"type":str}
              - start/end are character offsets in report_text
    Returns:
      - augmented_text: text after applying replacements
      - entity_map: mapping from mask_id string -> metadata dict for retrieval/repair
    """
    
    ents_sorted = sorted(entities, key=lambda e: (e['start'], -e['end']))
    augmented_parts = []
    last_idx = 0
    entity_map: Dict[str, Dict] = {}

    for ent in ents_sorted:
        s = ent['start']; e = ent['end']
        
        s = max(0, min(len(report_text), s))
        e = max(0, min(len(report_text), e))
        if s >= e:
            continue

        
        if s < last_idx:
            
            continue

        orig = report_text[s:e]
        ent_type = ent.get('type', '').strip()

        
        augmented_parts.append(report_text[last_idx:s])

        
        if ent_type in VERBATIM_KEEP:
            
            mask_id = str(uuid.uuid4()).replace('-', '')[:12]  
            token = f"[ENTITY_{mask_id}]"
            entity_map[mask_id] = {
                "action": "verbatim_mask",
                "orig_text": orig,
                "type": ent_type,
                "start": s,
                "end": e
            }
            augmented_parts.append(token)

        elif ent_type in NORMALIZE:
            
            chv = chv_find(orig)
            if chv and chv.get('explanation'):

                
                explanation = chv['explanation'].strip()
                explanation = replace_with_parentheses(explanation)


                
                mask_id = str(uuid.uuid4()).replace('-', '')[:12]
                
                
                augmented_parts.append(explanation)
                entity_map[mask_id] = {
                    "action": "normalized_inline",
                    "orig_text": orig,
                    "type": ent_type,
                    "chv_id": chv.get('chv_id'),
                    "canonical": chv.get('canonical'),
                    "explanation": explanation,
                    "start": s,
                    "end": e
                }
            else:
              
              definition = biomedical_definition_lookup(orig)

              if definition:
                  
                  
                  definition = definition.strip()
                  replacement = f"{definition} ({orig})"
                  replacement = replace_with_parentheses(replacement)


                  mask_id = str(uuid.uuid4()).replace('-', '')[:12]

                  augmented_parts.append(replacement)

                  entity_map[mask_id] = {
                      "action": "normalized_inline_biomed",
                      "orig_text": orig,
                      "type": ent_type,
                      "definition": definition,
                      "start": s,
                      "end": e
                  }

              else:
                  
                  mask_id = str(uuid.uuid4()).replace('-', '')[:12]
                  token = f"[ENTITY_{mask_id}]"

                  entity_map[mask_id] = {
                      "action": "mask_due_no_chv",
                      "orig_text": orig,
                      "type": ent_type,
                      "start": s,
                      "end": e
                  }
                  augmented_parts.append(token)

        else:
            
            augmented_parts.append(orig)
            
            
            
            

        last_idx = e

    
    augmented_parts.append(report_text[last_idx:])

    augmented_text = "".join(augmented_parts)
    return augmented_text, entity_map
!pip install deep_translator

from deep_translator import GoogleTranslator

def translate_to_telugu(sentence: str) -> str:
    """
    Translate English → Telugu using deep-translator's GoogleTranslator.
    Free, no API key, stable, and synchronous.
    """
    try:
        return GoogleTranslator(source='en', target='te').translate(sentence)
    except Exception as e:
        print("Translation error:", e)
        return sentence
def translate_to_hindi(sentence: str) -> str:
    """
    Translate English → Hindi using deep-translator's GoogleTranslator.
    Free, no API key, stable, and synchronous.
    """
    try:
        return GoogleTranslator(source='en', target='hi').translate(sentence)
    except Exception as e:
        print("Translation error:", e)
        return sentence

import re
from typing import Dict, Callable, Optional


ENTITY_TOKEN_RE = re.compile(r"\[ENTITY_([0-9a-fA-F]{8,})\]")

def restore_entities_from_ids(
    translated_text: str,
    entity_map: Dict[str, Dict],
    strategy: str = "orig",
    transliterate_fn: Optional[Callable[[str], str]] = None,
    translated_replacement_map: Optional[Dict[str, str]] = None
) -> str:
    """
    Replace [ENTITY_<id>] tokens in `translated_text` using metadata in entity_map.

    Args:
      translated_text: text after translation (contains tokens like [ENTITY_xxx]).
      entity_map: mapping mask_id -> metadata dict (as produced earlier).
      strategy: how to restore. One of:
        - "orig": replace token with entity_map[id]['orig_text']
        - "orig_paren_translated": replace token with "<token_kept_in_translation> (orig_text)"
            (This will just insert orig_text in parentheses because we can't recover a translator-kept token
             unless you provide translated_replacement_map.)
        - "both": if you supply translated_replacement_map (mask_id -> translated_string),
            replace token with translated_replacement_map[id] + " (" + orig_text + ")"
        - "transliterate_orig": will transliterate orig_text using transliterate_fn before inserting.
        - "leave_mask": leave token as-is (no-op)
      transliterate_fn: optional function(orig_text)->transliterated_text (for telugu)
      translated_replacement_map: optional dict of mask_id -> replacement (if translator returned something for masked region)

    Returns:
      new_text with tokens replaced.
    """
    def _replacement(match):
        mid = match.group(1)
        mid_lower = mid.lower()
        meta = entity_map.get(mid_lower) or entity_map.get(mid)  
        if meta is None:
            
            
            return match.group(0)

        orig = meta.get("orig_text", "")
        action = meta.get("action", "")

        if strategy == "leave_mask":
            return match.group(0)

        if strategy == "orig":
            return orig

        if strategy == "transliterate_orig":
            if transliterate_fn is None:
                
                return orig
            return transliterate_fn(orig)

        if strategy == "orig_paren_translated":
            
            return f"({orig})"

        if strategy == "both":
            
            if translated_replacement_map and mid in translated_replacement_map:
                translated_piece = translated_replacement_map[mid]
                return f"{translated_piece} ({orig})"
            else:
                
                return f"({orig})"

        
        return orig

    
    restored = ENTITY_TOKEN_RE.sub(_replacement, translated_text)
    return restored

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


model_name = "Qwen/Qwen1.5-4B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True
)

def simplify_to_lay_medical(text: str) -> str:
    """
    Simplify medical text using Qwen with a refined prompt for consistency.
    Emphasizes active rewriting and examples to guide the model.
    """
    prompt = (
        f"You are a kind doctor simplifying medical info for patients who aren't experts. "
        f"Always rewrite in simple words, short sentences. Explain hard terms briefly like 'ossicles' as 'small bones in the ear'. "
        f"Use everyday language: say 'stomach pain' not 'abdominal ache', 'swollen' not just 'swollen'. "
        f"Make it friendly and clear. Change structure if needed. Example: 'Abdomen' -> 'belly'. "
        f"Original: {text}\nSimplified:"
    )
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,  
                temperature=0.8,     
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1  
            )
        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        
        if generated.startswith(text):
            generated = generated[len(text):].strip()
        return generated if generated and len(generated) > 10 else text
    except Exception as e:
        print(f"Simplification error: {e}")
        return text
import csv
import os

OUTPUT_FILE = "output.csv"


if not os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "expert_original",
            "simple_original",
            "e_to_s",
            "restored_telugu",
            "direct_telugu",
            "lay_telugu",
            "restored_hindi",
            "direct_hindi",
            "lay_hindi"
        ])

VERBATIM_KEEP = {
    "Medication", "Dosage", "Lab_value",
    "Mass", "Volume", "Weight", "Height", "Distance",
    "Frequency", "Duration", "Time", "Date", "Age", "Quantitative_concept", "Lab_value", "Mass", "Volume",
    "Weight", "Height", "Distance", "Area",
}

NORMALIZE = {
    "Disease_disorder", "Diagnostic_procedure",
    "Therapeutic_procedure", "Clinical_event", "Outcome",
    "Severity", "Biological_structure", "Biological_attribute"
}

TRANSLATE = {
    "Color", "Shape", "Texture", "Detailed_description", "Qualitative_concept",
    "Personal_background", "Occupation", "Family_history", "History", "Subject", "Coreference",
    "Nonbiological_location", "Subject", "Other_entity",
    "Other_event", "Other_entity", "Coreference",
    "Family_history", "History", "Personal_background", "Outcome", "Sign_symptom"
}

print("processing reports")
with open(OUTPUT_FILE, "a", newline='', encoding="utf-8") as f:
  for i in range(l):
    print(i)
    report = data[i]['Expert']
    simple = data[i]['Simple']
    ner_results = ner_pipeline(report)
    entities = []
    print(ner_results)
    for entity in ner_results:
      entities += [{"start": entity['start'], "end": entity['end'], "text": entity['word'], "type": entity['entity_group']}]
    augmented_text, entity_map = process_report_and_entities(report, entities)
    translated_text = translate_to_telugu(augmented_text)
    telugu_restored = restore_entities_from_ids(translated_text, entity_map)
    telugu_direct = translate_to_telugu(simple)
    e_to_s = simplify_to_lay_medical(report)
    tlay = translate_to_telugu(e_to_s)

    print("Original report:\n", report, "\n")
    print("Augmented report:\n", augmented_text, "\n")
    print("Automated English Summary: \n", e_to_s, "\n")
    print("Telugu Translated report: \n", translated_text, "\n")
    print("Telugu Restored report: \n", telugu_restored, "\n")
    print("Direct Lay Translation: \n", telugu_direct)
    print("Simple Lay Translation: \n", tlay)



    translated_text = translate_to_hindi(augmented_text)
    hindi_restored_text = restore_entities_from_ids(translated_text, entity_map)
    dltranslation = translate_to_hindi(simple)
    hlay = translate_to_hindi(e_to_s)


    print("Hindi Translated report: \n", translated_text, "\n")
    print("Hindi Restored report: \n", hindi_restored_text, "\n")
    print("Direct Lay Translation: \n", dltranslation)
    print("Simple Lay Translation: \n", hlay)

    writer = csv.writer(f)
    writer.writerow([
            report,
            simple,
            e_to_s,
            telugu_restored,
            telugu_direct,
            tlay,
            hindi_restored_text,
            dltranslation,
            hlay
        ])

from google.colab import files
files.download("output.csv")

