import os
import json
import random
import faiss
import numpy as np
import streamlit as st
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai.errors import APIError
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

try:
    if GEMINI_API_KEY is None:
        raise ValueError("GEMINI_API_KEY not found in .env file")

    client = genai.Client(api_key=GEMINI_API_KEY)
    logging.info("Gemini API Client initialized successfully.")

except Exception as e:
    logging.error(f"Failed to initialize Gemini Client: {e}")
    st.error("Gemini Client Initialization Failed. Check console logs.")
    client = None

@st.cache_resource
def load_resources():
    try:
        index = faiss.read_index("faiss.index")
        with open("metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        with open("chapter_map.json", "r", encoding="utf-8") as f:
            cmap = json.load(f)
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        logging.info("Resources loaded successfully")
        return index, meta, cmap, model
    except Exception as e:
        logging.error(f"Failed to load resources: {e}")
        return None, None, None, None

faiss_index, metadata, chapter_map, embedder = load_resources()

if faiss_index is None:
    st.error("Critical resources missing. Please run ingestion first.")
    st.stop()
    
if client is None:
    st.error("Cannot generate questions: Gemini API client could not be set up.")


def get_chunk_embedding(text):
    e = embedder.encode([text], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(e)
    return e[0]

def retrieve_context_for_id(chunk_id):
    base_text = metadata[chunk_id]["text"]
    vec = get_chunk_embedding(base_text)
    D, I = faiss_index.search(np.array([vec]), 4)
    results = []
    for idx in I[0]:
        if idx >= 0 and idx < len(metadata):
            results.append(metadata[idx]["text"])
    return "\n".join(results)

def select_30_chunks_covering_chapters():
    valid_chapters = {k: v for k, v in chapter_map.items() if v and len(v) > 0}
    chapters = list(valid_chapters.keys())
    selected = set()
    
    for ch in chapters:
        ids = valid_chapters[ch]
        if ids:
            valid_ids = [i for i in ids if i < len(metadata)]
            if valid_ids:
                selected.add(random.choice(valid_ids))
    
    remaining = 30 - len(selected)
    all_ids = list(range(len(metadata)))
    pool = [i for i in all_ids if i not in selected]
    
    if remaining > 0:
        if remaining >= len(pool):
            selected.update(pool)
        else:
            selected.update(random.sample(pool, remaining))
            
    lst = list(selected)
    if len(lst) > 30:
        lst = random.sample(lst, 30)
        
    while len(lst) < 30 and len(lst) < len(metadata):
        x = random.randint(0, len(metadata) - 1)
        if x not in lst:
            lst.append(x)
            
    random.shuffle(lst)
    return lst[:30]

def clean_json_string(text):
    try:
        text = text.strip()
        if text.startswith('```'):
            text = text.lstrip('```json').lstrip('```').strip()
            if text.endswith('```'):
                text = text.rstrip('```').strip()

        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            return match.group(0)
        match_obj = re.search(r'\{.*\}', text, re.DOTALL)
        if match_obj:
            return f"[{match_obj.group(0)}]" 
        return text
    except:
        return text

def build_batch_prompt(batch_data, difficulty):
    prompt_context = ""
    for item in batch_data:
        prompt_context += f"--- START CONTEXT ID {item['id']} ---\n{item['context']}\n--- END CONTEXT ID {item['id']} ---\n\n"

    return f"""You are an expert science examiner.
I have provided {len(batch_data)} text segments above, labeled by ID.

TASK:
Generate a JSON LIST containing exactly {len(batch_data)} objects.
For EACH Context ID provided, create one Multiple Choice Question (MCQ).
Difficulty: {difficulty}.

REQUIREMENTS:
1. Return ONLY the JSON Array [{{}}, {{}}]. DO NOT include any introductory or concluding text, or markdown code fences (```json, ```).
2. Each object must contain the 'id' matching the context.
3. 4 choices per question, 1 correct.
4. Ensure the ID in the JSON matches the Context ID used to generate it.

JSON STRUCTURE per item:
{{
"id": 123,
"question_text": "...",
"choices": ["A", "B", "C", "D"],
"answer_index": 0,
"answer_text": "Correct Option Text",
"source_excerpt": "Quote",
"explanation": "A concise, teaching explanation that sounds natural and human-like. Do not reference 'the text' or 'the passage'; instead, explain the concept directly."
}}

CONTEXTS:
{prompt_context}
"""

def call_gemini_api_safe(prompt):
    global client
    try:
        if client is None:
            logging.error("Gemini client not initialized.")
            return None
            
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
        )
        return response.text
    except APIError as e:
        logging.error(f"Gemini API Error (APIError): {e}")
        return None
    except Exception as e:
        logging.error(f"General error during Gemini API call: {e}")
        return None

def process_batch(batch_ids, difficulty, cache, cache_path):
    
    missing_ids = []
    batch_contexts = []
    results = []

    for cid in batch_ids:
        key = f"{cid}:{difficulty}"
        if key in cache:
            results.append(cache[key])
        else:
            missing_ids.append(cid)
            ctx = retrieve_context_for_id(cid)
            if len(ctx) > 3000: 
                ctx = ctx[:3000]
            batch_contexts.append({"id": cid, "context": ctx})
    
    if not missing_ids:
        return results

    logging.info(f"Processing batch of {len(missing_ids)} new questions via Gemini API (Batch 10)")
    prompt = build_batch_prompt(batch_contexts, difficulty)
    
    raw_resp = call_gemini_api_safe(prompt)
    
    if not raw_resp:
        return results 

    cleaned_json = clean_json_string(raw_resp)
    
    try:
        qa_list = json.loads(cleaned_json)
        if isinstance(qa_list, dict): 
            qa_list = [qa_list]
            
        for qa in qa_list:
            if "id" not in qa: continue
            
            original_id = int(qa["id"])
            
            final_obj = {
                "chunk_id": original_id,
                "book": metadata[original_id].get("book", "Unknown"),
                "page": metadata[original_id].get("page", 0),
                "chapter": metadata[original_id].get("chapter", "Unknown"),
                "question_text": qa.get("question_text", "Error"),
                "choices": qa.get("choices", []),
                "answer_index": int(qa.get("answer_index", 0)),
                "answer_text": qa.get("answer_text", ""),
                "source_excerpt": qa.get("source_excerpt", ""),
                "explanation": qa.get("explanation", "Reason not provided by AI.") 
            }
            
            cache[f"{original_id}:{difficulty}"] = final_obj
            results.append(final_obj)

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
            
    except json.JSONDecodeError:
        logging.error("Failed to parse batch JSON response")
    except Exception as e:
        logging.error(f"Error processing batch items: {e}")

    return results

def get_all_qas_batched(ids, difficulty, batch_size=10):
    cache_path = "qa_cache.json"
    try:
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        else:
            cache = {}
    except:
        cache = {}

    all_results = []
    batches = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
    
    logging.info(f"Split {len(ids)} items into {len(batches)} batches of {batch_size}")

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_batch, b, difficulty, cache, cache_path) for b in batches]
        for f in as_completed(futures):
            res = f.result()
            if res:
                all_results.extend(res)
                
    return all_results

def init_session():
    if "page" not in st.session_state:
        st.session_state["page"] = "intro"
    if "quiz_data" not in st.session_state:
        st.session_state["quiz_data"] = []
    if "score" not in st.session_state:
        st.session_state["score"] = 0

def render_intro():
    st.title("Science Quiz Generator (Batch 10)")
    
    with st.form("setup_form"):
        name = st.text_input("Enter Student Name")
        email = st.text_input("Enter Email Address (Optional)")
        grade = st.selectbox("Select Grade Level", list(range(1, 11)))
        portion = st.selectbox("Select Exam Portion", ["Term 1", "Term 2", "Term 3", "Complete"])
        
        difficulty = st.selectbox("Select Exam Type (Difficulty)", ["Easy", "Medium", "Hard"])
        
        submitted = st.form_submit_button("Start Quiz")
        
        if submitted:
            if not name:
                st.warning("Please enter a name.")
            elif client is None:
                 st.error("Cannot start quiz: Gemini API client is not initialized.")
            else:
                st.session_state["name"] = name
                st.session_state["email"] = email
                st.session_state["grade"] = grade
                st.session_state["portion"] = portion
                st.session_state["difficulty"] = difficulty # Used by Gemini
                with st.spinner("Generating 30 Questions... (Batch Size 10)"):
                    selected_ids = select_30_chunks_covering_chapters()
                    st.session_state["quiz_data"] = get_all_qas_batched(selected_ids, difficulty, batch_size=10)
                st.session_state["page"] = "quiz"
                st.rerun()

def render_quiz():
    st.title(f"Quiz for {st.session_state.get('name')}")
    qas = st.session_state["quiz_data"]
    
    if not qas:
        st.error("No questions generated. Try again.")
        if st.button("Back"):
            st.session_state["page"] = "intro"
            st.rerun()
        return

    with st.form("quiz_form"):
        for idx, qa in enumerate(qas):
            st.markdown(f"### Q{idx+1}: {qa['question_text']}")
            st.radio(
                label="Choose:",
                options=qa["choices"],
                key=f"q_{idx}",
                label_visibility="collapsed"
            )
            st.markdown("---")
        
        submitted = st.form_submit_button("Submit Answers")
        
        if submitted:
            score = 0
            for idx, qa in enumerate(qas):
                user_choice = st.session_state.get(f"q_{idx}")
                correct_choice = qa["choices"][qa["answer_index"]] if qa["choices"] else ""
                if user_choice == correct_choice:
                    score += 1
            
            st.session_state["score"] = score
            st.session_state["page"] = "result"
            st.rerun()

def render_result():
    st.title("Quiz Results")
    score = st.session_state["score"]
    total = len(st.session_state["quiz_data"])
    
    st.metric("Final Score", f"{score} / {total}")
    
    if total > 0:
        percentage = (score / total) * 100
        if percentage >= 80:
            st.success("Excellent work!")
        elif percentage >= 50:
            st.warning("Good effort!")
        else:
            st.error("Keep practicing!")

    with st.expander("Review Answers"):
        options_letters = ["A", "B", "C", "D"]
        for i, qa in enumerate(st.session_state["quiz_data"]):
            user_val = st.session_state.get(f"q_{i}")
            correct_index = qa["answer_index"]
            correct_choice_text = qa["choices"][correct_index] if qa["choices"] else "Error"
            
            color = "green" if user_val == correct_choice_text else "red"
            
            st.markdown(f"**Q{i+1}: {qa['question_text']}**")
            
            st.markdown(f":{color}[Your Answer: {user_val}]")

            st.markdown("**All Options (Correct in Green):**")
            
            for j, choice in enumerate(qa["choices"]):
                display_text = f"{options_letters[j]}. {choice}"
                if j == correct_index:
                    st.markdown(f"- :green[**{display_text}**]")
                else:
                    st.markdown(f"- {display_text}")

            st.markdown(f"**Reason:** {qa.get('explanation', 'Reason not available.')}")
            st.markdown(f"*Source: {qa['book']} (Ch: {qa['chapter']})*")
            st.markdown("---")

    if st.button("Start New Quiz"):
        st.session_state.clear()
        st.session_state["page"] = "intro"
        st.rerun()

def main():
    init_session()
    if st.session_state["page"] == "intro":
        render_intro()
    elif st.session_state["page"] == "quiz":
        render_quiz()
    elif st.session_state["page"] == "result":
        render_result()

if __name__ == "__main__":
    main()