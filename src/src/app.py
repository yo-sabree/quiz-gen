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
MAX_QUIZ_SIZE = 30
BATCH_SIZE = 10

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
    
    remaining = MAX_QUIZ_SIZE - len(selected)
    all_ids = list(range(len(metadata)))
    pool = [i for i in all_ids if i not in selected]
    
    if remaining > 0:
        if remaining >= len(pool):
            selected.update(pool)
        else:
            selected.update(random.sample(pool, remaining))
            
    lst = list(selected)
    if len(lst) > MAX_QUIZ_SIZE:
        lst = random.sample(lst, MAX_QUIZ_SIZE)
        
    while len(lst) < MAX_QUIZ_SIZE and len(lst) < len(metadata):
        x = random.randint(0, len(metadata) - 1)
        if x not in lst:
            lst.append(x)
            
    random.shuffle(lst)
    return lst[:MAX_QUIZ_SIZE]

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

def is_valid_qa(qa):
    if "id" not in qa or "question_text" not in qa or "choices" not in qa or "answer_index" not in qa:
        return False
    if not isinstance(qa["choices"], list) or len(qa["choices"]) != 4:
        return False
    try:
        answer_index = int(qa["answer_index"])
        if not (0 <= answer_index < 4):
            return False
    except (ValueError, TypeError):
        return False
    return True

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

    logging.info(f"Processing batch of {len(missing_ids)} new questions via Gemini API (Batch {len(batch_ids)})")
    prompt = build_batch_prompt(batch_contexts, difficulty)
    
    raw_resp = call_gemini_api_safe(prompt)
    
    if not raw_resp:
        return results 

    cleaned_json = clean_json_string(raw_resp)
    
    try:
        qa_list = json.loads(cleaned_json)
        if isinstance(qa_list, dict): 
            qa_list = [qa_list]
            
        newly_generated_count = 0
        for qa in qa_list:
            if not is_valid_qa(qa):
                logging.warning(f"Skipping malformed QA object: {qa.get('id', 'Unknown ID')}")
                continue
            
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
            newly_generated_count += 1
        
        logging.info(f"Successfully generated and processed {newly_generated_count} items in this batch.")

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
            
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse batch JSON response: {e}")
        logging.error(f"Raw response: {raw_resp[:500]}...")
    except Exception as e:
        logging.error(f"Error processing batch items: {e}")

    return results

def get_all_qas_batched(ids, difficulty, batch_size=BATCH_SIZE):
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
                
    processed_ids = {qa['chunk_id'] for qa in all_results}

    remaining_needed = MAX_QUIZ_SIZE - len(all_results)
    
    if remaining_needed > 0:
        logging.info(f"Generated {len(all_results)} questions. Trying to fill {remaining_needed} from cache.")
        
        cache_pool = []
        for key, qa in cache.items():
            if key.endswith(f":{difficulty}") and qa['chunk_id'] not in processed_ids and is_valid_qa(qa):
                cache_pool.append(qa)
                
        if cache_pool:
            fill_count = min(remaining_needed, len(cache_pool))
            fill_qas = random.sample(cache_pool, fill_count)
            all_results.extend(fill_qas)
            logging.info(f"Filled {fill_count} questions from cache. Total questions: {len(all_results)}")
        else:
            logging.warning("No suitable questions found in cache to fill the quiz.")
            
    random.shuffle(all_results)
    return all_results[:MAX_QUIZ_SIZE]


def init_session():
    if "page" not in st.session_state:
        st.session_state["page"] = "intro"
    if "quiz_data" not in st.session_state:
        st.session_state["quiz_data"] = []
    if "score" not in st.session_state:
        st.session_state["score"] = 0

def render_intro():
    st.title("Science Quiz")
    
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
                st.session_state["difficulty"] = difficulty
                with st.spinner(f"Generating {MAX_QUIZ_SIZE} Questions"):
                    selected_ids = select_30_chunks_covering_chapters()
                    st.session_state["quiz_data"] = get_all_qas_batched(selected_ids, difficulty, batch_size=BATCH_SIZE)
                
                if not st.session_state["quiz_data"]:
                    st.error("Failed to generate any questions. Please check API key and logs.")
                else:
                    st.session_state["page"] = "quiz"
                    st.rerun()

def render_quiz():
    st.title(f"Science Quiz for {st.session_state.get('name')}")
    qas = st.session_state["quiz_data"]
    
    if not qas:
        st.error("No questions available. Try again.")
        if st.button("Back"):
            st.session_state["page"] = "intro"
            st.rerun()
        return

    st.info(f"Total Questions: {len(qas)}")
    
    with st.form("quiz_form"):
        for idx, qa in enumerate(qas):
            
            choices = qa.get("choices", [])
            if not choices or len(choices) < 4:
                st.error(f"Q{idx+1} is corrupted or incomplete and cannot be displayed.")
                continue

            st.markdown(f"### Q{idx+1}: {qa['question_text']}")
            st.radio(
                label="Choose:",
                options=choices,
                key=f"q_{idx}",
                label_visibility="collapsed"
            )
            st.markdown("---")
        
        submitted = st.form_submit_button("Submit Answers")
        
        if submitted:
            score = 0
            for idx, qa in enumerate(qas):
                choices = qa.get("choices", [])
                if not choices: continue 
                
                user_choice = st.session_state.get(f"q_{idx}")
                
                try:
                    correct_index = int(qa.get("answer_index", -1))
                except (ValueError, TypeError):
                    correct_index = -1
                    
                correct_choice = choices[correct_index] if 0 <= correct_index < len(choices) else None
                
                if user_choice is not None and user_choice == correct_choice:
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
            st.success("Excellent work! 🎉")
        elif percentage >= 50:
            st.warning("Good effort! Keep studying. 👍")
        else:
            st.error("Keep practicing! Let's review the answers. 📚")

    with st.expander("Review Answers"):
        options_letters = ["A", "B", "C", "D"]
        for i, qa in enumerate(st.session_state["quiz_data"]):
            
            choices = qa.get("choices", [])
            if not choices or len(choices) != 4:
                 st.markdown(f"**Q{i+1}: ERROR - Corrupted Question Data**")
                 st.markdown("---")
                 continue
                 
            try:
                correct_index = int(qa.get("answer_index", -1))
            except (ValueError, TypeError):
                correct_index = -1

            user_val = st.session_state.get(f"q_{i}")
            correct_choice_text = choices[correct_index] if 0 <= correct_index < len(choices) else "Error: Index Out of Bounds"
            
            color = "green" if user_val == correct_choice_text else "red"
            
            st.markdown(f"**Q{i+1}: {qa['question_text']}**")
            
            st.markdown(f":{color}[Your Answer: {user_val or 'Not Answered'}]")

            st.markdown("**All Options (Correct in Green):**")
            
            for j, choice in enumerate(choices):
                if j < len(options_letters):
                    display_text = f"{options_letters[j]}. {choice}"
                else:
                    display_text = f"Choice {j+1}. {choice}" 
                    
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