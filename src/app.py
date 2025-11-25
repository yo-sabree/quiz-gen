import os
import json
import random
import faiss
import numpy as np
import streamlit as st
import logging
import re
import math
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
NEW_QA_TARGET_RATIO = 0.8
MAX_SELECTION_MULTIPLIER = 1.0

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
def load_resources(subject):
    index_file = f"{subject}/faiss.index"
    metadata_file = f"{subject}/metadata.json"
    chapter_map_file = f"{subject}/chapter_map.json"
    
    try:
        index = faiss.read_index(index_file)
        with open(metadata_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        with open(chapter_map_file, "r", encoding="utf-8") as f:
            cmap = json.load(f)
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        logging.info(f"Resources for {subject} loaded successfully")
        return index, meta, cmap, model
    except Exception as e:
        logging.error(f"Failed to load resources for {subject}: {e}")
        return None, None, None, None

def get_resources_for_subject(subject):
    if subject == "Science":
        return load_resources("Science")
    elif subject == "Computer Science":
        return load_resources("Computer_Science")
    return None, None, None, None

def get_chunk_embedding(text, embedder):
    e = embedder.encode([text], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(e)
    return e[0]

def retrieve_context_for_id(chunk_id, faiss_index, metadata, embedder):
    base_text = metadata[chunk_id]["text"]
    vec = get_chunk_embedding(base_text, embedder)
    D, I = faiss_index.search(np.array([vec]), 4)
    results = []
    for idx in I[0]:
        if idx >= 0 and idx < len(metadata):
            results.append(metadata[idx]["text"])
    return "\n".join(results)

def select_30_chunks_covering_chapters(chapter_map, metadata):
    valid_chapters = {k: v for k, v in chapter_map.items() if v and len(v) > 0}
    chapters = list(valid_chapters.keys())
    selected = set()
    
    for ch in chapters:
        ids = valid_chapters[ch]
        if ids:
            valid_ids = [i for i in ids if i < len(metadata)]
            if valid_ids:
                selected.add(random.choice(valid_ids))
    
    target_size = MAX_QUIZ_SIZE
    
    remaining = target_size - len(selected)
    all_ids = list(range(len(metadata)))
    pool = [i for i in all_ids if i not in selected]
    
    if remaining > 0:
        if remaining >= len(pool):
            selected.update(pool)
        else:
            selected.update(random.sample(pool, remaining))
            
    lst = list(selected)
    
    if len(lst) > target_size:
        lst = random.sample(lst, target_size)
        
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

def build_batch_prompt(batch_data, difficulty, subject):
    prompt_context = ""
    for item in batch_data:
        prompt_context += f"--- START CONTEXT ID {item['id']} ---\n{item['context']}\n--- END CONTEXT ID {item['id']} ---\n\n"

    return f"""You are an expert {subject} examiner.
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
5. Create a new, unique question for each ID, even if similar concepts were questioned before.

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
    if "id" not in qa and "chunk_id" not in qa:
        return False
    if "question_text" not in qa or "choices" not in qa or "answer_index" not in qa:
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

def process_batch(batch_ids, difficulty, subject, faiss_index, metadata, embedder, cache, cache_path):
    
    batch_contexts = []
    results = []

    for cid in batch_ids:
        ctx = retrieve_context_for_id(cid, faiss_index, metadata, embedder)
        if len(ctx) > 3000: 
            ctx = ctx[:3000]
        batch_contexts.append({"id": cid, "context": ctx})
    
    if not batch_contexts:
        return results

    logging.info(f"Processing batch of {len(batch_contexts)} new questions for {subject} via Gemini API")
    prompt = build_batch_prompt(batch_contexts, difficulty, subject)
    
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
            
            original_id = int(qa.get("id", qa.get("chunk_id", -1)))
            if original_id == -1: continue 
            
            unique_key = f"{original_id}:{difficulty}:{random.randint(1000, 9999)}"
            
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
                "explanation": qa.get("explanation", "Reason not provided by AI."),
                "is_new": True
            }
            
            cache[unique_key] = final_obj
            results.append(final_obj)
            newly_generated_count += 1
        
        logging.info(f"Successfully generated and processed {newly_generated_count} items in this batch for {subject}.")

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
            
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse batch JSON response: {e}")
        logging.error(f"Raw response: {raw_resp[:500]}...")
    except Exception as e:
        logging.error(f"Error processing batch items: {e}")

    return results

def get_all_qas_batched(ids, difficulty, subject, faiss_index, metadata, embedder, batch_size=BATCH_SIZE):
    cache_path = f"{subject.replace(' ', '_').lower()}_qa_cache.json"
    try:
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        else:
            cache = {}
    except:
        cache = {}

    all_results = []
    
    target_new_qas = math.ceil(MAX_QUIZ_SIZE * NEW_QA_TARGET_RATIO) 
    
    if len(ids) < MAX_QUIZ_SIZE:
        logging.warning(f"Only {len(ids)} unique IDs available. Cannot guarantee {MAX_QUIZ_SIZE} questions.")
    
    
    if len(ids) >= target_new_qas:
        ids_for_generation = random.sample(ids, target_new_qas)
        logging.info(f"Generating {len(ids_for_generation)} mandatory new questions.")
        
        batches = [ids_for_generation[i:i + batch_size] for i in range(0, len(ids_for_generation), batch_size)]
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_batch, b, difficulty, subject, faiss_index, metadata, embedder, cache, cache_path) for b in batches]
            for f in as_completed(futures):
                res = f.result()
                if res:
                    all_results.extend(res)
    else:
        logging.warning(f"Not enough unique chunks ({len(ids)}) to meet the target of {target_new_qas} new questions. Using all available IDs for generation.")
        ids_for_generation = ids
        batches = [ids_for_generation[i:i + batch_size] for i in range(0, len(ids_for_generation), batch_size)]
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_batch, b, difficulty, subject, faiss_index, metadata, embedder, cache, cache_path) for b in batches]
            for f in as_completed(futures):
                res = f.result()
                if res:
                    all_results.extend(res)

    remaining_needed = MAX_QUIZ_SIZE - len(all_results)
    
    if remaining_needed > 0:
        logging.info(f"Generated {len(all_results)} questions. Filling {remaining_needed} remaining slots from cache.")
        
        cache_pool = [
            qa for key, qa in cache.items()
            if key.split(':')[1] == difficulty 
        ]
        
        if cache_pool:
            fill_count = min(remaining_needed, len(cache_pool))
            
            fill_qas = random.sample(cache_pool, fill_count)
            for qa in fill_qas:
                qa['is_new'] = False 
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
    if "subject" not in st.session_state:
        st.session_state["subject"] = "Science"

def render_intro():
    st.title("Quiz AI")
    
    with st.form("setup_form"):

        name = st.text_input("Enter Student Name")
        email = st.text_input("Enter Email Address (Optional)")
        subject = st.selectbox("Select Subject", ["Science", "Computer Science"])
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
                faiss_index, metadata, chapter_map, embedder = get_resources_for_subject(subject)
                
                if faiss_index is None:
                    st.error(f"Critical resources missing for {subject}. Please run ingestion for {subject} first.")
                    st.stop()
                    return
                
                st.session_state["name"] = name
                st.session_state["email"] = email
                st.session_state["grade"] = grade
                st.session_state["portion"] = portion
                st.session_state["difficulty"] = difficulty
                st.session_state["subject"] = subject
                st.session_state["faiss_index"] = faiss_index
                st.session_state["metadata"] = metadata
                st.session_state["chapter_map"] = chapter_map
                st.session_state["embedder"] = embedder

                with st.spinner(f"Generating {subject} Questions"):
                    selected_ids = select_30_chunks_covering_chapters(chapter_map, metadata)
                    st.session_state["quiz_data"] = get_all_qas_batched(
                        selected_ids, 
                        difficulty, 
                        subject, 
                        faiss_index, 
                        metadata, 
                        embedder, 
                        batch_size=BATCH_SIZE
                    )
                
                if not st.session_state["quiz_data"]:
                    st.error("Failed to generate any questions. Please check API key and logs.")
                else:
                    st.session_state["page"] = "quiz"
                    st.rerun()

def render_quiz():
    subject = st.session_state.get('subject', 'Quiz')
    st.title(f"{subject} Quiz for {st.session_state.get('name')}")
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
        new_qas = sum(1 for qa in st.session_state["quiz_data"] if qa.get("is_new", False))
        
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