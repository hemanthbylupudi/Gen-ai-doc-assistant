import streamlit as st
import string
import io
import pdfminer.high_level
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_option_menu import option_menu
from difflib import SequenceMatcher


@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    qa_model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    generator = pipeline("text-generation", model="gpt2", device=-1)
    return summarizer, tokenizer, qa_model, generator

@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

summarizer, tokenizer, qa_model, generator = load_models()
embedder = load_embedder()

st.markdown("""
    <style>
    .fade-in {
        animation: fadeIn 1s ease-in;
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .stTextInput > div > div > input {
        transition: all 0.3s ease-in-out;
    }
    .source-box {
        background-color: #eeeeff;
        padding: 0.75rem;
        border-left: 5px solid #6c63ff;
        margin-top: 0.5rem;
        font-size: 0.9rem;
        font-style: italic;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)
# ----------------------- Helper Functions -----------------------
def extract_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return pdfminer.high_level.extract_text(io.BytesIO(uploaded_file.read()))
    elif uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    return "Unsupported file type."

def safe_summarize(text, max_chunk=1000):
    chunks = [text[i:i + max_chunk] for i in range(0, len(text), max_chunk)]
    summaries = []
    for chunk in chunks:
        input_words = chunk.split()
        if len(input_words) < 30:
            summaries.append(" ".join(input_words))
            continue
        dynamic_max = min(150, len(input_words))
        dynamic_min = max(20, min(50, int(len(input_words) * 0.3)))
        summary = summarizer(chunk, max_length=dynamic_max, min_length=dynamic_min, do_sample=False)
        summaries.append(summary[0]["summary_text"])
    return " ".join(" ".join(summaries).split()[:150])

def select_relevant_window(context, question, window_size=1024):
    windows = [context[i:i + window_size] for i in range(0, len(context), window_size // 2)]
    question_embedding = embedder.encode(question)
    window_embeddings = embedder.encode(windows)
    best_idx = cosine_similarity([question_embedding], window_embeddings).argmax()
    return windows[best_idx]

def answer_question(context, question):
    try:
        best_window = select_relevant_window(context, question)
        inputs = tokenizer(question, best_window, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            outputs = qa_model(**inputs)

        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits) + 1
        answer_tokens = inputs["input_ids"][0][start_idx:end_idx]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

        window_start = context.find(best_window)
        window_end = window_start + len(best_window)
        paragraphs = [p.strip() for p in context.split('\n') if p.strip()]
        section = None

        for i, p in enumerate(paragraphs):
            if best_window in p:
                section = i + 1
                break

        citation = (f"This information comes from {'section ' + str(section) if section else 'the document'} "
                   f"[spanning positions {window_start}-{window_end} in the text].")

        if answer and answer.lower() in context.lower():
            return f"{answer} {citation}", best_window
        else:
            sentences = [s.strip() for s in best_window.split('.') if s.strip()]
            if sentences:
                question_embedding = embedder.encode(question)
                sentence_embeddings = embedder.encode(sentences)
                best_sentence = sentences[cosine_similarity([question_embedding], sentence_embeddings).argmax()]
                return f"Relevant information: {best_sentence} ({citation})", best_window
            return f"Could not find specific answer."
    except Exception as e:
        return f"Error: {e}", ""

def generate_questions(document_text, num=3, offset=0):
    chunk = document_text[offset:offset + 1000]
    if len(chunk.strip().split()) < 30:
        return ["Text too short to generate questions."]

    prompt = f"""Based on the following content, generate {num} exam-style, clear and specific questions:

""" + chunk + """

Make sure:
- Questions are strictly based on the text
- No external facts
- No duplicates
- Each question ends with a '?'
- Provide only questions without explanations
"""

    try:
        outputs = generator(
            prompt,
            max_new_tokens=256,
            num_return_sequences=1,
            temperature=0.7,
            top_k=40,
            do_sample=True
        )
        raw = outputs[0]['generated_text'].split('\n')
        questions = []
        for line in raw:
            line = line.strip()
            if line.endswith("?") and line.lower() not in (q.lower() for q in questions):
                questions.append(line)
            if len(questions) >= num:  # Ensure we only take 'num' questions
                break

        return questions  # Return only the specified number of questions
    except Exception as e:
        return [f"Question generation failed: {e}"]

# ----------------------- Streamlit App -----------------------
try:
    st.set_page_config(page_title="Gen AI Assistant", layout="wide")
    st.title("🧠 Gen AI Document Assistant")

    uploaded_file = st.file_uploader("Upload a PDF or TXT document", type=["pdf", "txt"])

    if uploaded_file:
        with st.spinner("Processing your document..."):
            st.session_state.clear()
            st.session_state.last_uploaded_file = uploaded_file.name
            text = extract_text(uploaded_file)
            st.session_state.text = text
            st.session_state.summary = safe_summarize(text)
            st.session_state.questions = generate_questions(text, 3)  # Generate exactly 3 questions
            st.session_state.offset = 0 

        st.success("✅ File uploaded successfully!")

        st.subheader("📄 Document Summary")
        if "summary" in st.session_state:
            st.markdown(f'<div class="fade-in">{st.session_state.summary}</div>', unsafe_allow_html=True)

        selected_mode = option_menu(
            menu_title=None,
            options=["Ask Anything", "Challenge Me"],
            icons=["question-circle", "target"],
            menu_icon=None,
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"padding": "0", "background-color": "#fafafa"},
                "nav-link": {"font-size": "16px", "margin": "0", "color": "black"},
                "nav-link-selected": {"background-color": "#6c63ff", "color": "white"},
            }
        )

        if selected_mode == "Ask Anything":
            st.subheader("❓ Ask Anything")
            question = st.text_input("Ask a question based on the document:")
            if question:
                answer, source = answer_question(st.session_state.text, question)
                st.markdown(f'<div class="fade-in"><strong>💡 Answer:</strong> {answer}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="fade-in source-box">📌 Context: {source}</div>', unsafe_allow_html=True)

        elif selected_mode == "Challenge Me":
            st.subheader("🎯 Challenge Mode: Comprehension Test")
    
            if st.button("🔄 Refresh Questions"):
                st.session_state.offset += 1000
                if st.session_state.offset >= len(st.session_state.text):
                    st.session_state.offset = 0
                st.session_state.questions = generate_questions(st.session_state.text, 3, st.session_state.offset)
                if 'evaluations' in st.session_state:
                    del st.session_state.evaluations

            if "questions" in st.session_state:
                st.markdown("<div class='fade-in'>📜 Answer the following questions:</div>", unsafe_allow_html=True)

                if 'evaluations' not in st.session_state:
                    st.session_state.evaluations = [None] * len(st.session_state.questions)

                # Store user answers
                user_answers = []
        
                for i, q in enumerate(st.session_state.questions):
                    answer = st.text_input(f"Q{i + 1}: {q}", key=f"user_answer_{i}")
                    user_answers.append(answer)

                all_answered = all(answer.strip() for answer in user_answers)

                if st.button("✅ Evaluate All", disabled=not all_answered, type="primary"):
                    for i, q in enumerate(st.session_state.questions):
                        if user_answers[i].strip():
                            full_answer, context = answer_question(st.session_state.text, q)
                    
                            # Extract just the core answer and citation
                            if "This information comes from" in full_answer:
                                core_answer, citation = full_answer.split("This information comes from", 1)
                                core_answer = core_answer.strip()
                                citation = "This information comes from" + citation
                            else:
                                core_answer = full_answer
                                citation = "Source not specified in document"
                    
                            # Clean answers for comparison
                            clean_user = user_answers[i].lower().translate(str.maketrans('', '', string.punctuation)).strip()
                            clean_correct = core_answer.lower().translate(str.maketrans('', '', string.punctuation)).strip()
                    
                            # Smart evaluation logic
                            is_correct = False
                            evaluation_notes = ""
                    
                            if not clean_correct:
                                is_correct = False
                                evaluation_notes = "No definite answer could be found in the document."
                            elif clean_user == clean_correct:
                                is_correct = True
                                evaluation_notes = "Your answer exactly matches the documented information."
                            elif clean_user in clean_correct:
                                is_correct = True
                                evaluation_notes = "Your answer is contained within the documented information."
                            elif clean_correct in clean_user:
                                is_correct = True
                                evaluation_notes = "The documented information is contained within your answer."
                            elif any(word in clean_correct.split() for word in clean_user.split()):
                                evaluation_notes = "Your answer contains some matching keywords but isn't fully accurate."
                            else:
                                evaluation_notes = "Your answer doesn't match the documented information."
                    
                            st.session_state.evaluations[i] = {
                                'user_answer': user_answers[i],
                                'core_answer': core_answer,
                                'citation': citation,
                                'context': context,
                                'is_correct': is_correct,
                                'evaluation_notes': evaluation_notes,
                                'similarity_score': SequenceMatcher(None, clean_user, clean_correct).ratio()
                            }

                # Show detailed evaluations
                for i in range(len(st.session_state.questions)):
                    if st.session_state.evaluations[i] is not None:
                        eval_data = st.session_state.evaluations[i]
                
                        with st.expander(f"Detailed Evaluation for Q{i + 1}", expanded=True):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("### Your Answer")
                                st.info(eval_data['user_answer'])
                            with col2:
                                st.markdown("### Documented Answer")
                                st.info(eval_data['core_answer'])
                    
                            st.markdown("### Evaluation")
                            if eval_data['is_correct']:
                                st.success(f"✅ Correct - {eval_data['evaluation_notes']}")
                                st.metric("Similarity Score", 
                                        value=f"{eval_data['similarity_score']*100:.1f}%",
                                        delta="High match with document")
                            else:
                                st.error(f"❌ Incorrect - {eval_data['evaluation_notes']}")
                                st.metric("Similarity Score",
                                        value=f"{eval_data['similarity_score']*100:.1f}%",
                                        delta_color="inverse")
                    
                            st.markdown("### Source Information")
                            st.markdown(eval_data['citation'])
                    
                            st.markdown("### Relevant Context from Document")
                            st.markdown(f'<div class="source-box">{eval_data["context"]}</div>', unsafe_allow_html=True)

except Exception as e:
    import traceback
    st.error("🚨 Application Error:")
    st.code(traceback.format_exc())

