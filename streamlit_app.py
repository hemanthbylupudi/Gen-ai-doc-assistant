import streamlit as st
import io
import pdfminer.high_level
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_option_menu import option_menu
from difflib import SequenceMatcher
import string


# ----------------------- Caching Models -----------------------
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

# ----------------------- CSS Styling -----------------------
st.markdown("""
    <style>
    .fade-in {
        animation: fadeIn 1s ease-in;
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
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
        summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
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

        if answer:
            return f"{answer} (context window matched)", best_window
        else:
            sentences = [s.strip() for s in best_window.split('.') if s.strip()]
            if sentences:
                question_embedding = embedder.encode(question)
                sentence_embeddings = embedder.encode(sentences)
                best_sentence = sentences[cosine_similarity([question_embedding], sentence_embeddings).argmax()]
                return f"Relevant info: {best_sentence}", best_window
        return "Could not find specific answer.", best_window
    except Exception as e:
        return f"Error: {e}", ""

def generate_all_chunks(text, chunk_size=1500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def generate_questions_from_chunk(chunk, num=3):
    if len(chunk.strip().split()) < 30:
        return []

    prompt = f"""Generate {num + 5} clear, specific, factual questions strictly based on the following content:

{chunk}

Rules:
- Only output questions ending with '?'
- No duplication
- No answers or explanations
"""

    try:
        outputs = generator(prompt, max_new_tokens=300, do_sample=True, top_k=60, temperature=0.9)
        raw = outputs[0]['generated_text'].split('\n')
        questions = []
        for line in raw:
            line = line.strip()
            if line.endswith('?') and len(line) > 10:
                questions.append(line)
                if len(questions) >= num:
                    break
        if len(questions) == 0:
            raw_questions = []
            for i in raw[3:]:
                if len(i.strip()) > 10:
                    raw_questions.append(i.strip())
                if len(raw_questions) >= num:
                    break
            return raw_questions
        return questions
    except Exception:
        return []

# ----------------------- Streamlit App -----------------------
st.set_page_config(page_title="Gen AI Assistant", layout="wide")
st.title("🧠 Gen AI Document Assistant")

uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

if uploaded_file:
    st.session_state.text = extract_text(uploaded_file)
    st.session_state.summary = safe_summarize(st.session_state.text)
    st.session_state.chunks = generate_all_chunks(st.session_state.text)
    st.session_state.chunk_index = 0
    st.session_state.questions = generate_questions_from_chunk(st.session_state.chunks[0], 3)

    st.subheader("📄 Summary")
    st.markdown(f'<div class="fade-in">{st.session_state.summary}</div>', unsafe_allow_html=True)

    mode = option_menu(None, ["Ask Anything", "Challenge Me"], icons=["question-circle", "target"],
                       default_index=0, orientation="horizontal",
                       styles={"nav-link-selected": {"background-color": "#6c63ff", "color": "white"}})

    if mode == "Ask Anything":
        st.subheader("❓ Ask a Question")
        question = st.text_input("Enter your question:")
        if question:
            answer, context = answer_question(st.session_state.text, question)
            st.markdown(f"**💡 Answer:** {answer}")
            st.markdown(f"<div class='source-box'>{context}</div>", unsafe_allow_html=True)

    elif mode == "Challenge Me":
        st.subheader("🎯 Comprehension Challenge")

        if st.session_state.questions:
            if st.button("🔄 Refresh Questions"):
                st.session_state.chunk_index += 1
                if st.session_state.chunk_index >= len(st.session_state.chunks):
                    st.session_state.chunk_index = 0
                st.session_state.questions = generate_questions_from_chunk(
                    st.session_state.chunks[st.session_state.chunk_index], 3
                )
                st.session_state.evaluations = [None] * 3

            answers = []
            all_answered = True
            for i, q in enumerate(st.session_state.questions):
                ans = st.text_input(f"Q{i+1}: {q}", key=f"qa_{i}")
                if not ans.strip():
                    all_answered = False
                answers.append(ans)

            if st.button("✅ Evaluate All"):
                if not all_answered:
                    st.warning("⚠️ Please answer all questions before evaluation.")
                else:
                    st.session_state.evaluations = []
                    for i, q in enumerate(st.session_state.questions):
                        user = answers[i].strip().lower().translate(str.maketrans('', '', string.punctuation))
                        gold, context = answer_question(st.session_state.text, q)
                        gold_clean = gold.strip().lower().translate(str.maketrans('', '', string.punctuation))
                        score = SequenceMatcher(None, user, gold_clean).ratio()
                        correct = user in gold_clean or gold_clean in user or score > 0.65
                        note = "High similarity" if correct else "Low similarity"
                        st.session_state.evaluations.append((answers[i], gold, note, score, context))

            if "evaluations" in st.session_state and st.session_state.evaluations:
                for i, evaluation in enumerate(st.session_state.evaluations):
                    if evaluation is None:
                        continue
                    user_ans, correct_ans, note, score, context = evaluation
                    with st.expander(f"Evaluation Q{i+1}"):
                        st.write(f"**Your Answer:** {user_ans}")
                        st.write(f"**Document Answer:** {correct_ans}")
                        st.write(f"**Result:** {'✅ Correct' if note=='High similarity' else '❌ Incorrect'}")
                        st.metric("Similarity", f"{score*100:.1f}%", note)
                        st.markdown(f"<div class='source-box'>{context}</div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Unable to generate valid questions from the document.")
