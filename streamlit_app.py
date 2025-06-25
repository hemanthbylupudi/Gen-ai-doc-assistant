import streamlit as st
import io
import pdfminer.high_level
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------- Load Models -----------------------
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
        return answer if answer else "Could not find a clear answer."
    except Exception as e:
        return f"Error: {e}"

def generate_questions(document_text, num=5, offset=0):
    chunk = document_text[offset:offset + 2000]
    prompt = f"""
    Generate exactly {num} questions based ONLY on this document segment:
    {chunk}

    Rules:
    1. Questions must be answerable from the given text
    2. Never reference external knowledge
    3. No duplicate questions
    4. End every question with '?'
    5. No explanatory text - just questions
    """
    try:
        outputs = generator(
            prompt,
            max_new_tokens=100,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            top_k=50
        )
        raw_questions = outputs[0]['generated_text'].split('\n')
        questions = []
        for q in raw_questions:
            q = q.strip()
            if q.endswith('?') and q.lower() not in (qq.lower() for qq in questions):
                questions.append(q)
            if len(questions) >= num:
                break
        return questions or ["No suitable questions found in this section."]
    except Exception as e:
        return [f"Question generation failed: {e}"]

# ----------------------- Streamlit App -----------------------
try:
    st.title("🧠 Gen AI Document Assistant")

    uploaded_file = st.file_uploader("Upload a PDF or TXT document", type=["pdf", "txt"])

    if uploaded_file:
        if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
            st.session_state.clear()
            st.session_state.last_uploaded_file = uploaded_file.name
            text = extract_text(uploaded_file)
            st.session_state.text = text
            st.session_state.summary = safe_summarize(text)
            st.session_state.offset = 0
            st.session_state.questions = generate_questions(text, offset=0)
        else:
            text = st.session_state.text

        st.success("✅ File uploaded successfully!")

        st.subheader("📄 Document Summary")
        st.write(st.session_state.summary)

        mode = st.radio("Choose your interaction mode:", ["Ask Anything", "Challenge Me"])

        if mode == "Ask Anything":
            st.subheader("❓ Ask Anything")
            question = st.text_input("Ask a question based on the document:")
            if question:
                st.write("💡 Answer:", answer_question(st.session_state.text, question))

        elif mode == "Challenge Me":
            st.subheader("🎯 Challenge Mode: Comprehension Test")

            if st.button("🔄 Refresh Questions"):
                st.session_state.offset += 2000
                if st.session_state.offset >= len(st.session_state.text):
                    st.session_state.offset = 0
                st.session_state.questions = generate_questions(st.session_state.text, offset=st.session_state.offset)

            if "questions" in st.session_state:
                st.write("📜 Answer the following questions:")
                all_answered = True
                user_answers = []

                for i, q in enumerate(st.session_state.questions):
                    answer = st.text_input(f"Q{i + 1}: {q}", key=f"user_answer_{i}")
                    user_answers.append(answer)
                    if not answer.strip():
                        all_answered = False

                if all_answered:
                    if st.button("✅ Evaluate Answers"):
                        st.subheader("🧪 Evaluation Results:")
                        for i, q in enumerate(st.session_state.questions):
                            correct = answer_question(st.session_state.text, q)
                            user = user_answers[i]
                            st.markdown(f"**Q{i + 1}: {q}**")
                            st.markdown(f"🧠 **Your Answer**: {user}")
                            st.markdown(f"🔬 **Correct Answer**: {correct}")
                            if correct.lower() in user.lower():
                                st.success("✅ Your answer is close!")
                            else:
                                st.warning("❌ Not quite. Review the correct answer.")
                else:
                    st.info("ℹ️ Please answer all questions before clicking **Evaluate**.")
except Exception as e:
    import traceback
    st.error("🚨 Application Error:")
    st.code(traceback.format_exc())
