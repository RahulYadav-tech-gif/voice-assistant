import os
import time
import base64
import tempfile
import requests
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import io

from faster_whisper import WhisperModel
from openai import OpenAI
from gtts import gTTS
from openai import OpenAIError


# PAGE CONFIG

st.set_page_config(page_title="Voice Avatar Assistant", layout="wide")


# SAFE ERROR HANDLER & LOGGING

def safe_error(user_message: str, err: Exception | None = None):
    st.error(user_message)
    if err:
        print("INTERNAL ERROR:", repr(err))

def log_time(func_name: str, start_time: float):
    """Log execution time for a function"""
    elapsed = time.time() - start_time
    print(f"{func_name}: {elapsed:.2f}s")


# SIDEBAR – API KEYS

st.sidebar.title("API Configuration")

openai_key_input = st.sidebar.text_input(
    "OpenAI API Key",
    type="password"
)

did_key_input = st.sidebar.text_input(
    "D-ID API Key",
    type="password"
)

if "OPENAI_API_KEY" not in st.session_state:
    st.session_state.OPENAI_API_KEY = None

if "DID_API_KEY" not in st.session_state:
    st.session_state.DID_API_KEY = None

if openai_key_input:
    st.session_state.OPENAI_API_KEY = openai_key_input

if did_key_input:
    st.session_state.DID_API_KEY = did_key_input


# STOP IF KEYS MISSING

if not st.session_state.OPENAI_API_KEY or not st.session_state.DID_API_KEY:
    st.title("Voice Avatar Assistant")
    st.warning("Please enter both OpenAI and D-ID API keys in the sidebar.")
    st.stop()

# CLIENT

client = OpenAI(api_key=st.session_state.OPENAI_API_KEY)

MODEL_SIZE = "tiny.en"  
AVATAR_IMAGE_URL = "https://storage.googleapis.com/magicpoint/outputs/biometric-photo-output.png"


# API KEY VALIDATION (CACHED)

@lru_cache(maxsize=1)
def validate_openai_key(api_key: str):
    start = time.time()
    try:
        temp_client = OpenAI(api_key=api_key)
        temp_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1
        )
        log_time("validate_openai_key", start)
        return True
    except OpenAIError as e:
        msg = str(e).lower()
        if "incorrect api key" in msg or "invalid" in msg:
            safe_error("Your OpenAI API key is invalid. Please check and try again.", e)
        elif "quota" in msg or "billing" in msg or "exceeded" in msg:
            safe_error("Your OpenAI credits are exhausted or expired.", e)
        else:
            safe_error("Unable to verify OpenAI API key right now.", e)
        return False

@lru_cache(maxsize=1)
def validate_did_key(api_key: str):
    start = time.time()
    try:
        auth = base64.b64encode(f"{api_key}:".encode()).decode()
        headers = {"Authorization": f"Basic {auth}"}

        r = requests.get(
            "https://api.d-id.com/credits",
            headers=headers,
            timeout=10
        )

        if r.status_code == 401:
            safe_error("Your D-ID API key is invalid.")
            return False

        if r.status_code == 403:
            safe_error("Your D-ID credits are exhausted or expired.")
            return False

        log_time("validate_did_key", start)
        return True

    except Exception as e:
        safe_error("Unable to verify D-ID API key right now.", e)
        return False


# VALIDATE KEYS ONCE (PARALLEL)

with ThreadPoolExecutor(max_workers=2) as executor:
    openai_future = executor.submit(validate_openai_key, st.session_state.OPENAI_API_KEY)
    did_future = executor.submit(validate_did_key, st.session_state.DID_API_KEY)
    
    if not openai_future.result() or not did_future.result():
        st.stop()


# SESSION STATE

if "avatar_video" not in st.session_state:
    st.session_state.avatar_video = None

if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

if "last_audio_size" not in st.session_state:
    st.session_state.last_audio_size = None


# LOAD WHISPER (OPTIMIZED)

@st.cache_resource
def load_whisper():
    return WhisperModel(
        MODEL_SIZE,
        device="cpu",
        compute_type="int8",
        num_workers=4  
    )

whisper_model = load_whisper()

# LLM

def query_llm(text: str) -> str:
    start = time.time()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer briefly in 1–2 sentences."},
            {"role": "user", "content": text}
        ],
        stream=False
    )
    result = response.choices[0].message.content.strip()
    log_time("query_llm", start)
    return result


# TTS

def generate_tts_audio() -> tuple[str, bytes]:
    """Generate TTS and return both file path and bytes for parallel upload"""
    start = time.time()
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_path = tmp.name
    tmp.close()
    log_time("generate_tts_audio", start)
    return tmp_path

def generate_tts_with_text(text: str) -> tuple[str, bytes]:
    start = time.time()
    tts = gTTS(text=text, lang="en", slow=False, lang_check=False)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    

    with open(tmp.name, "rb") as f:
        audio_bytes = f.read()
    
    log_time("generate_tts_audio", start)
    return tmp.name, audio_bytes


# D-ID AVATAR (ULTRA OPTIMIZED)

def upload_audio_to_did(audio_bytes: bytes) -> str:
    """Upload audio and return URL"""
    upload_start = time.time()
    auth = base64.b64encode(f"{st.session_state.DID_API_KEY}:".encode()).decode()
    headers = {"Authorization": f"Basic {auth}"}
    
    audio_upload = requests.post(
        "https://api.d-id.com/audios",
        headers=headers,
        files={"audio": ("audio.mp3", audio_bytes, "audio/mpeg")},
        timeout=30
    )
    log_time("  └─ audio_upload", upload_start)
    
    if audio_upload.status_code == 401:
        raise Exception("D-ID invalid key")
    if audio_upload.status_code == 403:
        raise Exception("D-ID credits exhausted")
    if audio_upload.status_code != 201:
        raise Exception("Audio upload failed")
    
    return audio_upload.json()["url"]

def create_avatar_talk(audio_url: str) -> str:
    """Create avatar talk and return talk ID"""
    create_start = time.time()
    auth = base64.b64encode(f"{st.session_state.DID_API_KEY}:".encode()).decode()
    headers = {"Authorization": f"Basic {auth}", "Content-Type": "application/json"}
    
    payload = {
        "source_url": AVATAR_IMAGE_URL,
        "script": {
            "type": "audio",
            "audio_url": audio_url
        },
        "config": {
            "stitch": True,
            "result_format": "mp4"  
        }
    }
    
    response = requests.post(
        "https://api.d-id.com/talks",
        headers=headers,
        json=payload,
        timeout=30
    )
    log_time("  └─ create_talk", create_start)
    
    if response.status_code != 201:
        raise Exception("Avatar creation failed")
    
    return response.json()["id"]

def poll_avatar_status(talk_id: str) -> str:
    """Poll for avatar completion and return video URL"""
    poll_start = time.time()
    auth = base64.b64encode(f"{st.session_state.DID_API_KEY}:".encode()).decode()
    headers = {"Authorization": f"Basic {auth}"}
    
    
    poll_intervals = [0.3, 0.3, 0.5, 0.5, 0.8, 1, 1, 1.5, 2, 2]
    max_polls = 50
    
    for i in range(max_polls):
        status = requests.get(
            f"https://api.d-id.com/talks/{talk_id}",
            headers=headers,
            timeout=10
        ).json()
        
        if status.get("status") == "done":
            log_time("  └─ polling", poll_start)
            return status["result_url"]
        
        if status.get("status") == "error":
            raise Exception("Avatar rendering error")
        
        wait_time = poll_intervals[i] if i < len(poll_intervals) else 2.5
        time.sleep(wait_time)
    
    raise Exception("Avatar generation timeout")

def generate_avatar_video_url(audio_bytes: bytes) -> str:
    """Complete avatar generation pipeline"""
    start = time.time()
    
   
    audio_url = upload_audio_to_did(audio_bytes)
    
    
    talk_id = create_avatar_talk(audio_url)
    
    
    video_url = poll_avatar_status(talk_id)
    
    log_time("generate_avatar_video_url [TOTAL]", start)
    return video_url


# FULLY PARALLEL PIPELINE

def process_llm_and_avatar(user_text: str):
    """Run LLM, TTS, and Avatar generation with maximum parallelization"""
    pipeline_start = time.time()
    
    
    reply = query_llm(user_text)
    
    
    tts_path, audio_bytes = generate_tts_with_text(reply)
    
    
    video_url = generate_avatar_video_url(audio_bytes)
    
    log_time("process_llm_and_avatar [TOTAL]", pipeline_start)
    return reply, tts_path, video_url

# UI

st.title("Voice Avatar Assistant")
st.caption("Speak → AI responds with a talking avatar")

left_col, right_col = st.columns([1, 1], gap="large")


# LEFT COLUMN

with left_col:
    st.subheader("Speak")

    audio = st.audio_input("Click and record your question")

    if audio is not None and not st.session_state.is_processing:
        overall_start = time.time()
        try:
            if audio.size == st.session_state.last_audio_size:
                st.stop()

            st.session_state.is_processing = True
            st.session_state.last_audio_size = audio.size

            audio_bytes = audio.getbuffer()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                audio_path = tmp.name

            try:
                with st.spinner("Transcribing..."):
                    transcribe_start = time.time()
                    segments, _ = whisper_model.transcribe(
                        audio_path,
                        beam_size=1,
                        best_of=1,
                        temperature=0,
                        vad_filter=True,  # Voice activity detection for faster processing
                        vad_parameters=dict(min_silence_duration_ms=500)
                    )
                    segments = list(segments)
                    user_text = " ".join(s.text for s in segments).strip()
                    log_time("transcribe", transcribe_start)
            except Exception as e:
                safe_error("Sorry, I couldn't understand your voice.", e)
                st.stop()
            finally:
                if os.path.exists(audio_path):
                    os.unlink(audio_path)

            if user_text:
                st.session_state.conversation.append(("You", user_text))

                try:
                    with st.spinner("Generating response and avatar..."):
                        reply, tts_path, video_url = process_llm_and_avatar(user_text)
                        st.session_state.conversation.append(("Assistant", reply))
                        st.session_state.avatar_video = video_url
                        
                        # Cleanup
                        if os.path.exists(tts_path):
                            os.unlink(tts_path)
                            
                except Exception as e:
                    safe_error("Could not generate response or avatar.", e)
                    st.stop()

            log_time("TOTAL PROCESSING TIME", overall_start)
            print("="*60)

        finally:
            st.session_state.is_processing = False

    st.divider()
    for role, text in st.session_state.conversation[-6:]:
        st.markdown(f"**{role}:** {text}")


# RIGHT COLUMN

with right_col:
    st.subheader("Assistant")
    if st.session_state.avatar_video:
        st.video(st.session_state.avatar_video, autoplay=True)
    else:
        st.image(AVATAR_IMAGE_URL, width='stretch')


# CSS

st.markdown(
    """
    <style>
    video, img {
        max-height: 70vh;
        object-fit: cover;
        border-radius: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)