import os
import time
import requests
import streamlit as st
from faster_whisper import WhisperModel
from openai import OpenAI
from gtts import gTTS
from dotenv import load_dotenv
import tempfile
from audio_recorder import AudioRecorder

# -------------------------------
# LOAD ENV
# -------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DID_API_KEY = os.getenv("DID_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------------
# CONFIG
# -------------------------------
MODEL_SIZE = "small.en"
AVATAR_IMAGE_URL = "https://storage.googleapis.com/magicpoint/outputs/biometric-photo-output.png"

# -------------------------------
# SESSION STATE INIT
# -------------------------------
if "avatar_video" not in st.session_state:
    st.session_state.avatar_video = None  # None = show image

if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "recorder" not in st.session_state:
    st.session_state.recorder = AudioRecorder(
        silence_threshold=500,
        silence_duration=2.0
    )

if "is_recording" not in st.session_state:
    st.session_state.is_recording = False

if "recorded_audio_path" not in st.session_state:
    st.session_state.recorded_audio_path = None

if "last_check_time" not in st.session_state:
    st.session_state.last_check_time = 0

# -------------------------------
# INIT MODELS (CACHED)
# -------------------------------
@st.cache_resource
def load_whisper():
    return WhisperModel(
        MODEL_SIZE,
        device="cpu",
        compute_type="int8"
    )

whisper_model = load_whisper()

# -------------------------------
# LLM
# -------------------------------
def query_llm(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer briefly in 1–2 sentences."},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content.strip()

# -------------------------------
# TTS
# -------------------------------
def generate_tts_audio(text):
    tts = gTTS(text=text, lang="en")
    tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp_audio.name)
    return tmp_audio.name

# -------------------------------
# D-ID AVATAR
# -------------------------------
def generate_avatar_video_url(audio_path):
    audio_upload = requests.post(
        "https://api.d-id.com/audios",
        headers={"Authorization": f"Basic {DID_API_KEY}"},
        files={"audio": open(audio_path, "rb")}
    )

    if audio_upload.status_code != 201:
        raise Exception(audio_upload.text)

    audio_url = audio_upload.json()["url"]

    payload = {
        "source_url": AVATAR_IMAGE_URL,
        "script": {
            "type": "audio",
            "audio_url": audio_url
        },
        "config": {"stitch": True}
    }

    response = requests.post(
        "https://api.d-id.com/talks",
        headers={
            "Authorization": f"Basic {DID_API_KEY}",
            "Content-Type": "application/json"
        },
        json=payload
    )

    if response.status_code != 201:
        raise Exception(response.text)

    talk_id = response.json()["id"]

    while True:
        status = requests.get(
            f"https://api.d-id.com/talks/{talk_id}",
            headers={"Authorization": f"Basic {DID_API_KEY}"}
        ).json()

        if status["status"] == "done":
            return status["result_url"]

        if status["status"] == "error":
            raise Exception(status)

        time.sleep(1)

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Voice Avatar Assistant", layout="wide")

st.title("Voice Avatar Assistant")
st.caption("Speak → AI responds with a talking avatar")

left_col, right_col = st.columns([1, 1], gap="large")

# -------------------------------
# LEFT SIDE: RECORD + CHAT
# -------------------------------
with left_col:
    # Spacer to push record section downward
    st.markdown("<div style='height: 3.5rem;'></div>", unsafe_allow_html=True)

    # Recording button
    if not st.session_state.is_recording:
        if st.button("Start Recording", type="secondary", use_container_width=True):
            try:
                # Start recording
                st.session_state.is_recording = True
                st.session_state.recorder.start_recording()
                
                # Show recording status
                with st.spinner("Recording... Speak your question."):
                    # Wait for recording to complete (silence detected)
                    while st.session_state.recorder.is_currently_recording():
                        time.sleep(0.1)
                    
                    # Recording stopped, get the audio file
                    audio_path = st.session_state.recorder.stop_recording()
                
                st.session_state.is_recording = False
                
                if audio_path:
                    with st.spinner("Transcribing..."):
                        segments, _ = whisper_model.transcribe(audio_path)
                        user_text = " ".join(s.text for s in segments).strip()

                    if user_text:
                        st.session_state.conversation.append(("You", user_text))

                        with st.spinner("Thinking..."):
                            llm_reply = query_llm(user_text)

                        st.session_state.conversation.append(("Assistant", llm_reply))

                        with st.spinner("Generating avatar..."):
                            tts_audio = generate_tts_audio(llm_reply)
                            try:
                                video_url = generate_avatar_video_url(tts_audio)
                                st.session_state.avatar_video = video_url
                            except Exception as e:
                                st.warning(f"Avatar generation failed: {str(e)}")
                                st.audio(tts_audio)
                        
                        st.rerun()
                    else:
                        st.warning("No speech detected. Please try again.")
                else:
                    st.error("Recording failed. No audio file was created.")
                    
            except Exception as e:
                st.session_state.is_recording = False
                st.error(f"Recording error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    st.divider()


# -------------------------------
# RIGHT SIDE: AVATAR (FIXED, FULL VIEW)
# -------------------------------
with right_col:
    st.markdown("### Assistant")

    if st.session_state.avatar_video:
        st.video(
            st.session_state.avatar_video,
            autoplay=True,
            loop=False
        )
    else:
        st.image(
            AVATAR_IMAGE_URL,
            width="stretch"
        )

# -------------------------------
# CSS
# -------------------------------
st.markdown(
    """
    <style>
    /* Overall page padding */
    .block-container {
        padding-top: 2.8rem;
    }

    /* Bring Assistant section upward */
    [data-testid="column"]:nth-child(2) {
        padding-top: 0rem;
    }

    /* Reduce Assistant title spacing */
    h3 {
        margin-top: 0.2rem;
        margin-bottom: 0.6rem;
    }

    /* Avatar sizing – no scroll */
    video, img {
        max-height: 68vh;
        width: 100%;
        object-fit: contain;
        border-radius: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)




# import os
# import time
# import tempfile
# import requests
# import streamlit as st
# from faster_whisper import WhisperModel
# from openai import OpenAI
# from gtts import gTTS
# from dotenv import load_dotenv

# # -------------------------------
# # LOAD ENV
# # -------------------------------
# load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# DID_API_KEY = os.getenv("DID_API_KEY")

# client = OpenAI(api_key=OPENAI_API_KEY)

# # -------------------------------
# # CONFIG
# # -------------------------------
# MODEL_SIZE = "small.en"
# AVATAR_IMAGE_URL = "https://storage.googleapis.com/magicpoint/outputs/biometric-photo-output.png"

# # -------------------------------
# # SESSION STATE
# # -------------------------------
# if "avatar_video" not in st.session_state:
#     st.session_state.avatar_video = None

# if "conversation" not in st.session_state:
#     st.session_state.conversation = []

# # -------------------------------
# # LOAD WHISPER (CACHED)
# # -------------------------------
# @st.cache_resource
# def load_whisper():
#     return WhisperModel(
#         MODEL_SIZE,
#         device="cpu",
#         compute_type="int8"
#     )

# whisper_model = load_whisper()

# # -------------------------------
# # LLM
# # -------------------------------
# def query_llm(text: str) -> str:
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": "Answer briefly in 1–2 sentences."},
#             {"role": "user", "content": text}
#         ]
#     )
#     return response.choices[0].message.content.strip()

# # -------------------------------
# # TTS
# # -------------------------------
# def generate_tts_audio(text: str) -> str:
#     tts = gTTS(text=text, lang="en")
#     tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
#     tts.save(tmp.name)
#     return tmp.name

# # -------------------------------
# # D-ID AVATAR
# # -------------------------------
# def generate_avatar_video_url(audio_path: str) -> str:
#     audio_upload = requests.post(
#         "https://api.d-id.com/audios",
#         headers={"Authorization": f"Basic {DID_API_KEY}"},
#         files={"audio": open(audio_path, "rb")}
#     )

#     if audio_upload.status_code != 201:
#         raise Exception(audio_upload.text)

#     audio_url = audio_upload.json()["url"]

#     payload = {
#         "source_url": AVATAR_IMAGE_URL,
#         "script": {
#             "type": "audio",
#             "audio_url": audio_url
#         },
#         "config": {"stitch": True}
#     }

#     response = requests.post(
#         "https://api.d-id.com/talks",
#         headers={
#             "Authorization": f"Basic {DID_API_KEY}",
#             "Content-Type": "application/json"
#         },
#         json=payload
#     )

#     if response.status_code != 201:
#         raise Exception(response.text)

#     talk_id = response.json()["id"]

#     while True:
#         status = requests.get(
#             f"https://api.d-id.com/talks/{talk_id}",
#             headers={"Authorization": f"Basic {DID_API_KEY}"}
#         ).json()

#         if status["status"] == "done":
#             return status["result_url"]

#         if status["status"] == "error":
#             raise Exception(status)

#         time.sleep(1)

# # -------------------------------
# # UI
# # -------------------------------
# st.set_page_config(page_title="Voice Avatar Assistant", layout="wide")

# st.title("Voice Avatar Assistant")
# st.caption("Speak → AI responds with a talking avatar")

# left_col, right_col = st.columns([1, 1], gap="large")

# # -------------------------------
# # LEFT: MIC + CHAT
# # -------------------------------
# with left_col:
#     st.markdown("<div style='height: 3.5rem;'></div>", unsafe_allow_html=True)

#     audio_input = st.audio_input("🎤 Record your question")

#     if audio_input:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
#             tmp_wav.write(audio_input.read())
#             audio_path = tmp_wav.name

#         with st.spinner("Transcribing..."):
#             segments, _ = whisper_model.transcribe(audio_path)
#             user_text = " ".join(s.text for s in segments).strip()

#         if user_text:
#             st.session_state.conversation.append(("You", user_text))

#             with st.spinner("Thinking..."):
#                 reply = query_llm(user_text)

#             st.session_state.conversation.append(("Assistant", reply))

#             with st.spinner("Generating avatar..."):
#                 tts_audio = generate_tts_audio(reply)
#                 try:
#                     video_url = generate_avatar_video_url(tts_audio)
#                     st.session_state.avatar_video = video_url
#                 except Exception as e:
#                     st.warning(f"Avatar failed: {e}")
#                     st.audio(tts_audio)

#             st.rerun()
#         else:
#             st.warning("No speech detected. Try again.")

#     st.divider()

#     for role, msg in st.session_state.conversation[-6:]:
#         st.markdown(f"**{role}:** {msg}")

# # -------------------------------
# # RIGHT: AVATAR
# # -------------------------------
# with right_col:
#     st.markdown("### Assistant")

#     if st.session_state.avatar_video:
#         st.video(
#             st.session_state.avatar_video,
#             autoplay=True,
#             loop=False
#         )
#     else:
#         st.image(
#             AVATAR_IMAGE_URL,
#             width="stretch"
#         )

# # -------------------------------
# # CSS
# # -------------------------------
# st.markdown(
#     """
#     <style>
#     .block-container {
#         padding-top: 2.8rem;
#     }

#     h3 {
#         margin-top: 0.2rem;
#         margin-bottom: 0.6rem;
#     }

#     video, img {
#         max-height: 68vh;
#         width: 100%;
#         object-fit: contain;
#         border-radius: 16px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )




# import os
# import time
# import tempfile
# import requests
# import numpy as np
# import streamlit as st
# import av
# from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
# from faster_whisper import WhisperModel
# from openai import OpenAI
# from gtts import gTTS
# from dotenv import load_dotenv

# # -------------------------------
# # LOAD ENV
# # -------------------------------
# load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# DID_API_KEY = os.getenv("DID_API_KEY")

# client = OpenAI(api_key=OPENAI_API_KEY)

# # -------------------------------
# # CONFIG
# # -------------------------------
# MODEL_SIZE = "small.en"
# AVATAR_IMAGE_URL = "https://storage.googleapis.com/magicpoint/outputs/biometric-photo-output.png"
# SAMPLE_RATE = 16000

# # -------------------------------
# # SESSION STATE
# # -------------------------------
# if "avatar_video" not in st.session_state:
#     st.session_state.avatar_video = None

# if "conversation" not in st.session_state:
#     st.session_state.conversation = []

# if "audio_frames" not in st.session_state:
#     st.session_state.audio_frames = []

# # -------------------------------
# # WHISPER (CACHED)
# # -------------------------------
# @st.cache_resource
# def load_whisper():
#     return WhisperModel(
#         MODEL_SIZE,
#         device="cpu",
#         compute_type="int8"
#     )

# whisper_model = load_whisper()

# # -------------------------------
# # AUDIO PROCESSOR (WEBRTC)
# # -------------------------------
# class AudioProcessor(AudioProcessorBase):
#     def __init__(self):
#         self.frames = []

#     def recv_audio(self, frame):
#         print("🎧 audio frame received")
#         audio = frame.to_ndarray()
#         self.frames.append(audio)
#         return frame

# # -------------------------------
# # LLM
# # -------------------------------
# def query_llm(text: str) -> str:
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": "Answer briefly in 1–2 sentences."},
#             {"role": "user", "content": text}
#         ]
#     )
#     return response.choices[0].message.content.strip()

# # -------------------------------
# # TTS
# # -------------------------------
# def generate_tts_audio(text: str) -> str:
#     tts = gTTS(text=text, lang="en")
#     tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
#     tts.save(tmp.name)
#     return tmp.name

# # -------------------------------
# # D-ID AVATAR
# # -------------------------------
# def generate_avatar_video_url(audio_path: str) -> str:
#     audio_upload = requests.post(
#         "https://api.d-id.com/audios",
#         headers={"Authorization": f"Basic {DID_API_KEY}"},
#         files={"audio": open(audio_path, "rb")}
#     )

#     if audio_upload.status_code != 201:
#         raise Exception(audio_upload.text)

#     audio_url = audio_upload.json()["url"]

#     payload = {
#         "source_url": AVATAR_IMAGE_URL,
#         "script": {"type": "audio", "audio_url": audio_url},
#         "config": {"stitch": True}
#     }

#     response = requests.post(
#         "https://api.d-id.com/talks",
#         headers={
#             "Authorization": f"Basic {DID_API_KEY}",
#             "Content-Type": "application/json"
#         },
#         json=payload
#     )

#     talk_id = response.json()["id"]

#     while True:
#         status = requests.get(
#             f"https://api.d-id.com/talks/{talk_id}",
#             headers={"Authorization": f"Basic {DID_API_KEY}"}
#         ).json()

#         if status["status"] == "done":
#             return status["result_url"]

#         if status["status"] == "error":
#             raise Exception(status)

#         time.sleep(1)

# # -------------------------------
# # UI
# # -------------------------------
# st.set_page_config(page_title="Voice Avatar Assistant", layout="wide")

# st.title("Voice Avatar Assistant")
# st.caption("Speak → AI responds with a talking avatar")

# left_col, right_col = st.columns([1, 1], gap="large")

# # -------------------------------
# # LEFT: WEBRTC MIC
# # -------------------------------
# with left_col:
#     st.markdown("<div style='height: 3rem;'></div>", unsafe_allow_html=True)

#     ctx = webrtc_streamer(
#         key="voice",
#         audio_processor_factory=AudioProcessor,
#         media_stream_constraints={"audio": True, "video": False},
#         async_processing=True,
#     )

#     if ctx.state == "READY":
#         st.info("🎤 Click the microphone icon to start recording")

#     elif ctx.state == "PLAYING":
#         st.success("🎙 Listening... Speak now")

#     elif ctx.state == "STOPPED":
#         st.warning("Recording stopped")

#     can_process = ctx.state == "PLAYING"

#     if st.button("Stop & Process", use_container_width=True, disabled=not can_process):
#         if ctx.audio_processor is None:
#             st.warning("Microphone not started.")
#         elif not ctx.audio_processor.frames:
#             st.warning("No audio captured.")
#         else:
#             import soundfile as sf
#             audio = np.concatenate(ctx.audio_processor.frames, axis=1)

#             wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
#             sf.write(wav_file.name, audio.T, 16000)

#             ctx.audio_processor.frames.clear()

#             with st.spinner("Transcribing..."):
#                 segments, _ = whisper_model.transcribe(wav_file.name)
#                 user_text = " ".join(s.text for s in segments).strip()

#             if user_text:
#                 st.session_state.conversation.append(("You", user_text))

#                 with st.spinner("Thinking..."):
#                     reply = query_llm(user_text)

#                 st.session_state.conversation.append(("Assistant", reply))

#                 with st.spinner("Generating avatar..."):
#                     tts_audio = generate_tts_audio(reply)
#                     try:
#                         st.session_state.avatar_video = generate_avatar_video_url(tts_audio)
#                     except Exception as e:
#                         st.warning(f"Avatar failed: {e}")
#                         st.audio(tts_audio)

#                 st.rerun()


#     st.divider()

#     for role, msg in st.session_state.conversation[-6:]:
#         st.markdown(f"**{role}:** {msg}")

# # -------------------------------
# # RIGHT: AVATAR
# # -------------------------------
# with right_col:
#     st.markdown("### Assistant")

#     if st.session_state.avatar_video:
#         st.video(st.session_state.avatar_video, autoplay=True)
#     else:
#         st.image(AVATAR_IMAGE_URL, width="stretch")

# # -------------------------------
# # CSS
# # -------------------------------
# st.markdown(
#     """
#     <style>
#     video, audio {
#         display: none !important;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )
