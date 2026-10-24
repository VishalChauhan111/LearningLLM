
import os
import tempfile
import threading

import gradio as gr
import ollama
import pyttsx3
from faster_whisper import WhisperModel

MODEL_NAME = "nemotron-3-nano:4b"

SYSTEM_PROMPT = (
    "You are a helpful local AI voice assistant. "
    "Answer naturally, clearly, and conversationally. "
    "Use simple English unless asked otherwise."
)

print("Loading local speech recognition model...")
stt_model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8",
)
print("Speech recognition loaded.")

tts_lock = threading.Lock()


def transcribe_audio(audio_path):
    if not audio_path:
        return ""

    segments, _ = stt_model.transcribe(
        audio_path,
        beam_size=1,
        vad_filter=True,
    )

    return " ".join(
        segment.text.strip() for segment in segments
    ).strip()


def speak_to_file(text):
    output_path = None

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    ) as temp_file:
        output_path = temp_file.name

    try:
        with tts_lock:
            engine = pyttsx3.init()
            engine.save_to_file(text, output_path)
            engine.runAndWait()
            engine.stop()

        if not os.path.exists(output_path):
            raise RuntimeError("Audio file was not created.")

        if os.path.getsize(output_path) == 0:
            raise RuntimeError("Audio file is empty.")

        return output_path

    except Exception:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise


def respond(text, audio_path, history):
    history = list(history or [])
    text = (text or "").strip()

    # Use typed text if supplied; otherwise transcribe audio.
    if not text and audio_path:
        try:
            text = transcribe_audio(audio_path)
        except Exception as error:
            raise gr.Error(
                f"Speech recognition failed: {error}"
            ) from error

    if not text:
        raise gr.Error("Type a message or record your voice.")

    history.append({"role": "user", "content": text})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    messages.extend(history[-12:])

    try:
        result = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            options={"num_ctx": 2048},
        )
        answer = (result.message.content or "").strip()

        if not answer:
            raise RuntimeError("Nemotron returned an empty answer.")

    except Exception as error:
        raise gr.Error(
            f"Nemotron request failed: {error}"
        ) from error

    history.append({"role": "assistant", "content": answer})

    try:
        audio_reply = speak_to_file(answer)
    except Exception as error:
        raise gr.Error(
            f"Text answer created, but speech failed: {error}"
        ) from error

    return history, "", None, audio_reply


with gr.Blocks(title="Local Nemotron Voice Assistant") as demo:
    gr.Markdown("# 🎙️ Local Nemotron Voice Assistant")
    gr.Markdown(
        "Speak or type. Nemotron, speech recognition, "
        "and speech output run locally."
    )

    chatbot = gr.Chatbot(
        type="messages",
        label="Conversation",
        height=420,
    )

    message = gr.Textbox(
        label="Type your message",
        placeholder="Ask Nemotron anything...",
        lines=2,
    )

    microphone = gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="Record a voice message",
    )

    send = gr.Button("Send", variant="primary")

    reply_audio = gr.Audio(
        label="Spoken response",
        autoplay=True,
    )

    send.click(
        respond,
        inputs=[message, microphone, chatbot],
        outputs=[chatbot, message, microphone, reply_audio],
    )

    message.submit(
        respond,
        inputs=[message, microphone, chatbot],
        outputs=[chatbot, message, microphone, reply_audio],
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
    )
