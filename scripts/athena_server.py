#!/usr/bin/env python3
"""
athena_server.py — WebSocket Chat + Voice + Sound Server for Athena

Accepts text or audio from a browser client, routes through Athena's brain,
returns spoken responses as text + TTS audio. Can also play sounds (music,
animal sounds, etc.) on request.

Endpoints:
    ws://host:8765/chat    — text chat (send text, receive text + TTS audio)
    ws://host:8765/voice   — voice chat (send audio, receive audio)
    GET /status             — brain health, bio stats
    GET /                   — serve the chat UI

Usage:
    python scripts/athena_server.py                          # Default brain
    python scripts/athena_server.py --brain athena.brain     # Load saved brain
    python scripts/athena_server.py --port 8765              # Custom port

Requirements:
    pip install fastapi uvicorn websockets
    Optional: pip install piper-tts   (local TTS)
    Optional: espeak (system package, fallback TTS)
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger("athena_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn
except ImportError:
    print("ERROR: Install dependencies: pip install fastapi uvicorn websockets")
    sys.exit(1)

import nimcp
from claude_teacher import encode_text

# Suppress noisy logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =============================================================================
# TTS Engine
# =============================================================================

class TTSEngine:
    """Text-to-speech with piper (preferred) or espeak (fallback)."""

    def __init__(self):
        self.engine = self._detect_engine()
        logger.info("TTS engine: %s", self.engine or "none")

    def _detect_engine(self):
        """Find available TTS engine."""
        # Try piper first (local, fast, high quality)
        try:
            result = subprocess.run(["piper", "--version"], capture_output=True, timeout=5)
            if result.returncode == 0:
                return "piper"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Try espeak (widely available, lower quality)
        try:
            result = subprocess.run(["espeak", "--version"], capture_output=True, timeout=5)
            if result.returncode == 0:
                return "espeak"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return None

    def synthesize(self, text):
        """Convert text to WAV audio bytes. Returns None if no engine."""
        if not self.engine or not text.strip():
            return None

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name

        try:
            if self.engine == "piper":
                proc = subprocess.run(
                    ["piper", "--output_file", wav_path],
                    input=text.encode(), capture_output=True, timeout=30
                )
            elif self.engine == "espeak":
                proc = subprocess.run(
                    ["espeak", "-w", wav_path, text],
                    capture_output=True, timeout=15
                )
            else:
                return None

            if proc.returncode == 0 and os.path.exists(wav_path):
                with open(wav_path, "rb") as f:
                    return f.read()
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("TTS failed: %s", e)
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

        return None

    def synthesize_b64(self, text):
        """Synthesize and return base64-encoded WAV."""
        audio = self.synthesize(text)
        if audio:
            return base64.b64encode(audio).decode("ascii")
        return None


# =============================================================================
# Sound Effects Engine
# =============================================================================

class SoundEngine:
    """Generate or play sound effects (music, animal sounds, etc.)."""

    # Built-in sound descriptions mapped to espeak/piper or simple tone generation
    SOUND_KEYWORDS = {
        "cat": ("meow", "A cat meows softly"),
        "dog": ("woof woof", "A dog barks happily"),
        "bird": ("tweet tweet tweet", "Birds are singing"),
        "cow": ("moo", "A cow moos"),
        "duck": ("quack quack", "A duck quacks"),
        "rooster": ("cock a doodle doo", "A rooster crows at dawn"),
        "lion": ("roar", "A lion roars"),
        "bee": ("buzz buzz buzz", "Bees are buzzing"),
        "rain": ("pitter patter pitter patter", "Rain falls gently"),
        "wind": ("whoosh", "The wind blows"),
        "thunder": ("boom rumble", "Thunder rolls across the sky"),
        "laugh": ("ha ha ha", "Someone laughs joyfully"),
        "cry": ("wah wah", "A baby cries"),
        "music": ("la la la, do re mi fa sol", "Beautiful music plays"),
        "piano": ("ding dong ding", "A piano plays a melody"),
        "drum": ("boom boom tap tap boom", "Drums beat rhythmically"),
    }

    def __init__(self, tts_engine, sounds_dir=None):
        self.tts = tts_engine
        self.sounds_dir = Path(sounds_dir) if sounds_dir else None

    def identify_sound(self, text):
        """Check if the request is for a sound effect."""
        text_lower = text.lower()
        for keyword, (onomatopoeia, description) in self.SOUND_KEYWORDS.items():
            if keyword in text_lower:
                return keyword, onomatopoeia, description
        return None, None, None

    def generate_sound(self, keyword):
        """Generate audio for a sound effect.

        First checks for a WAV file in sounds_dir, then falls back to
        TTS-based onomatopoeia.
        """
        if keyword not in self.SOUND_KEYWORDS:
            return None, None

        onomatopoeia, description = self.SOUND_KEYWORDS[keyword]

        # Check for pre-recorded sound file
        if self.sounds_dir:
            for ext in (".wav", ".mp3", ".ogg"):
                path = self.sounds_dir / f"{keyword}{ext}"
                if path.exists():
                    with open(path, "rb") as f:
                        audio = f.read()
                    return base64.b64encode(audio).decode("ascii"), description

        # Fall back to TTS onomatopoeia
        audio_b64 = self.tts.synthesize_b64(onomatopoeia)
        return audio_b64, description


# =============================================================================
# Sensory Feature Composer (simplified for server)
# =============================================================================

class SimpleComposer:
    """Lightweight feature composer for the server."""

    def __init__(self, num_inputs):
        self.num_inputs = num_inputs

    def compose(self, text):
        """Convert text to feature vector using sentence-transformers."""
        try:
            emb = encode_text(text)
            # Tile 384-dim embedding to fill num_inputs
            import numpy as np
            tiled = np.tile(emb, (self.num_inputs // len(emb)) + 1)[:self.num_inputs]
            return tiled.tolist()
        except Exception:
            # Fallback: simple hash-based features
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            features = [0.0] * self.num_inputs
            for i, b in enumerate(h):
                idx = (b * 7 + i * 13) % self.num_inputs
                features[idx] = (b / 255.0) * 2.0 - 1.0
            return features


# =============================================================================
# Athena Brain Interface
# =============================================================================

class AthenaBrain:
    """Wraps nimcp brain with language production."""

    def __init__(self, brain_path=None, num_inputs=1024, num_outputs=2048,
                 neuron_count=5000):
        nimcp.init()
        if brain_path and os.path.exists(brain_path):
            logger.info("Loading brain from %s", brain_path)
            self.brain = nimcp.Brain.load(brain_path)
        else:
            logger.info("Creating new brain (inputs=%d, outputs=%d, neurons=%d)",
                        num_inputs, num_outputs, neuron_count)
            self.brain = nimcp.Brain(
                "athena", num_inputs=num_inputs, num_outputs=num_outputs,
                neuron_count=neuron_count, init_mode="fast"
            )

        self.composer = SimpleComposer(num_inputs)
        self.conversation_history = []

    def think_and_speak(self, text):
        """Process text input and generate spoken response.

        Returns (response_text, confidence, fluency, output_vector).
        """
        features = self.composer.compose(text)
        result = self.brain.decide_full(features)
        output_vec = result.get("output_vector", [])

        # Generate speech
        response_text = ""
        confidence = 0.0
        fluency = 0.0
        if output_vec:
            try:
                spoken = self.brain.speak(output_vec)
                response_text = spoken.get("text", "")
                confidence = spoken.get("confidence", 0.0)
                fluency = spoken.get("fluency", 0.0)
            except Exception as e:
                logger.warning("speak() failed: %s", e)

        self.conversation_history.append({
            "role": "user", "text": text,
            "timestamp": time.time()
        })
        self.conversation_history.append({
            "role": "athena", "text": response_text,
            "confidence": confidence, "fluency": fluency,
            "timestamp": time.time()
        })

        return response_text, confidence, fluency, output_vec

    def get_status(self):
        """Get brain health/stats."""
        try:
            probe = self.brain.probe()
            return probe
        except Exception:
            return {"status": "ok"}


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(title="Athena Brain Server")
athena = None  # Initialized in main()
tts = None
sound_engine = None


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve a simple chat UI."""
    return CHAT_HTML


@app.get("/status")
async def get_status():
    """Brain health endpoint."""
    if not athena:
        return JSONResponse({"error": "Brain not loaded"}, status_code=503)
    return JSONResponse(athena.get_status())


@app.websocket("/chat")
async def chat_endpoint(websocket: WebSocket):
    """Text chat over WebSocket."""
    await websocket.accept()
    logger.info("Chat client connected")

    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                msg = {"type": "text", "message": data}

            msg_type = msg.get("type", "text")
            message = msg.get("message", "").strip()

            if not message:
                await websocket.send_json({"type": "error", "message": "Empty message"})
                continue

            # Check if this is a sound request
            keyword, onomatopoeia, description = sound_engine.identify_sound(message)
            if keyword:
                audio_b64, desc = sound_engine.generate_sound(keyword)
                response = {
                    "type": "sound",
                    "text": desc or f"*{onomatopoeia}*",
                    "sound_name": keyword,
                }
                if audio_b64:
                    response["audio_b64"] = audio_b64
                await websocket.send_json(response)
                continue

            # Normal text processing
            response_text, confidence, fluency, _ = athena.think_and_speak(message)

            response = {
                "type": "response",
                "text": response_text or "(Athena is thinking...)",
                "confidence": confidence,
                "fluency": fluency,
            }

            # Add TTS audio if available
            if response_text and tts:
                audio_b64 = tts.synthesize_b64(response_text)
                if audio_b64:
                    response["audio_b64"] = audio_b64

            await websocket.send_json(response)

    except WebSocketDisconnect:
        logger.info("Chat client disconnected")


@app.websocket("/voice")
async def voice_endpoint(websocket: WebSocket):
    """Voice chat over WebSocket.

    Client sends audio bytes (WebM/PCM), server processes through
    Athena's brain and returns audio response.
    """
    await websocket.accept()
    logger.info("Voice client connected")

    try:
        while True:
            # Receive audio data
            audio_data = await websocket.receive_bytes()

            # For now, convert audio to text using a simple placeholder
            # In production, wire to speech_cortex_process or whisper
            text = "(voice input)"

            response_text, confidence, fluency, _ = athena.think_and_speak(text)

            response = {
                "type": "voice_response",
                "text": response_text or "(silence)",
                "confidence": confidence,
            }

            if response_text and tts:
                audio_b64 = tts.synthesize_b64(response_text)
                if audio_b64:
                    response["audio_b64"] = audio_b64

            await websocket.send_json(response)

    except WebSocketDisconnect:
        logger.info("Voice client disconnected")


# =============================================================================
# Chat UI HTML
# =============================================================================

CHAT_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Athena Chat</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: #1a1a2e;
            color: #e0e0e0;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        header {
            background: #16213e;
            padding: 16px 24px;
            border-bottom: 1px solid #0f3460;
        }
        header h1 { font-size: 1.5rem; color: #e94560; }
        header .subtitle { font-size: 0.85rem; color: #888; margin-top: 4px; }
        #chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .message {
            max-width: 75%;
            padding: 12px 16px;
            border-radius: 12px;
            line-height: 1.5;
        }
        .message.user {
            align-self: flex-end;
            background: #0f3460;
            border-bottom-right-radius: 4px;
        }
        .message.athena {
            align-self: flex-start;
            background: #1a1a3e;
            border: 1px solid #333;
            border-bottom-left-radius: 4px;
        }
        .message.sound {
            align-self: flex-start;
            background: #2a1a3e;
            border: 1px solid #553;
            border-bottom-left-radius: 4px;
            font-style: italic;
        }
        .message .meta {
            font-size: 0.75rem;
            color: #666;
            margin-top: 4px;
        }
        .message .play-btn {
            background: #e94560;
            color: white;
            border: none;
            padding: 4px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.8rem;
            margin-top: 6px;
        }
        .message .play-btn:hover { background: #ff6b81; }
        #input-area {
            display: flex;
            gap: 8px;
            padding: 16px 24px;
            background: #16213e;
            border-top: 1px solid #0f3460;
        }
        #msg-input {
            flex: 1;
            padding: 12px 16px;
            border-radius: 8px;
            border: 1px solid #333;
            background: #1a1a2e;
            color: #e0e0e0;
            font-size: 1rem;
            outline: none;
        }
        #msg-input:focus { border-color: #e94560; }
        #send-btn, #mic-btn {
            padding: 12px 20px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            font-size: 1rem;
        }
        #send-btn {
            background: #e94560;
            color: white;
        }
        #send-btn:hover { background: #ff6b81; }
        #mic-btn {
            background: #333;
            color: #e0e0e0;
        }
        #mic-btn.recording { background: #e94560; animation: pulse 1s infinite; }
        @keyframes pulse { 50% { opacity: 0.7; } }
        .typing { color: #888; font-style: italic; padding: 8px 16px; }
    </style>
</head>
<body>
    <header>
        <h1>Athena</h1>
        <div class="subtitle">Biological Neural Brain &mdash; Language Production Interface</div>
    </header>
    <div id="chat-container"></div>
    <div id="input-area">
        <input id="msg-input" type="text" placeholder="Talk to Athena... (try: 'make a cat sound' or 'play bird sounds')" autocomplete="off">
        <button id="mic-btn" title="Hold to speak">&#127908;</button>
        <button id="send-btn">Send</button>
    </div>

    <script>
        const chatContainer = document.getElementById('chat-container');
        const msgInput = document.getElementById('msg-input');
        const sendBtn = document.getElementById('send-btn');
        const micBtn = document.getElementById('mic-btn');

        // WebSocket connection
        const wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const ws = new WebSocket(`${wsProtocol}//${location.host}/chat`);

        ws.onopen = () => addMessage('system', 'Connected to Athena');
        ws.onclose = () => addMessage('system', 'Disconnected');
        ws.onerror = () => addMessage('system', 'Connection error');

        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            removeTyping();

            if (msg.type === 'sound') {
                addMessage('sound', msg.text, msg);
            } else {
                const meta = `confidence: ${(msg.confidence || 0).toFixed(2)}, fluency: ${(msg.fluency || 0).toFixed(2)}`;
                addMessage('athena', msg.text, msg, meta);
            }
        };

        function sendMessage() {
            const text = msgInput.value.trim();
            if (!text) return;
            addMessage('user', text);
            ws.send(JSON.stringify({ type: 'text', message: text }));
            msgInput.value = '';
            addTyping();
        }

        function addMessage(role, text, data, meta) {
            const div = document.createElement('div');
            div.className = `message ${role}`;
            div.textContent = text;

            if (meta) {
                const metaDiv = document.createElement('div');
                metaDiv.className = 'meta';
                metaDiv.textContent = meta;
                div.appendChild(metaDiv);
            }

            if (data && data.audio_b64) {
                const btn = document.createElement('button');
                btn.className = 'play-btn';
                btn.textContent = '\\u25B6 Play';
                btn.onclick = () => {
                    const audio = new Audio('data:audio/wav;base64,' + data.audio_b64);
                    audio.play();
                };
                div.appendChild(btn);
            }

            chatContainer.appendChild(div);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function addTyping() {
            const div = document.createElement('div');
            div.className = 'typing';
            div.id = 'typing-indicator';
            div.textContent = 'Athena is thinking...';
            chatContainer.appendChild(div);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function removeTyping() {
            const el = document.getElementById('typing-indicator');
            if (el) el.remove();
        }

        sendBtn.onclick = sendMessage;
        msgInput.onkeydown = (e) => { if (e.key === 'Enter') sendMessage(); };

        // Microphone support
        let mediaRecorder = null;
        let audioChunks = [];

        micBtn.onmousedown = async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
                mediaRecorder.onstop = () => {
                    const blob = new Blob(audioChunks, { type: 'audio/webm' });
                    // Send audio to voice endpoint (future: speech recognition)
                    addMessage('user', '(voice message)');
                    addTyping();
                    blob.arrayBuffer().then(buf => {
                        const voiceWs = new WebSocket(`${wsProtocol}//${location.host}/voice`);
                        voiceWs.onopen = () => voiceWs.send(buf);
                        voiceWs.onmessage = (event) => {
                            removeTyping();
                            const msg = JSON.parse(event.data);
                            addMessage('athena', msg.text, msg);
                            voiceWs.close();
                        };
                    });
                    stream.getTracks().forEach(t => t.stop());
                };
                mediaRecorder.start();
                micBtn.classList.add('recording');
            } catch (e) {
                addMessage('system', 'Microphone access denied');
            }
        };

        micBtn.onmouseup = () => {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
                micBtn.classList.remove('recording');
            }
        };
    </script>
</body>
</html>
"""


# =============================================================================
# Main
# =============================================================================

def main():
    global athena, tts, sound_engine

    parser = argparse.ArgumentParser(description="Athena WebSocket Chat Server")
    parser.add_argument("--brain", type=str, default=None,
                        help="Path to saved brain file")
    parser.add_argument("--port", type=int, default=8765,
                        help="Server port (default: 8765)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server host (default: 0.0.0.0)")
    parser.add_argument("--neurons", type=int, default=5000,
                        help="Neuron count for new brain (default: 5000)")
    parser.add_argument("--sounds-dir", type=str, default=None,
                        help="Directory with sound effect WAV files")
    args = parser.parse_args()

    # Initialize TTS
    tts = TTSEngine()

    # Initialize sound engine
    sound_engine = SoundEngine(tts, sounds_dir=args.sounds_dir)

    # Initialize brain
    athena = AthenaBrain(
        brain_path=args.brain,
        neuron_count=args.neurons,
    )

    logger.info("Starting Athena server on %s:%d", args.host, args.port)
    logger.info("Open http://localhost:%d in your browser", args.port)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
