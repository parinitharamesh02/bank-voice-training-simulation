# api.py
from typing import Dict, Any, List, Optional
import uuid
import time

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pathlib import Path


from langchain_core.messages import HumanMessage, AIMessage

from config import OPENAI_API_KEY
from state import SimulationState
from scenarios import init_state_for_persona
from personas import PersonaId
from agents import simulation_app, MAX_TURNS, build_session_assessment
from voice_pipeline import transcribe_audio_file, synthesize_speech_to_file, save_upload_to_temp

app = FastAPI(title="Bank Voice Training Simulation")

# ---- CORS (so a small web demo can call this) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dev-only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- In-memory session store ----
SESSIONS: Dict[str, SimulationState] = {}


# ---------- Pydantic models ----------

class StartRequest(BaseModel):
    persona_id: PersonaId


class StartResponse(BaseModel):
    session_id: str
    persona: str
    difficulty: str
    context: str
    target_skills: List[str]
    mode: str
    starting_utterance: Optional[str] = None


class TextTurnRequest(BaseModel):
    session_id: str
    agent_text: str


class TextTurnResponse(BaseModel):
    session_id: str
    customer_reply: Optional[str]
    scores: Dict[str, float]
    coaching: str
    live_hint: str
    mode: str
    done: bool
    latency: Dict[str, float]


class AudioTurnResponse(BaseModel):
    session_id: str
    transcript: str
    customer_reply: Optional[str]
    scores: Dict[str, float]
    coaching: str
    live_hint: str
    mode: str
    done: bool
    latency: Dict[str, float]
    audio_path: str          # local path on disk (debug / screencast)
    audio_url: str           # browser URL like /audio/<filename>
    turn_log: List[Dict[str, Any]]
    latency_log: List[Dict[str, float]]
    cost_estimate: Dict[str, float]
    cost_log: List[Dict[str, float]]




class AssessmentResponse(BaseModel):
    session_id: str
    assessment: str


@app.get("/health")
def health():
    return {"status": "ok", "openai_key_loaded": bool(OPENAI_API_KEY)}


# ---------- Start a session ----------

@app.post("/session/start", response_model=StartResponse)
def start_session(req: StartRequest):
    state = init_state_for_persona(req.persona_id)

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = state

    return StartResponse(
        session_id=session_id,
        persona=state["persona"],
        difficulty=state["difficulty"],
        context=state["context"],
        target_skills=state["target_skills"],
        mode=state.get("mode", "normal"),
        starting_utterance=state.get("starting_utterance"),
    )


# ---------- Text turn (for debugging / non-voice) ----------

@app.post("/session/turn-text", response_model=TextTurnResponse)
def session_turn_text(req: TextTurnRequest):
    if req.session_id not in SESSIONS:
        raise ValueError("Unknown session_id")

    state = SESSIONS[req.session_id]

    if state.get("turn", 0) >= MAX_TURNS:
        done = True
        last_customer = _get_last_customer_message(state)
        return TextTurnResponse(
            session_id=req.session_id,
            customer_reply=last_customer,
            scores=state.get("evaluation", {}),
            coaching=state.get("coaching", ""),
            live_hint=state.get("live_hint", ""),
            mode=state.get("mode", "normal"),
            done=done,
            latency=state.get("latency", {}),
        )

    # Agent reply
    state["messages"].append(HumanMessage(content=req.agent_text))  # type: ignore[arg-type]
    state["turn"] = state.get("turn", 0) + 1

    # Measure LLM pipeline latency
    llm_start = time.time()
    state = simulation_app.invoke(state)
    llm_elapsed = time.time() - llm_start

    last_customer = _get_last_customer_message(state)
    scores = state.get("evaluation", {})
    coaching = state.get("coaching", "")
    live_hint = state.get("live_hint", "")
    mode = state.get("mode", "normal")

    # Update latency logs (no ASR/TTS here)
    latency = {"asr": 0.0, "llm": llm_elapsed, "tts": 0.0}
    state["latency"] = latency  # type: ignore[assignment]
    latency_log = state.get("latency_log") or []
    latency_log.append(latency)
    state["latency_log"] = latency_log  # type: ignore[assignment]

    # Turn log for assessment
    turn_log = state.get("turn_log") or []
    turn_log.append(
        {
            "turn": state.get("turn", 0),
            "agent": req.agent_text,
            "customer": last_customer or "",
            "scores": scores,
            "mode": mode,
        }
    )
    state["turn_log"] = turn_log  # type: ignore[assignment]

    SESSIONS[req.session_id] = state

    done = state.get("turn", 0) >= MAX_TURNS

    return TextTurnResponse(
        session_id=req.session_id,
        customer_reply=last_customer,
        scores=scores,
        coaching=coaching,
        live_hint=live_hint,
        mode=mode,
        done=done,
        latency=latency,
    )


# ---------- Audio turn (voice in + voice out) ----------

@app.post("/session/turn-audio", response_model=AudioTurnResponse)
async def session_turn_audio(
    session_id: str = Form(...),
    audio: UploadFile = File(...),
):
    if session_id not in SESSIONS:
        raise ValueError("Unknown session_id")

    state = SESSIONS[session_id]

    # If max turns reached, don't process further
    if state.get("turn", 0) >= MAX_TURNS:
        done = True
        last_customer = _get_last_customer_message(state)
        return AudioTurnResponse(
            session_id=session_id,
            transcript="",
            customer_reply=last_customer or "",
            scores=state.get("evaluation", {}),
            coaching=state.get("coaching", ""),
            live_hint=state.get("live_hint", ""),
            mode=state.get("mode", "normal"),
            done=done,
            latency=state.get("latency", {}),
            audio_path="",
            audio_url="",
        )

    # Save uploaded audio to a temp file and run ASR
    tmp_path = save_upload_to_temp(audio)
    transcript, asr_elapsed = transcribe_audio_file(tmp_path)

    # Append agent text
    state["messages"].append(HumanMessage(content=transcript))  # type: ignore[arg-type]
    state["turn"] = state.get("turn", 0) + 1

    # Run the simulation (customer + evaluator + coach + safety)
    llm_start = time.time()
    state = simulation_app.invoke(state)
    llm_elapsed = time.time() - llm_start

    last_customer = _get_last_customer_message(state)

    # TTS for customer reply
    tts_path = ""
    tts_elapsed = 0.0
    if last_customer:
        tts_path, tts_elapsed = synthesize_speech_to_file(
            last_customer,
            session_id=session_id,
            turn=state.get("turn", 0),
        )

        # Build latency dict
    latency = {"asr": asr_elapsed, "llm": llm_elapsed, "tts": tts_elapsed}
    state["latency"] = latency  # type: ignore[assignment]
    latency_log = state.get("latency_log") or []
    latency_log.append(latency)
    state["latency_log"] = latency_log  # type: ignore[assignment]

    # Turn log for assessment
    scores = state.get("evaluation", {})
    mode = state.get("mode", "normal")
    turn_log = state.get("turn_log") or []
    turn_log.append(
        {
            "turn": state.get("turn", 0),
            "agent": transcript,
            "customer": last_customer or "",
            "scores": scores,
            "mode": mode,
        }
    )
    state["turn_log"] = turn_log  # type: ignore[assignment]

    # Cost estimate for this turn + cost log
    cost_estimate = estimate_turn_cost(
        agent_text=transcript,
        customer_text=last_customer or "",
        latency=latency,
    )
    cost_log = state.get("cost_log") or []
    cost_log.append(cost_estimate)
    state["cost_log"] = cost_log  # type: ignore[assignment]

    SESSIONS[session_id] = state

    done = state.get("turn", 0) >= MAX_TURNS

    # Derive filename + audio_url for browser playback
    audio_filename = Path(tts_path).name if tts_path else ""
    audio_url = f"/audio/{audio_filename}" if audio_filename else ""


    return AudioTurnResponse(
        session_id=session_id,
        transcript=transcript,
        customer_reply=last_customer or "",
        scores=scores,
        coaching=state.get("coaching", ""),
        live_hint=state.get("live_hint", ""),
        mode=mode,
        done=done,
        latency=latency,
        audio_path=tts_path,
        audio_url=audio_url,
        turn_log=turn_log,
        latency_log=latency_log,
        cost_estimate=cost_estimate,
        cost_log=cost_log,
    )



# ---------- Optional: serve audio files back ----------

@app.get("/audio/{filename}")
def get_audio(filename: str):
    """
    Serve synthesized audio files so a simple HTML client can play them.
    """
    from pathlib import Path
    from voice_pipeline import AUDIO_OUT_DIR

    path = AUDIO_OUT_DIR / filename
    if not path.exists():
        return {"error": "file not found"}

    return FileResponse(path, media_type="audio/mpeg")


# ---------- End-of-session assessment ----------

@app.get("/session/assessment/{session_id}", response_model=AssessmentResponse)
def session_assessment(session_id: str):
    if session_id not in SESSIONS:
        raise ValueError("Unknown session_id")

    state = SESSIONS[session_id]
    assessment = build_session_assessment(state)
    return AssessmentResponse(session_id=session_id, assessment=assessment)

def estimate_turn_cost(
    agent_text: str,
    customer_text: str,
    latency: Dict[str, float],
) -> Dict[str, float]:
    """
    Very rough, demo-only cost estimate per turn.
    We approximate tokens from word count and multiply by tiny constants
    to show cost awareness. Numbers here are illustrative, not exact pricing.
    """
    def est_tokens(text: str) -> int:
        if not text:
            return 0
        # rough: ~1.3 tokens per word
        return int(len(text.split()) * 1.3)

    in_tokens = est_tokens(agent_text)
    out_tokens = est_tokens(customer_text)

    # Fake "prices" just for demo:
    # LLM: cheap model, per-token micro-costs
    llm_cost = in_tokens * 0.00000015 + out_tokens * 0.0000006

    # ASR / TTS: tiny per-second cost based on latency
    asr_cost = latency.get("asr", 0.0) * 0.000001
    tts_cost = latency.get("tts", 0.0) * 0.0000015

    total = llm_cost + asr_cost + tts_cost

    return {
        "asr": round(asr_cost, 6),
        "llm": round(llm_cost, 6),
        "tts": round(tts_cost, 6),
        "total": round(total, 6),
    }

# ---------- Helper ----------

def _get_last_customer_message(state: SimulationState) -> Optional[str]:
    from langchain_core.messages import BaseMessage

    messages: List[BaseMessage] = state.get("messages", [])  # type: ignore[assignment]
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg.content
    return None
