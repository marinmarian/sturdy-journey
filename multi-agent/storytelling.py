import logging
from dataclasses import dataclass
from dotenv import load_dotenv

from livekit import api
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ChatContext,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    # Audio imports
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
)
from livekit.agents.llm import function_tool
from livekit.plugins import openai, silero

logger = logging.getLogger("multi-agent")
load_dotenv()


@dataclass
class StoryData:
    name: str | None = None
    location: str | None = None
    genre: str | None = None
    problem: str | None = None


# --- Specialized Genre Agents ---


class ScifiAgent(Agent):
    def __init__(self, data: StoryData, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=(
                f"You are a Sci-Fi AI narrator. The hero is {data.name} on {data.location}. "
                f"The crisis: {data.problem}. Use a cinematic, tech-heavy tone. "
                "Keep the story interactive by asking the user what they do next."
            ),
            llm=openai.realtime.RealtimeModel(voice="alloy"),
            chat_ctx=chat_ctx,
        )

    async def on_enter(self):
        self.session.generate_reply()


class FantasyAgent(Agent):
    def __init__(self, data: StoryData, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=(
                f"You are a High Fantasy Bard. The hero is {data.name} in the realm of {data.location}. "
                f"The quest: {data.problem}. Use whimsical, epic language. "
                "Ask the user how they wish to proceed at key moments."
            ),
            llm=openai.realtime.RealtimeModel(voice="shimmer"),
            chat_ctx=chat_ctx,
        )

    async def on_enter(self):
        self.session.generate_reply()


class NoirAgent(Agent):
    def __init__(self, data: StoryData, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=(
                f"You are a gritty Noir Detective. The lead is {data.name} in the shadows of {data.location}. "
                f"The case: {data.problem}. Speak in short, punchy sentences with 1940s slang. "
                "Let the user make the tough choices."
            ),
            llm=openai.realtime.RealtimeModel(voice="onyx"),
            chat_ctx=chat_ctx,
        )

    async def on_enter(self):
        self.session.generate_reply()


# --- The Orchestrator ---


class IntroAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Your name is Echo. You are a story architect. "
                "Your goal is to gather four details: 1. Name, 2. Setting, 3. Genre (Sci-Fi, Fantasy, or Noir), "
                "and 4. The main problem/conflict. Ask for these details one or two at a time "
                "to keep it conversational. Once you have all four, call 'begin_adventure'."
            ),
        )

    async def on_enter(self):
        self.session.generate_reply()

    @function_tool
    async def begin_adventure(
        self,
        context: RunContext[StoryData],
        name: str,
        location: str,
        genre: str,
        problem: str,
    ):
        """Starts the story once the user has provided name, location, genre, and the problem."""
        context.userdata.name = name
        context.userdata.location = location
        context.userdata.genre = genre.lower()
        context.userdata.problem = problem

        logger.info(f"Handoff to {genre} for {name}")

        if "sci" in genre.lower():
            return ScifiAgent(context.userdata)
        elif "noir" in genre.lower():
            return NoirAgent(context.userdata)
        else:
            return FantasyAgent(context.userdata)


# --- Server Logic ---


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server = AgentServer(setup_fnc=prewarm)


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # 1. Setup the session
    session = AgentSession[StoryData](
        vad=ctx.proc.userdata["vad"],
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=openai.STT(),
        tts=openai.TTS(voice="echo"),
        userdata=StoryData(),
    )

    # 2. Start the session (connects to the room)
    await session.start(agent=IntroAgent(), room=ctx.room)

    # 3. Start background audio AFTER connection
    # KEYBOARD_TYPING creates the 'writing' effect while the AI thinks
    bg_audio = BackgroundAudioPlayer(
        thinking_sound=[AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.5)]
    )
    await bg_audio.start(room=ctx.room, agent_session=session)


if __name__ == "__main__":
    cli.run_app(server)
