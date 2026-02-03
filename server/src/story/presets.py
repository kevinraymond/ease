"""Pre-authored story presets for demonstration and templates."""

from .schema import StoryScript, SceneDefinition, SceneTrigger, SceneTransition


# Example: Skiing Adventure Story (from the user's vision)
SKIING_ADVENTURE = StoryScript(
    name="skiing_adventure",
    description="A woman skiing down a mountain, with dynamic action during energetic music and peaceful vistas during calm sections",
    default_negative_prompt="cartoon, anime, illustration, blurry, distorted, text, watermark, low quality, deformed",
    scenes=[
        SceneDefinition(
            id="wide_shot",
            base_prompt="woman skiing down snowy mountain, wide aerial shot, beautiful winter landscape, professional photography",
            energy_high_prompt="woman skiing fast through forest slalom, snow spray, dynamic action, speed blur, intense motion",
            energy_low_prompt="woman paused on mountain peak, panoramic vista, peaceful moment, contemplative, golden hour",
            duration_frames=150,  # ~5 seconds at 30fps
            trigger=SceneTrigger.TIME,
            transition=SceneTransition.ZOOM_IN,
            transition_frames=45,
        ),
        SceneDefinition(
            id="action",
            base_prompt="woman skiing, medium shot, athletic pose, snow trails, mountain backdrop",
            energy_high_prompt="woman doing ski jump, airborne, dramatic pose, blue sky, snow particles, action shot",
            energy_low_prompt="woman gliding gracefully, gentle turns, powder snow, serene",
            duration_frames=120,  # ~4 seconds
            trigger=SceneTrigger.BEAT_COUNT,
            trigger_value=8,  # Transition after 8 beats
            beat_prompt_modifier="dynamic pose, action moment",
            transition=SceneTransition.CROSSFADE,
            transition_frames=30,
        ),
        SceneDefinition(
            id="closeup",
            base_prompt="woman skier portrait, looking at camera, warm smile, snowy background, goggles on forehead",
            energy_high_prompt="woman skier portrait, exhilarated expression, windswept hair, bright eyes, adrenaline",
            energy_low_prompt="woman skier portrait, peaceful smile, soft light, contemplative mood",
            duration_frames=90,  # ~3 seconds
            trigger=SceneTrigger.TIME,
            transition=SceneTransition.CUT,
        ),
    ],
    loop=True,
    audio_reactive_keywords=True,
)


# Example: Dancing Figure Story
DANCING_FIGURE = StoryScript(
    name="dancing_figure",
    description="A dancer responding to music, with poses becoming more dramatic during energetic sections",
    default_negative_prompt="blurry, distorted, multiple people, text, watermark, bad anatomy",
    scenes=[
        SceneDefinition(
            id="intro",
            base_prompt="elegant dancer, graceful pose, studio lighting, artistic portrait, dramatic shadows",
            energy_high_prompt="dancer in dynamic leap, dramatic pose, motion blur, powerful movement, athletic",
            energy_low_prompt="dancer in gentle pose, soft movement, meditative, eyes closed, peaceful",
            duration_frames=120,
            trigger=SceneTrigger.TIME,
            beat_prompt_modifier="dramatic gesture",
            transition=SceneTransition.CROSSFADE,
            transition_frames=30,
        ),
        SceneDefinition(
            id="movement",
            base_prompt="dancer mid-movement, flowing dress, beautiful form, theatrical lighting",
            energy_high_prompt="dancer spinning, fabric flying, explosive movement, peak action, intense",
            energy_low_prompt="dancer swaying gently, soft fabric flow, dreamy atmosphere",
            duration_frames=90,
            trigger=SceneTrigger.ENERGY_PEAK,
            trigger_value=0.75,  # Trigger on high energy
            transition=SceneTransition.ZOOM_IN,
            transition_frames=45,
        ),
        SceneDefinition(
            id="climax",
            base_prompt="dancer dramatic pose, arms extended, powerful stance, spotlight",
            energy_high_prompt="dancer triumphant pose, arms raised high, victorious, dramatic lighting, climactic",
            duration_frames=60,
            trigger=SceneTrigger.BEAT_COUNT,
            trigger_value=4,
            transition=SceneTransition.ZOOM_OUT,
            transition_frames=45,
        ),
    ],
    loop=True,
    audio_reactive_keywords=True,
)


# Example: Abstract Landscape Story
ABSTRACT_LANDSCAPE = StoryScript(
    name="abstract_landscape",
    description="Evolving abstract landscapes that respond to music mood",
    default_negative_prompt="text, watermark, frame, border, signature",
    scenes=[
        SceneDefinition(
            id="calm_waters",
            base_prompt="serene abstract landscape, calm waters, soft gradients, ethereal light, digital art",
            energy_high_prompt="turbulent abstract seascape, crashing waves, dynamic motion, intense colors",
            energy_low_prompt="still abstract pond, mirror reflection, peaceful, minimal, zen",
            duration_frames=180,
            trigger=SceneTrigger.TIME,
            transition=SceneTransition.CROSSFADE,
            transition_frames=60,
        ),
        SceneDefinition(
            id="mountains",
            base_prompt="abstract mountain range, geometric peaks, atmospheric perspective, surreal",
            energy_high_prompt="volcanic abstract peaks, fire and ice, dramatic contrasts, powerful",
            energy_low_prompt="misty abstract hills, soft layers, peaceful valleys, dreamlike",
            duration_frames=180,
            trigger=SceneTrigger.TIME,
            transition=SceneTransition.CROSSFADE,
            transition_frames=60,
        ),
        SceneDefinition(
            id="sky",
            base_prompt="abstract sky, flowing clouds, color gradients, vast expanse, contemplative",
            energy_high_prompt="stormy abstract sky, lightning, dramatic clouds, intense atmosphere",
            energy_low_prompt="sunset abstract sky, warm colors, peaceful, golden hour, tranquil",
            duration_frames=180,
            trigger=SceneTrigger.TIME,
            transition=SceneTransition.CROSSFADE,
            transition_frames=60,
        ),
    ],
    loop=True,
    audio_reactive_keywords=True,
)


# Example: Minimal Portrait Story (for testing)
MINIMAL_PORTRAIT = StoryScript(
    name="minimal_portrait",
    description="Simple portrait with minimal scene changes, good for testing",
    default_negative_prompt="blurry, text, watermark",
    scenes=[
        SceneDefinition(
            id="portrait",
            base_prompt="portrait photography, beautiful lighting, professional",
            energy_high_prompt="portrait, expressive, dynamic lighting, intense gaze",
            energy_low_prompt="portrait, peaceful, soft lighting, gentle expression",
            duration_frames=300,  # 10 seconds
            trigger=SceneTrigger.TIME,
            transition=SceneTransition.CUT,
        ),
    ],
    loop=True,
    audio_reactive_keywords=True,
)


# Registry of all preset stories
STORY_PRESETS = {
    "skiing_adventure": SKIING_ADVENTURE,
    "dancing_figure": DANCING_FIGURE,
    "abstract_landscape": ABSTRACT_LANDSCAPE,
    "minimal_portrait": MINIMAL_PORTRAIT,
}


def get_story_preset(name: str) -> StoryScript:
    """Get a story preset by name.

    Args:
        name: Preset name (skiing_adventure, dancing_figure, etc.)

    Returns:
        StoryScript instance (returns minimal_portrait if name not found)
    """
    return STORY_PRESETS.get(name, MINIMAL_PORTRAIT)


def list_story_presets() -> list[str]:
    """Get list of available preset names."""
    return list(STORY_PRESETS.keys())
