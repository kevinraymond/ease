/**
 * Story preset definitions for the editor.
 * These mirror the presets in server/src/story/presets.py
 */

import { StoryConfig, SceneDefinition } from '../../core/types';

// Helper to create a scene with defaults
function scene(partial: Partial<SceneDefinition> & { id: string; basePrompt: string }): SceneDefinition {
  return {
    id: partial.id,
    basePrompt: partial.basePrompt,
    negativePrompt: partial.negativePrompt,
    durationFrames: partial.durationFrames ?? 120,
    trigger: partial.trigger ?? 'time',
    triggerValue: partial.triggerValue ?? 0,
    energyHighPrompt: partial.energyHighPrompt,
    energyLowPrompt: partial.energyLowPrompt,
    energyBlendRange: partial.energyBlendRange ?? [0.3, 0.6],
    beatPromptModifier: partial.beatPromptModifier,
    transition: partial.transition ?? 'crossfade',
    transitionFrames: partial.transitionFrames ?? 30,
  };
}

export const SKIING_ADVENTURE: StoryConfig = {
  name: 'skiing_adventure',
  description: 'A woman skiing down a mountain, with dynamic action during energetic music and peaceful vistas during calm sections',
  defaultNegativePrompt: 'cartoon, anime, illustration, blurry, distorted, text, watermark, low quality, deformed',
  scenes: [
    scene({
      id: 'wide_shot',
      basePrompt: 'woman skiing down snowy mountain, wide aerial shot, beautiful winter landscape, professional photography',
      energyHighPrompt: 'woman skiing fast through forest slalom, snow spray, dynamic action, speed blur, intense motion',
      energyLowPrompt: 'woman paused on mountain peak, panoramic vista, peaceful moment, contemplative, golden hour',
      durationFrames: 150,
      trigger: 'time',
      transition: 'zoom_in',
      transitionFrames: 45,
    }),
    scene({
      id: 'action',
      basePrompt: 'woman skiing, medium shot, athletic pose, snow trails, mountain backdrop',
      energyHighPrompt: 'woman doing ski jump, airborne, dramatic pose, blue sky, snow particles, action shot',
      energyLowPrompt: 'woman gliding gracefully, gentle turns, powder snow, serene',
      durationFrames: 120,
      trigger: 'beat_count',
      triggerValue: 8,
      beatPromptModifier: 'dynamic pose, action moment',
      transition: 'crossfade',
      transitionFrames: 30,
    }),
    scene({
      id: 'closeup',
      basePrompt: 'woman skier portrait, looking at camera, warm smile, snowy background, goggles on forehead',
      energyHighPrompt: 'woman skier portrait, exhilarated expression, windswept hair, bright eyes, adrenaline',
      energyLowPrompt: 'woman skier portrait, peaceful smile, soft light, contemplative mood',
      durationFrames: 90,
      trigger: 'time',
      transition: 'cut',
    }),
  ],
  loop: true,
  audioReactiveKeywords: true,
};

export const DANCING_FIGURE: StoryConfig = {
  name: 'dancing_figure',
  description: 'A dancer responding to music, with poses becoming more dramatic during energetic sections',
  defaultNegativePrompt: 'blurry, distorted, multiple people, text, watermark, bad anatomy',
  scenes: [
    scene({
      id: 'intro',
      basePrompt: 'elegant dancer, graceful pose, studio lighting, artistic portrait, dramatic shadows',
      energyHighPrompt: 'dancer in dynamic leap, dramatic pose, motion blur, powerful movement, athletic',
      energyLowPrompt: 'dancer in gentle pose, soft movement, meditative, eyes closed, peaceful',
      durationFrames: 120,
      trigger: 'time',
      beatPromptModifier: 'dramatic gesture',
      transition: 'crossfade',
      transitionFrames: 30,
    }),
    scene({
      id: 'movement',
      basePrompt: 'dancer mid-movement, flowing dress, beautiful form, theatrical lighting',
      energyHighPrompt: 'dancer spinning, fabric flying, explosive movement, peak action, intense',
      energyLowPrompt: 'dancer swaying gently, soft fabric flow, dreamy atmosphere',
      durationFrames: 90,
      trigger: 'energy_peak',
      triggerValue: 0.75,
      transition: 'zoom_in',
      transitionFrames: 45,
    }),
    scene({
      id: 'climax',
      basePrompt: 'dancer dramatic pose, arms extended, powerful stance, spotlight',
      energyHighPrompt: 'dancer triumphant pose, arms raised high, victorious, dramatic lighting, climactic',
      durationFrames: 60,
      trigger: 'beat_count',
      triggerValue: 4,
      transition: 'zoom_out',
      transitionFrames: 45,
    }),
  ],
  loop: true,
  audioReactiveKeywords: true,
};

export const ABSTRACT_LANDSCAPE: StoryConfig = {
  name: 'abstract_landscape',
  description: 'Evolving abstract landscapes that respond to music mood',
  defaultNegativePrompt: 'text, watermark, frame, border, signature',
  scenes: [
    scene({
      id: 'calm_waters',
      basePrompt: 'serene abstract landscape, calm waters, soft gradients, ethereal light, digital art',
      energyHighPrompt: 'turbulent abstract seascape, crashing waves, dynamic motion, intense colors',
      energyLowPrompt: 'still abstract pond, mirror reflection, peaceful, minimal, zen',
      durationFrames: 180,
      trigger: 'time',
      transition: 'crossfade',
      transitionFrames: 60,
    }),
    scene({
      id: 'mountains',
      basePrompt: 'abstract mountain range, geometric peaks, atmospheric perspective, surreal',
      energyHighPrompt: 'volcanic abstract peaks, fire and ice, dramatic contrasts, powerful',
      energyLowPrompt: 'misty abstract hills, soft layers, peaceful valleys, dreamlike',
      durationFrames: 180,
      trigger: 'time',
      transition: 'crossfade',
      transitionFrames: 60,
    }),
    scene({
      id: 'sky',
      basePrompt: 'abstract sky, flowing clouds, color gradients, vast expanse, contemplative',
      energyHighPrompt: 'stormy abstract sky, lightning, dramatic clouds, intense atmosphere',
      energyLowPrompt: 'sunset abstract sky, warm colors, peaceful, golden hour, tranquil',
      durationFrames: 180,
      trigger: 'time',
      transition: 'crossfade',
      transitionFrames: 60,
    }),
  ],
  loop: true,
  audioReactiveKeywords: true,
};

export const MINIMAL_PORTRAIT: StoryConfig = {
  name: 'minimal_portrait',
  description: 'Simple portrait with minimal scene changes, good for testing',
  defaultNegativePrompt: 'blurry, text, watermark',
  scenes: [
    scene({
      id: 'portrait',
      basePrompt: 'portrait photography, beautiful lighting, professional',
      energyHighPrompt: 'portrait, expressive, dynamic lighting, intense gaze',
      energyLowPrompt: 'portrait, peaceful, soft lighting, gentle expression',
      durationFrames: 300,
      trigger: 'time',
      transition: 'cut',
    }),
  ],
  loop: true,
  audioReactiveKeywords: true,
};

// Map of all presets
export const STORY_PRESETS: Record<string, StoryConfig> = {
  skiing_adventure: SKIING_ADVENTURE,
  dancing_figure: DANCING_FIGURE,
  abstract_landscape: ABSTRACT_LANDSCAPE,
  minimal_portrait: MINIMAL_PORTRAIT,
};

// Blank template for new stories
export const BLANK_STORY: StoryConfig = {
  name: 'custom_story',
  description: 'Custom story',
  defaultNegativePrompt: 'blurry, low quality, distorted',
  scenes: [
    scene({
      id: 'scene_1',
      basePrompt: '',
      durationFrames: 120,
      trigger: 'time',
      triggerValue: 0,
      energyBlendRange: [0.3, 0.6],
      transition: 'crossfade',
      transitionFrames: 30,
    }),
  ],
  loop: false,
  audioReactiveKeywords: true,
};
