
export type ShoeType = 'sneaker' | 'runner' | 'leather' | 'casual' | 'sandals' | 'boots' | 'luxury';

export interface PromptResult {
  sora_prompt: string;
  cinematography_notes: string;
  audio_timeline: string;
  dialogue: string;
  technical_specs: {
    resolution: string;
    frame_rate: string;
    camera_angle: string;
    lighting: string;
  };
}

export enum PromptMode {
  NORMAL = 'PROMPT 1 (Không Cameo)',
  CAMEO = 'PROMPT 2 (Có Cameo)'
}

/**
 * Scene styles available for the Sora AI prompt generator
 */
export enum SceneStyle {
  CINEMATIC = 'Cinematic',
  MINIMALIST = 'Minimalist',
  CYBERPUNK = 'Cyberpunk',
  VINTAGE = 'Vintage',
  HYPER_REALISTIC = 'Hyper-Realistic'
}
