const MUTE_KEY = "makerlab.recording.muted";

let ctx: AudioContext | null = null;

const getCtx = (): AudioContext => {
  if (!ctx) ctx = new AudioContext();
  return ctx;
};

export const getMuted = (): boolean => {
  return localStorage.getItem(MUTE_KEY) === "1";
};

export const setMuted = (value: boolean): void => {
  localStorage.setItem(MUTE_KEY, value ? "1" : "0");
};

const playTone = (frequency: number, durationMs: number, startOffsetMs = 0) => {
  if (getMuted()) return;
  const c = getCtx();
  const osc = c.createOscillator();
  const gain = c.createGain();
  osc.frequency.value = frequency;
  osc.type = "sine";
  gain.gain.value = 0;
  osc.connect(gain);
  gain.connect(c.destination);
  const start = c.currentTime + startOffsetMs / 1000;
  const stop = start + durationMs / 1000;
  gain.gain.setValueAtTime(0, start);
  gain.gain.linearRampToValueAtTime(0.2, start + 0.01);
  gain.gain.setValueAtTime(0.2, stop - 0.02);
  gain.gain.linearRampToValueAtTime(0, stop);
  osc.start(start);
  osc.stop(stop);
};

export const playRecordingStartCue = (): void => {
  playTone(660, 80, 0);
  playTone(880, 80, 90);
};

export const playResetStartCue = (): void => {
  playTone(660, 80, 0);
  playTone(440, 80, 90);
};

export const playAutoAdvanceWarning = (): void => {
  playTone(880, 70, 0);
  playTone(880, 70, 1000);
  playTone(880, 70, 2000);
};
