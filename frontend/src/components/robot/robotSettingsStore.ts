import { useSyncExternalStore } from "react";

/**
 * Module-level store so any surface (Home, sidebar gear, create flow) can open
 * the robot settings dialog without prop drilling. Mirrors the shared-store
 * pattern used by useRobots.
 */
interface RobotSettingsState {
  open: boolean;
  robotName: string | null;
}

let state: RobotSettingsState = { open: false, robotName: null };
const listeners = new Set<() => void>();

const emit = () => listeners.forEach((l) => l());

export const openRobotSettings = (name: string | null) => {
  state = { open: true, robotName: name };
  emit();
};

export const closeRobotSettings = () => {
  state = { ...state, open: false };
  emit();
};

const subscribe = (l: () => void) => {
  listeners.add(l);
  return () => listeners.delete(l);
};

const getSnapshot = () => state;

export const useRobotSettingsState = () =>
  useSyncExternalStore(subscribe, getSnapshot);
