/**
 * A Hub repo id: `namespace/name`, word chars / dot / dash in each segment.
 * Mirrors the backend's `_CUSTOM_REPO_RE` (lelab/server.py), which validates
 * /datasets/custom, /datasets/download, /models/custom, and /models/download —
 * shared here so the add-from-Hub dialogs never offer an id the backend would
 * reject.
 */
export const HUB_REPO_ID_RE = /^[\w.-]+\/[\w.-]+$/;
