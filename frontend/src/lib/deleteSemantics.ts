/**
 * Shared delete semantics for the dataset and model pickers/info cards.
 *
 * One resolver maps a listing row to what its trash button actually does, so
 * every entry point (picker row, info card) opens the same confirm dialog with
 * the same action. The rules (user-decided):
 *
 *  - Delete NEVER deletes or mutates a Hub repo.
 *  - "both" (local + hub): TWO-PRESS. First press removes only the local copy —
 *    the row flips to a plain hub row and stays listed, and the selection is
 *    kept (the dataset/model still exists). A second press (now hub-only)
 *    hides it.
 *  - EXCEPT a "both" row whose local side is a training RUN (`local_kind`).
 *    The two-press rule rests on the local copy being replaceable from the Hub,
 *    which holds for a downloaded model but not for a run: publishing uploads
 *    the checkpoints the user chose, so a run's unpublished checkpoints exist
 *    nowhere else and "the Hub copy stays" would understate the loss. Such a
 *    row gets the destructive local delete instead.
 *  - Pinned custom hub rows are unpinned (existing behavior).
 *  - Own-namespace hub-only rows are hidden via the persistent hidden-list
 *    (they'd otherwise resurface on every Hub listing).
 *  - Local-only rows are destructive local file deletes (existing routes).
 */

export type DeleteAction = "delete-local" | "delete-local-copy" | "unpin" | "hide";

export interface DeletableItem {
  source: "local" | "hub" | "both";
  saved_custom?: boolean;
  /** What the LOCAL side of this row is: a training run (holds every checkpoint
   * it saved, only some of which may be published) or a copy pulled from the
   * Hub / imported from disk. Absent for datasets and for hub-only rows. */
  local_kind?: "run" | "downloaded";
}

export interface DeleteResolution {
  action: DeleteAction;
  /** Dialog title verb; the caller interpolates the item label after it. */
  titlePrefix: string;
  description: string;
  confirmLabel: string;
  /** True when a confirmed action removes the row from the listing entirely
   * (hide / unpin / local-only delete) — the persisted selection should then
   * be cleared. False for the both→hub flip, where the row stays listed and
   * the selection is kept. */
  clearsSelection: boolean;
}

const LOCAL_DELETE_DESCRIPTION: Record<"dataset" | "model", string> = {
  dataset:
    "This permanently removes the dataset from local disk — including all " +
    "recorded episodes and videos. You can't undo this.",
  model:
    "This permanently removes the model's local files from disk — including " +
    "its checkpoints. You can't undo this. A Hub copy, if any, is not affected.",
};

export function resolveDeleteAction(
  kind: "dataset" | "model",
  item: DeletableItem,
): DeleteResolution {
  // A published training run is "both", but its local side is not a replaceable
  // copy — only the checkpoints the user chose to publish are on the Hub, so
  // this is a destructive delete and has to be described as one.
  if (item.source === "both" && item.local_kind === "run") {
    return {
      action: "delete-local",
      titlePrefix: "Delete",
      description:
        "This permanently removes the training run's local files from disk — " +
        "including every checkpoint, published or not. You can't undo this. " +
        "The checkpoints you already published to the Hub are not affected.",
      confirmLabel: "Delete",
      clearsSelection: true,
    };
  }
  // "both" outranks saved_custom: the first press always removes the local
  // copy; the (possibly pinned) hub row survives for a second press.
  if (item.source === "both") {
    return {
      action: "delete-local-copy",
      titlePrefix: "Remove local copy of",
      description:
        "This removes the local copy from disk — the Hub copy stays, and it " +
        `remains listed as a Hub ${kind}.`,
      confirmLabel: "Remove local copy",
      clearsSelection: false,
    };
  }
  if (item.source === "hub" && item.saved_custom) {
    return {
      action: "unpin",
      titlePrefix: "Remove",
      description:
        `This just removes the ${kind} from your list. The Hub repo and any ` +
        "local copy are untouched — you can re-add it any time from the " +
        `Add ${kind} menu.`,
      confirmLabel: "Remove",
      clearsSelection: true,
    };
  }
  if (item.source === "hub") {
    return {
      action: "hide",
      titlePrefix: "Remove",
      description:
        `This hides the ${kind} from your list. The Hub repo is not deleted — ` +
        `you can re-add it any time from the Add ${kind} menu.`,
      confirmLabel: "Remove",
      clearsSelection: true,
    };
  }
  return {
    action: "delete-local",
    titlePrefix: "Delete",
    description: LOCAL_DELETE_DESCRIPTION[kind],
    confirmLabel: "Delete",
    clearsSelection: true,
  };
}
