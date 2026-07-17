// Pure copy helpers for the recording session's two explicit exits, kept out of
// the component so the fresh-vs-resume wording is testable/greppable in one place.
//
// The two exits and what they do to the episodes:
//   Done — end now, KEEP everything saved so far, go to the upload page.
//   Quit — end WITHOUT saving. A FRESH session's whole dataset (this session's
//          own creation) is deleted; a RESUME session keeps every episode
//          already committed to the pre-existing dataset and only drops the
//          in-progress take. An unintentional page exit is treated as Quit.

export interface ExitConfirmCopy {
  title: string;
  description: string;
}

export function doneConfirmCopy(): ExitConfirmCopy {
  return {
    title: "Finish and save?",
    description:
      "Every episode saved so far is kept, and you'll go to the upload page. " +
      "The arm returns to its starting position, then goes limp.",
  };
}

export function quitConfirmCopy(resume: boolean): ExitConfirmCopy {
  return {
    title: "Quit without saving?",
    description: resume
      ? "Episodes already saved remain in the dataset; only the in-progress take " +
        "is discarded. The arm returns to its starting position, then goes limp."
      : "The recording and all its episodes will be deleted. The arm returns to " +
        "its starting position, then goes limp.",
  };
}

/**
 * Toast/confirm line for an UNINTENTIONAL leave (back button, tab close), which
 * is treated as Quit. Mirrors quitConfirmCopy's fresh-vs-resume distinction.
 */
export function leaveDiscardMessage(resume: boolean): string {
  return resume
    ? "Leaving quits the recording without saving — episodes already saved stay in the dataset."
    : "Leaving quits the recording without saving — the recording and all its episodes will be deleted.";
}
