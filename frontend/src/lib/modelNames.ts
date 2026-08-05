/**
 * Presentation helpers for model/skill card titles.
 *
 * The derivation itself is the backend's — `derive_imported_title` in
 * makermodslab/utils/naming.py peels an imported model's repo id down to its
 * task segment, and `dedupe_display_names` keeps two of them apart — so a card
 * only has to render the name it is given. What CSS can't do is what lives
 * here: `truncate` always eats the tail, which is the wrong half for a name
 * whose distinguishing text sits at the end.
 */

/**
 * Shorten `text` from the MIDDLE, keeping both ends readable.
 *
 * End-truncation is the wrong default for repo-derived names: their head is
 * boilerplate the author repeats across every model and their tail is the part
 * that identifies this one, so cutting the tail hides exactly what the user is
 * scanning for. Returns `text` untouched when it already fits, which is the
 * normal case for a derived title — this only bites on the fallback path, a
 * community repo whose name we can't parse and therefore can't shorten safely.
 */
export function middleEllipsis(text: string, max = 32): string {
  if (max < 5 || text.length <= max) return text;
  // One char of the budget goes to the ellipsis; the head keeps the odd char,
  // since a name's first word is usually the more recognizable of the two.
  const head = Math.ceil((max - 1) / 2);
  const tail = max - 1 - head;
  return `${text.slice(0, head)}…${text.slice(text.length - tail)}`;
}

/**
 * A trailing " (…)" the backend appended to break a name collision.
 *
 * `dedupe_display_names` (makermodslab/utils/naming.py) resolves two rows that
 * derived the same title by appending the one fact that separates them — the
 * run's date, its date and time, or an ordinal. That suffix is therefore the
 * MOST load-bearing text in the whole label and the LEAST survivable: it sits
 * at the very end, where `truncate` starts eating. "eraser_place (2026-08-…"
 * and "eraser_place (2026-08-…" are two rows the suffix was added to tell
 * apart, rendered identically.
 *
 * Splitting lets the caller render the base and the suffix as separate spans,
 * so only the base shrinks (see DisplayName). A name the user typed that
 * happens to end in parentheses splits too — harmless, since both halves are
 * rendered adjacent and read exactly as before.
 */
export function splitDedupeSuffix(name: string): {
  base: string;
  suffix: string | null;
} {
  const match = /^(.*\S)\s\(([^()]+)\)$/.exec(name);
  if (!match) return { base: name, suffix: null };
  return { base: match[1], suffix: match[2] };
}
