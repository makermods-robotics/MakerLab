import { cn } from "@/lib/utils";
import { splitDedupeSuffix } from "@/lib/modelNames";
import { useTruncationTitle } from "@/hooks/useTruncationTitle";

interface Props {
  /** The display name as the backend resolved it, collision suffix included. */
  name: string;
  className?: string;
}

/**
 * A model/run title line whose collision suffix survives truncation.
 *
 * The backend hands us names like `eraser_place (2026-08-02)`, where the
 * parenthesized tail is the ONLY thing separating that row from the one above
 * it (see `dedupe_display_names`). A single `truncate` span eats the tail
 * first, so in the narrow places these titles live — a card header, a select
 * item — the disambiguator is exactly what disappears, and two rows the backend
 * went to some trouble to distinguish arrive at the user identical, or worse,
 * distinguished only by a stump like "(2026-0…" that no longer says which.
 *
 * So the two parts get their own spans: the base truncates, the suffix does
 * not. The suffix is short and bounded by construction (a date, a date and
 * time, or a small ordinal), so protecting it costs a fixed sliver of the line
 * and the base absorbs the rest. When the base IS clipped, the full name is one
 * hover away — and only then, since a title on a name that already fits whole
 * just repeats it (see `useTruncationTitle`).
 *
 * The suffix is deliberately rendered verbatim rather than shortened to, say, a
 * dropped year: two runs a year apart would collapse to the same visible text,
 * which is the precise failure this exists to prevent.
 */
const DisplayName: React.FC<Props> = ({ name, className }) => {
  const { base, suffix } = splitDedupeSuffix(name);
  // The title hangs on the CONTAINER, and the hook sweeps its descendants —
  // the pair is one name, so either span clipping means the visible name is
  // incomplete, and the container itself never overflows (its flex children
  // shrink instead), which measuring it alone would misread as "fits".
  const hover = useTruncationTitle(name);
  // A flex row in both cases: `truncate` needs a block-level box to clip
  // against, and keeping one shape means the suffixed and unsuffixed titles
  // share a baseline wherever they sit side by side in a grid.
  return (
    <div className={cn("flex min-w-0 items-baseline", className)} {...hover}>
      <span className="truncate">{base}</span>
      {/* shrink-0: the base gives up its characters first, the disambiguator
          never does. */}
      {suffix === null ? null : (
        <span className="shrink-0 whitespace-nowrap">&nbsp;({suffix})</span>
      )}
    </div>
  );
};

export default DisplayName;
