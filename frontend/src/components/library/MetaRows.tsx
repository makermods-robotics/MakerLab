import React from "react";

/**
 * The unified metadata block for library cards: muted fixed-width labels with
 * truncating values, one row per fact. Every card family (dataset, job, model)
 * renders its details through this so metadata reads identically everywhere.
 * Callers pass only the rows they have — absent facts are simply omitted.
 */
const MetaRows: React.FC<{
  /** `[label, value]`, plus an optional tooltip for when the hover text should
   * differ from the rendered value — used where the display string is
   * shortened for readability but the underlying value is what someone
   * debugging the row actually needs (e.g. a bimanual dataset's camera keys,
   * shown bare but hovered in full). Defaults to `value`. */
  rows: Array<[label: string, value: string, title?: string]>;
}> = ({ rows }) => {
  if (rows.length === 0) return null;
  return (
    <div className="space-y-1 text-[11px]">
      {rows.map(([label, value, title]) => (
        <div key={label} className="flex items-baseline gap-1.5">
          <span className="w-14 shrink-0 text-muted-foreground">{label}</span>
          <span
            className="min-w-0 flex-1 truncate text-foreground"
            title={title ?? value}
          >
            {value}
          </span>
        </div>
      ))}
    </div>
  );
};

export default MetaRows;
