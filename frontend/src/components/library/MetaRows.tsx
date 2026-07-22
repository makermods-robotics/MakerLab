import React from "react";

/**
 * The unified metadata block for library cards: muted fixed-width labels with
 * truncating values, one row per fact. Every card family (dataset, job, model)
 * renders its details through this so metadata reads identically everywhere.
 * Callers pass only the rows they have — absent facts are simply omitted.
 */
const MetaRows: React.FC<{ rows: Array<[label: string, value: string]> }> = ({
  rows,
}) => {
  if (rows.length === 0) return null;
  return (
    <div className="space-y-1 text-[11px]">
      {rows.map(([label, value]) => (
        <div key={label} className="flex items-baseline gap-1.5">
          <span className="w-14 shrink-0 text-muted-foreground">{label}</span>
          <span className="min-w-0 flex-1 truncate text-foreground" title={value}>
            {value}
          </span>
        </div>
      ))}
    </div>
  );
};

export default MetaRows;
