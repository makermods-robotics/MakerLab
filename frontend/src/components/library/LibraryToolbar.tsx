import React from "react";
import { Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

/** Shared 2-up grid for every studio library (datasets, jobs, models) so the
 * three panels read as one system. Two cards per row at every width: each
 * library sits in a narrow studio column (one-third of the viewport), so three
 * cards squeeze the card icons/buttons past their card edges even on a
 * full-screen MacBook. Cards stretch to their row's height. */
export const LIBRARY_GRID = "grid grid-cols-2 items-stretch gap-2";

export interface LibraryFilterOption<K extends string> {
  key: K;
  label: string;
}

interface LibraryToolbarProps<K extends string> {
  query: string;
  onQueryChange: (query: string) => void;
  searchPlaceholder: string;
  filters: Array<LibraryFilterOption<K>>;
  filter: K;
  onFilterChange: (filter: K) => void;
}

/**
 * The unified library toolbar: a search input beside a segmented group of
 * filter pills. Used by the dataset, training-jobs, and model libraries so
 * searching and filtering look and behave identically in all three.
 */
function LibraryToolbar<K extends string>({
  query,
  onQueryChange,
  searchPlaceholder,
  filters,
  filter,
  onFilterChange,
}: LibraryToolbarProps<K>) {
  return (
    <div className="flex items-center gap-2">
      <div className="relative flex-1">
        <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
        <Input
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
          placeholder={searchPlaceholder}
          aria-label={searchPlaceholder}
          className="h-8 pl-8 text-sm"
        />
      </div>
      <div className="flex shrink-0 rounded-md border border-border bg-muted p-0.5">
        {filters.map(({ key, label }) => (
          <button
            key={key}
            type="button"
            onClick={() => onFilterChange(key)}
            aria-pressed={filter === key}
            className={cn(
              "rounded px-2 py-1 text-xs font-medium transition-colors",
              filter === key
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {label}
          </button>
        ))}
      </div>
    </div>
  );
}

export default LibraryToolbar;
