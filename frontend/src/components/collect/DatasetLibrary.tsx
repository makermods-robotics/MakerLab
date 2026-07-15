import { useMemo, useState } from "react";
import { ChevronRight, GitMerge, Search, Trash2 } from "lucide-react";

import { MarketListingCard } from "@/components/market/MarketListingCard";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { useHfAuth } from "@/contexts/HfAuthContext";
import { DatasetItem } from "@/lib/replayApi";
import { cn } from "@/lib/utils";

type DatasetFilter = "all" | "local" | "yours" | "public" | "private";

interface DatasetLibraryProps {
  // The parent's useDatasets instance — sharing it keeps the library in sync
  // with rename/merge/delete refreshes triggered elsewhere on the page.
  datasets: DatasetItem[];
  loading: boolean;
  selectedRepoId: string | null;
  onSelect: (repoId: string) => void;
  onMerge: () => void;
  onDelete?: (dataset: DatasetItem) => void;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}

const filters: Array<{ id: DatasetFilter; label: string }> = [
  { id: "all", label: "All" },
  { id: "local", label: "Local" },
  { id: "yours", label: "Yours on Hub" },
  { id: "public", label: "Public" },
  { id: "private", label: "Private" },
];

function relativeTime(iso: string | null, verb: string): string {
  if (!iso) return `${verb} unknown`;
  const time = Date.parse(iso);
  if (Number.isNaN(time)) return `${verb} unknown`;
  const seconds = Math.max(0, Math.floor((Date.now() - time) / 1000));
  if (seconds < 60) return `${verb} just now`;
  if (seconds < 3600) return `${verb} ${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${verb} ${Math.floor(seconds / 3600)}h ago`;
  return `${verb} ${Math.floor(seconds / 86400)}d ago`;
}

function repoNamespace(repoId: string): string | null {
  const separator = repoId.indexOf("/");
  return separator === -1 ? null : repoId.slice(0, separator);
}

function matchesFilter(
  dataset: DatasetItem,
  filter: DatasetFilter,
  username: string | null,
): boolean {
  const hubVisible = dataset.source !== "local";

  switch (filter) {
    case "local":
      return dataset.source !== "hub";
    case "yours":
      return (
        hubVisible &&
        username !== null &&
        repoNamespace(dataset.repo_id)?.toLowerCase() === username.toLowerCase()
      );
    case "public":
      return hubVisible && !dataset.private;
    case "private":
      return hubVisible && dataset.private;
    default:
      return true;
  }
}

export function DatasetLibrary({
  datasets,
  loading,
  selectedRepoId,
  onSelect,
  onMerge,
  onDelete,
  open,
  onOpenChange,
}: DatasetLibraryProps) {
  const { auth } = useHfAuth();
  const [filter, setFilter] = useState<DatasetFilter>("all");
  const [query, setQuery] = useState("");
  const [internalOpen, setInternalOpen] = useState(false);

  const expanded = open ?? internalOpen;
  const username = auth.status === "authenticated" ? auth.username : null;
  const visibleDatasets = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return datasets.filter(
      (dataset) =>
        matchesFilter(dataset, filter, username) &&
        (needle === "" || dataset.repo_id.toLowerCase().includes(needle)),
    );
  }, [datasets, filter, username, query]);

  const handleOpenChange = (nextOpen: boolean) => {
    if (open === undefined) setInternalOpen(nextOpen);
    onOpenChange?.(nextOpen);
  };

  return (
    <Card className="overflow-hidden shadow-sm">
      <Collapsible open={expanded} onOpenChange={handleOpenChange}>
        <div className="flex items-center justify-between gap-3 p-4 sm:px-6">
          <CollapsibleTrigger asChild>
            <button
              type="button"
              className="group flex min-w-0 items-center gap-2 rounded-md text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            >
              <ChevronRight className="h-4 w-4 shrink-0 transition-transform group-data-[state=open]:rotate-90" />
              <span className="font-display text-lg font-bold tracking-tight">
                Dataset library
              </span>
              <Badge variant="secondary">{datasets.length}</Badge>
            </button>
          </CollapsibleTrigger>
          <Button variant="outline" size="sm" onClick={onMerge}>
            <GitMerge className="h-4 w-4" />
            Merge
          </Button>
        </div>

        <CollapsibleContent>
          <div className="border-t border-border p-4 sm:p-6">
            <div className="mb-4 flex flex-wrap items-center gap-3">
              <div
                className="flex flex-wrap gap-1 rounded-full bg-secondary p-1 sm:w-fit"
                aria-label="Filter datasets"
              >
                {filters.map((item) => (
                  <Button
                    key={item.id}
                    type="button"
                    size="sm"
                    variant={filter === item.id ? "outline" : "ghost"}
                    aria-pressed={filter === item.id}
                    onClick={() => setFilter(item.id)}
                    className={cn(
                      "h-8 rounded-full px-3",
                      filter === item.id && "bg-card shadow-sm",
                    )}
                  >
                    {item.label}
                  </Button>
                ))}
              </div>
              <div className="relative w-full sm:w-64">
                <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  type="search"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Search datasets…"
                  aria-label="Search datasets"
                  className="pl-9"
                />
              </div>
            </div>

            {loading ? (
              <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                {Array.from({ length: 3 }).map((_, index) => (
                  <div
                    key={index}
                    className="h-[150px] animate-pulse rounded-xl border border-border bg-secondary"
                  />
                ))}
              </div>
            ) : visibleDatasets.length > 0 ? (
              <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                {visibleDatasets.map((dataset) => {
                  const selected = selectedRepoId === dataset.repo_id;
                  const hasLocalCopy = dataset.source === "both";

                  return (
                    <div
                      key={dataset.repo_id}
                      className={cn(
                        "relative rounded-xl transition-shadow",
                        selected &&
                          "ring-2 ring-primary ring-offset-2 ring-offset-background",
                      )}
                      onClick={(event) => {
                        if (
                          event.target instanceof Element &&
                          event.target.closest("button")
                        ) {
                          return;
                        }
                        onSelect(dataset.repo_id);
                      }}
                    >
                      <MarketListingCard
                        kind="dataset"
                        name={dataset.repo_id}
                        source={dataset.source === "local" ? "local" : "hub"}
                        meta={
                          dataset.created_at
                            ? relativeTime(dataset.created_at, "added")
                            : relativeTime(dataset.last_modified, "updated")
                        }
                        actionLabel="Select"
                        completeLabel="Selected"
                        complete={selected}
                        onAction={() => onSelect(dataset.repo_id)}
                        badges={
                          <>
                            {hasLocalCopy && (
                              <Badge variant="secondary">local copy</Badge>
                            )}
                            {dataset.private && (
                              <Badge variant="outline">private</Badge>
                            )}
                          </>
                        }
                        topRight={
                          onDelete && dataset.source !== "hub" ? (
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-6 px-1.5 text-destructive hover:text-destructive"
                              aria-label={`Delete ${dataset.repo_id}`}
                              onClick={() => onDelete(dataset)}
                            >
                              <Trash2 className="h-3.5 w-3.5" />
                            </Button>
                          ) : undefined
                        }
                      />
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="rounded-xl border border-dashed border-border p-8 text-center text-sm text-muted-foreground">
                {query.trim()
                  ? "No datasets match this search."
                  : "No datasets match this filter."}
              </div>
            )}
          </div>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  );
}
