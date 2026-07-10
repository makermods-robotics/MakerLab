import { useMemo, useState } from "react";
import { ChevronRight, Play, RefreshCw, SlidersHorizontal } from "lucide-react";

import { MarketListingCard } from "@/components/market/MarketListingCard";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { UserModel } from "@/lib/modelsApi";
import { cn } from "@/lib/utils";

type ModelFilter = "all" | "cloud" | "uploaded" | "public" | "private";

interface ModelLibraryProps {
  /** The user's Hub model repos — cloud-trained runs and hand-uploaded alike. */
  models: UserModel[];
  loading: boolean;
  authenticated: boolean;
  robotLabel: string;
  onRun: (repoId: string) => void;
  onFinetune: (repoId: string) => void;
  onRefresh: () => void;
}

const filters: Array<{ id: ModelFilter; label: string }> = [
  { id: "all", label: "All" },
  { id: "cloud", label: "Cloud runs" },
  { id: "uploaded", label: "Uploaded" },
  { id: "public", label: "Public" },
  { id: "private", label: "Private" },
];

function relativeTime(iso: string | null): string {
  if (!iso) return "updated unknown";
  const time = Date.parse(iso);
  if (Number.isNaN(time)) return "updated unknown";
  const seconds = Math.max(0, Math.floor((Date.now() - time) / 1000));
  if (seconds < 60) return "updated just now";
  if (seconds < 3600) return `updated ${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `updated ${Math.floor(seconds / 3600)}h ago`;
  return `updated ${Math.floor(seconds / 86400)}d ago`;
}

function matchesFilter(model: UserModel, filter: ModelFilter): boolean {
  switch (filter) {
    case "cloud":
      return model.cloud_run;
    case "uploaded":
      return !model.cloud_run;
    case "public":
      return !model.private;
    case "private":
      return model.private;
    default:
      return true;
  }
}

export function ModelLibrary({
  models,
  loading,
  authenticated,
  robotLabel,
  onRun,
  onFinetune,
  onRefresh,
}: ModelLibraryProps) {
  const [filter, setFilter] = useState<ModelFilter>("all");
  const [expanded, setExpanded] = useState(false);

  const visibleModels = useMemo(
    () => models.filter((model) => matchesFilter(model, filter)),
    [models, filter],
  );

  return (
    <Card className="overflow-hidden shadow-sm">
      <Collapsible open={expanded} onOpenChange={setExpanded}>
        <div className="flex items-center justify-between gap-3 p-4 sm:px-6">
          <CollapsibleTrigger asChild>
            <button
              type="button"
              className="group flex min-w-0 items-center gap-2 rounded-md text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            >
              <ChevronRight className="h-4 w-4 shrink-0 transition-transform group-data-[state=open]:rotate-90" />
              <span className="font-display text-lg font-bold tracking-tight">
                Model library
              </span>
              <Badge variant="secondary">{models.length}</Badge>
            </button>
          </CollapsibleTrigger>
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={onRefresh}
            aria-label="Refresh models"
          >
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>

        <CollapsibleContent>
          <div className="border-t border-border p-4 sm:p-6">
            <p className="mb-4 text-sm text-muted-foreground">
              Every model you own on Hugging Face — trained in the cloud or
              uploaded by hand. Run one on {robotLabel}, or fine-tune it on a
              new dataset.
            </p>
            <div
              className="mb-4 flex flex-wrap gap-1 rounded-full bg-secondary p-1 sm:w-fit"
              aria-label="Filter models"
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

            {loading ? (
              <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                {Array.from({ length: 3 }).map((_, index) => (
                  <div
                    key={index}
                    className="h-[280px] animate-pulse rounded-xl border border-border bg-secondary"
                  />
                ))}
              </div>
            ) : !authenticated ? (
              <div className="rounded-xl border border-dashed border-border p-8 text-center text-sm text-muted-foreground">
                Sign in to Hugging Face to see your models.
              </div>
            ) : visibleModels.length > 0 ? (
              <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                {visibleModels.map((model) => (
                  <div key={model.repo_id} className="relative rounded-xl">
                    <div className="absolute right-3 top-3 z-10 flex flex-wrap items-center justify-end gap-1.5">
                      {model.cloud_run && (
                        <Badge variant="secondary">cloud run</Badge>
                      )}
                      {model.lerobot && <Badge variant="outline">lerobot</Badge>}
                      {model.private && <Badge variant="outline">private</Badge>}
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 px-1.5"
                        aria-label={`Fine-tune ${model.repo_id}`}
                        title="Fine-tune on a new dataset"
                        onClick={() => onFinetune(model.repo_id)}
                      >
                        <SlidersHorizontal className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                    <MarketListingCard
                      kind="model"
                      name={model.repo_id}
                      source="hub"
                      meta={relativeTime(model.last_modified)}
                      actionLabel="Run"
                      actionIcon={<Play className="h-4 w-4" />}
                      completeLabel="Run"
                      complete={false}
                      onAction={() => onRun(model.repo_id)}
                    />
                  </div>
                ))}
              </div>
            ) : (
              <div className="rounded-xl border border-dashed border-border p-8 text-center text-sm text-muted-foreground">
                No models match this filter.
              </div>
            )}
          </div>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  );
}
