import type { ReactNode } from "react";
import { Check, Download, Plus } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

export type MarketListingKind = "dataset" | "model";

interface MarketListingCardProps {
  kind: MarketListingKind;
  name: string;
  source: string;
  meta: string;
  actionLabel: string;
  completeLabel: string;
  complete: boolean;
  loading?: boolean;
  /** Overrides the kind-derived action icon (Plus/Download). */
  actionIcon?: ReactNode;
  /** Extra badges rendered in the badge row, after kind/source. */
  badges?: ReactNode;
  /** Buttons/controls rendered on the title row, right-aligned. */
  topRight?: ReactNode;
  onAction: () => void;
}

export function MarketListingCard({
  kind,
  name,
  source,
  meta,
  actionLabel,
  completeLabel,
  complete,
  loading = false,
  actionIcon,
  badges,
  topRight,
  onAction,
}: MarketListingCardProps) {
  const ActionIcon = kind === "dataset" ? Plus : Download;

  return (
    <Card className="overflow-hidden rounded-xl p-0 shadow-sm">
      <div className="grid gap-3 p-4">
        <div className="flex min-w-0 items-start justify-between gap-2">
          {/* Repo ids have no spaces — break-all lets long names wrap instead
              of truncating away the distinguishing suffix. */}
          <h2 className="min-w-0 break-all text-base font-semibold leading-snug text-foreground">
            {name}
          </h2>
          {topRight && (
            <div className="flex shrink-0 flex-wrap items-center justify-end gap-1.5">
              {topRight}
            </div>
          )}
        </div>
        <div className="flex flex-wrap gap-1.5">
          <Badge variant={kind === "dataset" ? "secondary" : "outline"}>
            {kind}
          </Badge>
          <Badge variant="outline">{source}</Badge>
          {badges}
        </div>
        <p className="min-h-[20px] truncate text-[12.5px] text-muted-foreground" title={meta}>
          {meta}
        </p>
        <div className="mt-0.5 flex justify-end">
          <Button
            type="button"
            size="sm"
            variant={complete ? "ghost" : "default"}
            disabled={complete || loading}
            onClick={onAction}
            className={cn("h-9", complete && "text-muted-foreground")}
          >
            {complete ? (
              <Check className="h-4 w-4" />
            ) : (
              (actionIcon ?? <ActionIcon className="h-4 w-4" />)
            )}
            {complete ? completeLabel : loading ? "Working..." : actionLabel}
          </Button>
        </div>
      </div>
    </Card>
  );
}
