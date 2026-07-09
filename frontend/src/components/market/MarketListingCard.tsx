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
  onAction,
}: MarketListingCardProps) {
  const ActionIcon = kind === "dataset" ? Plus : Download;

  return (
    <Card className="overflow-hidden rounded-xl p-0 shadow-sm">
      <div
        className="media-slot h-[130px] min-h-[130px] rounded-none border-0 border-b border-dashed"
        data-label="task preview"
      />
      <div className="grid gap-3 p-4">
        <div className="min-w-0">
          <h2 className="truncate text-base font-semibold text-foreground" title={name}>
            {name}
          </h2>
        </div>
        <div className="flex flex-wrap gap-1.5">
          <Badge variant={kind === "dataset" ? "secondary" : "outline"}>
            {kind}
          </Badge>
          <Badge variant="outline">{source}</Badge>
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
              <ActionIcon className="h-4 w-4" />
            )}
            {complete ? completeLabel : loading ? "Working..." : actionLabel}
          </Button>
        </div>
      </div>
    </Card>
  );
}
