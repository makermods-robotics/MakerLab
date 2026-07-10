import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge, BadgeDot } from "@/components/ui/badge";
import { HubJob, isHubJobActive } from "@/lib/jobsApi";
import {
  ExternalLink,
  Trash2,
} from "lucide-react";

interface Props {
  job: HubJob;
  // Hide this job from the list (persisted backend-side; the Hub record is
  // untouched). The trash button is only offered on terminal stages — an
  // active run can't be dismissed out of sight.
  onDismiss?: (id: string) => void;
}

function relativeTime(iso: string | null): string {
  if (!iso) return "—";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "—";
  const diff = Math.max(0, (Date.now() - t) / 1000);
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

interface StagePresentation {
  label: string;
  variant: "default" | "ok" | "warn" | "destructive" | "outline";
  mark?: string;
  pulse?: boolean;
}

const stagePresentation: Record<string, StagePresentation> = {
  RUNNING: { label: "running", variant: "default", pulse: true },
  QUEUED: { label: "queued", variant: "warn" },
  SCHEDULING: { label: "scheduling", variant: "warn" },
  COMPLETED: { label: "done", variant: "ok", mark: "✓" },
  FAILED: { label: "failed", variant: "destructive", mark: "✕" },
  // HF API uses "CANCELED" (single L); accept both spellings.
  CANCELED: { label: "cancelled", variant: "warn" },
  CANCELLED: { label: "cancelled", variant: "warn" },
};

const HubJobCard: React.FC<Props> = ({ job, onDismiss }) => {
  const stage = job.status?.stage?.toUpperCase() ?? "";
  const present: StagePresentation = stagePresentation[stage] ?? {
    label: stage || "unknown",
    variant: "outline",
  };
  const title =
    job.docker_image ?? job.space_id ?? `Job ${job.id.slice(0, 12)}…`;

  return (
    <Card
      variant="flat"
      onClick={() => window.open(job.url, "_blank", "noopener,noreferrer")}
      className="rounded-xl bg-card cursor-pointer hover:border-input transition-colors"
    >
      <CardContent className="p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div
              className="truncate text-[15px] font-semibold text-foreground"
              title={title}
            >
              {title}
            </div>
            <div className="truncate font-mono text-[11px] text-muted-foreground">
              {job.flavor ?? "—"} · {relativeTime(job.created_at)}
              {job.owner ? ` · ${job.owner}` : ""}
            </div>
          </div>
          <div className="flex shrink-0 items-start gap-2">
            <Badge variant={present.variant}>
              {present.pulse ? <BadgeDot pulse /> : null}
              {present.label}
              {present.mark ? ` ${present.mark}` : ""}
            </Badge>
            <Button
              variant="ghost"
              size="icon"
              asChild
              className="h-7 w-7"
              aria-label="View on Hub"
            >
              <a
                href={job.url}
                target="_blank"
                rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
              >
                <ExternalLink className="w-3.5 h-3.5" />
              </a>
            </Button>
            {onDismiss && !isHubJobActive(job) ? (
              <Button
                variant="ghost"
                size="icon"
                onClick={(e) => {
                  e.stopPropagation();
                  if (
                    window.confirm(
                      "Remove this job from the list? The job record on Hugging Face is unaffected.",
                    )
                  )
                    onDismiss(job.id);
                }}
                className="h-7 w-7 hover:text-destructive"
                aria-label="Remove job from list"
                title="Remove from list"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </Button>
            ) : null}
          </div>
        </div>
        {job.status?.message ? (
          <div
            className="mt-3 truncate font-mono text-[11px] text-muted-foreground"
            title={job.status.message}
          >
            {job.status.message}
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
};

export default HubJobCard;
