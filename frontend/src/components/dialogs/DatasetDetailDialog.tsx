import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Boxes, Pause, Play, SkipBack, SkipForward, VideoOff } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { useStudio } from "@/contexts/StudioContext";
import { useSelectedDataset } from "@/hooks/useSelectedDataset";
import { useApi } from "@/contexts/ApiContext";
import DatasetInfoCard from "@/components/landing/DatasetInfoCard";
import {
  EpisodeJointSeries,
  EpisodeSummary,
  episodeVideoUrl,
  getDatasetInfo,
  getEpisodeJoints,
  listEpisodes,
} from "@/lib/replayApi";

export interface DatasetDetailDialogProps {
  repoId: string | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Called when an action navigates to the studio, so a parent surface that
   * would otherwise cover the studio (e.g. the library sheet) can close too. */
  onStudioAction?: () => void;
}

// Six SO-101 joints, in the dataset's fixed column order — a validated
// categorical set (see dataviz skill palette) so adjacent lines stay
// distinguishable; cycles defensively if a feature set ever carries more.
const JOINT_COLORS = [
  "#2a78d6",
  "#eb6834",
  "#1baf7a",
  "#eda100",
  "#e87ba4",
  "#008300",
];

// Best (cols, tileW, tileH) for `n` tiles inside a box of `boxW` x `boxH`:
// the search over column counts that a video-gallery layout needs so the
// camera grid keeps growing/reflowing as cameras are added or the window is
// resized, instead of a fixed N-up template.
function computeCameraLayout(
  n: number,
  boxW: number,
  boxH: number,
  gap = 8,
  aspect = 16 / 10,
) {
  let best = { cols: 1, tileW: boxW, tileH: boxW / aspect };
  let bestArea = -1;
  for (let cols = 1; cols <= n; cols++) {
    const rows = Math.ceil(n / cols);
    const cellW = (boxW - gap * (cols - 1)) / cols;
    const cellH = (boxH - gap * (rows - 1)) / rows;
    let tileW = cellW;
    let tileH = tileW / aspect;
    if (tileH > cellH) {
      tileH = cellH;
      tileW = tileH * aspect;
    }
    const area = tileW * tileH;
    if (area > bestArea) {
      bestArea = area;
      best = { cols, tileW, tileH };
    }
  }
  return best;
}

const fmtTime = (t: number) => `${t.toFixed(1)}s`;

/**
 * Episode viewer: a growable grid of camera feeds (one per dataset camera —
 * reflows on resize and as camera count changes, never a fixed 2-up) with
 * transport controls and a joint-position chart synced to the playhead. Real
 * <video> elements pointing at /datasets/episode-video (Range-request backed,
 * so scrubbing seeks without buffering the whole file); no local data means no
 * episode list, so this renders a locked "not viewable yet" state instead.
 */
const EpisodeViewer: React.FC<{
  repoId: string;
  cameras: string[];
  episodes: EpisodeSummary[];
  selectedEpisode: number;
  onSelectEpisode: (episodeIndex: number) => void;
}> = ({ repoId, cameras, episodes, selectedEpisode, onSelectEpisode }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [joints, setJoints] = useState<EpisodeJointSeries | null>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [videoErrors, setVideoErrors] = useState<Record<string, boolean>>({});
  const [hoverT, setHoverT] = useState<number | null>(null);

  const episode = episodes.find((e) => e.episode_index === selectedEpisode) ?? null;
  const videoRefs = useRef<Record<string, HTMLVideoElement | null>>({});
  const primaryCamera = cameras[0];

  useEffect(() => {
    const controller = new AbortController();
    setJoints(null);
    getEpisodeJoints(baseUrl, fetchWithHeaders, repoId, selectedEpisode, controller.signal)
      .then(setJoints)
      .catch(() => {
        if (!controller.signal.aborted) setJoints(null);
      });
    return () => controller.abort();
  }, [baseUrl, fetchWithHeaders, repoId, selectedEpisode]);

  useEffect(() => {
    setPlaying(false);
    setCurrentTime(0);
    setVideoErrors({});
  }, [selectedEpisode]);

  const forEachVideo = useCallback((fn: (v: HTMLVideoElement) => void) => {
    Object.values(videoRefs.current).forEach((v) => v && fn(v));
  }, []);

  const handlePlayPause = () => {
    if (playing) {
      forEachVideo((v) => v.pause());
      setPlaying(false);
    } else {
      forEachVideo((v) => {
        v.play().catch(() => {});
      });
      setPlaying(true);
    }
  };

  const handleSeek = (t: number) => {
    const clamped = Math.max(0, Math.min(episode?.duration ?? 0, t));
    forEachVideo((v) => {
      v.currentTime = clamped;
    });
    setCurrentTime(clamped);
  };

  const gridWrapRef = useRef<HTMLDivElement>(null);
  const [layout, setLayout] = useState({ cols: 1, tileW: 100, tileH: 60 });
  useEffect(() => {
    const el = gridWrapRef.current;
    if (!el) return;
    const recompute = () => {
      const rect = el.getBoundingClientRect();
      setLayout(
        computeCameraLayout(
          Math.max(1, cameras.length),
          Math.max(80, rect.width - 16),
          Math.max(80, rect.height - 16),
        ),
      );
    };
    recompute();
    const ro = new ResizeObserver(recompute);
    ro.observe(el);
    return () => ro.disconnect();
  }, [cameras.length]);

  const jointRanges = useMemo(() => {
    if (!joints || joints.values.length === 0) return [];
    return joints.joint_names.map((_, j) => {
      const vals = joints.values.map((frame) => frame[j]);
      return { min: Math.min(...vals), max: Math.max(...vals) };
    });
  }, [joints]);

  const chartRef = useRef<HTMLDivElement>(null);
  const handleChartHover = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!episode || !joints) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    setHoverT(frac * episode.duration);
  };

  const scrubFrac = episode?.duration ? currentTime / episode.duration : 0;

  const scrubRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef(false);
  const seekFromClientX = (clientX: number) => {
    const rect = scrubRef.current?.getBoundingClientRect();
    if (!rect || !episode) return;
    const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    handleSeek(frac * episode.duration);
  };

  const gotoEpisode = (delta: number) => {
    const i = episodes.findIndex((e) => e.episode_index === selectedEpisode);
    const next = episodes[i + delta];
    if (next) onSelectEpisode(next.episode_index);
  };

  if (cameras.length === 0) {
    return (
      <div className="flex flex-1 items-center justify-center rounded-md border border-border bg-muted/30 text-sm text-muted-foreground">
        This dataset has no camera data to view.
      </div>
    );
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <div
        ref={gridWrapRef}
        className="relative flex-1 overflow-auto rounded-md border border-border bg-[#0c0f14]"
      >
        <div
          className="grid justify-center content-center gap-2 p-2"
          style={{
            gridTemplateColumns: `repeat(${layout.cols}, ${layout.tileW}px)`,
            gridAutoRows: `${layout.tileH}px`,
          }}
        >
          {cameras.map((camera) => (
            <div
              key={camera}
              className="relative overflow-hidden rounded border border-white/10 bg-black"
            >
              <span className="absolute left-1.5 top-1.5 z-10 rounded bg-black/50 px-1.5 py-0.5 font-mono text-[10px] font-semibold text-zinc-100">
                {camera}
              </span>
              {videoErrors[camera] ? (
                <div className="flex h-full w-full flex-col items-center justify-center gap-1 px-3 text-center text-zinc-300">
                  <VideoOff className="h-4 w-4" />
                  <p className="text-[10px] leading-snug">
                    Can't decode this camera's video in this browser.
                  </p>
                </div>
              ) : (
                <video
                  key={`${repoId}:${selectedEpisode}:${camera}`}
                  ref={(el) => {
                    videoRefs.current[camera] = el;
                  }}
                  src={episodeVideoUrl(baseUrl, repoId, selectedEpisode, camera)}
                  className="h-full w-full object-cover"
                  muted
                  playsInline
                  onTimeUpdate={
                    camera === primaryCamera
                      ? (e) => setCurrentTime(e.currentTarget.currentTime)
                      : undefined
                  }
                  onEnded={camera === primaryCamera ? () => setPlaying(false) : undefined}
                  onError={() =>
                    setVideoErrors((prev) => ({ ...prev, [camera]: true }))
                  }
                />
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="flex items-center gap-2">
        <Button
          variant="outline"
          size="icon"
          className="h-8 w-8 shrink-0"
          onClick={() => gotoEpisode(-1)}
          disabled={!episode}
          aria-label="Previous episode"
        >
          <SkipBack className="h-3.5 w-3.5" />
        </Button>
        <Button
          size="icon"
          className="h-8 w-8 shrink-0"
          onClick={handlePlayPause}
          disabled={!episode}
          aria-label={playing ? "Pause" : "Play"}
        >
          {playing ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
        </Button>
        <Button
          variant="outline"
          size="icon"
          className="h-8 w-8 shrink-0"
          onClick={() => gotoEpisode(1)}
          disabled={!episode}
          aria-label="Next episode"
        >
          <SkipForward className="h-3.5 w-3.5" />
        </Button>
        <span className="shrink-0 font-mono text-xs tabular-nums text-muted-foreground">
          {episode ? `${fmtTime(currentTime)} / ${fmtTime(episode.duration)}` : "—"}
        </span>
        <div
          ref={scrubRef}
          className="relative h-5 flex-1 cursor-pointer"
          onPointerDown={(e) => {
            draggingRef.current = true;
            forEachVideo((v) => v.pause());
            setPlaying(false);
            seekFromClientX(e.clientX);
          }}
          onPointerMove={(e) => {
            if (draggingRef.current) seekFromClientX(e.clientX);
          }}
          onPointerUp={() => {
            draggingRef.current = false;
          }}
          onPointerLeave={() => {
            draggingRef.current = false;
          }}
        >
          <div className="absolute inset-y-0 my-auto h-1 w-full rounded-full bg-muted" />
          <div
            className="absolute inset-y-0 my-auto h-1 rounded-full bg-foreground"
            style={{ width: `${scrubFrac * 100}%` }}
          />
          <div
            className="absolute top-1/2 h-3 w-3 -translate-y-1/2 rounded-full bg-foreground shadow"
            style={{ left: `calc(${scrubFrac * 100}% - 6px)` }}
          />
        </div>
      </div>

      <div className="rounded-md border border-border bg-muted/40 p-2">
        <div className="mb-1 flex items-baseline justify-between">
          <span className="eyebrow">joint positions — synced to playhead</span>
          <span className="eyebrow">
            {episode ? `episode ${episode.episode_index}` : "—"}
          </span>
        </div>
        <div
          ref={chartRef}
          className="relative h-[72px]"
          onMouseMove={handleChartHover}
          onMouseLeave={() => setHoverT(null)}
        >
          {joints && episode && episode.duration > 0 ? (
            <svg viewBox="0 0 600 72" preserveAspectRatio="none" className="h-full w-full">
              {[0, 1, 2, 3, 4].map((i) => (
                <line
                  key={i}
                  x1={0}
                  x2={600}
                  y1={(i / 4) * 72}
                  y2={(i / 4) * 72}
                  stroke="hsl(var(--border))"
                  strokeWidth={1}
                />
              ))}
              {joints.joint_names.map((_, j) => {
                const range = jointRanges[j];
                if (!range) return null;
                const span = range.max - range.min || 1;
                const points = joints.timestamps
                  .map((t, i) => {
                    const x = (t / episode.duration) * 600;
                    const v = joints.values[i][j];
                    const y = 72 - ((v - range.min) / span) * 72;
                    return `${x.toFixed(1)},${y.toFixed(1)}`;
                  })
                  .join(" ");
                return (
                  <polyline
                    key={j}
                    points={points}
                    fill="none"
                    stroke={JOINT_COLORS[j % JOINT_COLORS.length]}
                    strokeWidth={1.5}
                    strokeLinejoin="round"
                    strokeLinecap="round"
                  />
                );
              })}
              <line
                x1={scrubFrac * 600}
                x2={scrubFrac * 600}
                y1={0}
                y2={72}
                stroke="hsl(var(--foreground))"
                strokeWidth={1}
                opacity={0.55}
              />
            </svg>
          ) : (
            <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
              {episode ? "Loading joint data…" : "No episode selected"}
            </div>
          )}
          {hoverT != null && joints && episode && (
            <div
              className="pointer-events-none absolute top-1 rounded border border-border bg-popover px-1.5 py-1 text-[10px] shadow"
              style={{
                left: Math.min(
                  Math.max((hoverT / episode.duration) * 100, 0),
                  85,
                ) + "%",
              }}
            >
              <div className="mb-0.5 font-semibold">t = {hoverT.toFixed(1)}s</div>
              {joints.joint_names.map((name, j) => {
                const idx = joints.timestamps.reduce(
                  (best, t, i) =>
                    Math.abs(t - hoverT) < Math.abs(joints.timestamps[best] - hoverT) ? i : best,
                  0,
                );
                return (
                  <div key={name} className="flex items-center gap-1 tabular-nums">
                    <span
                      className="h-1.5 w-1.5 shrink-0 rounded-sm"
                      style={{ background: JOINT_COLORS[j % JOINT_COLORS.length] }}
                    />
                    {name}: {joints.values[idx][j].toFixed(1)}
                  </div>
                );
              })}
            </div>
          )}
        </div>
        {joints && (
          <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-1">
            {joints.joint_names.map((name, j) => (
              <span key={name} className="flex items-center gap-1 text-[10px] text-muted-foreground">
                <span
                  className="h-2 w-2 shrink-0 rounded-sm"
                  style={{ background: JOINT_COLORS[j % JOINT_COLORS.length] }}
                />
                {name}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

const DatasetDetailDialog: React.FC<DatasetDetailDialogProps> = ({
  repoId,
  open,
  onOpenChange,
  onStudioAction,
}) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { openStudio } = useStudio();
  const { setSelectedDataset } = useSelectedDataset();

  const [episodes, setEpisodes] = useState<EpisodeSummary[] | null>(null);
  const [episodesLoading, setEpisodesLoading] = useState(true);
  const [cameras, setCameras] = useState<string[]>([]);
  const [selectedEpisode, setSelectedEpisode] = useState<number | null>(null);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    if (!repoId || !open) return;
    const controller = new AbortController();
    setEpisodesLoading(true);
    setEpisodes(null);
    setCameras([]);
    Promise.all([
      listEpisodes(baseUrl, fetchWithHeaders, repoId, controller.signal).catch(() => null),
      getDatasetInfo(baseUrl, fetchWithHeaders, repoId, controller.signal).catch(() => null),
    ]).then(([eps, info]) => {
      if (controller.signal.aborted) return;
      setEpisodes(eps);
      setCameras(info?.cameras ?? []);
      setSelectedEpisode(eps && eps.length > 0 ? eps[0].episode_index : null);
      setEpisodesLoading(false);
    });
    return () => controller.abort();
  }, [repoId, open, baseUrl, fetchWithHeaders, reloadKey]);

  if (!repoId) return null;

  const handleTrain = () => {
    setSelectedDataset(repoId);
    openStudio("train", { train: { datasetRepoId: repoId } });
    onOpenChange(false);
    onStudioAction?.();
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex h-[85vh] max-w-6xl flex-col gap-0 overflow-hidden p-0">
        <DialogHeader className="shrink-0 space-y-0 border-b border-border px-6 py-4 text-left">
          <p className="eyebrow">episodes · cameras · joint traces</p>
          <DialogTitle className="break-all pt-1 font-mono text-base font-semibold">
            {repoId}
          </DialogTitle>
        </DialogHeader>

        <div className="grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)_300px]">
          <div className="flex min-h-0 min-w-0 flex-col gap-3 p-4">
            {episodesLoading ? (
              <div className="flex flex-1 items-center justify-center text-sm text-muted-foreground">
                Loading episodes…
              </div>
            ) : episodes && episodes.length > 0 && selectedEpisode != null ? (
              <EpisodeViewer
                repoId={repoId}
                cameras={cameras}
                episodes={episodes}
                selectedEpisode={selectedEpisode}
                onSelectEpisode={setSelectedEpisode}
              />
            ) : (
              <div className="flex flex-1 flex-col items-center justify-center gap-2 rounded-md border border-dashed border-border bg-muted/30 px-6 text-center">
                <VideoOff className="h-6 w-6 text-muted-foreground" />
                <p className="text-sm font-medium text-foreground">
                  {episodes && episodes.length === 0
                    ? "No episodes recorded yet"
                    : "No viewable footage yet"}
                </p>
                <p className="max-w-sm text-xs text-muted-foreground">
                  {episodes && episodes.length === 0
                    ? "Record at least one episode into this dataset to view its camera footage here."
                    : "This dataset isn't downloaded to this machine yet, or predates the format this viewer reads. Downloading it below (if it's on the Hub) may make it viewable."}
                </p>
              </div>
            )}
          </div>

          <div className="flex min-h-0 flex-col divide-y divide-border overflow-y-auto border-l border-border">
            <div className="flex min-h-0 flex-1 flex-col p-3">
              <p className="eyebrow mb-2">episodes {episodes ? `(${episodes.length})` : ""}</p>
              <div className="min-h-0 flex-1 overflow-y-auto">
                {episodes && episodes.length > 0 ? (
                  <div className="space-y-0.5">
                    {episodes.map((ep) => (
                      <button
                        key={ep.episode_index}
                        type="button"
                        onClick={() => setSelectedEpisode(ep.episode_index)}
                        className={`flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-xs transition-colors hover:bg-accent ${
                          selectedEpisode === ep.episode_index
                            ? "border border-border bg-accent"
                            : "border border-transparent"
                        }`}
                      >
                        <span className="w-6 shrink-0 font-mono text-[11px] text-muted-foreground">
                          {String(ep.episode_index).padStart(2, "0")}
                        </span>
                        <span className="min-w-0 flex-1 truncate">
                          Episode {ep.episode_index}
                        </span>
                        <span className="shrink-0 font-mono text-[10.5px] tabular-nums text-muted-foreground">
                          {ep.duration.toFixed(1)}s
                        </span>
                      </button>
                    ))}
                  </div>
                ) : (
                  <p className="px-1 text-xs leading-relaxed text-muted-foreground">
                    Episodes appear here once this dataset is downloaded to your
                    machine.
                  </p>
                )}
              </div>
            </div>

            <div className="p-3">
              <DatasetInfoCard
                repoId={repoId}
                onDownloaded={() => setReloadKey((k) => k + 1)}
              />
            </div>

            <div className="p-3">
              <Button onClick={handleTrain} className="w-full gap-2">
                <Boxes className="h-4 w-4" />
                Train a skill from this
              </Button>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default DatasetDetailDialog;
