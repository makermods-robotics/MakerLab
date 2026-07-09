import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertTriangle, RefreshCw, Search } from "lucide-react";

import { MarketListingCard } from "@/components/market/MarketListingCard";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useApi } from "@/contexts/ApiContext";
import { useSelectedDataset } from "@/hooks/useSelectedDataset";
import { useToast } from "@/hooks/use-toast";
import {
  HubModel,
  importModel,
  listHubJobs,
  listJobs,
} from "@/lib/jobsApi";
import {
  DatasetInfo,
  DatasetItem,
  getDatasetInfo,
  listDatasets,
} from "@/lib/replayApi";

type MarketTab = "all" | "datasets" | "models";
type LoadErrors = Partial<Record<"datasets" | "models" | "imports", string>>;

const tabs: Array<{ id: MarketTab; label: string }> = [
  { id: "all", label: "All" },
  { id: "datasets", label: "Datasets" },
  { id: "models", label: "Models" },
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

function shortRepoName(repoId: string): string {
  if (!repoId.includes("/")) return repoId;
  return repoId.split("/").slice(1).join("/");
}

function datasetMeta(dataset: DatasetItem, info?: DatasetInfo): string {
  const privacy = dataset.private ? "private" : "public";
  const episodes =
    info && info.total_episodes > 0
      ? `${info.total_episodes} episode${info.total_episodes === 1 ? "" : "s"}`
      : "episodes unavailable";
  return `${episodes} · ${privacy} · ${relativeTime(dataset.last_modified)}`;
}

function modelMeta(model: HubModel): string {
  const privacy = model.private ? "private" : "public";
  return `Hub model · ${privacy} · ${relativeTime(model.last_modified)}`;
}

function matchesSearch(value: string, query: string): boolean {
  return !query || value.toLowerCase().includes(query);
}

function MarketSkeletonGrid() {
  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
      {Array.from({ length: 6 }).map((_, index) => (
        <div
          key={index}
          className="overflow-hidden rounded-xl border border-border bg-card shadow-sm"
        >
          <div className="h-[130px] border-b border-dashed border-border bg-muted" />
          <div className="grid gap-3 p-4">
            <div className="h-4 w-2/3 rounded bg-secondary animate-pulse" />
            <div className="flex gap-2">
              <div className="h-5 w-16 rounded-full bg-secondary animate-pulse" />
              <div className="h-5 w-14 rounded-full bg-secondary animate-pulse" />
            </div>
            <div className="h-3 w-full rounded bg-secondary animate-pulse" />
            <div className="ml-auto h-9 w-28 rounded-md bg-secondary animate-pulse" />
          </div>
        </div>
      ))}
    </div>
  );
}

const Market: React.FC = () => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { selectedDataset, setSelectedDataset } = useSelectedDataset();
  const { toast } = useToast();

  const [activeTab, setActiveTab] = useState<MarketTab>("all");
  const [search, setSearch] = useState("");
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [datasetInfos, setDatasetInfos] = useState<Record<string, DatasetInfo>>(
    {},
  );
  const [models, setModels] = useState<HubModel[]>([]);
  const [importedRepoIds, setImportedRepoIds] = useState<Set<string>>(
    () => new Set(),
  );
  const [importingRepoId, setImportingRepoId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [errors, setErrors] = useState<LoadErrors>({});

  const refresh = useCallback(
    async (signal?: AbortSignal) => {
      setLoading(true);
      setErrors({});

      const [datasetResult, hubResult, jobsResult] = await Promise.allSettled([
        listDatasets(baseUrl, fetchWithHeaders, signal),
        listHubJobs(baseUrl, fetchWithHeaders, signal),
        listJobs(baseUrl, fetchWithHeaders, 100, signal),
      ]);

      if (signal?.aborted) return;

      const nextErrors: LoadErrors = {};

      if (datasetResult.status === "fulfilled") {
        const hubDatasets = datasetResult.value.filter(
          (dataset) => dataset.source === "hub" || dataset.source === "both",
        );
        setDatasets(hubDatasets);

        const localInfoDatasets = hubDatasets.filter(
          (dataset) => dataset.source === "both",
        );
        const infoResults = await Promise.allSettled(
          localInfoDatasets.map(async (dataset) => {
            const info = await getDatasetInfo(
              baseUrl,
              fetchWithHeaders,
              dataset.repo_id,
              signal,
            );
            return [dataset.repo_id, info] as const;
          }),
        );

        if (signal?.aborted) return;

        const nextInfos: Record<string, DatasetInfo> = {};
        for (const result of infoResults) {
          if (result.status === "fulfilled") {
            nextInfos[result.value[0]] = result.value[1];
          }
        }
        setDatasetInfos(nextInfos);
      } else {
        const reason = datasetResult.reason;
        nextErrors.datasets =
          reason instanceof Error ? reason.message : String(reason);
      }

      if (hubResult.status === "fulfilled") {
        setModels(hubResult.value.models);
      } else {
        const reason = hubResult.reason;
        nextErrors.models =
          reason instanceof Error ? reason.message : String(reason);
      }

      if (jobsResult.status === "fulfilled") {
        setImportedRepoIds(
          new Set(
            jobsResult.value
              .map((job) => job.hf_repo_id?.toLowerCase())
              .filter((repoId): repoId is string => Boolean(repoId)),
          ),
        );
      } else {
        const reason = jobsResult.reason;
        nextErrors.imports =
          reason instanceof Error ? reason.message : String(reason);
      }

      setErrors(nextErrors);
      setLoading(false);
    },
    [baseUrl, fetchWithHeaders],
  );

  useEffect(() => {
    const controller = new AbortController();
    void refresh(controller.signal);
    return () => controller.abort();
  }, [refresh]);

  const query = search.trim().toLowerCase();

  const visibleDatasets = useMemo(
    () =>
      activeTab === "models"
        ? []
        : datasets.filter((dataset) => matchesSearch(dataset.repo_id, query)),
    [activeTab, datasets, query],
  );

  const visibleModels = useMemo(
    () =>
      activeTab === "datasets"
        ? []
        : models.filter((model) => matchesSearch(model.repo_id, query)),
    [activeTab, models, query],
  );

  const hasVisibleItems = visibleDatasets.length > 0 || visibleModels.length > 0;
  const errorMessages = Object.values(errors).filter(Boolean);

  const handleUseDataset = (repoId: string) => {
    if (selectedDataset === repoId) return;
    setSelectedDataset(repoId);
    toast({
      title: "Dataset selected",
      description: `${repoId} will appear in Collect.`,
    });
  };

  const handleImportModel = async (repoId: string) => {
    if (importedRepoIds.has(repoId.toLowerCase()) || importingRepoId) return;
    setImportingRepoId(repoId);
    try {
      const record = await importModel(baseUrl, fetchWithHeaders, repoId);
      setImportedRepoIds((prev) => {
        const next = new Set(prev);
        next.add((record.hf_repo_id ?? repoId).toLowerCase());
        return next;
      });
      toast({
        title: record.already_imported ? "Already imported" : "Model imported",
        description: `${repoId} will appear in Train & Deploy.`,
      });
    } catch (e) {
      toast({
        title: "Import failed",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setImportingRepoId(null);
    }
  };

  return (
    <main className="space-y-4">
      <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div>
          <h1 className="text-[22px] font-bold tracking-tight text-foreground">
            Market
          </h1>
          <p className="mt-1 text-[13.5px] text-muted-foreground">
            Hub datasets and models for your robot bench.
          </p>
        </div>
        <div className="relative w-full md:w-[300px]">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            className="pl-9"
            placeholder="Search datasets, models"
            aria-label="Search datasets and models"
          />
        </div>
      </div>

      <div className="flex w-full rounded-lg bg-secondary p-1 md:w-fit" aria-label="Market filters">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => setActiveTab(tab.id)}
            className={`flex-1 rounded-md px-4 py-1.5 text-sm font-medium transition-colors md:flex-none ${
              activeTab === tab.id
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {errorMessages.length > 0 ? (
        <div className="flex flex-col gap-3 rounded-xl border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive md:flex-row md:items-center md:justify-between">
          <div className="flex gap-2">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <div>
              <p className="font-medium">Some market listings could not load.</p>
              <p className="mt-1 text-destructive/90">
                {errorMessages.join(" · ")}
              </p>
            </div>
          </div>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => void refresh()}
            className="self-start border-destructive/40 text-destructive hover:bg-destructive/10 md:self-center"
          >
            <RefreshCw className="h-4 w-4" />
            Retry
          </Button>
        </div>
      ) : null}

      {loading ? (
        <MarketSkeletonGrid />
      ) : hasVisibleItems ? (
        <section
          className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3"
          aria-label="Market listings"
        >
          {visibleDatasets.map((dataset) => (
            <MarketListingCard
              key={`dataset:${dataset.repo_id}`}
              kind="dataset"
              name={dataset.repo_id}
              source={dataset.source}
              meta={datasetMeta(dataset, datasetInfos[dataset.repo_id])}
              actionLabel="Use in Collect"
              completeLabel="Selected ✓"
              complete={selectedDataset === dataset.repo_id}
              onAction={() => handleUseDataset(dataset.repo_id)}
            />
          ))}
          {visibleModels.map((model) => {
            const imported = importedRepoIds.has(model.repo_id.toLowerCase());
            return (
              <MarketListingCard
                key={`model:${model.repo_id}`}
                kind="model"
                name={shortRepoName(model.repo_id)}
                source="hub"
                meta={modelMeta(model)}
                actionLabel="Import"
                completeLabel="Imported ✓"
                complete={imported}
                loading={importingRepoId === model.repo_id}
                onAction={() => void handleImportModel(model.repo_id)}
              />
            );
          })}
        </section>
      ) : (
        <div className="rounded-xl border border-dashed border-border bg-card p-8 text-center">
          <Badge variant="outline">
            {activeTab === "datasets"
              ? "datasets"
              : activeTab === "models"
                ? "models"
                : "market"}
          </Badge>
          <p className="mt-3 text-sm font-medium text-foreground">
            No listings found
          </p>
          <p className="mt-1 text-sm text-muted-foreground">
            Try a different search or refresh the market.
          </p>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => void refresh()}
            className="mt-4"
          >
            <RefreshCw className="h-4 w-4" />
            Retry
          </Button>
        </div>
      )}

      <p className="text-xs text-muted-foreground">
        Datasets you use appear in Collect. Imported models appear in Train &
        Deploy.
      </p>
    </main>
  );
};

export default Market;
