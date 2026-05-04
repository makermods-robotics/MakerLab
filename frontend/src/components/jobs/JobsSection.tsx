import React, { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { JobRecord, listJobs, stopJob, deleteJob } from "@/lib/jobsApi";
import JobCard from "./JobCard";
import { RefreshCw } from "lucide-react";

const POLL_INTERVAL_MS = 5000;
const LIMIT = 10;

const JobsSection: React.FC = () => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();

  const [jobs, setJobs] = useState<JobRecord[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const next = await listJobs(baseUrl, fetchWithHeaders, LIMIT);
      setJobs(next);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [baseUrl, fetchWithHeaders]);

  useEffect(() => {
    let cancelled = false;
    refresh();
    const id = setInterval(() => {
      if (!cancelled) refresh();
    }, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [refresh]);

  const handleStop = async (id: string) => {
    try {
      await stopJob(baseUrl, fetchWithHeaders, id);
      toast({ title: "Job stopping" });
      refresh();
    } catch (e) {
      toast({
        title: "Stop failed",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await deleteJob(baseUrl, fetchWithHeaders, id);
      toast({ title: "Job removed" });
      refresh();
    } catch (e) {
      toast({
        title: "Delete failed",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  return (
    <section className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white">Jobs</h2>
        <Button
          variant="ghost"
          size="icon"
          onClick={refresh}
          className="h-7 w-7 text-slate-400 hover:text-white"
          aria-label="Refresh jobs"
        >
          <RefreshCw className="w-4 h-4" />
        </Button>
      </div>
      {error ? (
        <p className="text-sm text-red-300">Couldn't load jobs: {error}</p>
      ) : jobs.length === 0 ? (
        <p className="text-sm text-slate-500">
          No training jobs yet. Start one from the Training page.
        </p>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {jobs.map((job) => (
            <JobCard key={job.id} job={job} onStop={handleStop} onDelete={handleDelete} />
          ))}
        </div>
      )}
    </section>
  );
};

export default JobsSection;
