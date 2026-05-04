import { useCallback, useEffect, useMemo, useState } from "react";
import { useLocation } from "react-router-dom";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import type { CameraConfig } from "@/components/recording/CameraConfiguration";

export interface RobotRecord {
  name: string;
  leader_port: string;
  follower_port: string;
  leader_config: string;
  follower_config: string;
  cameras: CameraConfig[];
  is_clean: boolean;
}

const VISIBLE_KEY = "lelab.visibleRobots";

const readVisible = (): string[] => {
  try {
    const raw = localStorage.getItem(VISIBLE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((x) => typeof x === "string") : [];
  } catch {
    return [];
  }
};

const writeVisible = (names: string[]) => {
  try {
    localStorage.setItem(VISIBLE_KEY, JSON.stringify(names));
  } catch {
    // Storage may be unavailable (private mode, quota). Failures here are non-fatal.
  }
};

export const useRobots = () => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const location = useLocation();

  const [records, setRecords] = useState<Record<string, RobotRecord>>({});
  const [visibleNames, setVisibleNames] = useState<string[]>(() => readVisible());
  const [isLoading, setIsLoading] = useState(false);

  // Re-fetch records when location changes (RobotConfigManager mounts only on Landing,
  // so this fires on initial mount and on back-navigation to Landing)
  useEffect(() => {
    let cancelled = false;
    const fetchAll = async () => {
      setIsLoading(true);
      try {
        const res = await fetchWithHeaders(`${baseUrl}/robots`);
        const data = await res.json();
        if (cancelled) return;
        const next: Record<string, RobotRecord> = {};
        for (const r of data.robots ?? []) next[r.name] = r;
        setRecords(next);
        // Prune visible names whose records vanished (deleted from another tab)
        setVisibleNames((prev) => {
          const pruned = prev.filter((n) => n in next);
          if (pruned.length !== prev.length) writeVisible(pruned);
          return pruned;
        });
      } catch (e) {
        if (!cancelled) {
          console.error("Failed to fetch robots:", e);
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    };
    fetchAll();
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders, location.key]);

  // Persist visible names to localStorage
  useEffect(() => {
    writeVisible(visibleNames);
  }, [visibleNames]);

  const addToSession = useCallback((name: string) => {
    setVisibleNames((prev) => (prev.includes(name) ? prev : [...prev, name]));
  }, []);

  const removeFromSession = useCallback((name: string) => {
    setVisibleNames((prev) => prev.filter((n) => n !== name));
  }, []);

  const createRobot = useCallback(
    async (rawName: string): Promise<boolean> => {
      const name = rawName.trim();
      if (!name) {
        toast({ title: "Missing name", description: "Robot name cannot be empty.", variant: "destructive" });
        return false;
      }
      if (/[/\\]|\.\./.test(name)) {
        toast({ title: "Invalid name", description: "Robot names cannot contain '/', '\\', or '..'", variant: "destructive" });
        return false;
      }
      try {
        const res = await fetchWithHeaders(`${baseUrl}/robots/${encodeURIComponent(name)}?create=true`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: "{}",
        });
        if (res.status === 409) {
          toast({
            title: "Already exists",
            description: `A robot named "${name}" already exists. Pick it from the dropdown or choose a different name.`,
            variant: "destructive",
          });
          return false;
        }
        if (!res.ok) {
          const text = await res.text();
          toast({ title: "Failed to create", description: text, variant: "destructive" });
          return false;
        }
        const data = await res.json();
        if (data.robot) {
          setRecords((prev) => ({ ...prev, [name]: data.robot }));
          setVisibleNames((prev) => (prev.includes(name) ? prev : [...prev, name]));
        }
        return true;
      } catch (e) {
        toast({ title: "Network error", description: String(e), variant: "destructive" });
        return false;
      }
    },
    [baseUrl, fetchWithHeaders, toast]
  );

  const deleteRobot = useCallback(
    async (name: string): Promise<boolean> => {
      try {
        const res = await fetchWithHeaders(`${baseUrl}/robots/${encodeURIComponent(name)}`, {
          method: "DELETE",
        });
        if (!res.ok) {
          const text = await res.text();
          toast({ title: "Failed to delete", description: text, variant: "destructive" });
          return false;
        }
        setRecords((prev) => {
          const { [name]: _omit, ...rest } = prev;
          return rest;
        });
        setVisibleNames((prev) => prev.filter((n) => n !== name));
        return true;
      } catch (e) {
        toast({ title: "Network error", description: String(e), variant: "destructive" });
        return false;
      }
    },
    [baseUrl, fetchWithHeaders, toast]
  );

  const visibleRecords = useMemo(
    () => visibleNames.map((n) => records[n]).filter((r): r is RobotRecord => Boolean(r)),
    [visibleNames, records]
  );

  const hiddenNames = useMemo(
    () => Object.keys(records).filter((n) => !visibleNames.includes(n)).sort(),
    [records, visibleNames]
  );

  return {
    records,
    visibleRecords,
    hiddenNames,
    isLoading,
    addToSession,
    removeFromSession,
    createRobot,
    deleteRobot,
  };
};
