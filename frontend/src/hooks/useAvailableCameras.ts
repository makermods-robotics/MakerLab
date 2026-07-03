import { useCallback, useEffect, useState } from "react";
import { useApi } from "@/contexts/ApiContext";

export interface AvailableCamera {
  index: number;
  name: string;
  deviceId: string;
  available: boolean;
}

const norm = (s: string) => s.toLowerCase().replace(/\s+/g, " ").trim();

interface UseAvailableCamerasOptions {
  /** When false, do nothing. Use to gate on modal open. */
  enabled?: boolean;
}

/**
 * Enumerates cv2 camera indices from `/available-cameras` and merges each
 * with the matching browser deviceId (by AVFoundation localizedName) so
 * callers can render a preview alongside the bound dropdowns. Refreshes on
 * USB hotplug.
 */
export function useAvailableCameras({
  enabled = true,
}: UseAvailableCamerasOptions = {}) {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [cameras, setCameras] = useState<AvailableCamera[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const refresh = useCallback(async (): Promise<AvailableCamera[]> => {
    setIsLoading(true);
    try {
      // navigator.mediaDevices only exists in secure contexts (https or
      // localhost). Without it, skip browser matching — backend cv2 indices
      // still list and recording works, there are just no live previews.
      let browserDevices: { deviceId: string; label: string }[] = [];
      if (navigator.mediaDevices) {
        // Need a permission grant before enumerateDevices() returns labels.
        try {
          const probe = await navigator.mediaDevices.getUserMedia({ video: true });
          probe.getTracks().forEach((t) => t.stop());
        } catch {
          // ignore — we'll still try to enumerate, just without labels
        }

        browserDevices = (await navigator.mediaDevices.enumerateDevices())
          .filter((d) => d.kind === "videoinput")
          .map((d) => ({ deviceId: d.deviceId, label: d.label }));
      }

      const r = await fetchWithHeaders(`${baseUrl}/available-cameras`);
      if (!r.ok) {
        setCameras([]);
        return [];
      }
      const data = await r.json();
      const backendCams: {
        index: number;
        name?: string;
        available: boolean;
      }[] = data.cameras ?? [];

      // Browser's MediaDeviceInfo.label starts with AVFoundation's localizedName
      // but Chrome often appends "(vendorId:productId)". Match by exact, then
      // prefix, then either-contains.
      const used = new Set<string>();
      const merged: AvailableCamera[] = backendCams.map((cam) => {
        const label = cam.name || `Camera ${cam.index}`;
        const target = norm(label);
        const candidates = browserDevices.filter(
          (d) => !used.has(d.deviceId) && d.label
        );
        const match =
          candidates.find((d) => norm(d.label) === target) ||
          candidates.find((d) => norm(d.label).startsWith(target)) ||
          candidates.find(
            (d) => norm(d.label).includes(target) || target.includes(norm(d.label))
          );
        if (match) used.add(match.deviceId);
        return {
          index: cam.index,
          name: label,
          deviceId: match?.deviceId ?? "",
          available: cam.available,
        };
      });
      setCameras(merged);
      return merged;
    } catch {
      setCameras([]);
      return [];
    } finally {
      setIsLoading(false);
    }
  }, [baseUrl, fetchWithHeaders]);

  useEffect(() => {
    if (!enabled) return;
    refresh();
    const md = navigator.mediaDevices;
    if (!md) return; // insecure context: no hotplug events, refresh() ran once
    const handler = () => refresh();
    md.addEventListener("devicechange", handler);
    return () => md.removeEventListener("devicechange", handler);
  }, [enabled, refresh]);

  return { cameras, isLoading, refresh };
}
