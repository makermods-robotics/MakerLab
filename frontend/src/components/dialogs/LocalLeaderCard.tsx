import React, { useCallback, useEffect, useState } from "react";
import { Laptop, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import CalibrationLibrary from "@/components/calibration/CalibrationLibrary";
import { useApi } from "@/contexts/ApiContext";
import {
  EMPTY_LOCAL_LEADER,
  LocalLeaderConfig,
  readLocalLeader,
  writeLocalLeader,
} from "@/lib/localLeader";

interface LocalLeaderCardProps {
  /** Robot the leader belongs to (its record lives on the remote host). */
  robotName: string;
  /** Bimanual robots get a second leader pair. */
  bimanual: boolean;
}

/**
 * The leader arm's port + calibration when the leader is plugged into THIS
 * computer and the follower/cameras live on a remote host
 * (`leader_source === "remote"`).
 *
 * None of this is part of the remote's robot record — that record describes
 * the remote's own hardware. These values are read from this laptop's backend
 * and stored in this browser, and are what the leader-bridge is started with
 * (docs/remote-portal/SPEC.md, `POST /leader-bridge/start`).
 *
 * Calibrating the leader itself still happens with the corner switched to
 * "This computer": the calibration flow drives a serial port, and this window
 * is pointed at the remote machine.
 */
const LocalLeaderCard: React.FC<LocalLeaderCardProps> = ({
  robotName,
  bimanual,
}) => {
  const { baseUrl, localBaseUrl, fetchWithHeaders, activeHost } = useApi();

  const [value, setValue] = useState<LocalLeaderConfig>(EMPTY_LOCAL_LEADER);
  const [ports, setPorts] = useState<string[]>([]);
  const [portsLoading, setPortsLoading] = useState(false);

  useEffect(() => {
    setValue(readLocalLeader(baseUrl, robotName));
  }, [baseUrl, robotName]);

  // Ports come from THIS machine — the remote host's /available-ports lists
  // the server's buses, which the leader is not on.
  const fetchPorts = useCallback(async () => {
    setPortsLoading(true);
    try {
      const res = await fetchWithHeaders(`${localBaseUrl}/available-ports`);
      const data = await res.json();
      setPorts(Array.isArray(data.ports) ? data.ports : []);
    } catch (e) {
      console.error("Failed to list local ports:", e);
      setPorts([]);
    } finally {
      setPortsLoading(false);
    }
  }, [localBaseUrl, fetchWithHeaders]);

  useEffect(() => {
    fetchPorts();
  }, [fetchPorts]);

  const patch = useCallback(
    (next: Partial<LocalLeaderConfig>) => {
      setValue((prev) => {
        const merged = { ...prev, ...next };
        writeLocalLeader(baseUrl, robotName, merged);
        return merged;
      });
    },
    [baseUrl, robotName],
  );

  const leftSlot = {
    id: "left",
    label: bimanual ? "Left leader" : "Leader",
    port: value.leader_port,
    config: value.leader_config,
    setPort: (p: string) => patch({ leader_port: p }),
    setConfig: (c: string) => patch({ leader_config: c }),
  };
  const rightSlot = {
    id: "right",
    label: "Right leader",
    port: value.right_leader_port,
    config: value.right_leader_config,
    setPort: (p: string) => patch({ right_leader_port: p }),
    setConfig: (c: string) => patch({ right_leader_config: c }),
  };
  const slots = bimanual ? [leftSlot, rightSlot] : [leftSlot];

  return (
    <div className="space-y-4 rounded-md border border-border bg-muted/30 p-3">
      <div className="flex items-start gap-2">
        <Laptop className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
        <div className="space-y-1">
          <p className="text-sm font-medium">Leader on this computer</p>
          <p className="text-xs text-muted-foreground">
            Read from this laptop and kept here — not saved to{" "}
            {activeHost.name}'s record for {robotName}. The bridge sends these
            when a session starts. To calibrate this leader, switch the corner
            to “This computer” first.
          </p>
        </div>
        <Button
          type="button"
          variant="outline"
          size="icon"
          onClick={fetchPorts}
          disabled={portsLoading}
          title="Rescan this computer's ports"
          aria-label="Rescan this computer's ports"
          className="ml-auto shrink-0"
        >
          <RefreshCw
            className={portsLoading ? "h-4 w-4 animate-spin" : "h-4 w-4"}
          />
        </Button>
      </div>

      {slots.map((slot) => (
        <div key={slot.id} className="space-y-2">
          <Label htmlFor={`local-leader-port-${slot.id}`}>
            {slot.label} port
          </Label>
          <Select value={slot.port} onValueChange={slot.setPort}>
            <SelectTrigger id={`local-leader-port-${slot.id}`}>
              <SelectValue
                placeholder={
                  ports.length
                    ? "Select a port on this computer"
                    : "No arms detected here — plug in & rescan"
                }
              />
            </SelectTrigger>
            <SelectContent>
              {ports.map((p) => (
                <SelectItem key={p} value={p}>
                  <span className="font-mono text-xs">{p}</span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Label className="block">{slot.label} calibration</Label>
          <CalibrationLibrary
            device="teleop"
            apiBaseUrl={localBaseUrl}
            assignedConfig={slot.config}
            onAssignOverride={slot.setConfig}
          />
        </div>
      ))}
    </div>
  );
};

export default LocalLeaderCard;
