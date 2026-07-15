import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { RobotRecord, RobotMode } from "@/hooks/useRobots";
import RobotTile from "./RobotTile";

interface RobotConfigManagerProps {
  records: Record<string, RobotRecord>;
  selectedName: string | null;
  selectedRecord: RobotRecord | null;
  availableNames: string[];
  isLoading: boolean;
  selectRobot: (name: string) => void;
  clearSelection: () => void;
  createRobot: (name: string, mode: RobotMode) => Promise<boolean>;
  renameRobot: (oldName: string, newName: string) => Promise<boolean>;
  deleteRobot: (name: string) => Promise<boolean>;
}

const RobotConfigManager: React.FC<RobotConfigManagerProps> = ({
  records,
  selectedName,
  selectedRecord,
  availableNames,
  isLoading,
  selectRobot,
  clearSelection,
  createRobot,
  renameRobot,
  deleteRobot,
}) => {
  const navigate = useNavigate();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();

  // Layout filter for the robot dropdown. This is a *view* filter, never a
  // mutator — the record's `mode` is immutable (backend 409s changes). The
  // active side doubles as the layout indicator, so the selection is the
  // source of truth: whenever a robot is selected/created, the filter snaps
  // to that robot's mode so the selected robot is never hidden.
  const [modeFilter, setModeFilter] = useState<RobotMode>(
    selectedRecord?.mode ?? "single"
  );

  // Keep the filter in sync with the selected record's mode. Records load
  // asynchronously (and a persisted selection may resolve after mount), so the
  // initializer above can run before selectedRecord exists. This guarantees
  // the active side always matches the selected robot without hiding it.
  useEffect(() => {
    if (selectedRecord) setModeFilter(selectedRecord.mode);
  }, [selectedRecord]);

  // Names of the two modes, in dropdown order (mirrors availableNames sort).
  const namesByMode = useMemo(() => {
    const single: string[] = [];
    const bimanual: string[] = [];
    for (const name of availableNames) {
      const rec = records[name];
      if (!rec) continue;
      (rec.mode === "bimanual" ? bimanual : single).push(name);
    }
    return { single, bimanual };
  }, [availableNames, records]);

  const filteredNames = namesByMode[modeFilter];

  const handleFilterChange = (mode: RobotMode) => {
    if (mode === modeFilter) return;
    setModeFilter(mode);
    // Move the selection to the first robot of that mode (availableNames is
    // sorted alphabetically; there is no MRU tracking, so this is the first
    // alphabetically). If none exist, clear the selection so the selector
    // shows its filtered empty state.
    const next = namesByMode[mode];
    if (next.length > 0) {
      selectRobot(next[0]);
    } else {
      clearSelection();
    }
  };

  // Wrap create so the filter follows a newly created robot's mode — the user
  // may pick the other layout inside the dialog, and the new robot must not
  // vanish from view. useRobots selects the new robot on success.
  const handleCreateNew = async (name: string, mode: RobotMode) => {
    const ok = await createRobot(name, mode);
    if (ok) setModeFilter(mode);
    return ok;
  };

  // Selecting an existing robot (from the dropdown) also snaps the filter to
  // its mode — selection is the source of truth for the active side.
  const handleSelect = (name: string) => {
    const rec = records[name];
    if (rec) setModeFilter(rec.mode);
    selectRobot(name);
  };

  const handleConfigure = (name: string) => {
    navigate("/calibration", { state: { robot_name: name } });
  };

  const handleTeleop = async (robot: RobotRecord) => {
    try {
      const res = await fetchWithHeaders(`${baseUrl}/move-arm`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          leader_port: robot.leader_port,
          follower_port: robot.follower_port,
          leader_config: robot.leader_config,
          follower_config: robot.follower_config,
          // Bimanual: include the mode + right arm so the backend builds a BiSO pair.
          mode: robot.mode,
          right_leader_port: robot.right_leader_port,
          right_follower_port: robot.right_follower_port,
          right_leader_config: robot.right_leader_config,
          right_follower_config: robot.right_follower_config,
          // Robot name → BiSO staging base id (bimanual). Names the per-session
          // staging dir; does not affect which calibration drives which arm.
          robot_name: robot.name,
          // Raw follower torque limit for the session (0-1000, default 400).
          max_torque_limit: robot.max_torque_limit ?? 400,
        }),
      });
      const data = await res.json();
      // The backend returns HTTP 200 with `{ success: false }` for logical
      // failures (arm not connected, already active), so gate on `data.success`
      // — not just `res.ok` — or we'd navigate to an empty teleop screen.
      if (res.ok && data.success) {
        // A success can carry a warn-but-allow arm-identity finding (e.g. the
        // arm's servos hold a different saved calibration). Make it visible —
        // it used to be silently dropped.
        if (data.warning) {
          toast({
            title: "Started with a warning",
            description: data.warning,
            duration: 10000,
          });
        } else {
          toast({
            title: "Teleoperation Started",
            description: data.message || `Started teleoperation for ${robot.name}.`,
          });
        }
        navigate("/teleoperation");
      } else {
        toast({
          title: "Error Starting Teleoperation",
          description: data.message || "Failed to start.",
          variant: "destructive",
        });
      }
    } catch (e) {
      toast({
        title: "Connection Error",
        description: "Could not connect to the backend server.",
        variant: "destructive",
      });
    }
  };

  return (
    <RobotTile
      robot={selectedRecord}
      selectedName={selectedName}
      availableNames={filteredNames}
      modeFilter={modeFilter}
      onFilterChange={handleFilterChange}
      isLoading={isLoading}
      onSelect={handleSelect}
      onCreateNew={handleCreateNew}
      onConfigure={handleConfigure}
      onTeleop={handleTeleop}
      onRename={renameRobot}
      onDelete={deleteRobot}
    />
  );
};

export default RobotConfigManager;
