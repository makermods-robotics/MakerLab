import React from "react";
import { useNavigate } from "react-router-dom";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { RobotRecord, RobotMode } from "@/hooks/useRobots";
import RobotTile from "./RobotTile";

interface RobotConfigManagerProps {
  selectedName: string | null;
  selectedRecord: RobotRecord | null;
  availableNames: string[];
  isLoading: boolean;
  selectRobot: (name: string) => void;
  createRobot: (name: string) => Promise<boolean>;
  renameRobot: (oldName: string, newName: string) => Promise<boolean>;
  setRobotMode: (name: string, mode: RobotMode) => Promise<boolean>;
  deleteRobot: (name: string) => Promise<boolean>;
}

const RobotConfigManager: React.FC<RobotConfigManagerProps> = ({
  selectedName,
  selectedRecord,
  availableNames,
  isLoading,
  selectRobot,
  createRobot,
  renameRobot,
  setRobotMode,
  deleteRobot,
}) => {
  const navigate = useNavigate();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();

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
          // Follower torque limit for the session (10-100% of full power).
          motor_power: robot.motor_power ?? 100,
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
      availableNames={availableNames}
      isLoading={isLoading}
      onSelect={selectRobot}
      onCreateNew={createRobot}
      onConfigure={handleConfigure}
      onTeleop={handleTeleop}
      onRename={renameRobot}
      onSetMode={setRobotMode}
      onDelete={deleteRobot}
    />
  );
};

export default RobotConfigManager;
