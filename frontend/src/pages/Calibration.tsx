import { useLocation } from "react-router-dom";
import { AppShell } from "@/components/shell/AppShell";
import { PageHeader } from "@/components/ui/page-header";
import RobotSettingsPanel from "@/components/robot/RobotSettingsPanel";

/**
 * Full-page host for the robot settings surface. The same
 * `RobotSettingsPanel` also renders inside `RobotSettingsDialog`; this route
 * keeps deep links (the sidebar gear used to navigate here) working.
 */
const Calibration = () => {
  const location = useLocation();
  const robotName =
    (location.state as { robot_name?: string } | null)?.robot_name ?? null;

  return (
    <AppShell back={{ label: "back" }}>
      <div className="mx-auto max-w-4xl space-y-6">
        <PageHeader
          eyebrow="[ Calibration ]"
          title={robotName ? `Calibrate "${robotName}"` : "Device calibration"}
        />
        <RobotSettingsPanel robotName={robotName} variant="page" />
      </div>
    </AppShell>
  );
};

export default Calibration;
