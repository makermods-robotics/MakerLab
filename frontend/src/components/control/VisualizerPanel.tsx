import React from "react";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import UrdfViewer from "../UrdfViewer";

interface VisualizerPanelProps {
  onGoBack: () => void;
  className?: string;
  /** Render a second arm viewer (driven by the "joints_right" stream). */
  bimanual?: boolean;
  /** Optional content rendered as a column beside the 3D viewer (e.g. a camera panel). */
  rightSlot?: React.ReactNode;
}

const VisualizerPanel: React.FC<VisualizerPanelProps> = ({
  className,
  bimanual = false,
  rightSlot,
}) => {
  return (
    <div
      className={cn(
        "flex w-full flex-col gap-4 p-3 sm:p-4 lg:flex-row",
        className
      )}
    >
      <Card variant="flat" className="flex flex-1 flex-col overflow-hidden p-4">
        {/* No standing torque warning here: stops are graceful (the arm
            drives back to its session-start pose before torque releases) and
            the stop toast explains the behavior at the moment it happens.
            Only error stops release in place. */}
        {bimanual ? (
          <div className="flex min-h-[50vh] flex-1 flex-col gap-3 sm:flex-row lg:min-h-0">
            <div className="flex flex-1 flex-col">
              <span className="mb-2 font-mono text-xs text-muted-foreground">Left arm</span>
              <Card variant="flat" className="min-h-[25vh] flex-1 overflow-hidden">
                <UrdfViewer jointsKey="joints" />
              </Card>
            </div>
            <div className="flex flex-1 flex-col">
              <span className="mb-2 font-mono text-xs text-muted-foreground">Right arm</span>
              <Card variant="flat" className="min-h-[25vh] flex-1 overflow-hidden">
                <UrdfViewer jointsKey="joints_right" />
              </Card>
            </div>
          </div>
        ) : (
          <Card variant="flat" className="min-h-[50vh] flex-1 overflow-hidden lg:min-h-0">
            <UrdfViewer />
          </Card>
        )}
      </Card>
      {rightSlot && (
        <div className="flex flex-col lg:w-96">{rightSlot}</div>
      )}
    </div>
  );
};

export default VisualizerPanel;
