import React from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import UrdfViewer from "../UrdfViewer";
import Logo from "@/components/Logo";

interface VisualizerPanelProps {
  onGoBack: () => void;
  className?: string;
  /** Render a second arm viewer (driven by the "joints_right" stream). */
  bimanual?: boolean;
}

const VisualizerPanel: React.FC<VisualizerPanelProps> = ({
  onGoBack,
  className,
  bimanual = false,
}) => {
  return (
    <div
      className={cn(
        "w-full p-2 sm:p-4 space-y-4 lg:space-y-0 lg:space-x-4 flex flex-col lg:flex-row",
        className
      )}
    >
      <div className="bg-gray-900 rounded-lg p-4 flex-1 flex flex-col">
        <div className="flex items-center gap-4 mb-4">
          <Logo iconOnly={true} />
          <div className="w-px h-6 bg-gray-700" />
          <h2 className="text-xl font-medium text-gray-200">Teleoperation</h2>
          <Button
            onClick={onGoBack}
            className="ml-auto bg-red-500 hover:bg-red-600 text-white flex-shrink-0"
          >
            Done
          </Button>
        </div>
        {/* No standing torque warning here: stops are graceful (the arm
            drives back to its session-start pose before torque releases) and
            the stop toast explains the behavior at the moment it happens.
            Only error stops release in place. */}
        {bimanual ? (
          <div className="flex-1 flex flex-col sm:flex-row gap-2 min-h-[50vh] lg:min-h-0">
            <div className="flex-1 flex flex-col">
              <span className="text-xs text-gray-400 mb-1">Left arm</span>
              <div className="flex-1 bg-black rounded border border-gray-800 min-h-[25vh]">
                <UrdfViewer jointsKey="joints" />
              </div>
            </div>
            <div className="flex-1 flex flex-col">
              <span className="text-xs text-gray-400 mb-1">Right arm</span>
              <div className="flex-1 bg-black rounded border border-gray-800 min-h-[25vh]">
                <UrdfViewer jointsKey="joints_right" />
              </div>
            </div>
          </div>
        ) : (
          <div className="flex-1 bg-black rounded border border-gray-800 min-h-[50vh] lg:min-h-0">
            <UrdfViewer />
          </div>
        )}
      </div>
    </div>
  );
};

export default VisualizerPanel;
