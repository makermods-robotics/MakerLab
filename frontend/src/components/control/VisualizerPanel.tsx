import React from "react";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { ArrowLeft, AlertTriangle } from "lucide-react";
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
          <Button
            variant="ghost"
            size="icon"
            onClick={onGoBack}
            className="text-gray-400 hover:text-white hover:bg-gray-800 flex-shrink-0"
          >
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <Logo iconOnly={true} />
          <div className="w-px h-6 bg-gray-700" />
          <h2 className="text-xl font-medium text-gray-200">Teleoperation</h2>
        </div>
        {/* Stopping (back arrow, closing the tab, an error) disconnects the
            follower, which releases motor torque — warn before it happens. */}
        <Alert className="mb-4 bg-amber-900/40 border-amber-700 text-amber-100">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            Stopping releases motor torque — the follower arm will go limp and
            fall under gravity. Move it to a low, supported pose before
            stopping or leaving this page.
          </AlertDescription>
        </Alert>
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
