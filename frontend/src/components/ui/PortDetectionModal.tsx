import React, { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Loader2, CheckCircle, AlertCircle } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useApi } from "@/contexts/ApiContext";

interface PortDetectionModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  robotType: "leader" | "follower";
  onPortDetected: (port: string) => void;
}

const SUCCESS_HOLD_MS = 2000;

const PortDetectionModal: React.FC<PortDetectionModalProps> = ({
  open,
  onOpenChange,
  robotType,
  onPortDetected,
}) => {
  const [step, setStep] = useState<"detecting" | "success" | "error">(
    "detecting"
  );
  const [detectedPort, setDetectedPort] = useState<string>("");
  const [error, setError] = useState<string>("");
  const cancelledRef = useRef(false);
  const abortRef = useRef<AbortController | null>(null);
  const successTimerRef = useRef<number | null>(null);
  const { toast } = useToast();
  const { baseUrl, fetchWithHeaders } = useApi();

  const runDetection = async () => {
    try {
      abortRef.current = new AbortController();
      const startResponse = await fetchWithHeaders(
        `${baseUrl}/start-port-detection`,
        {
          method: "POST",
          body: JSON.stringify({ robot_type: robotType }),
          signal: abortRef.current.signal,
        }
      );
      const startData = await startResponse.json();
      if (cancelledRef.current) return;
      if (startData.status !== "success") {
        throw new Error(startData.message || "Failed to start port detection");
      }
      const portsBefore: string[] = startData.data.ports_before;

      // Poll the backend in a loop. Each call waits up to 15s for an unplug;
      // we silently retry on timeout so the user has unlimited time to read
      // and act. The loop ends on success, abort, or a non-timeout failure.
      while (!cancelledRef.current) {
        abortRef.current = new AbortController();
        const response = await fetchWithHeaders(
          `${baseUrl}/detect-port-after-disconnect`,
          {
            method: "POST",
            body: JSON.stringify({ ports_before: portsBefore }),
            signal: abortRef.current.signal,
          }
        );
        const data = await response.json();
        if (cancelledRef.current) return;

        if (data.status === "success") {
          setDetectedPort(data.port);
          await savePort(data.port);
          if (cancelledRef.current) return;
          setStep("success");
          toast({
            title: "Port Detected Successfully",
            description: `${robotType} port detected: ${data.port}`,
          });
          successTimerRef.current = window.setTimeout(() => {
            if (cancelledRef.current) return;
            onPortDetected(data.port);
            onOpenChange(false);
          }, SUCCESS_HOLD_MS);
          return;
        }

        const message =
          typeof data.message === "string" ? data.message : "";
        if (message.includes("Timed out")) continue;
        throw new Error(message || "Failed to detect port");
      }
    } catch (e) {
      if (cancelledRef.current) return;
      if (e instanceof DOMException && e.name === "AbortError") return;
      console.error("Port detection failed:", e);
      setError(e instanceof Error ? e.message : "Unknown error");
      setStep("error");
    }
  };

  const savePort = async (port: string) => {
    try {
      await fetchWithHeaders(`${baseUrl}/save-robot-port`, {
        method: "POST",
        body: JSON.stringify({ robot_type: robotType, port }),
      });
    } catch (e) {
      console.error("Error saving port:", e);
    }
  };

  useEffect(() => {
    if (!open) return;
    cancelledRef.current = false;
    setStep("detecting");
    setError("");
    setDetectedPort("");
    runDetection();
    return () => {
      cancelledRef.current = true;
      abortRef.current?.abort();
      if (successTimerRef.current !== null) {
        window.clearTimeout(successTimerRef.current);
        successTimerRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const handleCancel = () => {
    onOpenChange(false);
  };

  const handleRetry = () => {
    cancelledRef.current = false;
    abortRef.current?.abort();
    setStep("detecting");
    setError("");
    setDetectedPort("");
    runDetection();
  };

  const renderStepContent = () => {
    switch (step) {
      case "detecting":
        return (
          <div className="space-y-6 text-center">
            <Loader2 className="w-16 h-16 text-blue-500 mx-auto animate-spin" />
            <div className="space-y-2">
              <h3 className="text-lg font-semibold text-white">
                Unplug the {robotType} arm
              </h3>
              <p className="text-gray-400">
                Disconnect the {robotType} robot arm from USB. The port will be
                detected automatically.
              </p>
            </div>
            <div className="flex justify-center">
              <Button
                onClick={handleCancel}
                variant="outline"
                className="border-gray-500 hover:border-gray-200 text-gray-700 hover:text-gray-900 dark:text-gray-300 dark:hover:text-white px-8 py-2"
              >
                Cancel
              </Button>
            </div>
          </div>
        );

      case "success":
        return (
          <div className="space-y-6 text-center">
            <CheckCircle className="w-16 h-16 text-green-500 mx-auto" />
            <div className="space-y-2">
              <h3 className="text-lg font-semibold text-white">
                Port Detected
              </h3>
              <p className="text-xl font-mono text-green-400 bg-gray-800 px-4 py-2 rounded inline-block">
                {detectedPort}
              </p>
            </div>
          </div>
        );

      case "error":
        return (
          <div className="space-y-6 text-center">
            <AlertCircle className="w-16 h-16 text-red-500 mx-auto" />
            <div className="space-y-2">
              <h3 className="text-lg font-semibold text-white">
                Detection Failed
              </h3>
              <div className="bg-red-900/20 border border-red-800 rounded-lg p-3">
                <p className="text-red-400 text-sm">{error}</p>
              </div>
            </div>
            <div className="flex gap-4 justify-center">
              <Button
                onClick={handleRetry}
                className="bg-blue-500 hover:bg-blue-600 text-white px-8 py-2"
              >
                Try Again
              </Button>
              <Button
                onClick={handleCancel}
                variant="outline"
                className="border-gray-500 hover:border-gray-200 text-gray-700 hover:text-gray-900 dark:text-gray-300 dark:hover:text-white px-8 py-2"
              >
                Cancel
              </Button>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-gray-900 border-gray-800 text-white sm:max-w-[500px] p-8">
        <DialogHeader>
          <DialogTitle className="text-white text-center text-xl font-bold">
            Port Detection
          </DialogTitle>
          <DialogDescription className="text-gray-400 text-center">
            Detect the USB port for your {robotType} arm
          </DialogDescription>
        </DialogHeader>

        <div className="py-4">{renderStepContent()}</div>
      </DialogContent>
    </Dialog>
  );
};

export default PortDetectionModal;
