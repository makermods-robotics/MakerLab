import React, { useEffect, useRef } from "react";
import { useApi } from "@/contexts/ApiContext";

interface BackendCameraStreamProps {
  /** cv2 camera index on the server (CameraConfig.camera_index). */
  cameraIndex: number;
  /** Fired when the stream fails (camera busy, recording active, server down). */
  onError?: () => void;
  className?: string;
}

/**
 * MJPEG `<img>` stream of a camera attached to the *server* machine
 * (GET /camera-preview/{index}).
 *
 * Fallback for headless deployments (e.g. a Jetson on the LAN): the browser's
 * getUserMedia only sees the viewing machine's cameras, so a camera plugged
 * into the server has no matching deviceId — this streams it from the backend
 * instead. The parent must unmount this component to pause/release the
 * preview; the unmount cleanup clears the img src so the browser drops the
 * HTTP connection and the server releases the shared capture.
 */
const BackendCameraStream: React.FC<BackendCameraStreamProps> = ({
  cameraIndex,
  onError,
  className,
}) => {
  const { baseUrl } = useApi();
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    const img = imgRef.current;
    return () => {
      // Detaching an <img> doesn't reliably abort its in-flight request;
      // clearing src does, which is what lets the backend release the camera.
      if (img) img.src = "";
    };
  }, []);

  return (
    <img
      ref={imgRef}
      src={`${baseUrl}/camera-preview/${cameraIndex}`}
      onError={onError}
      className={className}
      alt="Server camera preview"
    />
  );
};

export default BackendCameraStream;
