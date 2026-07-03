import React from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { ArrowRight, CheckCircle, Trash2 } from "lucide-react";
import { useSelectedDataset } from "@/hooks/useSelectedDataset";

interface RecordedDatasetInfo {
  dataset_repo_id: string;
  // Set by the recording flow when the session saved zero episodes and the
  // backend discarded the (empty) dataset directory — nothing is on disk.
  discarded_empty?: boolean;
}

/**
 * Post-recording landing page. Recording flow routes here (see Recording.tsx)
 * once a session ends. Upload no longer happens here — it moved to the dataset
 * info card on the home page, alongside the Hub sync status. This page is now a
 * thin pointer: it confirms the dataset was saved locally and sends the user
 * home with that dataset preselected, so the info card (and its "Upload to Hub"
 * action) is one click away. The /upload route is kept so the recording flow's
 * navigate("/upload", …) keeps working.
 */
const Upload = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { setSelectedDataset } = useSelectedDataset();

  const recorded = location.state?.datasetInfo as
    | RecordedDatasetInfo
    | undefined;
  const discardedEmpty = recorded?.discarded_empty ?? false;
  // A discarded (empty) session left nothing on disk, so there's no repo id to
  // link or preselect.
  const repoId = discardedEmpty ? null : recorded?.dataset_repo_id ?? null;

  const goHome = () => {
    // Preselect the just-recorded dataset so the home page opens straight onto
    // its info card (same mechanism training reads — useSelectedDataset).
    if (repoId) setSelectedDataset(repoId);
    navigate("/");
  };

  if (discardedEmpty) {
    return (
      <div className="min-h-screen bg-black text-white flex items-center justify-center p-8">
        <div className="max-w-md w-full text-center space-y-6">
          <div className="flex justify-center">
            <Trash2 className="w-12 h-12 text-gray-500" />
          </div>
          <div className="space-y-2">
            <h1 className="text-2xl font-bold">No episodes were recorded</h1>
            <p className="text-gray-400">
              Nothing was saved — the empty dataset was discarded so it doesn't
              take up disk space.
            </p>
          </div>
          <Button
            onClick={goHome}
            className="bg-blue-500 hover:bg-blue-600 text-white"
          >
            Go to home page
            <ArrowRight className="w-4 h-4 ml-2" />
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black text-white flex items-center justify-center p-8">
      <div className="max-w-md w-full text-center space-y-6">
        <div className="flex justify-center">
          <CheckCircle className="w-12 h-12 text-green-500" />
        </div>
        <div className="space-y-2">
          <h1 className="text-2xl font-bold">Dataset saved locally</h1>
          {repoId && (
            <p className="font-mono text-sm text-gray-400 break-all">{repoId}</p>
          )}
          <p className="text-gray-400">
            Review it and upload it to the Hub from the home page.
          </p>
        </div>
        <Button
          onClick={goHome}
          className="bg-blue-500 hover:bg-blue-600 text-white"
        >
          Go to home page
          <ArrowRight className="w-4 h-4 ml-2" />
        </Button>
      </div>
    </div>
  );
};

export default Upload;
