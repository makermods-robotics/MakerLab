import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import ReplayHeader from "@/components/replay/ReplayHeader";
import DatasetCombobox from "@/components/replay/DatasetCombobox";
import { useApi } from "@/contexts/ApiContext";
import { DatasetItem, listDatasets } from "@/lib/replayApi";

const ReplayDataset: React.FC = () => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const navigate = useNavigate();

  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [datasetsLoading, setDatasetsLoading] = useState(true);

  useEffect(() => {
    setDatasetsLoading(true);
    listDatasets(baseUrl, fetchWithHeaders)
      .then(setDatasets)
      .catch(() => setDatasets([]))
      .finally(() => setDatasetsLoading(false));
  }, [baseUrl, fetchWithHeaders]);

  const handleDatasetChange = (repoId: string | null) => {
    if (!repoId) return;
    const found = datasets.find((d) => d.repo_id === repoId);
    // Private/unknown repos bounce through huggingface.co/login?next=… so the user has a browser session before the Space tries to fetch them.
    const needsAuth = !found || found.private;
    const spacePath = `/spaces/lerobot/visualize_dataset?path=${encodeURIComponent(`/${repoId}`)}`;
    const target = needsAuth
      ? `https://huggingface.co/login?next=${encodeURIComponent(spacePath)}`
      : `https://huggingface.co${spacePath}`;
    window.open(target, "_blank", "noopener,noreferrer");
    navigate("/");
  };

  return (
    <div className="h-screen bg-black text-white flex flex-col p-4 gap-6">
      <ReplayHeader />

      <div className="flex-1 flex flex-col items-center justify-center gap-4 px-4">
        <p className="text-gray-400 text-sm text-center max-w-md">
          Pick a dataset to open it in the LeRobot dataset viewer in a new tab.
        </p>
        <div className="w-full max-w-xl">
          <DatasetCombobox
            datasets={datasets}
            loading={datasetsLoading}
            value={null}
            onChange={handleDatasetChange}
          />
        </div>
      </div>
    </div>
  );
};

export default ReplayDataset;
