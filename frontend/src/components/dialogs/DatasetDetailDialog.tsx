import React from "react";
import { Boxes } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { useStudio } from "@/contexts/StudioContext";
import { useSelectedDataset } from "@/hooks/useSelectedDataset";
import DatasetInfoCard from "@/components/landing/DatasetInfoCard";

export interface DatasetDetailDialogProps {
  repoId: string | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Called when an action navigates to the studio, so a parent surface that
   * would otherwise cover the studio (e.g. the library sheet) can close too. */
  onStudioAction?: () => void;
}

/**
 * Dataset detail — wraps the existing DatasetInfoCard (unmodified) in a dialog
 * and adds the market action: train a skill from this dataset (→ Train panel,
 * dataset preselected). It stamps the shared selected-dataset key so the
 * studio panels pick it up, exactly like DatasetInfoCard's own ecosystem.
 * (Recording on top of an existing dataset was removed — grow a dataset by
 * recording a new one and merging.)
 */
const DatasetDetailDialog: React.FC<DatasetDetailDialogProps> = ({
  repoId,
  open,
  onOpenChange,
  onStudioAction,
}) => {
  const { openStudio } = useStudio();
  const { setSelectedDataset } = useSelectedDataset();

  if (!repoId) return null;

  const handleTrain = () => {
    setSelectedDataset(repoId);
    openStudio("train", { train: { datasetRepoId: repoId } });
    onOpenChange(false);
    onStudioAction?.();
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="break-all font-mono text-base">
            {repoId}
          </DialogTitle>
        </DialogHeader>

        <DatasetInfoCard repoId={repoId} />

        <div className="flex flex-col gap-2">
          <Button onClick={handleTrain} className="w-full gap-2">
            <Boxes className="h-4 w-4" />
            Train a skill from this
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default DatasetDetailDialog;
