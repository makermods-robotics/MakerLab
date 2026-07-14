import React from "react";
import { Globe, Lock } from "lucide-react";

/**
 * Segmented Public|Private visibility toggle (Globe/Lock, active side filled).
 * Shared by UploadDatasetDialog (upload-time visibility) and DatasetInfoCard's
 * post-upload editor so both render the identical control. `value` is the
 * PRIVATE flag (true = Private selected). `idBase` seeds the aria ids so several
 * toggles on one page stay distinct.
 */
const VisibilityToggle: React.FC<{
  value: boolean;
  onChange: (isPrivate: boolean) => void;
  idBase: string;
  disabled?: boolean;
}> = ({ value, onChange, idBase, disabled = false }) => (
  <div
    role="radiogroup"
    aria-labelledby={idBase}
    className="flex rounded-md border border-gray-700 bg-gray-800 p-0.5"
  >
    <button
      type="button"
      role="radio"
      aria-checked={!value}
      disabled={disabled}
      onClick={() => onChange(false)}
      className={`flex flex-1 items-center justify-center gap-1.5 rounded px-2 py-1 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${
        !value ? "bg-gray-600 text-white" : "text-gray-400 hover:text-gray-200"
      }`}
    >
      <Globe className="h-3 w-3" />
      Public
    </button>
    <button
      type="button"
      role="radio"
      aria-checked={value}
      disabled={disabled}
      onClick={() => onChange(true)}
      className={`flex flex-1 items-center justify-center gap-1.5 rounded px-2 py-1 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${
        value ? "bg-gray-600 text-white" : "text-gray-400 hover:text-gray-200"
      }`}
    >
      <Lock className="h-3 w-3" />
      Private
    </button>
  </div>
);

export default VisibilityToggle;
