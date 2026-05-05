import React from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import Logo from "@/components/Logo";
import HfAuthChip from "@/components/landing/HfAuthChip";

const ReplayHeader: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div className="flex items-center gap-4 text-3xl">
      <Button
        variant="ghost"
        size="icon"
        onClick={() => navigate("/")}
        className="text-slate-400 hover:bg-slate-800 hover:text-white rounded-lg"
      >
        <ArrowLeft className="w-5 h-5" />
      </Button>
      <Logo />
      <h1 className="font-bold text-white text-2xl">Replay Dataset</h1>
      <div className="ml-auto">
        <HfAuthChip />
      </div>
    </div>
  );
};

export default ReplayHeader;
