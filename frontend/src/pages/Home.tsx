import React from "react";
import { useNavigate } from "react-router-dom";
import BoothHero from "@/components/home/BoothHero";
import { useRobots } from "@/hooks/useRobots";
import { Button } from "@/components/ui/button";

/** Placeholder home — implemented fully by worker W1 (morph transition, action
 * cards, hints). Functional core: pick a robot, land in Collect. */
const Home: React.FC = () => {
  const navigate = useNavigate();
  const { records, selectRobot } = useRobots();
  return (
    <div className="grid min-h-screen place-items-center bg-background">
      <div className="flex w-full max-w-[680px] flex-col items-center px-4 py-8">
        <BoothHero className="w-full max-w-[520px]" />
        <h1 className="mt-3 text-[19px] font-bold tracking-tight">MakerLab</h1>
        <p className="text-[13px] text-muted-foreground">Choose a robot to start</p>
        <div className="mt-6 w-full max-w-[520px]">
          {records.map((r) => (
            <Button
              key={r.name}
              variant="ghost"
              className="w-full justify-between"
              onClick={() => {
                selectRobot(r.name);
                navigate("/collect");
              }}
            >
              <span>{r.name}</span>
              <span className="font-mono text-[11px] text-muted-foreground">
                {r.mode} · {r.is_clean ? "ready" : "needs calibration"}
              </span>
            </Button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Home;
