import React from "react";
import { Outlet } from "react-router-dom";
import RobotsSidebar from "@/components/shell/RobotsSidebar";
import StageDock from "@/components/shell/StageDock";

/**
 * Layout route for the stage pages (Collect / Train & Deploy / Market):
 * persistent robots sidebar on the left, floating stage dock at the bottom,
 * page content offset to the right of the rail with room above the dock.
 */
const StageLayout: React.FC = () => (
  <div className="min-h-screen bg-background">
    <RobotsSidebar />
    <main className="ml-72 min-h-screen">
      <div className="mx-auto max-w-[1120px] px-6 pb-28 pt-7">
        <Outlet />
      </div>
    </main>
    <StageDock />
  </div>
);

export default StageLayout;
