
import React from "react";
import { AppShell } from "@/components/shell/AppShell";
import { Card, CardContent } from "@/components/ui/card";
import { Eyebrow } from "@/components/ui/eyebrow";

const EditDataset = () => {
  return (
    <AppShell back={{ to: "/" }}>
      <div className="flex min-h-[calc(100vh-7rem)] items-center justify-center">
        <Card className="w-full max-w-md">
          <CardContent className="p-8 text-center">
            <Eyebrow className="mb-4">[ Under construction ]</Eyebrow>
            <h1 className="font-display text-3xl font-bold tracking-tight">
              edit dataset
            </h1>
            <p className="mt-3 text-sm text-muted-foreground">
              Dataset editing tools are being prepared for this workspace.
            </p>
          </CardContent>
        </Card>
      </div>
    </AppShell>
  );
};

export default EditDataset;
