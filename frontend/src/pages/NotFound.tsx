import { useLocation } from "react-router-dom";
import { useEffect } from "react";
import { AppShell } from "@/components/shell/AppShell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    console.error(
      "404 Error: User attempted to access non-existent route:",
      location.pathname
    );
  }, [location.pathname]);

  return (
    <AppShell showAuthChip={false}>
      <div className="flex min-h-[calc(100vh-7rem)] items-center justify-center">
        <div className="flex flex-col items-center gap-5 text-center">
          <Badge variant="stencil">[ 404 ]</Badge>
          <h1 className="font-display text-4xl font-bold tracking-tight">
            page not found
          </h1>
          <Button variant="ghost" asChild>
            <a href="/">← back to the workshop</a>
          </Button>
        </div>
      </div>
    </AppShell>
  );
};

export default NotFound;
