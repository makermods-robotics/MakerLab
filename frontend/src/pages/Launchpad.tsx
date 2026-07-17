import { useState } from "react";
import { Library } from "lucide-react";
import { Button } from "@/components/ui/button";
import BrandMark from "@/components/BrandMark";
import Footer from "@/components/Footer";
import HfAuthChip from "@/components/landing/HfAuthChip";
import UsageInstructionsModal from "@/components/landing/UsageInstructionsModal";
import Hero from "@/components/launchpad/Hero";
import SkillSlider from "@/components/launchpad/SkillSlider";
import NewSkillBanner from "@/components/launchpad/NewSkillBanner";
import ActivityStrip from "@/components/launchpad/ActivityStrip";
import LibrarySheet from "@/components/launchpad/LibrarySheet";
import RobotCorner from "@/components/launchpad/RobotCorner";
import CollectHandoff from "@/components/studio/CollectHandoff";
import StudioOverlay from "@/components/studio/StudioOverlay";
import { isHostedSpace } from "@/lib/isHostedSpace";

const ON_SPACE = isHostedSpace();

/**
 * Layout D "Launchpad" — the single dashboard route. Marketplace-first hero
 * with the skill slider, the "+ New Skill" banner that slides the studio up,
 * and the always-visible robot corner. Config happens in dialogs; live
 * hardware sessions live on their own immersive routes.
 */
const Launchpad = () => {
  const [showUsageModal, setShowUsageModal] = useState(ON_SPACE);
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [search, setSearch] = useState("");

  return (
    <div className="flex min-h-screen flex-col bg-background text-foreground">
      <header className="flex items-center justify-between gap-3 px-4 py-3 sm:px-6">
        <div className="flex items-center gap-3">
          <BrandMark />
          <HfAuthChip />
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            className="h-8 gap-1.5 rounded-full px-3"
            onClick={() => setLibraryOpen(true)}
          >
            <Library className="h-3.5 w-3.5" />
            My library
          </Button>
          <RobotCorner />
        </div>
      </header>

      {/* justify-center holds the whole stack (hero → banner) in the middle
          of the viewport rather than hugging the header. */}
      <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col items-center justify-center gap-10 px-4 py-8 sm:px-6">
        <CollectHandoff />
        <Hero search={search} onSearchChange={setSearch} />
        <SkillSlider search={search} />
        <ActivityStrip />
        <div className="w-full">
          <NewSkillBanner />
        </div>
      </main>

      <Footer />

      <UsageInstructionsModal
        open={showUsageModal}
        onOpenChange={setShowUsageModal}
        dismissible={!ON_SPACE}
      />
      <LibrarySheet open={libraryOpen} onOpenChange={setLibraryOpen} />
      <StudioOverlay />
    </div>
  );
};

export default Launchpad;
