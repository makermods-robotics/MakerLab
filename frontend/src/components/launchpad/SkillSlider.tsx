import React, { useMemo, useRef, useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { useModels } from "@/hooks/useModels";
import { useHfAuth } from "@/contexts/HfAuthContext";
import { ModelItem } from "@/lib/modelsApi";
import SkillCard, {
  classifySkill,
  skillNamespace,
  skillTitle,
} from "@/components/launchpad/SkillCard";
import SkillDetailDialog from "@/components/dialogs/SkillDetailDialog";

export interface SkillSliderProps {
  /** Live search filter from the hero search box. */
  search: string;
}

/** A single loading skeleton shaped like a SkillCard. */
const CardSkeleton: React.FC = () => (
  <div className="flex w-64 shrink-0 flex-col overflow-hidden rounded-lg border border-border bg-card shadow-1">
    <div className="aspect-[4/3] w-full animate-pulse bg-muted" />
    <div className="flex flex-col gap-2 p-3">
      <div className="h-4 w-3/4 animate-pulse rounded bg-muted" />
      <div className="h-3 w-1/2 animate-pulse rounded bg-muted" />
      <div className="h-3 w-2/3 animate-pulse rounded bg-muted" />
    </div>
  </div>
);

/**
 * Horizontal skill slider — every /models row rendered as a card, mixing
 * MakerMods-highlighted, community, and the user's own skills. Scroll-snap track
 * with ‹ › arrow buttons; the hero search box filters live by name/author. Card
 * click opens the skill detail dialog.
 */
const SkillSlider: React.FC<SkillSliderProps> = ({ search }) => {
  const { models, loading } = useModels();
  const { auth } = useHfAuth();
  const trackRef = useRef<HTMLDivElement>(null);
  const [detail, setDetail] = useState<ModelItem | null>(null);
  const [detailOpen, setDetailOpen] = useState(false);

  const username = auth.status === "authenticated" ? auth.username : null;

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return models;
    return models.filter((m) => {
      const ns = skillNamespace(m) ?? "";
      return (
        skillTitle(m).toLowerCase().includes(q) ||
        m.id.toLowerCase().includes(q) ||
        ns.toLowerCase().includes(q)
      );
    });
  }, [models, search]);

  const scrollBy = (dir: 1 | -1) => {
    const el = trackRef.current;
    if (!el) return;
    el.scrollBy({ left: dir * Math.round(el.clientWidth * 0.8), behavior: "smooth" });
  };

  const openDetail = (model: ModelItem) => {
    setDetail(model);
    setDetailOpen(true);
  };

  return (
    <section className="w-full" aria-label="Skills">
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => scrollBy(-1)}
          aria-label="Previous skills"
          className="hidden h-9 w-9 shrink-0 items-center justify-center rounded-full border border-border bg-card text-muted-foreground shadow-1 transition-colors hover:border-ring hover:text-foreground sm:flex"
        >
          <ChevronLeft className="h-4 w-4" />
        </button>

        <div
          ref={trackRef}
          className="no-scrollbar flex flex-1 snap-x snap-mandatory gap-4 overflow-x-auto scroll-smooth pb-2"
        >
          {loading ? (
            <>
              <CardSkeleton />
              <CardSkeleton />
              <CardSkeleton />
              <CardSkeleton />
            </>
          ) : filtered.length === 0 ? (
            <div className="flex min-h-[13rem] w-full items-center justify-center rounded-lg border border-dashed border-border bg-card/50 px-6 py-10 text-center text-sm text-muted-foreground">
              {models.length === 0
                ? "No skills yet — create the first one below."
                : "No skills match your search."}
            </div>
          ) : (
            filtered.map((model) => (
              <SkillCard
                key={model.id}
                model={model}
                badge={classifySkill(model, username)}
                onOpen={openDetail}
              />
            ))
          )}
        </div>

        <button
          type="button"
          onClick={() => scrollBy(1)}
          aria-label="Next skills"
          className="hidden h-9 w-9 shrink-0 items-center justify-center rounded-full border border-border bg-card text-muted-foreground shadow-1 transition-colors hover:border-ring hover:text-foreground sm:flex"
        >
          <ChevronRight className="h-4 w-4" />
        </button>
      </div>

      <SkillDetailDialog
        model={detail}
        open={detailOpen}
        onOpenChange={setDetailOpen}
      />
    </section>
  );
};

export default SkillSlider;
