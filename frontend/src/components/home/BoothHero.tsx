import React, { useEffect, useRef } from "react";

/**
 * OpenBooth wireframe hero — bimanual SO-101 line-drawn arms passing blocks
 * inside the booth. Pure SVG driven by requestAnimationFrame with 2-link IK;
 * colors come from the design tokens so it adapts to theme. Respects
 * prefers-reduced-motion (renders a fixed pose).
 */

const NS = "http://www.w3.org/2000/svg";

type Attrs = Record<string, string | number>;

const el = (tag: string, attrs: Attrs, parent: Element) => {
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, String(attrs[k]));
  parent.appendChild(n);
  return n;
};

const ease = (u: number) =>
  u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;

const STYLE = `
  .bo-line { stroke: hsl(var(--muted-foreground)); stroke-opacity: .45; stroke-width: 1.2; fill: none; }
  .bo-line.soft { stroke-opacity: .22; }
  .bo-link { stroke: hsl(var(--foreground)); stroke-width: 7; stroke-linecap: round; }
  .bo-joint { fill: hsl(var(--background)); stroke: hsl(var(--foreground)); stroke-width: 2; }
  .bo-base { fill: hsl(var(--muted)); stroke: hsl(var(--foreground)); stroke-width: 1.5; }
  .bo-grip { stroke: hsl(var(--foreground)); stroke-width: 4; stroke-linecap: round; fill: none; }
  .bo-block { fill: hsl(var(--foreground)); }
  .bo-block.ghost { fill: hsl(var(--muted-foreground)); opacity: .4; }
  .bo-label { font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 10px; fill: hsl(var(--muted-foreground)); letter-spacing: .08em; }
`;

interface ArmOpts {
  sx: number;
  sy: number;
  L1: number;
  L2: number;
  G: number;
  elbow?: number;
}

function makeArm(parent: Element, opts: ArmOpts) {
  const g = el("g", {}, parent);
  const { sx, sy, L1, L2, G } = opts;
  el("rect", { class: "bo-base", x: sx - 9, y: sy, width: 18, height: 26, rx: 3 }, g);
  const link1 = el("line", { class: "bo-link" }, g);
  const link2 = el("line", { class: "bo-link" }, g);
  const wristSeg = el("line", { class: "bo-grip" }, g);
  const f1 = el("polyline", { class: "bo-grip" }, g);
  const f2 = el("polyline", { class: "bo-grip" }, g);
  el("circle", { class: "bo-joint", r: 5, cx: sx, cy: sy }, g);
  const jE = el("circle", { class: "bo-joint", r: 4.5 }, g);
  const jW = el("circle", { class: "bo-joint", r: 4 }, g);

  return function pose(tx: number, ty: number, grip: number) {
    const wx = tx,
      wy = ty - G;
    let dx = wx - sx,
      dy = wy - sy;
    let d = Math.hypot(dx, dy);
    const max = L1 + L2 - 1;
    if (d > max) {
      dx *= max / d;
      dy *= max / d;
      d = max;
    }
    const alpha = Math.atan2(dy, dx);
    const cosA = Math.min(1, Math.max(-1, (L1 * L1 + d * d - L2 * L2) / (2 * L1 * d)));
    const a1 = alpha - Math.acos(cosA) * (opts.elbow ?? 1);
    const ex = sx + L1 * Math.cos(a1),
      ey = sy + L1 * Math.sin(a1);
    link1.setAttribute("x1", String(sx));
    link1.setAttribute("y1", String(sy));
    link1.setAttribute("x2", String(ex));
    link1.setAttribute("y2", String(ey));
    link2.setAttribute("x1", String(ex));
    link2.setAttribute("y1", String(ey));
    link2.setAttribute("x2", String(sx + dx));
    link2.setAttribute("y2", String(sy + dy));
    wristSeg.setAttribute("x1", String(sx + dx));
    wristSeg.setAttribute("y1", String(sy + dy));
    wristSeg.setAttribute("x2", String(tx));
    wristSeg.setAttribute("y2", String(ty - 8));
    const spread = 3 + grip * 6;
    f1.setAttribute(
      "points",
      `${tx - spread},${ty - 9} ${tx - spread},${ty} ${tx - spread + 2.5},${ty + 2}`
    );
    f2.setAttribute(
      "points",
      `${tx + spread},${ty - 9} ${tx + spread},${ty} ${tx + spread - 2.5},${ty + 2}`
    );
    jE.setAttribute("cx", String(ex));
    jE.setAttribute("cy", String(ey));
    jW.setAttribute("cx", String(sx + dx));
    jW.setAttribute("cy", String(sy + dy));
  };
}

interface Frame {
  t: number;
  x: number;
  y: number;
  g: number;
}

function sample(frames: Frame[], t: number, P: number) {
  const tt = t % P;
  let i = 0;
  while (i < frames.length - 1 && frames[i + 1].t <= tt) i++;
  const a = frames[i],
    b = frames[(i + 1) % frames.length];
  const span = (b.t > a.t ? b.t : P + b.t) - a.t;
  const u = span === 0 ? 0 : Math.min(1, (tt - a.t) / span);
  const e = ease(u);
  return {
    x: a.x + (b.x - a.x) * e,
    y: a.y + (b.y - a.y) * e,
    g: a.g + (b.g - a.g) * e,
  };
}

const FLOOR = 384;
const PERIOD = 12;

const FRAMES_L: Frame[] = [
  { t: 0.0, x: 250, y: 320, g: 1 },
  { t: 1.0, x: 239, y: FLOOR, g: 1 },
  { t: 1.5, x: 239, y: FLOOR, g: 0 },
  { t: 2.6, x: 320, y: 300, g: 0 },
  { t: 3.8, x: 400, y: FLOOR, g: 0 },
  { t: 4.3, x: 400, y: FLOOR, g: 1 },
  { t: 5.6, x: 250, y: 320, g: 1 },
  { t: 11.4, x: 250, y: 323, g: 1 },
];

const FRAMES_R: Frame[] = [
  { t: 0.0, x: 550, y: 320, g: 1 },
  { t: 5.4, x: 550, y: 323, g: 1 },
  { t: 6.4, x: 400, y: FLOOR, g: 1 },
  { t: 6.9, x: 400, y: FLOOR, g: 0 },
  { t: 8.0, x: 480, y: 300, g: 0 },
  { t: 9.2, x: 561, y: FLOOR, g: 0 },
  { t: 9.7, x: 561, y: FLOOR, g: 1 },
  { t: 11.0, x: 550, y: 320, g: 1 },
];

const BoothHero: React.FC<{ className?: string }> = ({ className }) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    svg.replaceChildren();

    el("style", {}, svg).textContent = STYLE;
    const defs = el("defs", {}, svg);
    const grad = el(
      "linearGradient",
      { id: "bo-wash", x1: 0, y1: 0, x2: 0, y2: 1 },
      defs
    );
    el("stop", { offset: 0, "stop-color": "#60A5FA", "stop-opacity": 0.1 }, grad);
    el("stop", { offset: 1, "stop-color": "#60A5FA", "stop-opacity": 0.03 }, grad);

    el(
      "polygon",
      {
        points: "210,70 590,70 590,310 710,390 90,390 210,310",
        fill: "url(#bo-wash)",
        stroke: "none",
      },
      svg
    );
    el("rect", { x: 210, y: 70, width: 380, height: 240, class: "bo-line" }, svg);
    el("polyline", { points: "210,70 90,30 90,390 210,310", class: "bo-line" }, svg);
    el("polyline", { points: "590,70 710,30 710,390 590,310", class: "bo-line" }, svg);
    el("line", { x1: 90, y1: 30, x2: 710, y2: 30, class: "bo-line" }, svg);
    el("line", { x1: 90, y1: 390, x2: 710, y2: 390, class: "bo-line" }, svg);
    el("line", { x1: 210, y1: 310, x2: 590, y2: 310, class: "bo-line soft" }, svg);
    el("line", { x1: 150, y1: 350, x2: 650, y2: 350, class: "bo-line soft" }, svg);
    el(
      "text",
      { x: 400, y: 24, "text-anchor": "middle", class: "bo-label" },
      svg
    ).textContent = "[ OpenBooth ]";

    el("rect", { class: "bo-block ghost", x: 196, y: 370, width: 14, height: 14, rx: 2.5 }, svg);
    el("rect", { class: "bo-block ghost", x: 214, y: 370, width: 14, height: 14, rx: 2.5 }, svg);
    el("rect", { class: "bo-block ghost", x: 576, y: 370, width: 14, height: 14, rx: 2.5 }, svg);
    const blockA = el("rect", { class: "bo-block", x: 232, y: 370, width: 14, height: 14, rx: 2.5 }, svg);

    const poseL = makeArm(svg, { sx: 285, sy: 314, L1: 74, L2: 62, G: 24 });
    const poseR = makeArm(svg, { sx: 515, sy: 314, L1: 74, L2: 62, G: 24, elbow: -1 });

    const frame = (t: number) => {
      const pl = sample(FRAMES_L, t, PERIOD);
      const pr = sample(FRAMES_R, t, PERIOD);
      poseL(pl.x, pl.y, pl.g);
      poseR(pr.x, pr.y, pr.g);
      const tt = t % PERIOD;
      let bx = 232,
        by = 370,
        o = 1;
      if (tt >= 1.5 && tt < 4.3) {
        bx = pl.x - 7;
        by = pl.y - 12;
      } else if (tt >= 4.3 && tt < 6.9) {
        bx = 393;
        by = 370;
      } else if (tt >= 6.9 && tt < 9.7) {
        bx = pr.x - 7;
        by = pr.y - 12;
      } else if (tt >= 9.7) {
        bx = 554;
        by = 370;
        o = Math.max(0, 1 - (tt - 11.2) / 0.6);
      } else {
        o = Math.min(1, tt / 0.4);
      }
      blockA.setAttribute("x", String(bx));
      blockA.setAttribute("y", String(by));
      blockA.setAttribute("opacity", String(o));
    };

    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      frame(2.2);
      return;
    }
    let raf = 0;
    const loop = (ms: number) => {
      frame(ms / 1000);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <svg
      ref={svgRef}
      viewBox="0 0 800 440"
      className={className}
      aria-label="OpenBooth wireframe animation"
    />
  );
};

export default BoothHero;
