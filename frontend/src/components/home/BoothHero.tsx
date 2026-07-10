import React, { useEffect, useRef } from "react";

/**
 * OpenBooth wireframe hero — bimanual SO-101 line-drawn arms passing a block
 * inside the booth, seen from a corner at a low elevation (flat axonometric:
 * verticals stay vertical, the floor is a shallow diamond). Pure SVG driven by
 * requestAnimationFrame with 2-link IK in the arms' working plane; colors come
 * from the design tokens so it adapts to theme. Respects
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
  .bo-line.soft { stroke-opacity: .2; }
  .bo-link { stroke: hsl(var(--foreground)); stroke-width: 7; stroke-linecap: round; }
  .bo-joint { fill: hsl(var(--background)); stroke: hsl(var(--foreground)); stroke-width: 2; }
  .bo-base { fill: hsl(var(--muted)); stroke: hsl(var(--foreground)); stroke-width: 1.5; }
  .bo-grip { stroke: hsl(var(--foreground)); stroke-width: 4; stroke-linecap: round; fill: none; }
  .bo-block { fill: hsl(var(--foreground)); stroke: none; }
  .bo-block.ghost { fill: hsl(var(--muted-foreground)); }
  .bo-label { font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 10px; fill: hsl(var(--muted-foreground)); letter-spacing: .08em; }
`;

/**
 * Corner axonometric projection, low camera: world x runs right-down, world z
 * runs left-down, world y is straight up. S controls the elevation (pitch) —
 * small S = camera close to the tabletop.
 */
const C = 0.9;
const S = 0.26;
const OX = 400;
const OY = 168;

const pt = (x: number, y: number, z: number): readonly [number, number] => [
  OX + (x - z) * C,
  OY + (x + z) * S - y,
];

const pts = (...ps: Array<readonly [number, number]>) =>
  ps.map((p) => `${p[0]},${p[1]}`).join(" ");

/** Booth floor extent and the depth of the plane the arms work in. */
const W = 300;
const D = 300;
const AZ = 150;
const WALL_H = 128;

/** Project a point in the arms' working plane (u along x, v = height). */
const pp = (u: number, v: number) => pt(u, v, AZ);

/**
 * Small extruded block (top + two visible faces). Returns an updater so the
 * carried block can move each frame; static blocks just never update.
 */
function makeBlock(parent: Element, cls: string, ghost = false) {
  const g = el("g", ghost ? { opacity: 0.35 } : {}, parent);
  const faceL = el("polygon", { class: cls, "fill-opacity": 0.55 }, g);
  const faceR = el("polygon", { class: cls, "fill-opacity": 0.75 }, g);
  const top = el("polygon", { class: cls }, g);
  const s = 8; // half-size in world units
  const h = 13;
  const place = (x: number, y: number, z: number) => {
    const tNE = pt(x + s, y + h, z - s);
    const tNW = pt(x - s, y + h, z - s);
    const tSE = pt(x + s, y + h, z + s);
    const tSW = pt(x - s, y + h, z + s);
    const bSW = pt(x - s, y, z + s);
    const bSE = pt(x + s, y, z + s);
    const bNE = pt(x + s, y, z - s);
    top.setAttribute("points", pts(tNW, tNE, tSE, tSW));
    // faces adjacent to the near (south-west / south-east) corner
    faceL.setAttribute("points", pts(tSW, tSE, bSE, bSW));
    faceR.setAttribute("points", pts(tSE, tNE, bNE, bSE));
    return g;
  };
  return { place, g };
}

interface ArmOpts {
  bu: number; // base position along the working plane
  L1: number;
  L2: number;
  G: number; // wrist standoff above the grip target
  elbow?: number;
}

function makeArm(parent: Element, opts: ArmOpts) {
  const g = el("g", {}, parent);
  const { bu, L1, L2, G } = opts;

  // Pedestal: a squat extruded base under the shoulder joint.
  makeBlock(g, "bo-base").place(bu, 0, AZ);

  const link1 = el("line", { class: "bo-link" }, g);
  const link2 = el("line", { class: "bo-link" }, g);
  const wristSeg = el("line", { class: "bo-grip" }, g);
  const f1 = el("polyline", { class: "bo-grip" }, g);
  const f2 = el("polyline", { class: "bo-grip" }, g);
  const SHOULDER_H = 16;
  const [bx, by] = pp(bu, SHOULDER_H);
  el("circle", { class: "bo-joint", r: 5, cx: bx, cy: by }, g);
  const jE = el("circle", { class: "bo-joint", r: 4.5 }, g);
  const jW = el("circle", { class: "bo-joint", r: 4 }, g);

  const setLine = (
    n: Element,
    a: readonly [number, number],
    b: readonly [number, number]
  ) => {
    n.setAttribute("x1", String(a[0]));
    n.setAttribute("y1", String(a[1]));
    n.setAttribute("x2", String(b[0]));
    n.setAttribute("y2", String(b[1]));
  };

  /** Pose toward a target (tu, tv) in the working plane; v is height. */
  return function pose(tu: number, tv: number, grip: number) {
    const wu = tu;
    const wv = tv + G;
    let du = wu - bu;
    let dv = wv - SHOULDER_H;
    let d = Math.hypot(du, dv);
    const max = L1 + L2 - 1;
    if (d > max) {
      du *= max / d;
      dv *= max / d;
      d = max;
    }
    const alpha = Math.atan2(dv, du);
    const cosA = Math.min(
      1,
      Math.max(-1, (L1 * L1 + d * d - L2 * L2) / (2 * L1 * d))
    );
    // Elbow bends upward: add the interior angle on the side that lifts it.
    const a1 = alpha + Math.acos(cosA) * (opts.elbow ?? 1);
    const eu = bu + L1 * Math.cos(a1);
    const ev = SHOULDER_H + L1 * Math.sin(a1);

    const B = pp(bu, SHOULDER_H);
    const E = pp(eu, ev);
    const Wp = pp(bu + du, SHOULDER_H + dv);
    setLine(link1, B, E);
    setLine(link2, E, Wp);
    setLine(wristSeg, Wp, pp(tu, tv + 8));

    const spread = 3 + grip * 6;
    f1.setAttribute(
      "points",
      pts(pp(tu - spread, tv + 9), pp(tu - spread, tv), pp(tu - spread + 2.5, tv - 2))
    );
    f2.setAttribute(
      "points",
      pts(pp(tu + spread, tv + 9), pp(tu + spread, tv), pp(tu + spread - 2.5, tv - 2))
    );
    jE.setAttribute("cx", String(E[0]));
    jE.setAttribute("cy", String(E[1]));
    jW.setAttribute("cx", String(Wp[0]));
    jW.setAttribute("cy", String(Wp[1]));
  };
}

interface Frame {
  t: number;
  u: number;
  v: number;
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
    u: a.u + (b.u - a.u) * e,
    v: a.v + (b.v - a.v) * e,
    g: a.g + (b.g - a.g) * e,
  };
}

const PERIOD = 12;
const PICK_U = 23;
const MID_U = 150;
const DROP_U = 277;

const FRAMES_L: Frame[] = [
  { t: 0.0, u: 30, v: 74, g: 1 },
  { t: 1.0, u: PICK_U, v: 2, g: 1 },
  { t: 1.5, u: PICK_U, v: 2, g: 0 },
  { t: 2.6, u: 88, v: 96, g: 0 },
  { t: 3.8, u: MID_U, v: 2, g: 0 },
  { t: 4.3, u: MID_U, v: 2, g: 1 },
  { t: 5.6, u: 30, v: 74, g: 1 },
  { t: 11.4, u: 30, v: 70, g: 1 },
];

const FRAMES_R: Frame[] = [
  { t: 0.0, u: 270, v: 74, g: 1 },
  { t: 5.4, u: 270, v: 70, g: 1 },
  { t: 6.4, u: MID_U, v: 2, g: 1 },
  { t: 6.9, u: MID_U, v: 2, g: 0 },
  { t: 8.0, u: 212, v: 96, g: 0 },
  { t: 9.2, u: DROP_U, v: 2, g: 0 },
  { t: 9.7, u: DROP_U, v: 2, g: 1 },
  { t: 11.0, u: 270, v: 74, g: 1 },
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

    // Floor: shallow diamond (the low camera flattens it).
    const fA = pt(0, 0, 0);
    const fB = pt(W, 0, 0);
    const fC = pt(W, 0, D);
    const fD = pt(0, 0, D);
    el(
      "polygon",
      { points: pts(fA, fB, fC, fD), fill: "url(#bo-wash)", class: "bo-line" },
      svg
    );
    // Iso floor grid.
    for (let gx = 60; gx < W; gx += 60)
      el(
        "line",
        {
          x1: pt(gx, 0, 0)[0], y1: pt(gx, 0, 0)[1],
          x2: pt(gx, 0, D)[0], y2: pt(gx, 0, D)[1],
          class: "bo-line soft",
        },
        svg
      );
    for (let gz = 60; gz < D; gz += 60)
      el(
        "line",
        {
          x1: pt(0, 0, gz)[0], y1: pt(0, 0, gz)[1],
          x2: pt(W, 0, gz)[0], y2: pt(W, 0, gz)[1],
          class: "bo-line soft",
        },
        svg
      );

    // Back walls meeting at the far corner; verticals stay vertical.
    el(
      "polygon",
      {
        points: pts(fA, fB, pt(W, WALL_H, 0), pt(0, WALL_H, 0)),
        class: "bo-line",
        fill: "none",
      },
      svg
    );
    el(
      "polygon",
      {
        points: pts(fA, fD, pt(0, WALL_H, D), pt(0, WALL_H, 0)),
        class: "bo-line",
        fill: "none",
      },
      svg
    );
    // Soft top rails hinting at the booth frame.
    el(
      "line",
      {
        x1: pt(W, WALL_H, 0)[0], y1: pt(W, WALL_H, 0)[1],
        x2: pt(W, WALL_H, D)[0], y2: pt(W, WALL_H, D)[1],
        class: "bo-line soft",
      },
      svg
    );
    el(
      "line",
      {
        x1: pt(0, WALL_H, D)[0], y1: pt(0, WALL_H, D)[1],
        x2: pt(W, WALL_H, D)[0], y2: pt(W, WALL_H, D)[1],
        class: "bo-line soft",
      },
      svg
    );

    el(
      "text",
      { x: 400, y: 24, "text-anchor": "middle", class: "bo-label" },
      svg
    ).textContent = "[ OpenBooth ]";

    // Staging blocks (ghosts) and the live block.
    makeBlock(svg, "bo-block ghost", true).place(10, 0, AZ + 38);
    makeBlock(svg, "bo-block ghost", true).place(30, 0, AZ + 22);
    makeBlock(svg, "bo-block ghost", true).place(DROP_U + 16, 0, AZ + 30);
    const live = makeBlock(svg, "bo-block");

    const poseL = makeArm(svg, { bu: 60, L1: 80, L2: 66, G: 22 });
    const poseR = makeArm(svg, { bu: 240, L1: 80, L2: 66, G: 22 });

    const frame = (t: number) => {
      const pl = sample(FRAMES_L, t, PERIOD);
      const pr = sample(FRAMES_R, t, PERIOD);
      poseL(pl.u, pl.v, pl.g);
      poseR(pr.u, pr.v, pr.g);
      const tt = t % PERIOD;
      let bu = PICK_U,
        bv = 0,
        o = 1;
      if (tt >= 1.5 && tt < 4.3) {
        bu = pl.u;
        bv = Math.max(0, pl.v - 11);
      } else if (tt >= 4.3 && tt < 6.9) {
        bu = MID_U;
        bv = 0;
      } else if (tt >= 6.9 && tt < 9.7) {
        bu = pr.u;
        bv = Math.max(0, pr.v - 11);
      } else if (tt >= 9.7) {
        bu = DROP_U;
        bv = 0;
        o = Math.max(0, 1 - (tt - 11.2) / 0.6);
      } else {
        o = Math.min(1, tt / 0.4);
      }
      live.place(bu, bv, AZ);
      live.g.setAttribute("opacity", String(o));
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
