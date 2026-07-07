import { useCallback, useEffect, useRef, useState, ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";

type Peer = { id: string; openedAt: number; lastSeen: number };

const CHANNEL = "makerlab-tabs-v1";
const HEARTBEAT_MS = 1000;
const PEER_TIMEOUT_MS = 3000;

// crypto.randomUUID only exists in secure contexts (https/localhost); when the
// UI is served over plain HTTP on a LAN host, fall back to a non-crypto id —
// the tab election only needs uniqueness.
const newTabId = (): string =>
  typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
    ? crypto.randomUUID()
    : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;

const SingleTabGuard = ({ children }: { children: ReactNode }) => {
  const [isPrimary, setIsPrimary] = useState(true);
  const peersRef = useRef<Map<string, Peer>>(new Map());
  const myIdRef = useRef<string>("");
  const myOpenedAtRef = useRef<number>(0);
  const channelRef = useRef<BroadcastChannel | null>(null);

  const recompute = useCallback(() => {
    const peers = peersRef.current;
    const cutoff = Date.now() - PEER_TIMEOUT_MS;
    for (const [id, peer] of peers) {
      if (peer.lastSeen < cutoff) peers.delete(id);
    }
    let winnerId = myIdRef.current;
    let winnerOpenedAt = myOpenedAtRef.current;
    for (const peer of peers.values()) {
      if (
        peer.openedAt < winnerOpenedAt ||
        (peer.openedAt === winnerOpenedAt && peer.id < winnerId)
      ) {
        winnerId = peer.id;
        winnerOpenedAt = peer.openedAt;
      }
    }
    setIsPrimary(winnerId === myIdRef.current);
  }, []);

  useEffect(() => {
    if (typeof window === "undefined" || typeof BroadcastChannel === "undefined") {
      return;
    }

    myIdRef.current = newTabId();
    myOpenedAtRef.current = Date.now();

    const channel = new BroadcastChannel(CHANNEL);
    channelRef.current = channel;

    const send = (type: string) => {
      channel.postMessage({
        type,
        id: myIdRef.current,
        openedAt: myOpenedAtRef.current,
      });
    };

    channel.onmessage = (e) => {
      const msg = e.data;
      if (!msg || msg.id === myIdRef.current) return;
      const peers = peersRef.current;

      if (msg.type === "HEARTBEAT") {
        peers.set(msg.id, {
          id: msg.id,
          openedAt: msg.openedAt,
          lastSeen: Date.now(),
        });
      } else if (msg.type === "RELEASE") {
        peers.delete(msg.id);
      } else if (msg.type === "TAKEOVER") {
        peers.set(msg.id, {
          id: msg.id,
          openedAt: msg.openedAt,
          lastSeen: Date.now(),
        });
        // Move ourselves behind the taker so the election flips.
        if (myOpenedAtRef.current <= msg.openedAt) {
          myOpenedAtRef.current = msg.openedAt + 1;
        }
      }
      recompute();
    };

    send("HEARTBEAT");
    const interval = setInterval(() => {
      send("HEARTBEAT");
      recompute();
    }, HEARTBEAT_MS);

    const onUnload = () => send("RELEASE");
    window.addEventListener("beforeunload", onUnload);

    return () => {
      window.removeEventListener("beforeunload", onUnload);
      clearInterval(interval);
      send("RELEASE");
      channel.close();
      channelRef.current = null;
    };
  }, [recompute]);

  const takeOver = useCallback(() => {
    myOpenedAtRef.current = 0;
    channelRef.current?.postMessage({
      type: "TAKEOVER",
      id: myIdRef.current,
      openedAt: 0,
    });
    recompute();
  }, [recompute]);

  return (
    <>
      {children}
      {!isPrimary && (
        <div
          className="fixed inset-0 z-[9999] flex items-center justify-center bg-foreground/80 p-4"
          role="dialog"
          aria-modal="true"
        >
          <Card className="w-full max-w-md">
            <CardContent className="space-y-4 p-6 text-center">
              <Badge variant="stencil">[ single tab ]</Badge>
              <h2 className="font-display text-xl font-bold tracking-tight">
                MakerLab is already open in another tab
              </h2>
              <p className="text-sm text-muted-foreground">
                Only one tab can control the robot at a time. Switch back to the
                original tab, or take over here; the other tab will lock.
              </p>
              <Button onClick={takeOver}>use this tab</Button>
            </CardContent>
          </Card>
        </div>
      )}
    </>
  );
};

export default SingleTabGuard;
