"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRaceList, useRaceStateAtLap } from "@/hooks/useRaceState";
import { usePitwall } from "@/hooks/usePitwall";
import RaceSelector from "@/components/strategy/RaceSelector";
import LapControl from "@/components/strategy/LapControl";
import TelemetryPanel from "@/components/strategy/TelemetryPanel";
import StrategyChat from "@/components/strategy/StrategyChat";
import LapTimePanel from "@/components/strategy/LapTimePanel";
import LiveBadge from "@/components/ui/LiveBadge";

export default function StrategyPage() {
  const { data: races } = useRaceList();
  const [sessionKey, setSessionKey] = useState<number | null>(null);
  const [lap, setLap] = useState(15);
  const [selectedDriverNumber, setSelectedDriverNumber] = useState<number | null>(null);

  useEffect(() => {
    if (sessionKey === null && races && races.length > 0) {
      setSessionKey(races[0].session_key);
    }
  }, [races, sessionKey]);

  // A driver picked for one race won't exist in another — reset on switch.
  useEffect(() => {
    setSelectedDriverNumber(null);
  }, [sessionKey]);

  const { data: raceState, isLoading } = useRaceStateAtLap(sessionKey, lap);
  const { messages, send, isStreaming } = usePitwall(sessionKey, lap);

  const selectedDriver = useMemo(() => {
    if (!raceState) return undefined;
    return (
      raceState.drivers.find((d) => d.driver_number === selectedDriverNumber) ??
      raceState.drivers.find((d) => d.position === 1)
    );
  }, [raceState, selectedDriverNumber]);

  const latestRecommendation = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      if (messages[i].strategyData) return messages[i].strategyData!;
    }
    return null;
  }, [messages]);

  // Clamp the lap slider once we learn the race's real total lap count.
  useEffect(() => {
    if (raceState && lap > raceState.total_laps) {
      setLap(raceState.total_laps);
    }
  }, [raceState, lap]);

  return (
    <main className="flex h-screen flex-col overflow-hidden bg-carbon text-text-primary">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-border-dim px-6 py-3">
        <div className="flex items-center gap-3">
          <Link href="/" className="font-display text-lg font-bold uppercase text-text-primary">
            PITWALL
          </Link>
          <LiveBadge label="REPLAY" />
        </div>
        <div className="flex flex-wrap items-center gap-4">
          <RaceSelector races={races ?? []} value={sessionKey} onChange={setSessionKey} />
          <LapControl lap={lap} totalLaps={raceState?.total_laps ?? 60} onChange={setLap} />
        </div>
      </header>

      <div className="grid min-h-0 flex-1 grid-cols-1 overflow-hidden lg:grid-cols-[280px_1fr_320px]">
        <TelemetryPanel
          raceState={raceState}
          loading={isLoading}
          latestRecommendation={latestRecommendation}
          selectedDriverNumber={selectedDriver?.driver_number ?? null}
          onSelectDriver={setSelectedDriverNumber}
        />
        <StrategyChat
          messages={messages}
          onSend={send}
          isStreaming={isStreaming}
          raceState={raceState}
        />
        <LapTimePanel driver={selectedDriver} weather={raceState?.weather} />
      </div>
    </main>
  );
}
