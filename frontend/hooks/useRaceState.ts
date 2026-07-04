"use client";

import useSWR from "swr";
import { getRaces, getRaceStateAtLap } from "@/lib/api";

export function useRaceList(year?: number) {
  return useSWR(["races", year ?? "all"], () => getRaces(year));
}

export function useRaceStateAtLap(sessionKey: number | null, lap: number) {
  return useSWR(
    sessionKey ? ["race-state", sessionKey, lap] : null,
    () => getRaceStateAtLap(sessionKey as number, lap),
    { revalidateOnFocus: false }
  );
}
