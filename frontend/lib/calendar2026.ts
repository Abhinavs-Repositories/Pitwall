export interface RaceEvent {
  round: number;
  name: string;
  location: string;
  dateRange: string;
  shapeFile: string;
}

// 2026 F1 season — 24-race calendar (a full-season regulation-change year:
// Audi and Cadillac join the grid, Madrid debuts).
export const CALENDAR_2026: RaceEvent[] = [
  { round: 1, name: "Australian GP", location: "Melbourne", dateRange: "Mar 6–8", shapeFile: "melbourne.svg" },
  { round: 2, name: "Chinese GP", location: "Shanghai", dateRange: "Mar 13–15", shapeFile: "shanghai.svg" },
  { round: 3, name: "Japanese GP", location: "Suzuka", dateRange: "Mar 27–29", shapeFile: "suzuka.svg" },
  { round: 4, name: "Bahrain GP", location: "Sakhir", dateRange: "Apr 10–12", shapeFile: "sakhir.svg" },
  { round: 5, name: "Saudi Arabian GP", location: "Jeddah", dateRange: "Apr 17–19", shapeFile: "jeddah.svg" },
  { round: 6, name: "Miami GP", location: "Miami", dateRange: "May 1–3", shapeFile: "miami.svg" },
  { round: 7, name: "Canadian GP", location: "Montreal", dateRange: "May 22–24", shapeFile: "montreal.svg" },
  { round: 8, name: "Monaco GP", location: "Monaco", dateRange: "Jun 5–7", shapeFile: "monaco.svg" },
  { round: 9, name: "Spanish GP", location: "Catalunya", dateRange: "Jun 12–14", shapeFile: "catalunya.svg" },
  { round: 10, name: "Austrian GP", location: "Red Bull Ring", dateRange: "Jun 26–28", shapeFile: "red-bull-ring.svg" },
  { round: 11, name: "British GP", location: "Silverstone", dateRange: "Jul 3–5", shapeFile: "silverstone.svg" },
  { round: 12, name: "Belgian GP", location: "Spa-Francorchamps", dateRange: "Jul 17–19", shapeFile: "spa.svg" },
  { round: 13, name: "Hungarian GP", location: "Hungaroring", dateRange: "Jul 24–26", shapeFile: "hungaroring.svg" },
  { round: 14, name: "Dutch GP", location: "Zandvoort", dateRange: "Aug 21–23", shapeFile: "zandvoort.svg" },
  { round: 15, name: "Italian GP", location: "Monza", dateRange: "Sep 4–6", shapeFile: "monza.svg" },
  { round: 16, name: "Spanish GP (Madrid)", location: "Madrid", dateRange: "Sep 11–13", shapeFile: "madrid.svg" },
  { round: 17, name: "Azerbaijan GP", location: "Baku", dateRange: "Sep 24–26", shapeFile: "baku.svg" },
  { round: 18, name: "Singapore GP", location: "Marina Bay", dateRange: "Oct 9–11", shapeFile: "marina-bay.svg" },
  { round: 19, name: "United States GP", location: "Austin (COTA)", dateRange: "Oct 23–25", shapeFile: "cota.svg" },
  { round: 20, name: "Mexico City GP", location: "Mexico City", dateRange: "Oct 30–Nov 1", shapeFile: "mexico-city.svg" },
  { round: 21, name: "Brazilian GP", location: "Interlagos", dateRange: "Nov 6–8", shapeFile: "interlagos.svg" },
  { round: 22, name: "Las Vegas GP", location: "Las Vegas", dateRange: "Nov 19–21", shapeFile: "las-vegas.svg" },
  { round: 23, name: "Qatar GP", location: "Losail", dateRange: "Nov 27–29", shapeFile: "losail.svg" },
  { round: 24, name: "Abu Dhabi GP", location: "Yas Marina", dateRange: "Dec 4–6", shapeFile: "yas-marina.svg" },
];
