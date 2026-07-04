# PITWALL_UI_SPEC.md
# Pitwall — Immersive Frontend Design & Implementation Specification

> **Status:** Draft v1.0  
> **Author:** Abhinav  
> **Companion doc:** `PITWALL_AI_SPEC.md` (backend architecture)  
> **Target:** Replace Streamlit frontend with a production-grade, immersive Next.js experience

---

## 1. Vision & Design Philosophy

Pitwall is not a dashboard. It is a pit wall — the nerve centre of an F1 race weekend, running on adrenaline, precision, and split-second decisions. The frontend must feel like that.

**The governing principle:** scroll is a playhead, not navigation. Every user interaction — scrolling, hovering, clicking — should feel like it has weight, momentum, and intent. Nothing snaps. Nothing is static. The page breathes.

**Aesthetic DNA:**
- Dark, cinematic, carbon-black base
- F1 red (`#E8002D`) as the single accent — used sparingly, earned
- Monospaced telemetry data, bold editorial typography
- Parallax depth, physics-based momentum, cursor-reactive 3D
- Feels like the inside of a Red Bull pit wall monitor, not a SaaS product

**Inspiration reference:** Lando Norris portfolio site — scroll-driven timeline, immersive 3D hero, kinetic typography, axis-hijacked carousels, asymmetric parallax gallery.

---

## 2. Tech Stack

### Core Framework
| Package | Version | Purpose |
|---|---|---|
| `next` | 14.x (App Router) | Framework, SSR, routing, image optimization |
| `react` | 18.x | UI library |
| `typescript` | 5.x | Type safety across all components |

### Animation Layer
| Package | Version | Purpose | Cost |
|---|---|---|---|
| `gsap` | 3.x | Scroll-triggered animations, staggered reveals, timeline scrubbing | Free (core + ScrollTrigger) |
| `@gsap/react` | latest | React integration for GSAP | Free |
| `lenis` | latest | Smooth scroll LERP momentum (replaces GSAP ScrollSmoother) | Free |
| `framer-motion` | 11.x | Component-level micro-interactions, hover physics, staggered grids | Free |

### 3D Layer
| Package | Version | Purpose | Cost |
|---|---|---|---|
| `three` | latest | WebGL 3D engine | Free |
| `@react-three/fiber` | latest | React wrapper for Three.js | Free |
| `@react-three/drei` | latest | Helpers: environment maps, orbit controls, loaders | Free |

### Styling
| Package | Purpose |
|---|---|
| `tailwindcss` | Utility-first CSS, responsive layout |
| `tailwind-merge` | Merge Tailwind classes safely |
| `clsx` | Conditional class composition |

### Data & API
| Package | Purpose |
|---|---|
| `swr` or `@tanstack/react-query` | Data fetching, caching, real-time polling |
| `socket.io-client` | WebSocket connection to FastAPI `/ws` endpoint for live race data |

### Fonts
| Font | Role | Source |
|---|---|---|
| `Rajdhani` | Display / hero typography | Google Fonts |
| `JetBrains Mono` | Telemetry data, lap times, agent trace | Google Fonts |
| `Inter` | Body copy, UI labels | Google Fonts |

### Deployment
| Service | Tier | Cost |
|---|---|---|
| Vercel | Hobby | Free |

### Full cost: **$0**

---

## 3. Repository Structure

Add a `frontend/` directory at the root of the existing Pitwall repo. The Python backend is entirely untouched.

```
Pitwall/
├── src/                          ← existing Python backend (unchanged)
│   ├── agents/
│   ├── analysis/
│   ├── api/
│   ├── core/
│   ├── data/
│   ├── rag/
│   └── ui/                       ← Streamlit (can be kept or deprecated)
├── scripts/                      ← existing (unchanged)
├── tests/                        ← existing (unchanged)
│
├── frontend/                     ← NEW: entire Next.js app lives here
│   ├── app/
│   │   ├── layout.tsx            ← root layout, fonts, Lenis provider
│   │   ├── page.tsx              ← landing/marketing page (immersive scroll)
│   │   ├── strategy/
│   │   │   └── page.tsx          ← strategy console (chat + telemetry)
│   │   ├── replay/
│   │   │   └── page.tsx          ← lap replay scrubber
│   │   └── api/                  ← Next.js API routes (proxy to FastAPI)
│   │       └── chat/route.ts
│   │
│   ├── components/
│   │   ├── hero/
│   │   │   ├── HeroSection.tsx   ← split-screen + 3D circuit cursor tracking
│   │   │   ├── CircuitCanvas.tsx ← R3F scene: 3D circuit wireframe
│   │   │   └── HeroText.tsx      ← kinetic typography reveal
│   │   ├── scroll/
│   │   │   ├── LenisProvider.tsx ← smooth scroll context
│   │   │   └── ScrollTriggerPin.tsx
│   │   ├── sections/
│   │   │   ├── KineticText.tsx   ← "STRATEGY IS EVERYTHING" scroll reveal
│   │   │   ├── ParallaxGallery.tsx ← floating race footage cards
│   │   │   ├── SplitSection.tsx  ← "On Pit Wall / On Track" split screen
│   │   │   ├── StaggeredGrid.tsx ← cascading circuit/helmet grid
│   │   │   └── HorizontalCarousel.tsx ← axis-hijacked team logo strip
│   │   ├── strategy/
│   │   │   ├── StrategyConsole.tsx ← main chat interface
│   │   │   ├── TelemetryPanel.tsx  ← live data sidebar
│   │   │   ├── RecommendationCard.tsx
│   │   │   ├── AgentTrace.tsx
│   │   │   └── LapTimeChart.tsx
│   │   └── ui/
│   │       ├── Chip.tsx
│   │       ├── LiveBadge.tsx
│   │       └── DriverCard.tsx
│   │
│   ├── hooks/
│   │   ├── useLiveRace.ts        ← WebSocket hook for live data
│   │   ├── useCursorTrack.ts     ← normalised mouse X/Y for 3D parallax
│   │   ├── useScrollProgress.ts  ← 0–1 scroll progress for GSAP
│   │   └── usePitwall.ts         ← chat state, message history
│   │
│   ├── lib/
│   │   ├── api.ts                ← typed fetch wrappers for FastAPI endpoints
│   │   ├── constants.ts          ← team colours, compound colours, circuit data
│   │   └── utils.ts
│   │
│   ├── public/
│   │   ├── fonts/
│   │   ├── images/               ← race photography, circuit maps
│   │   └── models/               ← .glb 3D assets (circuit wireframes)
│   │
│   ├── styles/
│   │   └── globals.css           ← CSS variables, Tailwind base
│   │
│   ├── next.config.js
│   ├── tailwind.config.ts
│   ├── tsconfig.json
│   └── package.json
│
├── PITWALL_AI_SPEC.md            ← existing backend spec
├── PITWALL_UI_SPEC.md            ← this file
├── README.md
├── docker-compose.yml
└── .env.example
```

---

## 4. Page Architecture

### 4.1 Landing Page (`/`) — The Immersive Experience

This is the showpiece. Five scroll-driven sections, each with distinct animation mechanics.

#### Section 1 — 3D Hero
**Layout:** Full-viewport split screen. Left half: high-contrast race photography (still). Right half: 3D F1 circuit wireframe rendered in R3F.

**Interaction:**
- The 3D circuit on the right tracks cursor movement via `useCursorTrack` — X movement tilts the circuit on the Y axis, Y movement tilts on the X axis. Range: ±15°.
- Implemented with `useFrame` in R3F — cursor delta is lerped (factor 0.05) for smooth lag.
- Circuit wireframe: white lines on black, very low opacity backdrop, `MeshBasicMaterial` with `wireframe: true`.

**Text:** `PITWALL` in Rajdhani 700 at ~14vw, with a 2px F1 red underline that draws in on load (SVG path animation, 1.2s ease-out).

**Entry animation:** On mount, hero fades in with a staggered sequence — background first (0s), image (0.3s), text (0.6s), 3D canvas (0.9s). GSAP timeline.

#### Section 2 — Kinetic Typography
**Trigger:** ScrollTrigger pin — hero scales down to 0 and fades as user scrolls past.

**Canvas shift:** Background transitions from `#0a0a0a` to `#0d0d0d` with faint topographic SVG lines fading in (opacity 0 → 0.04).

**Typography:** Three stacked text blocks in Rajdhani 900:
```
STRATEGY
IS
EVERYTHING.
```
Each line slides in from opposing directions (left, right, left) with a 0.15s stagger, triggered when the section enters viewport. Font size: `clamp(4rem, 12vw, 10rem)`.

**Sub-copy:** Below the headline, a one-liner in JetBrains Mono fades up: `// 7 agents. live telemetry. pit-wall grade answers.`

#### Section 3 — Parallax Gallery
**Layout:** Asymmetric, intentionally collage-like. 6–8 media blocks (race footage stills, circuit maps, tire close-ups) arranged at irregular offsets.

**Animation:** Each block has its own parallax `y` factor set in data attributes:
- Fast blocks (`factor: 1.4`): move 40% faster than scroll
- Slow blocks (`factor: 0.6`): lag behind scroll by 40%
- GSAP ScrollTrigger `scrub: 1` for smooth, momentum-tied movement

**Hover micro-interaction:** On hover, each card scales to 1.04 and applies a subtle CSS `rotate3d` based on mouse position within the card (max ±5°). Framer Motion `whileHover` + `onMouseMove`.

#### Section 4 — Split Screen: On Pit Wall / On Track
**Transition:** Two halves slide in from opposite sides — left panel ("On Pit Wall") from the left, right panel ("On Track") from the right — as the section enters viewport. GSAP `fromTo` with `xPercent`.

**Left panel content:** Screenshots/mockups of the strategy console UI — chat interface, telemetry panels, recommendation card.

**Right panel content:** Race photography. Driver in cockpit, pit crew, tyre change.

**Grid reveal:** Below the split, a 3×3 grid of circuit cards appears with a cascading stagger — each card fades and scales in from 0.9 → 1, top-left to bottom-right, 0.08s delay per item.

#### Section 5 — Horizontal Carousel (Axis Hijack)
**Mechanic:** Vertical scrolling is converted to horizontal translation. GSAP `ScrollTrigger` with a pinned container. As the user scrolls down, the inner strip translates left, revealing cards in sequence.

**Content:** Race calendar cards — one per Grand Prix. Each card shows: circuit name, flag, date, Pitwall's prediction accuracy for that race (if available).

**Physics:** Lenis smooth scroll + GSAP `scrub: 1.5` gives a natural deceleration — cards don't stop abruptly.

**After the carousel:** The page unpins, transitions to the footer with the Pitwall logo and a CTA: `Launch Strategy Console →`.

---

### 4.2 Strategy Console (`/strategy`) — The Tool

Functional UI. The immersive landing is the showcase; this is where work happens.

**Layout:** Three-column dark panel layout (as per the UI prototype built in Claude):
- Left: Running order, tire compounds, gap analysis, pit window
- Center: Chat interface — messages, recommendation cards, agent trace, quick chips
- Right: Last lap hero time, lap time trend chart, stint data, weather grid

**Real-time data:** WebSocket connection via `socket.io-client` to `ws://localhost:8000/ws`. On message, update relevant panels reactively via React state / SWR mutation.

**Chat behaviour:**
- User message → POST to `/api/chat` → stream response tokens if the FastAPI endpoint supports SSE (or full response if not)
- Recommendation card animates in with a border-top draw effect (CSS `scaleX` from 0 → 1, 0.4s)
- Agent trace appears 200ms after recommendation card with a typewriter effect (JetBrains Mono, character-by-character)

**Session selector:** Dropdown in the header allows switching between live session and historical races. Populates from `GET /api/races`.

---

### 4.3 Lap Replay (`/replay`)
**Layout:** Fullscreen dark canvas. A horizontal scrubber at the bottom (0 → total laps). Dragging or clicking the scrubber calls `GET /api/races/{session_key}/lap/{lap}` and re-renders all telemetry panels.

**Visualisation:** A miniature circuit SVG with animated driver position dots — positions update as lap changes.

**Chat:** Collapsed by default. Expand button in the bottom right opens the strategy console as a slide-up panel, pre-seeded with the selected lap context.

---

## 5. Animation System

### 5.1 Smooth Scroll (Lenis)

```typescript
// app/layout.tsx
const lenis = new Lenis({
  duration: 1.4,
  easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
  smoothWheel: true,
})

// Sync with GSAP ticker
gsap.ticker.add((time) => {
  lenis.raf(time * 1000)
})
gsap.ticker.lagSmoothing(0)
```

### 5.2 Cursor Tracking (3D Hero)

```typescript
// hooks/useCursorTrack.ts
export function useCursorTrack() {
  const cursor = useRef({ x: 0, y: 0 })
  const lerped = useRef({ x: 0, y: 0 })

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      cursor.current.x = (e.clientX / window.innerWidth - 0.5) * 2   // -1 to 1
      cursor.current.y = (e.clientY / window.innerHeight - 0.5) * 2  // -1 to 1
    }
    window.addEventListener('mousemove', onMove)
    return () => window.removeEventListener('mousemove', onMove)
  }, [])

  // Lerp in useFrame (R3F) or rAF
  return lerped
}
```

### 5.3 Kinetic Text Reveal

```typescript
// GSAP staggered text reveal on ScrollTrigger
gsap.fromTo(
  '.kinetic-line',
  { xPercent: -100, opacity: 0 },
  {
    xPercent: 0,
    opacity: 1,
    duration: 1.0,
    ease: 'power3.out',
    stagger: 0.15,
    scrollTrigger: {
      trigger: '.kinetic-section',
      start: 'top 70%',
      toggleActions: 'play none none reverse',
    },
  }
)
```

### 5.4 Parallax Gallery

```typescript
// Each card registered with its own ScrollTrigger
cards.forEach((card) => {
  const factor = parseFloat(card.dataset.parallax ?? '1')
  gsap.to(card, {
    yPercent: -20 * factor,
    ease: 'none',
    scrollTrigger: {
      trigger: card,
      start: 'top bottom',
      end: 'bottom top',
      scrub: 1,
    },
  })
})
```

### 5.5 Axis-Hijacked Horizontal Carousel

```typescript
const totalWidth = carouselInner.scrollWidth - carouselOuter.clientWidth

gsap.to(carouselInner, {
  x: -totalWidth,
  ease: 'none',
  scrollTrigger: {
    trigger: carouselOuter,
    start: 'top top',
    end: `+=${totalWidth}`,
    pin: true,
    scrub: 1.5,   // the 1.5 gives the physics lag / deceleration feel
    anticipatePin: 1,
  },
})
```

### 5.6 Staggered Grid Reveal

```typescript
// Framer Motion variant — cascading fade-in
const gridVariants = {
  hidden: {},
  visible: {
    transition: { staggerChildren: 0.08, delayChildren: 0.1 }
  }
}

const itemVariants = {
  hidden: { opacity: 0, scale: 0.9, y: 20 },
  visible: { opacity: 1, scale: 1, y: 0, transition: { duration: 0.5, ease: [0.25, 0.46, 0.45, 0.94] } }
}
```

---

## 6. Design Tokens

```css
/* styles/globals.css */
:root {
  /* Base palette */
  --carbon:        #0a0a0a;
  --carbon-2:      #111111;
  --carbon-3:      #181818;
  --carbon-4:      #222222;

  /* Accent */
  --f1-red:        #E8002D;
  --f1-red-dim:    rgba(232, 0, 45, 0.12);
  --f1-red-glow:   rgba(232, 0, 45, 0.06);

  /* Text */
  --text-primary:  #f5f5f5;
  --text-secondary: #a0a0a0;
  --text-muted:    #606060;

  /* Data colours */
  --data-green:    #00d26a;
  --data-amber:    #ffb800;
  --data-blue:     #00aaff;

  /* Borders */
  --border-dim:    rgba(255, 255, 255, 0.06);
  --border-bright: rgba(255, 255, 255, 0.12);
  --border-accent: rgba(232, 0, 45, 0.4);

  /* Team colours (for driver cards) */
  --team-redbull:   #3671C6;
  --team-ferrari:   #E8002D;
  --team-mercedes:  #27F4D2;
  --team-mclaren:   #FF8000;
  --team-alpine:    #FF87BC;
  --team-aston:     #229971;
  --team-williams:  #64C4FF;
  --team-haas:      #B6BABD;
  --team-sauber:    #52E252;
  --team-rb:        #6692FF;

  /* Compound colours */
  --compound-soft:   #E8002D;
  --compound-medium: #FFD700;
  --compound-hard:   #F0F0F0;
  --compound-inter:  #39B54A;
  --compound-wet:    #0067FF;

  /* Typography scale */
  --font-display: 'Rajdhani', sans-serif;
  --font-mono:    'JetBrains Mono', monospace;
  --font-body:    'Inter', sans-serif;

  /* Spacing */
  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-lg: 12px;
}
```

---

## 7. API Integration

### 7.1 FastAPI Backend Endpoints Used

| Frontend action | Endpoint | Method |
|---|---|---|
| Load race list | `/api/races?year=2024` | GET |
| Load race state | `/api/races/{session_key}` | GET |
| Get lap state (replay) | `/api/races/{session_key}/lap/{lap}` | GET |
| Send chat message | `/api/chat` | POST |
| Live race stream | `ws://host/ws` | WebSocket |

### 7.2 Typed API Client

```typescript
// lib/api.ts
const BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

export async function sendChatMessage(payload: ChatRequest): Promise<ChatResponse> {
  const res = await fetch(`${BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) throw new Error(`Chat API error: ${res.status}`)
  return res.json()
}

export async function getRaceState(sessionKey: number): Promise<RaceState> {
  const res = await fetch(`${BASE}/api/races/${sessionKey}`)
  if (!res.ok) throw new Error(`Race state error: ${res.status}`)
  return res.json()
}
```

### 7.3 Chat Request/Response Types

```typescript
// Mirrors PITWALL_AI_SPEC.md exactly
interface ChatRequest {
  session_key: number
  current_lap: number
  message: string
  conversation_history: Message[]
}

interface ChatResponse {
  response: string
  strategy_data: StrategyRecommendation | null
  agents_used: string[]
  processing_time_ms: number
}

interface StrategyRecommendation {
  driver_number: number
  recommended_action: 'PIT' | 'STAY_OUT' | 'MONITOR'
  recommended_compound: 'SOFT' | 'MEDIUM' | 'HARD' | 'INTERMEDIATE' | 'WET'
  confidence: number          // 0–1
  undercut_viable: boolean
  overcut_viable: boolean
  reasoning: string
}
```

---

## 8. Environment Variables

```bash
# frontend/.env.local
NEXT_PUBLIC_API_URL=http://localhost:8000      # FastAPI backend
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws      # WebSocket
NEXT_PUBLIC_ENV=development
```

For production (Vercel):
```bash
NEXT_PUBLIC_API_URL=https://your-api.render.com
NEXT_PUBLIC_WS_URL=wss://your-api.render.com/ws
```

---

## 9. Performance Considerations

**Asset optimization:**
- All race photography served via `next/image` with WebP conversion and lazy loading
- 3D `.glb` models loaded with `useGLTF` from `@react-three/drei` — suspense boundary wraps the canvas so the rest of the page renders immediately
- Lenis + GSAP animations only initialised client-side (`'use client'` + `useEffect`) — no SSR animation code

**Reducing 3D cost:**
- The circuit wireframe uses `MeshBasicMaterial` (no lighting calculations) — minimal GPU load
- `pixelRatio` capped at `Math.min(window.devicePixelRatio, 1.5)` in the R3F canvas
- The 3D canvas is unmounted when scrolled past (IntersectionObserver)

**Bundle size:**
- Three.js tree-shakes well with R3F — only import the geometries/materials used
- GSAP plugins imported individually (`gsap/ScrollTrigger`), not the full bundle

---

## 10. Build & Dev Commands

```bash
# From repo root
cd frontend

# Install dependencies
npm install

# Dev server (with HMR)
npm run dev          # → http://localhost:3000

# Production build
npm run build
npm run start

# Type check
npm run type-check

# Lint
npm run lint
```

`package.json` dependencies:
```json
{
  "dependencies": {
    "next": "^14.2.0",
    "react": "^18.3.0",
    "react-dom": "^18.3.0",
    "gsap": "^3.12.5",
    "@gsap/react": "^2.1.1",
    "lenis": "^1.1.14",
    "framer-motion": "^11.3.0",
    "three": "^0.167.0",
    "@react-three/fiber": "^8.17.0",
    "@react-three/drei": "^9.109.0",
    "swr": "^2.2.5",
    "clsx": "^2.1.1",
    "tailwind-merge": "^2.4.0"
  },
  "devDependencies": {
    "typescript": "^5.5.0",
    "@types/react": "^18.3.0",
    "@types/three": "^0.167.0",
    "tailwindcss": "^3.4.0",
    "autoprefixer": "^10.4.0",
    "postcss": "^8.4.0"
  }
}
```

---

## 11. Implementation Phases

### Phase 1 — Foundation (Week 1)
- [ ] Scaffold `frontend/` with Next.js 14 + TypeScript + Tailwind
- [ ] Configure Lenis smooth scroll provider in root layout
- [ ] Set up GSAP + ScrollTrigger integration
- [ ] Build typed API client (`lib/api.ts`) wired to FastAPI
- [ ] Deploy skeleton to Vercel, confirm backend connectivity

### Phase 2 — Strategy Console (Week 2)
- [ ] Build `StrategyConsole` — chat UI, message history, input
- [ ] Build `TelemetryPanel` — running order, tire compounds, gap analysis
- [ ] Build `RecommendationCard` with border-draw entry animation
- [ ] Build `AgentTrace` with typewriter effect
- [ ] Build `LapTimeChart` (Chart.js or Recharts)
- [ ] Wire WebSocket for live data updates

### Phase 3 — Immersive Landing (Week 3)
- [ ] Build 3D hero with R3F circuit wireframe + cursor tracking
- [ ] Build kinetic typography section with GSAP ScrollTrigger
- [ ] Build parallax gallery — 6 cards at varying speeds
- [ ] Build split-screen "On Pit Wall / On Track" section

### Phase 4 — Polish (Week 4)
- [ ] Axis-hijacked horizontal race calendar carousel
- [ ] Staggered circuit grid with cascading reveal
- [ ] Page transitions (Framer Motion `AnimatePresence`)
- [ ] Mobile responsiveness audit
- [ ] Performance audit — Lighthouse score ≥ 90
- [ ] Final Vercel production deployment

---

## 12. Relationship to Existing Backend

The frontend is a pure consumer of the existing FastAPI backend. Zero changes are required to `src/` for Phase 1–3. The only backend work that may be needed for Phase 4 (live mode) is enabling the existing WebSocket endpoint in `src/api/websocket.py` — the scaffold is already there per the backend spec.

The Streamlit frontend (`src/ui/app.py`) can remain running in parallel during development. It is not deprecated until the Next.js frontend reaches feature parity.

---

*End of spec. For questions on the backend contract (request/response shapes, available endpoints), refer to `PITWALL_AI_SPEC.md`.*
