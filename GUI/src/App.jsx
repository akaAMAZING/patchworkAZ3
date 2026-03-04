import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

/**
 * Patchwork Pro GUI (Production-grade overhaul)
 *
 * Goals:
 * - Accurate rules: never fabricate moves; apply only server-provided legal actions.
 * - Snappy: avoid heavy recompute, prevent request races, keep UI responsive.
 * - QoL: edit mode, JSON import/export, full piece atlas + circle editor, model manager, move log.
 * - Visual fidelity: keep pieces visible; render income squares (val=2) from action.cells locally.
 *
 * Works with patchwork_api.py endpoints:
 *  GET  /new, /pieces, /nn/status
 *  POST /legal, /apply, /solve, /solve_nn, /nn/load, /nn/unload
 */

const DEFAULT_API =
  // Vite
  (typeof import.meta !== "undefined" && import.meta.env && (import.meta.env.VITE_PATCHWORK_API || import.meta.env.VITE_API_URL)) ||
  // CRA
  (typeof process !== "undefined" && process.env && (process.env.REACT_APP_PATCHWORK_API || process.env.REACT_APP_API_URL)) ||
  "http://127.0.0.1:8000";

// #region agent log
function _dbgLog(location, message, data, hypothesisId) {
  const payload = { sessionId: "285c74", location, message, data: data || {}, timestamp: Date.now(), hypothesisId };
  fetch("http://127.0.0.1:7772/ingest/33e70ed1-8e9b-4e3a-8ab0-9483b5954fab", { method: "POST", headers: { "Content-Type": "application/json", "X-Debug-Session-Id": "285c74" }, body: JSON.stringify(payload) }).catch(() => {});
}
// #endregion

const BOARD_N = 9;

const COLORS = {
  bg: "#0a0e1a",
  panel: "#0f172a",
  panel2: "#0b1224",
  border: "#1e293b",
  text: "#e5e7eb",
  muted: "#94a3b8",
  dim: "#64748b",
  dangerBg: "#3f1d1d",
  danger: "#fecaca",
  warn: "#f59e0b",
  ok: "#22c55e",
  blue: "#3b82f6",
  green: "#16a34a",
  indigo: "#4f46e5",
  purple: "#a855f7",
  amber: "#fbbf24",
};

/** Colors for last-turn placements (1st..4th), shown only after turn has ended */
const PLACEMENT_COLORS = ["#00FF6A", "#FFEA00", "#FF00C8", "#0044FF"]; // 1st Neon Green, 2nd Pure Yellow, 3rd Hot Magenta, 4th Electric Blue

/** Per-piece hex colors (piece id 0..32); index 33 = patch (Abyss Navy) */
const PIECE_HEX = [
  "#2C3E50", "#8B2635", "#2D6A4F", "#7B3F00", "#4A235A", "#1A5276", "#6B4226", "#1C4E40", "#7D6608", "#512E5F",
  "#922B21", "#1F618D", "#5D4037", "#2E4057", "#6D4C41", "#3B5323", "#7C3626", "#1B4F72", "#4E342E", "#424242",
  "#5C4033", "#2C6975", "#6A1E55", "#3D3D3D", "#7A4419", "#1E5631", "#5B2C6F", "#884EA0", "#2C3A47", "#A04000",
  "#1A3A4A", "#4D5656", "#6E2C00",
];
const PATCH_HEX = "#7F8C8D";
const DEFAULT_FILL_HEX = "#1e3a5f";

function getCellFillHex(pieceId) {
  if (pieceId == null) return DEFAULT_FILL_HEX;
  if (pieceId === "patch" || pieceId === 34) return PATCH_HEX;
  const i = Number(pieceId);
  return Number.isInteger(i) && i >= 0 && i < PIECE_HEX.length ? PIECE_HEX[i] : DEFAULT_FILL_HEX;
}

function clamp(n, lo, hi) {
  return Math.max(lo, Math.min(hi, n));
}

function emptyBoard() {
  return Array.from({ length: BOARD_N }, () => Array(BOARD_N).fill(0));
}

function emptyPieceIdBoard() {
  return Array.from({ length: BOARD_N }, () => Array(BOARD_N).fill(null));
}

function cloneBoard(b) {
  return b.map((r) => r.slice());
}

function boardFromRows(rows) {
  const out = emptyBoard();
  if (!Array.isArray(rows) || rows.length !== BOARD_N) return out;
  for (let r = 0; r < BOARD_N; r++) {
    const line = String(rows[r] ?? "");
    for (let c = 0; c < BOARD_N; c++) {
      const ch = line[c];
      out[r][c] = ch === "2" ? 2 : ch === "1" ? 1 : 0;
    }
  }
  return out;
}

function boardToRows(board) {
  return board.map((row) => row.map((v) => (v === 0 ? "." : String(v))).join(""));
}

function countCoverage(board) {
  let n = 0;
  for (const row of board) for (const v of row) if (v !== 0) n++;
  return n;
}

function basename(p) {
  if (!p) return "";
  const s = String(p);
  return s.replace(/\\/g, "/").split("/").pop();
}

/**
 * Derive whose turn it is from game state. Mirrors API current_player logic.
 * IMPORTANT: Patch placement takes absolute precedence over timeboard.
 * When either player has bought a piece with patches to place, they must place ALL
 * before the turn can switch. Only after pending_patches === 0 do we use timeboard.
 */
function currentPlayerFromState(st) {
  if (!st?.players?.length) return 0;
  const pending = Number(st.pending_patches ?? 0);
  if (pending > 0) return Number(st.pending_owner ?? 0); // Patch placement mode
  const p0pos = Number(st.players[0]?.position ?? 0);
  const p1pos = Number(st.players[1]?.position ?? 0);
  if (p0pos < p1pos) return 0; // P0 behind on timeboard
  if (p1pos < p0pos) return 1; // P1 behind on timeboard
  return Number(st.tie_player ?? 0);
}

function useLocalStorageState(key, initialValue) {
  const [value, setValue] = useState(() => {
    try {
      const raw = localStorage.getItem(key);
      return raw == null ? initialValue : JSON.parse(raw);
    } catch {
      return initialValue;
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      // ignore
    }
  }, [key, value]);
  return [value, setValue];
}

function useAbortableRequest() {
  // Allows concurrent requests. Aborts all in-flight on unmount.
  const ctrlsRef = useRef(new Set());
  const abortAll = useCallback(() => {
    for (const ctrl of ctrlsRef.current) ctrl.abort();
    ctrlsRef.current.clear();
  }, []);

  const requestJson = useCallback(async (url, options = {}) => {
    const ctrl = new AbortController();
    ctrlsRef.current.add(ctrl);
    let res;
    let txt = "";
    try {
      res = await fetch(url, { ...options, signal: ctrl.signal });
      txt = await res.text();
    } finally {
      ctrlsRef.current.delete(ctrl);
    }
    let data = {};
    try {
      data = txt ? JSON.parse(txt) : {};
    } catch {
      // ignore
    }
    if (!res.ok) {
      const msg = (data && data.error) || txt || `Request failed (${res.status})`;
      throw new Error(msg);
    }
    return data;
  }, []);

  useEffect(() => abortAll, [abortAll]);
  return { requestJson, abortAll };
}

function IconDot({ color = COLORS.muted }) {
  return <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: 999, background: color, marginRight: 6 }} />;
}

function Pill({ children, tone = "neutral" }) {
  const bg = tone === "danger" ? "#3f1d1d" : tone === "warn" ? "#3b2b0b" : tone === "ok" ? "#13351f" : "#101a2f";
  const bd = tone === "danger" ? "#7f1d1d" : tone === "warn" ? "#854d0e" : tone === "ok" ? "#14532d" : COLORS.border;
  const fg = tone === "danger" ? COLORS.danger : tone === "warn" ? "#fde68a" : tone === "ok" ? "#bbf7d0" : COLORS.muted;
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        padding: "2px 8px",
        borderRadius: 999,
        border: `1px solid ${bd}`,
        background: bg,
        color: fg,
        fontSize: 12,
        lineHeight: 1.6,
      }}
    >
      {children}
    </span>
  );
}

function MiniShape({ cells, cellPx = 10 }) {
  if (!cells || cells.length === 0) return null;
  let maxR = 0,
    maxC = 0;
  for (const x of cells) {
    maxR = Math.max(maxR, x.r);
    maxC = Math.max(maxC, x.c);
  }
  const h = maxR + 1;
  const w = maxC + 1;
  const key = (r, c) => `${r},${c}`;
  const map = new Map(cells.map((x) => [key(x.r, x.c), x.val]));
  return (
    <div style={{ display: "inline-grid", gridTemplateColumns: `repeat(${w}, ${cellPx}px)`, gap: 1 }}>
      {Array.from({ length: h }).flatMap((_, r) =>
        Array.from({ length: w }).map((_, c) => {
          const v = map.get(key(r, c)) || 0;
          const bg = v === 0 ? "transparent" : v === 2 ? COLORS.green : COLORS.blue;
          const bd = v === 0 ? `1px solid ${COLORS.border}` : "1px solid rgba(0,0,0,0.25)";
          return <div key={key(r, c)} style={{ width: cellPx, height: cellPx, borderRadius: 2, border: bd, background: bg }} />;
        }),
      )}
    </div>
  );
}

function computeLocalCellsFromShape(shape) {
  if (!shape || !Array.isArray(shape) || shape.length === 0) return [];
  const cells = [];
  for (let r = 0; r < shape.length; r++) {
    for (let c = 0; c < shape[0].length; c++) {
      const v = shape[r][c];
      if (v !== 0) cells.push({ r, c, val: v });
    }
  }
  return cells;
}

/** Normalize placement cells to origin (0,0) for mini preview */
function cellsToRelative(cells) {
  if (!cells?.length) return [];
  const minR = Math.min(...cells.map((c) => c.r));
  const minC = Math.min(...cells.map((c) => c.c));
  return cells.map((c) => ({ r: c.r - minR, c: c.c - minC, val: c.val ?? 1 }));
}

function BoardGrid({
  title,
  subtitle,
  active,
  accent,
  board,
  pieceIdBoard,
  onHoverCell,
  onLeave,
  onClickCell,
  overlayCells,
  highlightIdxSet,
  hintTopLeftSet,
  lastMovePlacements,
  editMode,
}) {
  const overlayMap = useMemo(() => {
    const m = new Map();
    if (!overlayCells) return m;
    for (const x of overlayCells) m.set(`${x.r},${x.c}`, x.val);
    return m;
  }, [overlayCells]);

  const placementColorByKey = useMemo(() => {
    const m = new Map();
    if (!Array.isArray(lastMovePlacements)) return m;
    for (let i = 0; i < Math.min(lastMovePlacements.length, 4); i++) {
      const cells = lastMovePlacements[i];
      if (!Array.isArray(cells)) continue;
      const hex = PLACEMENT_COLORS[i] ?? PLACEMENT_COLORS[0];
      for (const cell of cells) {
        if (cell?.r != null && cell?.c != null) m.set(`${cell.r},${cell.c}`, hex);
      }
    }
    return m;
  }, [lastMovePlacements]);

  return (
    <div style={{ background: COLORS.panel, border: `2px solid ${active ? accent : COLORS.border}`, borderRadius: 14, padding: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 8, gap: 10 }}>
        <div>
          <div style={{ fontWeight: 800, fontSize: 14 }}>{title}</div>
          <div style={{ fontSize: 12, color: COLORS.muted }}>{subtitle}</div>
        </div>
        <Pill>
          <IconDot color={accent} />
          {countCoverage(board)}/81
        </Pill>
      </div>

      <div
        onMouseLeave={onLeave}
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(9, 34px)",
          gap: 2,
          background: COLORS.panel2,
          borderRadius: 10,
          padding: 6,
          userSelect: "none",
        }}
      >
        {board.map((row, r) =>
          row.map((v, c) => {
            const idx = r * BOARD_N + c;
            const over = overlayMap.has(`${r},${c}`);
            const ov = overlayMap.get(`${r},${c}`);
            const patchLegal = highlightIdxSet && highlightIdxSet.has(idx);
            const hintTopLeft = hintTopLeftSet && hintTopLeftSet.has(idx);
            const placementHex = placementColorByKey.get(`${r},${c}`);
            const isLastMove = !!placementHex;
            const filled = v !== 0;
            const pieceId = pieceIdBoard?.[r]?.[c];
            const filledBg = isLastMove ? (placementHex + "99") : filled ? getCellFillHex(pieceId) : "#0b1430";
            const bg = over
              ? "rgba(251,191,36,0.22)"
              : patchLegal
                ? "rgba(34,197,94,0.20)"
                : hintTopLeft
                  ? "rgba(59,130,246,0.14)"
                  : filledBg;
            const border = over
              ? `2px solid ${COLORS.amber}`
              : isLastMove
                ? `2px solid ${placementHex}`
                : patchLegal
                  ? `2px solid ${COLORS.ok}`
                  : hintTopLeft
                    ? `1px dashed rgba(147,197,253,0.7)`
                    : `1px solid ${COLORS.border}`;
            const symbol = over ? (ov === 2 ? "●" : "■") : v === 2 ? "●" : v === 1 ? "■" : "";
            return (
              <button
                key={`${r}-${c}`}
                type="button"
                onMouseEnter={() => onHoverCell?.(r, c, idx)}
                onClick={() => onClickCell?.(r, c, idx)}
                style={{
                  width: 34,
                  height: 34,
                  borderRadius: 6,
                  border,
                  background: bg,
                  color: "#dbeafe",
                  fontWeight: 900,
                  cursor: editMode || patchLegal || hintTopLeft || over ? "pointer" : "default",
                  outline: "none",
                }}
              >
                {symbol}
              </button>
            );
          }),
        )}
      </div>
    </div>
  );
}

function Section({ title, right, children }) {
  return (
    <div style={{ background: COLORS.panel, border: `1px solid ${COLORS.border}`, borderRadius: 12, padding: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
        <div style={{ fontSize: 12, fontWeight: 800, letterSpacing: 0.4, color: COLORS.dim }}>{title}</div>
        {right}
      </div>
      {children}
    </div>
  );
}

function Button({ children, onClick, disabled, tone = "neutral", title }) {
  const bg = tone === "primary" ? COLORS.indigo : tone === "ok" ? COLORS.ok : tone === "warn" ? COLORS.warn : COLORS.panel2;
  const fg = tone === "primary" || tone === "ok" || tone === "warn" ? "#ffffff" : COLORS.text;
  const bd = tone === "neutral" ? `1px solid ${COLORS.border}` : "none";
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      disabled={disabled}
      style={{
        padding: "8px 12px",
        borderRadius: 10,
        border: bd,
        background: disabled ? "#0a132b" : bg,
        color: disabled ? COLORS.dim : fg,
        cursor: disabled ? "not-allowed" : "pointer",
        fontWeight: 800,
      }}
    >
      {children}
    </button>
  );
}

function Field({ label, children }) {
  return (
    <label style={{ display: "grid", gap: 6, fontSize: 12, color: COLORS.muted }}>
      <span>{label}</span>
      {children}
    </label>
  );
}

function Input({ value, onChange, type = "text", min, max, step, placeholder }) {
  return (
    <input
      value={value}
      onChange={onChange}
      type={type}
      min={min}
      max={max}
      step={step}
      placeholder={placeholder}
      style={{ width: "100%", padding: "8px 10px", borderRadius: 10, background: COLORS.panel2, border: `1px solid ${COLORS.border}`, color: COLORS.text }}
    />
  );
}

function Select({ value, onChange, children, disabled }) {
  return (
    <select
      value={value}
      onChange={onChange}
      disabled={disabled}
      style={{ width: "100%", padding: "8px 10px", borderRadius: 10, background: COLORS.panel2, border: `1px solid ${COLORS.border}`, color: COLORS.text }}
    >
      {children}
    </select>
  );
}

function MarketRing({ piecesById, circle, neutral, selectedPieceId, onSelectPiece, compact = false }) {
  const n = circle?.length || 0;
  if (!n) return <div style={{ color: COLORS.dim, fontSize: 12 }}>Circle is empty.</div>;
  return (
    <div style={{ display: "grid", gap: 8 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <Pill>
            <span style={{ color: COLORS.dim }}>Neutral</span>
            <span style={{ color: COLORS.text, fontWeight: 800 }}>idx {neutral}</span>
          </Pill>
          <Pill>
            <span style={{ color: COLORS.dim }}>Pieces left</span>
            <span style={{ color: COLORS.text, fontWeight: 800 }}>{n}</span>
          </Pill>
        </div>
      </div>
      <div style={{ display: "flex", gap: 6, overflowX: "auto", paddingBottom: 4 }}>
        {circle.map((pid, idx) => {
          const piece = piecesById[pid];
          const isNeutral = idx === neutral;
          const isSelected = pid === selectedPieceId;
          const border = isSelected ? `2px solid ${COLORS.amber}` : `1px solid ${COLORS.border}`;
          const bg = isNeutral ? "rgba(168,85,247,0.18)" : "rgba(59,130,246,0.10)";
          const localCells = piece?.shape ? computeLocalCellsFromShape(piece.shape) : [];
          return (
            <button
              key={`${idx}-${pid}`}
              type="button"
              onClick={() => onSelectPiece?.(pid)}
              style={{
                minWidth: compact ? 86 : 120,
                padding: 8,
                borderRadius: 12,
                border,
                background: bg,
                color: COLORS.text,
                textAlign: "left",
                cursor: "pointer",
              }}
              title={isNeutral ? "Neutral pawn here" : ""}
            >
              <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
                <div style={{ display: "grid", gap: 2 }}>
                  <div style={{ fontWeight: 900, fontSize: 12 }}>#{pid}</div>
                  <div style={{ fontSize: 11, color: COLORS.muted }}>{piece ? `${piece.cost_buttons}b / ${piece.cost_time}t / +${piece.income}` : "…"}</div>
                  {isNeutral && <div style={{ fontSize: 11, color: COLORS.purple, fontWeight: 800 }}>NEUTRAL</div>}
                </div>
                {!compact && piece?.shape && <MiniShape cells={localCells} cellPx={8} />}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function TopMoves({ solveResult }) {
  if (!solveResult?.top?.length) return <div style={{ color: COLORS.dim, fontSize: 12 }}>No search breakdown yet.</div>;
  return (
    <div style={{ display: "grid", gap: 6 }}>
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
        {solveResult.total_sims != null && (
          <Pill>
            total_sims <b style={{ color: COLORS.text }}>{solveResult.total_sims}</b>
          </Pill>
        )}
        {solveResult.search_time != null && (
          <Pill>
            search_time <b style={{ color: COLORS.text }}>{solveResult.search_time}s</b>
          </Pill>
        )}
      </div>
      <div style={{ display: "grid", gap: 4 }}>
        {solveResult.top.map((x, i) => (
          <div key={i} style={{ display: "flex", justifyContent: "space-between", gap: 10, fontSize: 12, color: COLORS.muted }}>
            <div style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{x.pretty}</div>
            <div style={{ display: "flex", gap: 10, flexShrink: 0 }}>
              {x.visits != null && <span>v={x.visits}</span>}
              {x.winProb != null && <span>p={(100 * x.winProb).toFixed(1)}%</span>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [apiBase, setApiBase] = useLocalStorageState("pw_api_base", DEFAULT_API);
  const { requestJson } = useAbortableRequest();

  const [gameState, setGameState] = useState(null);
  const [toMoveServer, setToMoveServer] = useState(null);
  const [boards, setBoards] = useState([emptyBoard(), emptyBoard()]);
  const [pieces, setPieces] = useState([]);
  const piecesById = useMemo(() => Object.fromEntries((pieces || []).map((p) => [p.id, p])), [pieces]);

  const [legal, setLegal] = useState(null);
  const [selectedKey, setSelectedKey] = useState(null);
  const [selectedOrient, setSelectedOrient] = useState(null);
  const [previewAction, setPreviewAction] = useState(null);
  const [hintTopLeftSet, setHintTopLeftSet] = useState(null);
  const [patchIdxToAction, setPatchIdxToAction] = useState(new Map());

  const [showSetup, setShowSetup] = useLocalStorageState("pw_show_setup", true);
  const [editMode, setEditMode] = useLocalStorageState("pw_edit_mode", false);
  const [atlasOpen, setAtlasOpen] = useLocalStorageState("pw_atlas_open", false);
  const [buildMode, setBuildMode] = useLocalStorageState("pw_build_mode", false);

  const [gameMode, setGameMode] = useLocalStorageState("pw_game_mode", "human_vs_ai");
  const [humanPlayer, setHumanPlayer] = useLocalStorageState("pw_human_player", 1);
  const [autoPlayAivsAi, setAutoPlayAivsAi] = useLocalStorageState("pw_autoplay", false);
  const [autoPlayDelayMs, setAutoPlayDelayMs] = useLocalStorageState("pw_autoplay_delay", 350);

  const [engine, setEngine] = useLocalStorageState("pw_engine", "nn");
  const [mctsIterations, setMctsIterations] = useLocalStorageState("pw_mcts_iters", 50000);
  const [mctsWorkers, setMctsWorkers] = useLocalStorageState("pw_mcts_workers", 4);
  const [nnSimulations, setNnSimulations] = useLocalStorageState("pw_nn_sims", 3000);

  const [nnStatus, setNnStatus] = useState(null);
  // Best model: committed iter69. Fallback default: latest_model.pt
  const ITER69_PATH = "C:\\Users\\Shanks\\Desktop\\Codes\\patchworkaz - Copy - v2\\runs\\patchwork_production\\committed\\iter_069\\iteration_069.pt";
  const [nnPath, setNnPath] = useLocalStorageState("pw_nn_path", ITER69_PATH);
  const [nnConfig, setNnConfig] = useLocalStorageState("pw_nn_cfg", "C:\\Users\\Shanks\\Desktop\\Codes\\patchworkaz - Copy - v2\\configs\\config_best.yaml");
  useEffect(() => {
    try {
      const raw = localStorage.getItem("pw_nn_cfg");
      if (raw != null && raw.includes("configs/config_best.yaml") && !raw.includes("Users\\\\Shanks")) {
        localStorage.setItem("pw_nn_cfg", JSON.stringify("C:\\Users\\Shanks\\Desktop\\Codes\\patchworkaz - Copy - v2\\configs\\config_best.yaml"));
        setNnConfig("C:\\Users\\Shanks\\Desktop\\Codes\\patchworkaz - Copy - v2\\configs\\config_best.yaml");
      }
      const pathRaw = localStorage.getItem("pw_nn_path");
      if (pathRaw != null && (pathRaw.includes("latest_model.pt") || pathRaw.endsWith("checkpoints\\\\latest_model.pt"))) {
        localStorage.setItem("pw_nn_path", JSON.stringify(ITER69_PATH));
        setNnPath(ITER69_PATH);
      }
    } catch (_) {}
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const [nnDevice, setNnDevice] = useLocalStorageState("pw_nn_dev", "cuda");

  const [jsonText, setJsonText] = useState("{");
  const [jsonDirty, setJsonDirty] = useState(false);
  const [manualCircle, setManualCircle] = useState(false);

  const [busy, setBusy] = useState({ thinking: false, legal: false, loading: false, nn: false });
  const [banner, setBanner] = useState({ kind: "info", text: "Ready." });
  const [moveLog, setMoveLog] = useState([]);
  const [solveBreakdown, setSolveBreakdown] = useState(null);
  const [lastMovePlacementsP0, setLastMovePlacementsP0] = useState([]);
  const [lastMovePlacementsP1, setLastMovePlacementsP1] = useState([]);
  const [pieceIdBoardP0, setPieceIdBoardP0] = useState(() => emptyPieceIdBoard());
  const [pieceIdBoardP1, setPieceIdBoardP1] = useState(() => emptyPieceIdBoard());
  const [gameStarted, setGameStarted] = useState(false);

  const boardsRef = useRef(boards);
  useEffect(() => {
    boardsRef.current = boards;
  }, [boards]);

  const solveAndPlayRef = useRef(null);

  const setInfo = useCallback((text) => setBanner({ kind: "info", text }), []);
  const setWarn = useCallback((text) => setBanner({ kind: "warn", text }), []);
  const setErr = useCallback((text) => setBanner({ kind: "error", text }), []);

  const api = useMemo(() => {
    const base = String(apiBase || "").replace(/\/$/, "");
    return {
      base,
      get: (path) => requestJson(`${base}${path}`),
      post: (path, body) =>
        requestJson(`${base}${path}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }),
    };
  }, [apiBase, requestJson]);

  const toMove = useMemo(() => {
    if (legal?.to_move != null) return legal.to_move;
    if (toMoveServer != null) return toMoveServer;
    if (gameState) return currentPlayerFromState(gameState);
    return 0;
  }, [legal?.to_move, toMoveServer, gameState]);

  const isTerminal = Boolean(legal?.terminal);
  const isHumanTurn = gameMode === "human_vs_ai" && toMove === humanPlayer && !isTerminal;
  const nnAvailable = Boolean(nnStatus?.nn_loaded);

  const circle = gameState?.circle || [];
  const neutral = gameState?.neutral ?? 0;
  const next3 = useMemo(() => {
    const n = circle.length;
    const out = [];
    for (let off = 1; off <= Math.min(3, n); off++) out.push({ off, piece_id: circle[(neutral + off) % n] });
    return out;
  }, [circle, neutral]);

  const stateForApi = useMemo(() => {
    if (!gameState) return null;
    return {
      ...gameState,
      players: [
        { ...gameState.players[0], board: boardToRows(boards[0]) },
        { ...gameState.players[1], board: boardToRows(boards[1]) },
      ],
      randomize_circle: manualCircle ? false : undefined,
    };
  }, [gameState, boards, manualCircle]);

  const selectedGroup = useMemo(() => {
    if (!legal?.buy_groups || !selectedKey) return null;
    const [offset, pid] = selectedKey.split("-").map((x) => parseInt(x, 10));
    return legal.buy_groups.find((g) => g.offset === offset && g.piece_id === pid) || null;
  }, [legal?.buy_groups, selectedKey]);

  const placementIndex = useMemo(() => {
    if (!selectedGroup?.placements) return null;
    const byOrient = new Map();
    const orientList = [];
    for (const a of selectedGroup.placements) {
      const oi = Number(a.orient);
      if (!byOrient.has(oi)) {
        byOrient.set(oi, new Map());
        orientList.push(oi);
      }
      byOrient.get(oi).set(`${a.top},${a.left}`, a);
    }
    orientList.sort((a, b) => a - b);
    return { byOrient, orientList };
  }, [selectedGroup]);

  useEffect(() => {
    if (!placementIndex) {
      setSelectedOrient(null);
      return;
    }
    if (selectedOrient == null || !placementIndex.orientList.includes(selectedOrient)) {
      setSelectedOrient(placementIndex.orientList[0] ?? null);
    }
  }, [placementIndex, selectedOrient]);

  useEffect(() => {
    if (!placementIndex || selectedOrient == null) {
      setHintTopLeftSet(null);
      return;
    }
    const m = placementIndex.byOrient.get(selectedOrient);
    if (!m) {
      setHintTopLeftSet(null);
      return;
    }
    const s = new Set();
    for (const key of m.keys()) {
      const [top, left] = key.split(",").map((x) => parseInt(x, 10));
      s.add(top * BOARD_N + left);
    }
    setHintTopLeftSet(s);
  }, [placementIndex, selectedOrient]);

  const applyServerState = useCallback(
    (payload) => {
      const st = payload?.state || payload;
      setGameState(st);
      if (payload?.to_move != null) setToMoveServer(payload.to_move);

      // #region agent log
      const p0Rows = st?.players?.[0]?.board;
      const p1Rows = st?.players?.[1]?.board;
      const serverP0Twos = Array.isArray(p0Rows) ? p0Rows.join("").split("2").length - 1 : 0;
      const serverP1Twos = Array.isArray(p1Rows) ? p1Rows.join("").split("2").length - 1 : 0;
      _dbgLog("App.jsx:applyServerState", "applyServerState entry", { to_move: payload?.to_move, terminal: payload?.terminal, serverP0Twos, serverP1Twos }, "H1");
      // #endregion

      const occ0 = boardFromRows(st?.players?.[0]?.board);
      const occ1 = boardFromRows(st?.players?.[1]?.board);
      setPieceIdBoardP0((prev) => {
        const out = prev.map((row) => row.slice());
        for (let r = 0; r < BOARD_N; r++) for (let c = 0; c < BOARD_N; c++) if (occ0[r][c] === 0) out[r][c] = null;
        return out;
      });
      setPieceIdBoardP1((prev) => {
        const out = prev.map((row) => row.slice());
        for (let r = 0; r < BOARD_N; r++) for (let c = 0; c < BOARD_N; c++) if (occ1[r][c] === 0) out[r][c] = null;
        return out;
      });
      setBoards((prev) => {
        const next = [cloneBoard(prev[0]), cloneBoard(prev[1])];
        for (let p = 0; p < 2; p++) {
          const occ = p === 0 ? occ0 : occ1;
          for (let r = 0; r < BOARD_N; r++) {
            for (let c = 0; c < BOARD_N; c++) {
              if (occ[r][c] === 0) next[p][r][c] = 0;
              else if (next[p][r][c] === 0) next[p][r][c] = 1;
            }
          }
        }
        // #region agent log
        const n0 = next[0].flat();
        const n1 = next[1].flat();
        _dbgLog("App.jsx:applyServerState:setBoards", "after merge", { nextP0zeros: n0.filter((v) => v === 0).length, nextP0ones: n0.filter((v) => v === 1).length, nextP0twos: n0.filter((v) => v === 2).length, nextP1twos: n1.filter((v) => v === 2).length }, "H1");
        // #endregion
        if (!jsonDirty) {
          const derived = {
            ...st,
            players: [
              { ...st.players?.[0], board: boardToRows(next[0]) },
              { ...st.players?.[1], board: boardToRows(next[1]) },
            ],
            randomize_circle: manualCircle ? false : undefined,
          };
          setJsonText(JSON.stringify(derived, null, 2));
        }
        return next;
      });
    },
    [jsonDirty, manualCircle],
  );

  const refreshPieces = useCallback(async () => {
    const data = await api.get("/pieces");
    setPieces(data.pieces || []);
  }, [api]);

  const refreshNnStatus = useCallback(async () => {
    try {
      const s = await api.get("/nn/status");
      setNnStatus(s);
      if (!nnPath && s?.model_path) setNnPath(s.model_path);
      if (engine === "nn" && !s?.nn_loaded) {
        setWarn("NN not loaded on server; switched to Pure MCTS.");
        setEngine("mcts");
      }
    } catch {
      // ignore
    }
  }, [api, nnPath, setNnPath, engine, setEngine, setWarn]);

  const newGame = useCallback(async () => {
    setBusy((b) => ({ ...b, loading: true }));
    try {
      const [st, pc] = await Promise.all([api.get("/new"), api.get("/pieces")]);
      setPieces(pc.pieces || []);
      setGameState(st.state);
      setBoards([boardFromRows(st.state.players?.[0]?.board), boardFromRows(st.state.players?.[1]?.board)]);
      setManualCircle(false);
      setLegal(null);
      setSelectedKey(null);
      setSelectedOrient(null);
      setPreviewAction(null);
      setHintTopLeftSet(null);
      setSolveBreakdown(null);
      setMoveLog([]);
      setLastMovePlacementsP0([]);
      setLastMovePlacementsP1([]);
      setPieceIdBoardP0(emptyPieceIdBoard());
      setPieceIdBoardP1(emptyPieceIdBoard());
      setGameStarted(false);
      setJsonDirty(false);
      setJsonText(JSON.stringify(st.state, null, 2));
      // Fetch legal to get correct to_move (for display); game doesn't "start" until Start Game is pressed
      const legalData = await api.post("/legal", { state: st.state });
      setLegal(legalData);
      setToMoveServer(legalData?.to_move ?? null);
      if (legalData?.mode === "patch" && Array.isArray(legalData?.actions)) {
        const map = new Map();
        for (const a of legalData.actions) map.set(a.idx, a);
        setPatchIdxToAction(map);
      } else {
        setPatchIdxToAction(new Map());
      }
      setInfo("Board set up. Press Start Game to begin.");
    } catch (e) {
      setErr(String(e?.message || e));
    } finally {
      setBusy((b) => ({ ...b, loading: false }));
    }
  }, [api, setInfo, setErr]);

  const fetchLegal = useCallback(async () => {
    if (!stateForApi) return null;
    setBusy((b) => ({ ...b, legal: true }));
    try {
      const data = await api.post("/legal", { state: stateForApi });
      // #region agent log
      const b0 = stateForApi?.players?.[0]?.board?.join?.("") ?? "";
      const b1 = stateForApi?.players?.[1]?.board?.join?.("") ?? "";
      _dbgLog("App.jsx:fetchLegal", "legal response", { to_move: data?.to_move, terminal: data?.terminal, mode: data?.mode, pass_allowed: data?.pass_allowed, stateBoard0Twos: (b0.match(/2/g) || []).length, stateBoard1Twos: (b1.match(/2/g) || []).length }, "H2");
      // #endregion
      setLegal(data);
      setToMoveServer(data?.to_move ?? null);
      setSelectedKey(null);
      setSelectedOrient(null);
      setPreviewAction(null);
      // Keep solveBreakdown (win prob) - don't clear; user may want to see last evaluation
      if (data?.mode === "patch" && Array.isArray(data?.actions)) {
        const map = new Map();
        for (const a of data.actions) map.set(a.idx, a);
        setPatchIdxToAction(map);
      } else {
        setPatchIdxToAction(new Map());
      }
      setInfo(`Legal moves loaded for P${data?.to_move}.`);
      return data;
    } catch (e) {
      // #region agent log
      _dbgLog("App.jsx:fetchLegal", "API error", { error: String(e?.message || e) }, "H3");
      // #endregion
      setErr(String(e?.message || e));
      return null;
    } finally {
      setBusy((b) => ({ ...b, legal: false }));
    }
  }, [api, stateForApi, setInfo, setErr]);

  const stampActionCells = useCallback((playerIdx, actionObj) => {
    if (!actionObj?.cells) return;
    const pieceIdOrPatch = actionObj.piece_id !== undefined ? actionObj.piece_id : "patch";
    setBoards((prev) => {
      const next = [cloneBoard(prev[0]), cloneBoard(prev[1])];
      const b = next[playerIdx];
      for (const cell of actionObj.cells) {
        if (cell?.r == null || cell?.c == null) continue;
        b[cell.r][cell.c] = cell.val === 2 ? 2 : 1;
      }
      return next;
    });
    if (playerIdx === 0) {
      setPieceIdBoardP0((prev) => {
        const next = prev.map((row) => row.slice());
        for (const cell of actionObj.cells) if (cell?.r != null && cell?.c != null) next[cell.r][cell.c] = pieceIdOrPatch;
        return next;
      });
    } else {
      setPieceIdBoardP1((prev) => {
        const next = prev.map((row) => row.slice());
        for (const cell of actionObj.cells) if (cell?.r != null && cell?.c != null) next[cell.r][cell.c] = pieceIdOrPatch;
        return next;
      });
    }
  }, []);

  const applyAction = useCallback(
    async (playerIdx, actionObj, prettyOverride) => {
      if (!stateForApi) return null;
      setBusy((b) => ({ ...b, thinking: true }));
      const prevBoards = boardsRef.current;
      try {
        stampActionCells(playerIdx, actionObj);
        const data = await api.post("/apply", { state: stateForApi, action: actionObj });
        applyServerState(data);
        const cells = actionObj?.cells ?? [];
        const placement = Array.isArray(cells) && cells.length ? cells : [];
        if (playerIdx === 0) {
          setLastMovePlacementsP0((prev) => [...prev, placement].slice(-4));
          setLastMovePlacementsP1([]);
        } else {
          setLastMovePlacementsP1((prev) => [...prev, placement].slice(-4));
          setLastMovePlacementsP0([]);
        }
        setMoveLog((prev) => [...prev, { player: playerIdx, text: prettyOverride || actionObj?.pretty || actionObj?.type || "MOVE" }]);
        if (data?.terminal) setLegal({ terminal: true, to_move: data.to_move });
        else setLegal(null);
        setSelectedKey(null);
        setSelectedOrient(null);
        setPreviewAction(null);
        setHintTopLeftSet(null);
        // Don't fetchLegal here - it would use stale stateForApi from closure. The effect
        // (toMove === humanPlayer) fetches legal with correct state after re-render.
        // Human vs AI: when turn switches to AI, auto-play. Server's to_move already accounts
        // for patch placement (pending_owner) so we never trigger during human's patch placement.
        if (gameMode === "human_vs_ai" && data?.to_move != null && data.to_move !== humanPlayer && !data?.terminal) {
          setTimeout(() => solveAndPlayRef.current?.(), 50);
        }
        return data;
      } catch (e) {
        // #region agent log
        _dbgLog("App.jsx:applyAction", "apply API error", { error: String(e?.message || e), playerIdx, actionType: actionObj?.type }, "H3");
        // #endregion
        // Roll back optimistic visual stamp if server rejected.
        setBoards(prevBoards);
        setErr(String(e?.message || e));
        return null;
      } finally {
        setBusy((b) => ({ ...b, thinking: false }));
      }
    },
    [api, stateForApi, stampActionCells, applyServerState, setErr, gameMode, humanPlayer],
  );

  const solveAndPlay = useCallback(async () => {
    if (!stateForApi) return;
    // #region agent log
    _dbgLog("App.jsx:solveAndPlay", "solveAndPlay called", { engine, nnAvailable, circleLen: stateForApi?.circle?.length, neutral: stateForApi?.neutral }, "H2");
    // #endregion
    setBusy((b) => ({ ...b, thinking: true }));
    try {
      const useNn = engine === "nn" && nnAvailable;
      const endpoint = useNn ? "/solve_nn" : "/solve";
      const body = useNn
        ? { state: stateForApi, simulations: clamp(Number(nnSimulations) || 3000, 50, 20000), temperature: 0.0 }
        : {
            state: stateForApi,
            iterations: clamp(Number(mctsIterations) || 50000, 500, 2000000),
            workers: clamp(Number(mctsWorkers) || 4, 1, 64),
            seed: 0,
            exploration: 1.4,
            only_player0: false,
          };
      const data = await api.post(endpoint, body);
      // #region agent log
      _dbgLog("App.jsx:solveAndPlay", "solve response", { to_move: data?.to_move, terminal: data?.terminal, hasBestAction: Boolean(data?.best?.action), needs_opponent_move: data?.needs_opponent_move }, "H2");
      // #endregion
      setSolveBreakdown(data);
      if (data?.terminal) {
        setLegal({ terminal: true, to_move: data.to_move ?? toMove });
        return;
      }
      if (data?.best?.action) {
        const mover = data.to_move ?? toMove;
        // In human_vs_ai, never apply AI moves for the human player
        if (gameMode === "human_vs_ai" && mover === humanPlayer) return;
        await applyAction(mover, data.best.action, data.best.pretty);
        setInfo(`Applied AI move: ${data.best.pretty}`);
      }
    } catch (e) {
      // #region agent log
      _dbgLog("App.jsx:solveAndPlay", "solve API error", { error: String(e?.message || e) }, "H3");
      // #endregion
      setErr(String(e?.message || e));
    } finally {
      setBusy((b) => ({ ...b, thinking: false }));
    }
  }, [api, engine, nnAvailable, nnSimulations, mctsIterations, mctsWorkers, stateForApi, applyAction, toMove, gameMode, humanPlayer, setInfo, setErr]);

  useEffect(() => {
    solveAndPlayRef.current = solveAndPlay;
  }, [solveAndPlay]);

  const onHover = useCallback(
    (playerIdx, r, c) => {
      if (editMode) return;
      if (!isHumanTurn || playerIdx !== humanPlayer) return;
      if (legal?.mode === "patch") {
        const idx = r * BOARD_N + c;
        const a = patchIdxToAction.get(idx);
        setPreviewAction(a || null);
        return;
      }
      if (!placementIndex || selectedOrient == null) return;
      const a = placementIndex.byOrient.get(selectedOrient)?.get(`${r},${c}`);
      setPreviewAction(a || null);
    },
    [editMode, isHumanTurn, humanPlayer, legal?.mode, patchIdxToAction, placementIndex, selectedOrient],
  );

  const onClickCell = useCallback(
    async (playerIdx, r, c, idx) => {
      if (!gameState) return;
      if (!gameStarted && !editMode) {
        setWarn("Press Start Game first.");
        return;
      }
      if (editMode) {
        setBoards((prev) => {
          const next = [cloneBoard(prev[0]), cloneBoard(prev[1])];
          next[playerIdx][r][c] = (next[playerIdx][r][c] + 1) % 3;
          return next;
        });
        if (playerIdx === 0) {
          setPieceIdBoardP0((prev) => {
            const out = prev.map((row) => row.slice());
            out[r][c] = null;
            return out;
          });
        } else {
          setPieceIdBoardP1((prev) => {
            const out = prev.map((row) => row.slice());
            out[r][c] = null;
            return out;
          });
        }
        setJsonDirty(true);
        return;
      }
      if (!isHumanTurn || playerIdx !== humanPlayer) return;
      if (!legal) {
        setWarn("Legal moves not loaded yet. Click 'Refresh Legal'.");
        return;
      }
      if (legal.mode === "patch") {
        const a = patchIdxToAction.get(idx);
        if (a) await applyAction(humanPlayer, a, `PLACE_PATCH (${r + 1},${c + 1})`);
        return;
      }
      if (!placementIndex || selectedOrient == null) {
        setWarn("Select a piece (next 3) and orientation first.");
        return;
      }
      const a = placementIndex.byOrient.get(selectedOrient)?.get(`${r},${c}`);
      if (a) await applyAction(humanPlayer, a, `BUY piece ${a.piece_id} @ (${r + 1},${c + 1})`);
    },
    [gameState, gameStarted, editMode, isHumanTurn, humanPlayer, legal, patchIdxToAction, applyAction, placementIndex, selectedOrient, setWarn],
  );

  const onLeaveBoard = useCallback(() => setPreviewAction(null), []);

  const passTurn = useCallback(async () => {
    if (!isHumanTurn || !legal?.pass_allowed) return;
    await applyAction(humanPlayer, { type: "pass" }, "PASS");
  }, [isHumanTurn, legal?.pass_allowed, applyAction, humanPlayer]);

  const loadNnCheckpoint = useCallback(async () => {
    setBusy((b) => ({ ...b, nn: true }));
    try {
      const model_path = String(nnPath || "").trim();
      if (!model_path) throw new Error("Model path is empty.");
      await api.post("/nn/load", {
        model_path,
        config_path: String(nnConfig || "").trim() || "configs/config_best.yaml",
        device: String(nnDevice || "cuda").trim() || "cuda",
        simulations: clamp(Number(nnSimulations) || 3000, 50, 20000),
      });
      await refreshNnStatus();
      setEngine("nn");
      setInfo(`Loaded NN: ${basename(model_path)}`);
    } catch (e) {
      setErr(String(e?.message || e));
    } finally {
      setBusy((b) => ({ ...b, nn: false }));
    }
  }, [api, nnPath, nnConfig, nnDevice, nnSimulations, refreshNnStatus, setEngine, setInfo, setErr]);

  const unloadNnCheckpoint = useCallback(async () => {
    setBusy((b) => ({ ...b, nn: true }));
    try {
      await api.post("/nn/unload", {});
      await refreshNnStatus();
      setEngine("mcts");
      setInfo("NN unloaded.");
    } catch (e) {
      setErr(String(e?.message || e));
    } finally {
      setBusy((b) => ({ ...b, nn: false }));
    }
  }, [api, refreshNnStatus, setEngine, setInfo, setErr]);

  const buySelect = useCallback(
    (off, pieceId) => {
      if (!legal?.buy_groups) return;
      const key = `${off}-${pieceId}`;
      setSelectedKey(key);
      setPreviewAction(null);
      setSolveBreakdown(null);
    },
    [legal?.buy_groups],
  );

  const toggleCircleBuild = useCallback(() => {
    setBuildMode((v) => !v);
    setManualCircle(true);
    setInfo("Circle builder toggled. When manual circle is ON, server won't randomize circle on /new.");
  }, [setBuildMode, setManualCircle, setInfo]);

  const addPieceToCircle = useCallback(
    (pid) => {
      if (!gameState) return;
      setGameState((prev) => {
        const circle2 = [...(prev.circle || []), pid];
        return { ...prev, circle: circle2 };
      });
      setJsonDirty(true);
    },
    [gameState],
  );

  const removePieceFromCircle = useCallback(
    (pid) => {
      if (!gameState) return;
      setGameState((prev) => {
        const circle2 = (prev.circle || []).filter((x) => x !== pid);
        const neutral2 = clamp(prev.neutral ?? 0, 0, Math.max(0, circle2.length - 1));
        return { ...prev, circle: circle2, neutral: neutral2 };
      });
      setJsonDirty(true);
    },
    [gameState],
  );

  const setNeutralIdx = useCallback(
    (idx) => {
      if (!gameState) return;
      setGameState((prev) => {
        const n = prev.circle?.length || 0;
        return { ...prev, neutral: clamp(idx, 0, Math.max(0, n - 1)) };
      });
      setJsonDirty(true);
    },
    [gameState],
  );

  const loadJson = useCallback(() => {
    try {
      const parsed = JSON.parse(jsonText);
      setGameState(parsed);
      setBoards([boardFromRows(parsed.players?.[0]?.board), boardFromRows(parsed.players?.[1]?.board)]);
      setLegal(null);
      setLastMovePlacementsP0([]);
      setLastMovePlacementsP1([]);
      setSelectedKey(null);
      setSelectedOrient(null);
      setPreviewAction(null);
      setHintTopLeftSet(null);
      setSolveBreakdown(null);
      setMoveLog([]);
      setInfo("JSON loaded into client state. Click 'Refresh Legal' to validate with server.");
      setJsonDirty(false);
    } catch (e) {
      setErr(String(e?.message || e));
    }
  }, [jsonText, setInfo, setErr]);

  const syncJsonFromState = useCallback(() => {
    if (!stateForApi) return;
    setJsonText(JSON.stringify(stateForApi, null, 2));
    setJsonDirty(false);
  }, [stateForApi]);

  // Clear current mover's last-move placements when their turn starts (avoids AI "consideration" artifacts)
  useEffect(() => {
    if (toMove === 0) setLastMovePlacementsP0([]);
    if (toMove === 1) setLastMovePlacementsP1([]);
  }, [toMove]);

  // Auto-fetch legal when it becomes human's turn (only after game has started)
  useEffect(() => {
    if (!gameStarted || !gameState || editMode) return;
    if (gameMode === "human_vs_ai" && toMove === humanPlayer) fetchLegal();
  }, [gameStarted, gameState, editMode, gameMode, toMove, humanPlayer, fetchLegal]);

  // Human vs AI: auto-play AI turns only (never for human; only after game started)
  const autoMoveLock = useRef(false);
  useEffect(() => {
    if (!gameStarted || !gameState || editMode || busy.thinking || busy.loading) return;
    if (gameMode !== "human_vs_ai") return;
    if (toMove === humanPlayer) return;
    if (isTerminal) return;
    if (autoMoveLock.current) return;
    autoMoveLock.current = true;
    const t = setTimeout(async () => {
      try {
        const leg = await fetchLegal();
        if (leg?.terminal) {
          setLegal(leg);
          return;
        }
        if (leg?.pass_allowed && (!leg?.buy_groups || leg.buy_groups.length === 0)) {
          await applyAction(toMove, { type: "pass" }, "PASS");
          setInfo("AI passed (no piece to place).");
          return;
        }
        await solveAndPlay();
      } finally {
        autoMoveLock.current = false;
      }
    }, 0);
    return () => clearTimeout(t);
  }, [gameStarted, gameState, editMode, busy.thinking, gameMode, toMove, humanPlayer, isTerminal, fetchLegal, applyAction, solveAndPlay, setInfo]);

  // AI vs AI autoplay (only after game started)
  useEffect(() => {
    if (!gameStarted || !gameState || editMode || busy.thinking) return;
    if (gameMode !== "ai_vs_ai" || !autoPlayAivsAi || isTerminal) return;
    const t = setTimeout(() => solveAndPlay(), clamp(Number(autoPlayDelayMs) || 350, 60, 5000));
    return () => clearTimeout(t);
  }, [gameStarted, gameState, editMode, busy.thinking, gameMode, autoPlayAivsAi, autoPlayDelayMs, solveAndPlay, isTerminal]);

  const refreshPiecesCb = refreshPieces;
  const refreshNnStatusCb = refreshNnStatus;

  // Startup
  useEffect(() => {
    (async () => {
      try {
        setBusy((b) => ({ ...b, loading: true }));
        await Promise.all([refreshPiecesCb(), refreshNnStatusCb()]);
        await newGame();
      } catch (e) {
        setErr(String(e?.message || e));
      } finally {
        setBusy((b) => ({ ...b, loading: false }));
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const onKey = (ev) => {
      if (ev.target && (ev.target.tagName === "INPUT" || ev.target.tagName === "TEXTAREA" || ev.target.isContentEditable)) return;
      if (ev.key === " ") {
        ev.preventDefault();
        solveAndPlay();
      }
      if (ev.key.toLowerCase() === "l") fetchLegal();
      if (ev.key.toLowerCase() === "n") newGame();
      if (ev.key.toLowerCase() === "e") setEditMode((v) => !v);
      if (ev.key.toLowerCase() === "p") passTurn();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [solveAndPlay, fetchLegal, newGame, passTurn, setEditMode]);

  const overlayCellsP0 = useMemo(() => (previewAction?.cells && isHumanTurn && humanPlayer === 0 ? previewAction.cells : null), [previewAction, isHumanTurn, humanPlayer]);
  const overlayCellsP1 = useMemo(() => (previewAction?.cells && isHumanTurn && humanPlayer === 1 ? previewAction.cells : null), [previewAction, isHumanTurn, humanPlayer]);
  const lastMovePlacementsForP0 = useMemo(
    () => (Array.isArray(lastMovePlacementsP0) && lastMovePlacementsP0.length ? lastMovePlacementsP0.slice(-4) : null),
    [lastMovePlacementsP0]
  );
  const lastMovePlacementsForP1 = useMemo(
    () => (Array.isArray(lastMovePlacementsP1) && lastMovePlacementsP1.length ? lastMovePlacementsP1.slice(-4) : null),
    [lastMovePlacementsP1]
  );
  const patchHighlightSet = useMemo(() => {
    if (legal?.mode !== "patch" || !Array.isArray(legal?.actions)) return null;
    const s = new Set();
    for (const a of legal.actions) s.add(a.idx);
    return s;
  }, [legal]);

  const bannerStyle = useMemo(() => {
    const kind = banner.kind;
    const isErr = kind === "error";
    const isWarn = kind === "warn";
    return {
      padding: "10px 12px",
      borderRadius: 12,
      border: `1px solid ${isErr ? "#ef4444" : isWarn ? "#f59e0b" : COLORS.border}`,
      background: isErr ? COLORS.dangerBg : isWarn ? "#2a200b" : COLORS.panel,
      color: isErr ? COLORS.danger : isWarn ? "#fde68a" : COLORS.muted,
    };
  }, [banner.kind]);

  const p0 = gameState?.players?.[0] || { position: 0, buttons: 0, income: 0, board: [] };
  const p1 = gameState?.players?.[1] || { position: 0, buttons: 0, income: 0, board: [] };

  function computeFinalScore(board, buttons, bonusOwner, playerIdx) {
    const empty = board.flat().filter((v) => v === 0).length;
    return Number(buttons) - 2 * empty + (bonusOwner === playerIdx ? 7 : 0);
  }
  const finalScores = useMemo(() => {
    if (!isTerminal || !gameState) return null;
    const bonusOwner = gameState.bonus_owner ?? -1;
    return {
      p0: computeFinalScore(boards[0], p0.buttons, bonusOwner, 0),
      p1: computeFinalScore(boards[1], p1.buttons, bonusOwner, 1),
    };
  }, [isTerminal, gameState, boards, p0.buttons, p1.buttons]);

  return (
    <div style={{ minHeight: "100vh", background: COLORS.bg, color: COLORS.text, fontFamily: "Inter, ui-sans-serif, system-ui, -apple-system" }}>
      <div style={{ maxWidth: 1400, margin: "0 auto", padding: 16 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10, marginBottom: 12, flexWrap: "wrap" }}>
          <div>
            <div style={{ fontSize: 22, fontWeight: 900 }}>Patchwork Pro GUI</div>
            <div style={{ fontSize: 12, color: COLORS.muted }}>Hotkeys: Space=AI, L=legal, N=new, E=edit, P=pass</div>
          </div>
          <div style={{ flex: 1, display: "flex", justifyContent: "center", alignItems: "center", gap: 10, minWidth: 0 }}>
            <span style={{ fontSize: 11, color: COLORS.muted }}>Last turn order:</span>
            {PLACEMENT_COLORS.map((hex, i) => (
              <span key={i} style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                <span style={{ width: 14, height: 14, borderRadius: 4, background: hex, border: `1px solid ${COLORS.border}` }} title={`Placement ${i + 1}`} />
                <span style={{ fontSize: 11, color: COLORS.muted }}>{i + 1}</span>
              </span>
            ))}
          </div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <Button onClick={() => setShowSetup((v) => !v)}>{showSetup ? "Hide Setup" : "Show Setup"}</Button>
            <Button tone="primary" onClick={newGame} disabled={busy.loading}>
              New Game
            </Button>
            <Button tone="ok" onClick={() => setGameStarted(true)} disabled={busy.loading || gameStarted}>
              Start Game
            </Button>
          </div>
        </div>

        <div style={{ ...bannerStyle, marginBottom: 12, display: "flex", justifyContent: "space-between", gap: 10, flexWrap: "wrap" }}>
          <div>{banner.text}</div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <Pill>
              Turn <b style={{ color: COLORS.text }}>P{toMove}</b>
              {isTerminal ? " (terminal)" : isHumanTurn ? " (human)" : " (AI)"}
            </Pill>
            {solveBreakdown?.best?.winProb != null && !isTerminal && (
              <Pill>
                P0 win <b style={{ color: COLORS.text }}>{(100 * ((solveBreakdown.to_move ?? toMove) === 0 ? solveBreakdown.best.winProb : 1 - solveBreakdown.best.winProb)).toFixed(1)}%</b>
              </Pill>
            )}
            {solveBreakdown?.total_sims != null && !isTerminal && (
              <Pill title="MCTS simulations (paths explored from root)">
                sims <b style={{ color: COLORS.text }}>{solveBreakdown.total_sims}</b>
              </Pill>
            )}
            {solveBreakdown?.search_time != null && !isTerminal && (
              <Pill title="Time spent on last search">
                <b style={{ color: COLORS.text }}>{solveBreakdown.search_time}s</b>
              </Pill>
            )}
            {legal?.mode === "patch" && <Pill tone="warn">PATCH MODE ×{legal?.pending_patches ?? "?"}</Pill>}
            {finalScores != null && (
              <Pill tone="ok">
                Game Over — P0: <b style={{ color: COLORS.text }}>{finalScores.p0}</b> pts | P1: <b style={{ color: COLORS.text }}>{finalScores.p1}</b> pts
              </Pill>
            )}
            {!gameStarted && gameState && (
              <Pill tone="warn">Press Start Game to begin</Pill>
            )}
          </div>
        </div>

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 12 }}>
          <div style={{ minWidth: 180 }}>
            <Field label="Game Mode">
              <Select value={gameMode} onChange={(e) => setGameMode(e.target.value)}>
                <option value="human_vs_ai">Human vs AI</option>
                <option value="ai_vs_ai">AI vs AI</option>
              </Select>
            </Field>
          </div>
          <div style={{ minWidth: 180 }}>
            <Field label="Human Player">
              <Select value={humanPlayer} onChange={(e) => setHumanPlayer(parseInt(e.target.value, 10))} disabled={gameMode !== "human_vs_ai"}>
                <option value={0}>You = P0</option>
                <option value={1}>You = P1</option>
              </Select>
            </Field>
          </div>
          <div style={{ minWidth: 180 }}>
            <Field label="Engine">
              <Select value={engine} onChange={(e) => setEngine(e.target.value)}>
                <option value="nn" disabled={!nnAvailable}>
                  AlphaZero NN
                </option>
                <option value="mcts">Pure MCTS</option>
              </Select>
            </Field>
          </div>

          <div style={{ display: "flex", alignItems: "end", gap: 8, flexWrap: "wrap" }}>
            <Button onClick={fetchLegal} disabled={busy.legal || !stateForApi || !gameStarted}>
              {busy.legal ? "Loading…" : "Refresh Legal"}
            </Button>
            <Button tone="primary" onClick={solveAndPlay} disabled={!gameStarted || busy.thinking || !stateForApi || isTerminal}>
              {busy.thinking ? "Thinking…" : "AI Move"}
            </Button>
            <Button tone="warn" onClick={passTurn} disabled={!gameStarted || !isHumanTurn || !legal?.pass_allowed || busy.thinking || legal?.mode === "patch"}>
              Pass
            </Button>
            <label
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                padding: "8px 10px",
                borderRadius: 10,
                border: `1px solid ${COLORS.border}`,
                background: COLORS.panel2,
                color: COLORS.muted,
                fontSize: 12,
              }}
            >
              <input type="checkbox" checked={editMode} onChange={(e) => setEditMode(e.target.checked)} />
              Edit Mode
            </label>
            {gameMode === "ai_vs_ai" && (
              <label
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  padding: "8px 10px",
                  borderRadius: 10,
                  border: `1px solid ${COLORS.border}`,
                  background: COLORS.panel2,
                  color: COLORS.muted,
                  fontSize: 12,
                }}
              >
                <input type="checkbox" checked={autoPlayAivsAi} onChange={(e) => setAutoPlayAivsAi(e.target.checked)} />
                Autoplay
                <input
                  type="number"
                  value={autoPlayDelayMs}
                  onChange={(e) => setAutoPlayDelayMs(clamp(parseInt(e.target.value || "350", 10), 60, 5000))}
                  style={{ width: 90, marginLeft: 6, padding: "6px 8px", borderRadius: 8, background: COLORS.panel, border: `1px solid ${COLORS.border}`, color: COLORS.text }}
                />
                <span style={{ color: COLORS.dim }}>ms</span>
              </label>
            )}
          </div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 360px 1fr", gap: 12, alignItems: "start" }}>
          <BoardGrid
            title={`Player 0 ${humanPlayer === 0 && gameMode === "human_vs_ai" ? "(You)" : ""}`}
            subtitle={`pos ${p0.position} • buttons ${p0.buttons} • income +${p0.income}`}
            active={toMove === 0}
            accent={COLORS.blue}
            board={boards[0]}
            pieceIdBoard={pieceIdBoardP0}
            overlayCells={overlayCellsP0}
            highlightIdxSet={humanPlayer === 0 && isHumanTurn && legal?.mode === "patch" ? patchHighlightSet : null}
            hintTopLeftSet={humanPlayer === 0 && isHumanTurn && legal?.mode === "normal" ? hintTopLeftSet : null}
            lastMovePlacements={toMove === 1 ? lastMovePlacementsForP0 : null}
            editMode={editMode}
            onHoverCell={(r, c) => onHover(0, r, c)}
            onLeave={onLeaveBoard}
            onClickCell={(r, c, idx) => onClickCell(0, r, c, idx)}
          />

          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <Section title="MARKET / CIRCLE" right={<Button onClick={() => setAtlasOpen((v) => !v)}>{atlasOpen ? "Hide atlas" : "Show atlas"}</Button>}>
              <div style={{ display: "grid", gap: 10 }}>
                <div>
                  <div style={{ fontSize: 12, color: COLORS.muted, marginBottom: 6 }}>Next 3 (click to select)</div>
                  <div style={{ display: "grid", gap: 8 }}>
                    {next3.map(({ off, piece_id }) => {
                      const piece = piecesById[piece_id];
                      const isSelected = selectedKey === `${off}-${piece_id}`;
                      const canBuy = legal?.mode === "normal" && legal?.buy_groups?.some((g) => g.offset === off && g.piece_id === piece_id);
                      const unaffordable = legal?.mode === "normal" && !canBuy;
                      const bg = isSelected ? "rgba(251,191,36,0.18)" : canBuy ? "rgba(34,197,94,0.08)" : unaffordable ? "rgba(239,68,68,0.08)" : "rgba(59,130,246,0.10)";
                      const bd = isSelected ? `2px solid ${COLORS.amber}` : canBuy ? `2px solid ${COLORS.ok}` : unaffordable ? `2px solid ${COLORS.danger}` : `1px solid ${COLORS.border}`;
                      const localCells = piece?.shape ? computeLocalCellsFromShape(piece.shape) : [];
                      return (
                        <button
                          key={`${off}-${piece_id}`}
                          type="button"
                          onClick={() => buySelect(off, piece_id)}
                          disabled={legal?.mode === "patch" || !legal?.buy_groups?.some((g) => g.offset === off && g.piece_id === piece_id)}
                          style={{
                            padding: 10,
                            borderRadius: 12,
                            border: bd,
                            background: bg,
                            color: COLORS.text,
                            cursor: "pointer",
                            opacity: legal?.mode === "patch" ? 0.5 : 1,
                            textAlign: "left",
                          }}
                        >
                          <div style={{ display: "flex", justifyContent: "space-between", gap: 10 }}>
                            <div style={{ display: "grid", gap: 2 }}>
                              <div style={{ fontWeight: 900 }}>offset {off} • #{piece_id}</div>
                              <div style={{ fontSize: 12, color: COLORS.muted }}>{piece ? `${piece.cost_buttons} buttons • ${piece.cost_time} time • +${piece.income} income` : "…"}</div>
                              {isSelected && selectedOrient != null && (
                                <div style={{ fontSize: 12, color: COLORS.amber, fontWeight: 800 }}>Orientation: {selectedOrient}</div>
                              )}
                            </div>
                            {piece?.shape && <MiniShape cells={localCells} cellPx={10} />}
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>

                {selectedGroup && placementIndex && (
                  <div style={{ display: "grid", gap: 8 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
                      <Pill tone="ok">
                        placements <b style={{ color: COLORS.text }}>{selectedGroup.placements?.length ?? 0}</b>
                      </Pill>
                      <div>
                        <div style={{ fontSize: 12, color: COLORS.muted, marginBottom: 6 }}>Pick orientation (click shape)</div>
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "flex-start" }}>
                          {placementIndex.orientList.map((oi) => {
                            const placeMap = placementIndex.byOrient.get(oi);
                            const firstPlace = placeMap && placeMap.size ? Array.from(placeMap.values())[0] : null;
                            const relCells = firstPlace?.cells ? cellsToRelative(firstPlace.cells) : [];
                            const isSelected = selectedOrient === oi;
                            return (
                              <button
                                key={oi}
                                type="button"
                                onClick={() => setSelectedOrient(oi)}
                                style={{
                                  padding: 6,
                                  borderRadius: 10,
                                  border: `2px solid ${isSelected ? COLORS.amber : COLORS.border}`,
                                  background: isSelected ? "rgba(251,191,36,0.15)" : COLORS.panel2,
                                  cursor: "pointer",
                                  display: "inline-flex",
                                  alignItems: "center",
                                  justifyContent: "center",
                                }}
                                title={`Orientation ${oi}`}
                              >
                                {relCells.length > 0 ? <MiniShape cells={relCells} cellPx={12} /> : <span style={{ fontSize: 11, color: COLORS.dim }}>{oi}</span>}
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    </div>
                    <div style={{ fontSize: 12, color: COLORS.muted }}>
                      Tip: hover a square to preview; click a highlighted square (blue dashed) to place.
                    </div>
                  </div>
                )}

                <Section
                  title="Circle builder"
                  right={
                    <div style={{ display: "flex", gap: 8 }}>
                      <Button onClick={() => setManualCircle((v) => !v)}>{manualCircle ? "Manual circle: ON" : "Manual circle: OFF"}</Button>
                      <Button onClick={toggleCircleBuild}>{buildMode ? "Close builder" : "Open builder"}</Button>
                    </div>
                  }
                >
                  <MarketRing piecesById={piecesById} circle={circle} neutral={neutral} selectedPieceId={null} onSelectPiece={() => {}} compact />
                  {buildMode && (
                    <div style={{ marginTop: 10, display: "grid", gap: 10 }}>
                      <Field label="Neutral index">
                        <Input type="number" value={neutral} min={0} max={Math.max(0, circle.length - 1)} onChange={(e) => setNeutralIdx(parseInt(e.target.value || "0", 10))} />
                      </Field>
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                        <div>
                          <div style={{ fontSize: 12, color: COLORS.muted, marginBottom: 6 }}>Add piece (greyed = already in circle)</div>
                          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 6, maxHeight: 260, overflow: "auto" }}>
                            {pieces.map((p) => {
                              const inCircle = circle.includes(p.id);
                              const cells = p.shape ? computeLocalCellsFromShape(p.shape) : [];
                              return (
                                <button
                                  key={`add-${p.id}`}
                                  type="button"
                                  onClick={() => !inCircle && addPieceToCircle(p.id)}
                                  disabled={inCircle}
                                  style={{
                                    padding: 8,
                                    borderRadius: 10,
                                    border: `1px solid ${COLORS.border}`,
                                    background: inCircle ? COLORS.panel : COLORS.panel2,
                                    color: inCircle ? COLORS.dim : COLORS.text,
                                    cursor: inCircle ? "not-allowed" : "pointer",
                                    textAlign: "left",
                                    display: "flex",
                                    justifyContent: "space-between",
                                    gap: 8,
                                    alignItems: "center",
                                    opacity: inCircle ? 0.55 : 1,
                                  }}
                                  title={inCircle ? "Already in circle — remove it below to add again" : "Add to circle"}
                                >
                                  <div>
                                    <div style={{ fontWeight: 900, fontSize: 12 }}>#{p.id}</div>
                                    <div style={{ fontSize: 11, color: COLORS.muted }}>{p.cost_buttons}b/{p.cost_time}t</div>
                                  </div>
                                  {cells.length > 0 && <MiniShape cells={cells} cellPx={8} />}
                                </button>
                              );
                            })}
                          </div>
                        </div>
                        <div>
                          <div style={{ fontSize: 12, color: COLORS.muted, marginBottom: 6 }}>Remove piece (click)</div>
                          <div style={{ display: "grid", gap: 6, maxHeight: 260, overflow: "auto" }}>
                            {circle.map((pid, i) => {
                              const p = piecesById[pid];
                              const cells = p?.shape ? computeLocalCellsFromShape(p.shape) : [];
                              return (
                                <button
                                  key={`rm-${i}-${pid}`}
                                  type="button"
                                  onClick={() => removePieceFromCircle(pid)}
                                  style={{ padding: 8, borderRadius: 10, border: `1px solid ${COLORS.border}`, background: COLORS.panel2, color: COLORS.text, cursor: "pointer", textAlign: "left", display: "flex", justifyContent: "space-between", gap: 8, alignItems: "center" }}
                                  title="Remove from circle"
                                >
                                  <div>
                                    <b>#{pid}</b> <span style={{ color: COLORS.dim }}>idx {i}</span>
                                  </div>
                                  {cells.length > 0 && <MiniShape cells={cells} cellPx={8} />}
                                </button>
                              );
                            })}
                          </div>
                        </div>
                      </div>
                      <Button onClick={syncJsonFromState}>Sync JSON from current state</Button>
                    </div>
                  )}
                </Section>
              </div>
            </Section>

            <Section title="ENGINE SETTINGS">
              <div style={{ display: "grid", gap: 10 }}>
                <Field label="NN simulations (for /solve_nn)">
                  <Input type="number" value={nnSimulations} onChange={(e) => setNnSimulations(parseInt(e.target.value || "3000", 10))} min={50} max={20000} step={10} />
                </Field>
                <Field label="Pure MCTS iterations (for /solve)">
                  <Input type="number" value={mctsIterations} onChange={(e) => setMctsIterations(parseInt(e.target.value || "50000", 10))} min={500} max={2000000} step={500} />
                </Field>
                <Field label="Pure MCTS workers">
                  <Input type="number" value={mctsWorkers} onChange={(e) => setMctsWorkers(parseInt(e.target.value || "4", 10))} min={1} max={64} step={1} />
                </Field>
              </div>
            </Section>

            <Section title="SEARCH BREAKDOWN">
              <TopMoves solveResult={solveBreakdown} />
            </Section>

            {atlasOpen && (
              <Section title="PIECE ATLAS">
                <div style={{ display: "grid", gap: 10 }}>
                  <div style={{ fontSize: 12, color: COLORS.muted }}>
                    All pieces (click to add to circle). Greyed = already in circle — remove from circle to add again.
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 8, maxHeight: 420, overflow: "auto", paddingRight: 4 }}>
                    {pieces.map((p) => {
                      const inCircle = circle.includes(p.id);
                      const cells = p.shape ? computeLocalCellsFromShape(p.shape) : [];
                      return (
                        <button
                          key={`atlas-${p.id}`}
                          type="button"
                          onClick={() => {
                            if (inCircle) return;
                            if (!buildMode) {
                              setBuildMode(true);
                              setManualCircle(true);
                            }
                            addPieceToCircle(p.id);
                            setInfo(`Added piece #${p.id} to circle.`);
                          }}
                          disabled={inCircle}
                          style={{
                            padding: 10,
                            borderRadius: 12,
                            border: `1px solid ${COLORS.border}`,
                            background: inCircle ? COLORS.panel : "rgba(59,130,246,0.08)",
                            color: inCircle ? COLORS.dim : COLORS.text,
                            cursor: inCircle ? "not-allowed" : "pointer",
                            textAlign: "left",
                            display: "flex",
                            justifyContent: "space-between",
                            gap: 10,
                            opacity: inCircle ? 0.55 : 1,
                          }}
                          title={inCircle ? "Already in circle — remove it in Circle builder to add again" : "Add to circle"}
                        >
                          <div style={{ display: "grid", gap: 2 }}>
                            <div style={{ fontWeight: 900 }}>#{p.id}</div>
                            <div style={{ fontSize: 12, color: COLORS.muted }}>
                              {p.cost_buttons}b • {p.cost_time}t • +{p.income}
                            </div>
                          </div>
                          <MiniShape cells={cells} cellPx={10} />
                        </button>
                      );
                    })}
                  </div>
                </div>
              </Section>
            )}

            <Section title="MODEL MANAGER">
              <div style={{ display: "grid", gap: 10 }}>
                <Field label="API URL">
                  <Input value={apiBase} onChange={(e) => setApiBase(e.target.value)} />
                </Field>
                <Pill tone={nnAvailable ? "ok" : "warn"}>
                  {nnAvailable ? "NN loaded" : "NN not loaded"}
                  {nnAvailable && nnStatus?.model_path && <span style={{ marginLeft: 8, color: COLORS.text }}>{basename(nnStatus.model_path)}</span>}
                </Pill>
                <Field label="Checkpoint path">
                  <Input value={nnPath} onChange={(e) => setNnPath(e.target.value)} placeholder="/abs/path/to/model.pt" />
                </Field>
                {nnStatus?.checkpoints?.length > 0 && (
                  <Field label="Discovered checkpoints">
                    <Select value={nnPath} onChange={(e) => setNnPath(e.target.value)}>
                      <option value="">Select…</option>
                      {nnStatus.checkpoints.map((p) => (
                        <option key={p} value={p}>
                          {basename(p)}
                        </option>
                      ))}
                    </Select>
                  </Field>
                )}
                <Field label="Config path">
                  <Input value={nnConfig} onChange={(e) => setNnConfig(e.target.value)} />
                </Field>
                <Field label="Device">
                  <Select value={nnDevice} onChange={(e) => setNnDevice(e.target.value)}>
                    <option value="cuda">cuda</option>
                    <option value="cpu">cpu</option>
                  </Select>
                </Field>
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  <Button tone="ok" onClick={loadNnCheckpoint} disabled={busy.nn}>
                    Load
                  </Button>
                  <Button onClick={unloadNnCheckpoint} disabled={busy.nn}>
                    Unload
                  </Button>
                  <Button onClick={refreshNnStatus} disabled={busy.nn}>
                    Refresh
                  </Button>
                </div>
              </div>
            </Section>

            <Section title="JSON (import/export)" right={jsonDirty ? <Pill tone="warn">Unsaved edits</Pill> : <Pill tone="ok">Synced</Pill>}>
              <textarea
                value={jsonText}
                onChange={(e) => {
                  setJsonText(e.target.value);
                  setJsonDirty(true);
                }}
                style={{
                  width: "100%",
                  minHeight: 220,
                  padding: 10,
                  borderRadius: 12,
                  background: COLORS.panel2,
                  border: `1px solid ${COLORS.border}`,
                  color: COLORS.text,
                  fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
                  fontSize: 12,
                }}
              />
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 10 }}>
                <Button
                  onClick={() => {
                    navigator.clipboard.writeText(jsonText);
                    setInfo("JSON copied.");
                  }}
                >
                  Copy
                </Button>
                <Button onClick={loadJson} tone="primary">
                  Load JSON
                </Button>
                <Button
                  onClick={() => {
                    setJsonText("{}");
                    setJsonDirty(true);
                  }}
                  title="Clear editor"
                >
                  Clear
                </Button>
                <Button onClick={syncJsonFromState} title="Overwrite editor from current client state">
                  Sync from state
                </Button>
              </div>
              <div style={{ marginTop: 8, fontSize: 12, color: COLORS.dim }}>
                Tip: Load JSON updates client state; click “Refresh Legal” to validate server-side legality.
              </div>
            </Section>
          </div>

          <BoardGrid
            title={`Player 1 ${humanPlayer === 1 && gameMode === "human_vs_ai" ? "(You)" : ""}`}
            subtitle={`pos ${p1.position} • buttons ${p1.buttons} • income +${p1.income}`}
            active={toMove === 1}
            accent={COLORS.purple}
            board={boards[1]}
            pieceIdBoard={pieceIdBoardP1}
            overlayCells={overlayCellsP1}
            highlightIdxSet={humanPlayer === 1 && isHumanTurn && legal?.mode === "patch" ? patchHighlightSet : null}
            hintTopLeftSet={humanPlayer === 1 && isHumanTurn && legal?.mode === "normal" ? hintTopLeftSet : null}
            lastMovePlacements={toMove === 0 ? lastMovePlacementsForP1 : null}
            editMode={editMode}
            onHoverCell={(r, c) => onHover(1, r, c)}
            onLeave={onLeaveBoard}
            onClickCell={(r, c, idx) => onClickCell(1, r, c, idx)}
          />
        </div>

        <div style={{ marginTop: 12, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          <Section title="STATE SUMMARY">
            {!gameState ? (
              <div style={{ color: COLORS.dim, fontSize: 12 }}>No state loaded.</div>
            ) : (
              <div style={{ display: "grid", gap: 8, fontSize: 12, color: COLORS.muted }}>
                <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                  <Pill>
                    circle <b style={{ color: COLORS.text }}>{circle.length}</b>
                  </Pill>
                  <Pill>
                    neutral <b style={{ color: COLORS.text }}>{neutral}</b>
                  </Pill>
                  <Pill>
                    mode <b style={{ color: COLORS.text }}>{legal?.mode || "?"}</b>
                  </Pill>
                  {legal?.pass_allowed && <Pill tone="ok">pass allowed</Pill>}
                </div>
                <div style={{ color: COLORS.dim }}>
                  This UI always applies server-produced legal actions (buys + patch placements). Edit Mode only affects the visual boards and JSON.
                </div>
              </div>
            )}
          </Section>

          <Section title="MOVE LOG">
            {moveLog.length === 0 ? (
              <div style={{ color: COLORS.dim, fontSize: 12 }}>No moves yet.</div>
            ) : (
              <div style={{ display: "grid", gap: 6, maxHeight: 240, overflow: "auto", paddingRight: 4 }}>
                {moveLog.map((m, i) => (
                  <div key={i} style={{ display: "flex", gap: 10, fontSize: 12, color: COLORS.muted }}>
                    <div style={{ width: 48, color: m.player === 0 ? COLORS.blue : COLORS.purple }}>
                      P{m.player}
                    </div>
                    <div style={{ flex: 1 }}>{m.text}</div>
                  </div>
                ))}
              </div>
            )}
          </Section>
        </div>
      </div>
    </div>
  );
}