import React, { useEffect, useMemo, useState } from "react";
import {
  Copy, Download, Info, Zap, RotateCcw, Eye, Check,
  Undo2, Trash2, RefreshCw, ChevronDown, ChevronUp
} from "lucide-react";

const emptyBoard = () => Array.from({ length: 9 }, () => Array(9).fill(0));

// Convert API board rows (array of 9 strings of "."/"1"/"2") into the GUI's numeric 9x9 grid.
const boardFromRows = (rows) => {
  if (!Array.isArray(rows) || rows.length !== 9) return emptyBoard();
  const b = emptyBoard();
  for (let r = 0; r < 9; r++) {
    const line = rows[r] || "";
    for (let c = 0; c < 9; c++) {
      const ch = line[c];
      b[r][c] = ch === "1" ? 1 : ch === "2" ? 2 : 0;
    }
  }
  return b;
};

function MiniShape({ shape, cell = 12 }) {
  if (!shape) return null;
  const h = shape.length, w = shape[0].length;
  return (
    <div style={{ display: "grid", gridTemplateColumns: `repeat(${w}, ${cell}px)`, gap: 2 }}>
      {shape.flatMap((row, r) =>
        row.map((v, c) => {
          const bg = v === 0 ? "#0b1220" : v === 1 ? "#3b82f6" : "#22c55e";
          return (
            <div
              key={`${r}-${c}`}
              style={{
                width: cell,
                height: cell,
                background: bg,
                border: "1px solid #334155",
                borderRadius: 3
              }}
            />
          );
        })
      )}
    </div>
  );
}

// Build a mini grid from an action.cells list (absolute r/c) by normalizing to top-left
function miniFromCells(cells) {
  if (!cells || cells.length === 0) return null;
  let minR = 999, minC = 999, maxR = -999, maxC = -999;
  for (const c of cells) {
    minR = Math.min(minR, c.r);
    minC = Math.min(minC, c.c);
    maxR = Math.max(maxR, c.r);
    maxC = Math.max(maxC, c.c);
  }
  const h = maxR - minR + 1;
  const w = maxC - minC + 1;
  const g = Array.from({ length: h }, () => Array(w).fill(0));
  for (const c of cells) g[c.r - minR][c.c - minC] = c.val;
  return g;
}

export default function PatchworkSolver() {
  // ===== game state =====
  const [p0Pos, setP0Pos] = useState(0);
  const [p0Buttons, setP0Buttons] = useState(5);
  const [p0Board, setP0Board] = useState(emptyBoard());

  const [p1Pos, setP1Pos] = useState(0);
  const [p1Buttons, setP1Buttons] = useState(5);
  const [p1Board, setP1Board] = useState(emptyBoard());

  const [circleArr, setCircleArr] = useState([]);
  const [neutral, setNeutral] = useState(0);

  // ===== rules bookkeeping (must be preserved across server calls) =====
  const [bonusOwner, setBonusOwner] = useState(-1);
  const [pendingPatches, setPendingPatches] = useState(0);
  const [pendingOwner, setPendingOwner] = useState(-1);
  const [tiePlayer, setTiePlayer] = useState(0);

  // ===== solver =====
  const [iterations, setIterations] = useState(100000);
  const [workers, setWorkers] = useState(4);
  const [autoSolveAfterOpponent, setAutoSolveAfterOpponent] = useState(true);

  // ===== catalog =====
  const [pieces, setPieces] = useState([]);
  const [piecesById, setPiecesById] = useState({});

  // ===== turn control =====
  const [serverToMove, setServerToMove] = useState(null); // 0/1
  const [loadingSolve, setLoadingSolve] = useState(false);
  const [loadingLegal, setLoadingLegal] = useState(false);

  const [solveResult, setSolveResult] = useState(null);   // AI result (only when p0 to move)
  const [legalResult, setLegalResult] = useState(null);   // legal moves for current player

  // ===== preview =====
  const [preview, setPreview] = useState(null);
  const [previewPlayer, setPreviewPlayer] = useState(0);

  // ===== opponent selection =====
  const [opSelectedGroupKey, setOpSelectedGroupKey] = useState(null); // "offset-pieceid"
  const [opOrient, setOpOrient] = useState(null); // integer orient index

  // ===== patch mode =====
  const [patchMode, setPatchMode] = useState(false);
  const [patchLegalSet, setPatchLegalSet] = useState(new Set());

  // ===== circle builder =====
  const [buildMode, setBuildMode] = useState(false);

  // ===== piece atlas collapse =====
  const [atlasOpen, setAtlasOpen] = useState(true);

  // ===== derived =====
  const circleString = useMemo(() => circleArr.join(","), [circleArr]);

  const toMoveGuess = p0Pos < p1Pos ? 0 : p1Pos < p0Pos ? 1 : 0;
  const toMove = (serverToMove ?? toMoveGuess);
  const toMoveButtons = toMove === 0 ? p0Buttons : p1Buttons;

  const turnLabel = toMove === 0 ? "Player 0 (AI) turn" : "Player 1 (Opponent) turn";

  const next3 = useMemo(() => {
    const n = circleArr.length;
    if (!n) return [];
    const picks = [];
    for (let off = 1; off <= Math.min(3, n); off++) {
      const idx = (neutral + off) % n;
      const id = circleArr[idx];
      picks.push({ off, idx, id });
    }
    return picks;
  }, [circleArr, neutral]);


  const loadNewGame = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/new");
      const txt = await res.text();
      if (!res.ok) throw new Error(txt);
      const data = JSON.parse(txt);
      const st = data.state || data;

      setP0Pos(st.players?.[0]?.position ?? 0);
      setP0Buttons(st.players?.[0]?.buttons ?? 5);
      setP0Board(boardFromRows(st.players?.[0]?.board));

      setP1Pos(st.players?.[1]?.position ?? 0);
      setP1Buttons(st.players?.[1]?.buttons ?? 5);
      setP1Board(boardFromRows(st.players?.[1]?.board));

      setCircleArr(st.circle || []);
      setNeutral(st.neutral || 0);

      setBonusOwner(st.bonus_owner ?? -1);
      setPendingPatches(st.pending_patches ?? 0);
      setPendingOwner(st.pending_owner ?? -1);
      setTiePlayer(st.tie_player ?? 0);

      setServerToMove(data.to_move ?? null);

      // UI bookkeeping
      setSolveResult(null);
      setLegalResult(null);
      setPreview(null);
      setPreviewPlayer(0);
      setPatchMode(false);
      setPatchLegalSet(new Set());
      setOpSelectedGroupKey(null);
      setOpOrient(null);
    } catch (e) {
      // If the /new endpoint isn't reachable, fall back to the original deterministic setup.
      setNeutral(0);
    }
  };

  useEffect(() => {
    fetch("http://127.0.0.1:8000/pieces")
      .then(r => r.json())
      .then(async data => {
        const list = data.pieces || [];
        const map = {};
        for (const p of list) map[p.id] = p;
        setPieces(list);
        setPiecesById(map);
        // Initialize a fresh randomized game (random circle, neutral pawn after id 32)
        await loadNewGame();
      })
      .catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const generateStateObject = () => {
    const toStr = (b) => b.map(row => row.map(c => (c === 0 ? "." : c.toString())).join(""));
    return {
      players: [
        { position: p0Pos, buttons: p0Buttons, board: toStr(p0Board) },
        { position: p1Pos, buttons: p1Buttons, board: toStr(p1Board) }
      ],
      circle: circleArr,
      neutral,
      bonus_owner: bonusOwner,
      pending_patches: pendingPatches,
      pending_owner: pendingOwner,
      tie_player: tiePlayer
    };
  };

  const generateJSON = () => JSON.stringify(generateStateObject(), null, 2);

  const copyJSON = async () => {
    await navigator.clipboard.writeText(generateJSON());
    alert("JSON copied!");
  };

  const downloadJSON = () => {
    const blob = new Blob([generateJSON()], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "patchwork_state.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  const resetToStart = async () => {
    setP0Pos(0); setP0Buttons(5); setP0Board(emptyBoard());
    setP1Pos(0); setP1Buttons(5); setP1Board(emptyBoard());
    setBonusOwner(-1);
    setPendingPatches(0);
    setPendingOwner(-1);
    setTiePlayer(0);
    setServerToMove(null);
    setSolveResult(null);
    setLegalResult(null);
    setPreview(null);
    setPreviewPlayer(0);
    setPatchMode(false);
    setPatchLegalSet(new Set());
    setOpSelectedGroupKey(null);
    setOpOrient(null);
    await loadNewGame();
  };

  // ===== preview overlay map =====
  const overlayMap = useMemo(() => {
    const m = new Map();
    if (!preview?.cells) return m;
    for (const cell of preview.cells) m.set(`${cell.r},${cell.c}`, cell.val);
    return m;
  }, [preview]);

  // ===== neutral helper =====
  const setNeutralBeforePieceId = (pieceId) => {
    const n = circleArr.length;
    if (!n) return;
    const idx = circleArr.indexOf(pieceId);
    if (idx === -1) return;
    const neu = (idx - 1 + n) % n;
    setNeutral(neu);
  };

  // ===== manual board cycling (only for editing; disabled during opponent placement) =====
  const updateCellCycle = (player, row, col) => {
    const update = (board, setBoard) => {
      const nextVal = (board[row][col] + 1) % 3;
      const newBoard = board.map((r, rIdx) =>
        rIdx === row ? r.map((c, cIdx) => (cIdx === col ? nextVal : c)) : r
      );
      setBoard(newBoard);
    };
    if (player === 0) update(p0Board, setP0Board);
    else update(p1Board, setP1Board);
  };

  const clearBoard = (player) => {
    if (player === 0) setP0Board(emptyBoard());
    else setP1Board(emptyBoard());
  };

  // ===== Apply action =====
  const applyAction = async (actionObj, mover) => {
    try {
      const state = generateStateObject();
      const res = await fetch("http://127.0.0.1:8000/apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ state, action: actionObj })
      });
      const txt = await res.text();
      if (!res.ok) throw new Error(txt);
      const data = JSON.parse(txt);

      // Stamp cells on local board so income squares are preserved visually.
      const clone = (b) => b.map(r => r.slice());
      let newP0 = clone(p0Board);
      let newP1 = clone(p1Board);

      if (actionObj?.cells) {
        const target = mover === 0 ? newP0 : newP1;
        for (const cell of actionObj.cells) target[cell.r][cell.c] = cell.val;
      }

      setP0Pos(data.state.players[0].position);
      setP0Buttons(data.state.players[0].buttons);
      setP1Pos(data.state.players[1].position);
      setP1Buttons(data.state.players[1].buttons);
      setCircleArr(data.state.circle || []);
      setNeutral(data.state.neutral || 0);
      setBonusOwner(data.state.bonus_owner ?? -1);
      setPendingPatches(data.state.pending_patches ?? 0);
      setPendingOwner(data.state.pending_owner ?? -1);
      setTiePlayer(data.state.tie_player ?? 0);

      setP0Board(newP0);
      setP1Board(newP1);

      setPreview(null);
      setSolveResult(null);
      setLegalResult(null);
      setOpSelectedGroupKey(null);
      setOpOrient(null);

      setServerToMove(data.to_move);

      if (data.to_move === 1) {
        await fetchLegal();
      } else if (data.to_move === 0 && autoSolveAfterOpponent) {
        await runSolve();
      }
    } catch (e) {
      alert("Apply error: " + e.message);
    }
  };

  // ===== legal moves =====
  const fetchLegal = async () => {
    setLoadingLegal(true);
    setSolveResult(null);
    setLegalResult(null);
    setPreview(null);
    setPreviewPlayer(0);
    setPatchMode(false);
    setPatchLegalSet(new Set());
    setOpSelectedGroupKey(null);
    setOpOrient(null);

    try {
      const state = generateStateObject();
      const res = await fetch("http://127.0.0.1:8000/legal", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ state })
      });
      const txt = await res.text();
      if (!res.ok) throw new Error(txt);
      const data = JSON.parse(txt);

      setLegalResult(data);
      setServerToMove(data.to_move);

      if (data.mode === "patch" && data.to_move === 1) {
        const s = new Set((data.actions || []).map(a => a.idx));
        setPatchMode(true);
        setPatchLegalSet(s);
      }
    } catch (e) {
      alert("Legal error: " + e.message);
    } finally {
      setLoadingLegal(false);
    }
  };

  // ===== solve (ONLY for Player 0) =====
  const runSolve = async () => {
    setLoadingSolve(true);
    setSolveResult(null);
    setPreview(null);
    setPreviewPlayer(0);

    try {
      const state = generateStateObject();
      const res = await fetch("http://127.0.0.1:8000/solve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          state,
          iterations,
          workers,
          seed: 0,
          exploration: 1.4,
          only_player0: true
        })
      });
      const txt = await res.text();
      if (!res.ok) throw new Error(txt);
      const data = JSON.parse(txt);

      setServerToMove(data.to_move ?? serverToMove);

      if (data.needs_opponent_move) {
        await fetchLegal();
        return;
      }

      setSolveResult(data);

      if (!data.terminal && data.best?.action) {
        setPreview(data.best.action);
        setPreviewPlayer(0);
      }
    } catch (e) {
      alert("Solve error: " + e.message);
    } finally {
      setLoadingSolve(false);
    }
  };

  // ===== circle builder =====
  const clearCircle = () => setCircleArr([]);
  const undoCircle = () => setCircleArr(prev => prev.slice(0, -1));
  const clickPiece = (id) => {
    if (!buildMode) return;
    setCircleArr(prev => (prev.includes(id) ? prev : [...prev, id]));
  };
  const setCircleFromText = (txt) => {
    const ids = txt.split(",").map(x => parseInt(x.trim(), 10)).filter(x => !Number.isNaN(x));
    setCircleArr(ids);
  };

  // ===== Opponent placement: hover+click from LEGAL placements only =====
  const selectedGroup = useMemo(() => {
    if (!legalResult?.buy_groups) return null;
    if (!opSelectedGroupKey) return null;
    const [offStr, pidStr] = opSelectedGroupKey.split("-");
    const off = parseInt(offStr, 10);
    const pid = parseInt(pidStr, 10);
    return legalResult.buy_groups.find(g => g.offset === off && g.piece_id === pid) || null;
  }, [legalResult, opSelectedGroupKey]);

  const opPlacementIndex = useMemo(() => {
    // Map (orient,top,left) -> action for fast lookup
    if (!selectedGroup?.placements) return null;
    const map = new Map();
    const orients = new Set();
    const exampleByOrient = new Map();
    for (const a of selectedGroup.placements) {
      map.set(`${a.orient}:${a.top}:${a.left}`, a);
      orients.add(a.orient);
      if (!exampleByOrient.has(a.orient)) exampleByOrient.set(a.orient, a);
    }
    const orientList = Array.from(orients).sort((a, b) => a - b);
    return { map, orientList, exampleByOrient };
  }, [selectedGroup]);

  useEffect(() => {
    if (!opPlacementIndex) return;
    // auto pick first orientation when selecting a piece
    if (opOrient === null || !opPlacementIndex.orientList.includes(opOrient)) {
      setOpOrient(opPlacementIndex.orientList[0] ?? null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [opPlacementIndex, selectedGroup]);

  const opponentHoverAt = (r, c) => {
    if (toMove !== 1) return;
    if (patchMode) return;
    if (!selectedGroup || !opPlacementIndex) return;
    if (opOrient === null) return;

    const a = opPlacementIndex.map.get(`${opOrient}:${r}:${c}`);
    if (a) {
      setPreview(a);
      setPreviewPlayer(1);
    } else {
      // no legal placement at that top-left for that orientation
      // keep last preview, but you can also clear if you want:
      // setPreview(null);
    }
  };

  const opponentClickAt = async (r, c) => {
    if (toMove !== 1) return;
    if (patchMode) return;
    if (!selectedGroup || !opPlacementIndex) return;
    if (opOrient === null) return;

    const a = opPlacementIndex.map.get(`${opOrient}:${r}:${c}`);
    if (!a) return; // illegal square
    await applyAction(a, 1);
  };

  // ===== Board UI component =====
  const BoardEditor = ({ board, player, label }) => (
    <div style={{ border: "1px solid #334155", borderRadius: 12, padding: 12, background: "#0b1220" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
        <div style={{ fontWeight: 900, color: "white" }}>{label}</div>
        <button onClick={() => clearBoard(player)} style={{ padding: "6px 10px", borderRadius: 10 }}>Clear</button>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(9, 30px)", gap: 3, background: "#111827", padding: 8, borderRadius: 10 }}>
        {board.map((row, r) =>
          row.map((cell, c) => {
            const idx = r * 9 + c;
            const key = `${r},${c}`;

            const isOverlay = (preview && player === previewPlayer && overlayMap.has(key));
            const overlayVal = isOverlay ? overlayMap.get(key) : null;

            const patchLegal = patchMode && player === 1 && patchLegalSet.has(idx);

            const opponentPlacementActive = (
              player === 1 &&
              toMove === 1 &&
              !patchMode &&
              selectedGroup &&
              opPlacementIndex &&
              opOrient !== null
            );

            const baseBg = cell === 0 ? "#0f172a" : cell === 1 ? "#2563eb" : "#16a34a";
            const bg = isOverlay ? "#f59e0b33" : patchLegal ? "#22c55e22" : baseBg;

            const labelTxt = isOverlay ? (overlayVal === 2 ? "⚫" : "■") : (cell === 2 ? "⚫" : cell === 1 ? "■" : "");

            const onClick = async () => {
              if (patchMode && player === 1) {
                if (!patchLegalSet.has(idx)) return;
                await applyAction({ type: "patch", idx, row: r, col: c, cells: [{ r, c, val: 1 }] }, 1);
                return;
              }

              if (opponentPlacementActive) {
                await opponentClickAt(r, c);
                return;
              }

              updateCellCycle(player, r, c);
            };

            const onEnter = () => {
              if (opponentPlacementActive) opponentHoverAt(r, c);
            };

            return (
              <button
                key={key}
                onMouseEnter={onEnter}
                onClick={onClick}
                style={{
                  width: 30, height: 30,
                  background: bg,
                  border: isOverlay ? "3px solid #f59e0b" : patchLegal ? "2px solid #22c55e" : "1px solid #334155",
                  borderRadius: 6,
                  fontWeight: 900,
                  color: (cell === 0 && !isOverlay) ? "#cbd5e1" : "white",
                  cursor: (patchLegal || opponentPlacementActive) ? "pointer" : "default"
                }}
                title={`Row ${r+1}, Col ${c+1}${isOverlay ? " (preview)" : ""}${patchLegal ? " (legal patch)" : ""}`}
              >
                {labelTxt}
              </button>
            );
          })
        )}
      </div>

      <div style={{ fontSize: 12, color: "#cbd5e1", marginTop: 10 }}>
        {player === 1 && toMove === 1 && selectedGroup && opOrient !== null && !patchMode ? (
          <span style={{ color: "#f59e0b", fontWeight: 900 }}>
            Opponent placement mode: hover to preview, click to place.
          </span>
        ) : (
          <span>Click cells to cycle: Empty → Filled → Income.</span>
        )}
        {preview && player === previewPlayer ? (
          <span style={{ color: "#f59e0b", fontWeight: 900 }}> Preview highlighted in ORANGE.</span>
        ) : null}
        {patchMode && player === 1 ? (
          <span style={{ color: "#22c55e", fontWeight: 900 }}> Opponent must place patch: click GREEN squares.</span>
        ) : null}
      </div>
    </div>
  );

  // ===== AI move card =====
  const MoveCard = ({ entry, isBest }) => {
    const action = entry.action;
    const t = action?.type;

    let subtitle = "";
    let shape = null;

    if (t === "buy") {
      const p = piecesById[action.piece_id];
      shape = p?.shape;
      subtitle = `Piece ${action.piece_id} | cost ${action.cost_buttons}b, ${action.cost_time}t | income ${p?.income ?? "?"}`;
    } else if (t === "pass") {
      subtitle = "Pass / advance";
    } else if (t === "patch") {
      subtitle = `Place patch at row ${action.row + 1}, col ${action.col + 1}`;
    }

    return (
      <div style={{ border: `2px solid ${isBest ? "#22c55e" : "#334155"}`, borderRadius: 12, padding: 12, background: "#0b1220", color: "white" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start", gap: 12 }}>
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: 900 }}>{isBest ? "✅ AI Best Move (Player 0)" : "Candidate"}</div>
            <div style={{ fontFamily: "monospace", marginTop: 6 }}>{entry.pretty}</div>
            <div style={{ color: "#cbd5e1", fontSize: 13, marginTop: 6 }}>{subtitle}</div>
            <div style={{ display: "flex", gap: 12, marginTop: 8, flexWrap: "wrap" }}>
              <div>Win: <b>{(entry.winProb * 100).toFixed(1)}%</b></div>
              <div>Score: <b>{Number(entry.scoreDiff).toFixed(2)}</b></div>
              <div>
                Visits: <b>{entry.visits}</b>{" "}
                <span style={{ color:"#94a3b8" }}>/ {Number(solveResult?.total_sims ?? 0).toLocaleString()} sims</span>
              </div>
            </div>
          </div>
          <div style={{ width: 120 }}>
            {shape ? <MiniShape shape={shape} /> : null}
          </div>
        </div>

        <div style={{ display: "flex", gap: 10, marginTop: 10, flexWrap: "wrap" }}>
          <button
            onClick={() => { setPreview(action); setPreviewPlayer(0); }}
            style={{ display: "flex", alignItems: "center", gap: 6, padding: "8px 10px", borderRadius: 10 }}
          >
            <Eye size={16} /> Preview
          </button>

          <button
            onClick={() => applyAction(action, 0)}
            style={{ display: "flex", alignItems: "center", gap: 6, padding: "8px 10px", borderRadius: 10, background: "#22c55e", color: "#052e16", fontWeight: 900, border: "none" }}
          >
            <Check size={16} /> Apply (Player 0)
          </button>
        </div>
      </div>
    );
  };

  // ===== Piece atlas collapse button text =====
  const atlasButton = atlasOpen ? (
    <button
      onClick={() => setAtlasOpen(false)}
      style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 10px", borderRadius: 10 }}
    >
      <ChevronUp size={18} /> Hide Piece Atlas
    </button>
  ) : (
    <button
      onClick={() => setAtlasOpen(true)}
      style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 10px", borderRadius: 10 }}
    >
      <ChevronDown size={18} /> Show Piece Atlas
    </button>
  );

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: 18, color: "white" }}>
      <h1 style={{ margin: 0, fontSize: 30 }}>🧩 Patchwork Human vs AI</h1>
      <div style={{ color: "#cbd5e1", marginTop: 6, marginBottom: 12 }}>
        AI only recommends for <b>Player 0</b>. On Player 1 turns, you select a legal move and place it by hovering + clicking the board.
      </div>

      {/* Boards */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 14 }}>
        <div style={{ background: "#0b1220", border: "2px solid #1d4ed8", borderRadius: 12, padding: 12 }}>
          <h2 style={{ margin: "0 0 10px 0" }}>Player 0 (AI)</h2>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 10 }}>
            <label>
              Position
              <input type="number" value={p0Pos}
                onChange={(e)=>setP0Pos(Math.max(0, Math.min(53, parseInt(e.target.value||"0",10))))}
                style={{ width: "100%", padding: 8, borderRadius: 8, border: "1px solid #334155", background: "#111827", color: "white" }} />
            </label>
            <label>
              Buttons
              <input type="number" value={p0Buttons}
                onChange={(e)=>setP0Buttons(Math.max(0, parseInt(e.target.value||"0",10)))}
                style={{ width: "100%", padding: 8, borderRadius: 8, border: "1px solid #334155", background: "#111827", color: "white" }} />
            </label>
          </div>
          <BoardEditor board={p0Board} player={0} label="Board (9×9)" />
        </div>

        <div style={{ background: "#0b1220", border: "2px solid #7c3aed", borderRadius: 12, padding: 12 }}>
          <h2 style={{ margin: "0 0 10px 0" }}>Player 1 (Opponent)</h2>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 10 }}>
            <label>
              Position
              <input type="number" value={p1Pos}
                onChange={(e)=>setP1Pos(Math.max(0, Math.min(53, parseInt(e.target.value||"0",10))))}
                style={{ width: "100%", padding: 8, borderRadius: 8, border: "1px solid #334155", background: "#111827", color: "white" }} />
            </label>
            <label>
              Buttons
              <input type="number" value={p1Buttons}
                onChange={(e)=>setP1Buttons(Math.max(0, parseInt(e.target.value||"0",10)))}
                style={{ width: "100%", padding: 8, borderRadius: 8, border: "1px solid #334155", background: "#111827", color: "white" }} />
            </label>
          </div>
          <BoardEditor board={p1Board} player={1} label="Board (9×9)" />
        </div>
      </div>

      {/* Game State */}
      <div style={{ background: "#0b1220", border: "2px solid #334155", borderRadius: 12, padding: 14, marginBottom: 14 }}>
        <h2 style={{ margin: "0 0 10px 0" }}>Game State</h2>

        <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 10 }}>
          <label>
            Circle (IDs in exact table order)
            <input value={circleString} onChange={(e)=>setCircleFromText(e.target.value)}
              style={{ width: "100%", padding: 8, borderRadius: 8, border: "1px solid #334155", background: "#111827", color: "white", fontFamily: "monospace" }} />
          </label>

          <label>
            Neutral (pawn)
            <input
              type="number"
              value={neutral}
              onChange={(e)=>setNeutral(Math.max(0, parseInt(e.target.value||"0",10)))}
              style={{ width: "100%", padding: 8, borderRadius: 8, border: "1px solid #334155", background: "#111827", color: "white" }}
            />
            <div style={{ marginTop: 8, fontSize: 12, color: "#cbd5e1" }}>
              OR choose which piece the pawn is <b>before</b>:
            </div>
            <select
              value={circleArr.length ? circleArr[(neutral + 1) % circleArr.length] : ""}
              onChange={(e) => setNeutralBeforePieceId(parseInt(e.target.value, 10))}
              style={{ marginTop: 6, width: "100%", padding: 8, borderRadius: 8, border: "1px solid #334155", background: "#111827", color: "white" }}
            >
              {circleArr.length === 0 ? <option value="">(circle empty)</option> : null}
              {circleArr.map((id) => (
                <option key={id} value={id}>ID {id}</option>
              ))}
            </select>
          </label>
        </div>

        {/* Next 3 pieces panel */}
        <div style={{ marginTop: 12, padding: 12, borderRadius: 12, border: "1px solid #334155", background: "#111827" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12, flexWrap: "wrap" }}>
            <div style={{ fontWeight: 900 }}>Next 3 pieces you are allowed to buy</div>
            <div style={{ color: "#94a3b8", fontSize: 13 }}>
              Neutral = {neutral} • Turn: {turnLabel} • Buttons: {toMoveButtons}
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 10, marginTop: 10 }}>
            {next3.map(({ off, idx, id }) => {
              const p = piecesById[id];
              const affordable = p ? (toMoveButtons >= p.cost_buttons) : true;

              return (
                <div key={`${off}-${idx}-${id}`} style={{
                  border: "1px solid #334155",
                  borderRadius: 12,
                  padding: 10,
                  background: affordable ? "#0b1220" : "#1f2937"
                }}>
                  <div style={{ display: "flex", justifyContent: "space-between", gap: 10 }}>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontWeight: 900 }}>Offset {off} (idx {idx})</div>
                      <div style={{ fontFamily: "monospace", marginTop: 4 }}>Piece ID {id}</div>
                      {p ? (
                        <div style={{ color: "#cbd5e1", fontSize: 13, marginTop: 6 }}>
                          cost {p.cost_buttons}b • {p.cost_time}t • income {p.income} • size {p.size}
                        </div>
                      ) : (
                        <div style={{ color: "#cbd5e1", fontSize: 13, marginTop: 6 }}>(piece data loading...)</div>
                      )}
                      <div style={{ marginTop: 8, color: affordable ? "#22c55e" : "#f59e0b", fontWeight: 900 }}>
                        {affordable ? "Affordable" : "Not affordable"}
                      </div>
                    </div>

                    <div style={{ width: 120 }}>
                      {p?.shape ? <MiniShape shape={p.shape} /> : null}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* solver controls */}
        <div style={{ marginTop: 12, display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
          <label>
            Iterations
            <input type="number" value={iterations}
              onChange={(e)=>setIterations(Math.max(200, parseInt(e.target.value||"100000",10)))}
              style={{ width: "100%", padding: 8, borderRadius: 8, border: "1px solid #334155", background: "#111827", color: "white" }} />
          </label>
          <label>
            Workers
            <input type="number" value={workers}
              onChange={(e)=>setWorkers(Math.max(1, Math.min(64, parseInt(e.target.value||"1",10))))}
              style={{ width: "100%", padding: 8, borderRadius: 8, border: "1px solid #334155", background: "#111827", color: "white" }} />
          </label>
          <label style={{ display: "flex", alignItems: "end", gap: 10 }}>
            <input type="checkbox" checked={autoSolveAfterOpponent} onChange={(e)=>setAutoSolveAfterOpponent(e.target.checked)} />
            <span style={{ color: "#cbd5e1" }}>Auto-solve after opponent plays</span>
          </label>
        </div>
      </div>

      {/* Piece Atlas (collapsible) */}
      <div style={{ background: "#0b1220", border: "2px solid #334155", borderRadius: 12, padding: 14, marginBottom: 14 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <h2 style={{ margin: 0 }}>Piece Atlas</h2>
          {atlasButton}
        </div>

        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 10 }}>
          <button
            onClick={() => setBuildMode(v => !v)}
            style={{
              padding: "8px 10px",
              borderRadius: 10,
              background: buildMode ? "#f59e0b" : "#111827",
              border: "1px solid #334155",
              color: buildMode ? "#111827" : "white",
              fontWeight: 900
            }}
          >
            {buildMode ? "✅ Build Circle Mode: ON" : "Build Circle Mode: OFF"}
          </button>

          <button onClick={undoCircle} style={{ display: "flex", gap: 6, alignItems: "center", padding: "8px 10px", borderRadius: 10 }}>
            <Undo2 size={16} /> Undo last
          </button>
          <button onClick={clearCircle} style={{ display: "flex", gap: 6, alignItems: "center", padding: "8px 10px", borderRadius: 10 }}>
            <Trash2 size={16} /> Clear circle
          </button>
        </div>

        {atlasOpen && (
          <div style={{ marginTop: 12, display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(170px, 1fr))", gap: 10, maxHeight: 340, overflow: "auto", paddingRight: 6 }}>
            {pieces.map(p => {
              const selected = circleArr.includes(p.id);
              return (
                <button
                  key={p.id}
                  onClick={() => clickPiece(p.id)}
                  style={{
                    textAlign: "left",
                    padding: 10,
                    borderRadius: 12,
                    border: selected ? "2px solid #22c55e" : "1px solid #334155",
                    background: selected ? "#052e16" : "#111827",
                    color: "white"
                  }}
                  title={buildMode ? "Click to append to circle order" : "Enable Build Circle Mode to click-add"}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start", gap: 10 }}>
                    <div>
                      <div style={{ fontWeight: 900 }}>ID {p.id}</div>
                      <div style={{ fontSize: 12, color: "#cbd5e1" }}>
                        cost {p.cost_buttons}b • {p.cost_time}t • income {p.income} • size {p.size}
                      </div>
                    </div>
                    <MiniShape shape={p.shape} cell={12} />
                  </div>
                  <div style={{ marginTop: 8, fontSize: 12, color: buildMode ? "#f59e0b" : "#94a3b8" }}>
                    {buildMode ? "Click to append to circle order" : "Enable Build Circle Mode to click"}
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Turn Control */}
      <div style={{ background: "#0b1220", border: "2px solid #334155", borderRadius: 12, padding: 14, marginBottom: 14 }}>
        <h2 style={{ margin: "0 0 10px 0" }}>Turn Control</h2>

        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          <button
            onClick={runSolve}
            disabled={loadingSolve}
            style={{
              display: "flex", alignItems: "center", gap: 8,
              padding: "10px 14px",
              background: loadingSolve ? "#9ca3af" : "#2563eb",
              color: "white",
              border: "none",
              borderRadius: 10,
              fontWeight: 900
            }}
          >
            <Zap size={18} />
            {loadingSolve ? "Analyzing..." : "Solve (Player 0 only)"}
          </button>

          <button
            onClick={fetchLegal}
            disabled={loadingLegal}
            style={{
              display: "flex", alignItems: "center", gap: 8,
              padding: "10px 14px",
              background: loadingLegal ? "#9ca3af" : "#111827",
              color: "white",
              border: "1px solid #334155",
              borderRadius: 10,
              fontWeight: 900
            }}
          >
            <RefreshCw size={18} />
            {loadingLegal ? "Loading legal..." : "Show legal moves (current player)"}
          </button>

          <button onClick={copyJSON} style={{ display: "flex", alignItems: "center", gap: 8, padding: "10px 14px", borderRadius: 10 }}>
            <Copy size={18} /> Copy JSON
          </button>

          <button onClick={downloadJSON} style={{ display: "flex", alignItems: "center", gap: 8, padding: "10px 14px", borderRadius: 10 }}>
            <Download size={18} /> Download JSON
          </button>

          <button onClick={resetToStart} style={{ display: "flex", alignItems: "center", gap: 8, padding: "10px 14px", borderRadius: 10 }}>
            <RotateCcw size={18} /> Reset
          </button>
        </div>

        <div style={{ marginTop: 10, padding: 10, background: "#111827", border: "1px solid #334155", borderRadius: 10 }}>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <Info size={18} />
            <div style={{ fontSize: 13, color: "#cbd5e1" }}>
              Current: <b>{turnLabel}</b>. If it’s Opponent turn, click “Show legal moves”, select a piece, pick orientation, then hover+click the opponent board.
            </div>
          </div>
        </div>
      </div>

      {/* Opponent Turn Panel */}
      {toMove === 1 && (
        <div style={{ background: "#0b1220", border: "2px solid #7c3aed", borderRadius: 12, padding: 14, marginBottom: 14 }}>
          <h2 style={{ margin: "0 0 10px 0" }}>Opponent Turn (Player 1)</h2>

          {!legalResult && (
            <div style={{ color: "#cbd5e1" }}>
              Click <b>“Show legal moves”</b> above to load what Player 1 is allowed to do right now.
            </div>
          )}

          {legalResult?.mode === "patch" && (
            <div style={{ color: "#cbd5e1" }}>
              Opponent must place <b>{legalResult.pending_patches}</b> patch(es).
              <div style={{ marginTop: 6 }}>
                ✅ Click a <b>green</b> square on the Player 1 board to place a patch.
              </div>
            </div>
          )}

          {legalResult?.mode === "normal" && (
            <>
              <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 10 }}>
                {legalResult.pass_allowed && (
                  <button
                    onClick={() => applyAction({ type: "pass" }, 1)}
                    style={{ padding: "10px 14px", borderRadius: 10, background: "#f59e0b", border: "none", fontWeight: 900 }}
                  >
                    Opponent PASS
                  </button>
                )}
              </div>

              <div style={{ fontWeight: 900, marginBottom: 8 }}>Opponent BUY options</div>

              {(legalResult.buy_groups || []).length === 0 ? (
                <div style={{ color: "#cbd5e1" }}>No affordable buy options (opponent likely must pass).</div>
              ) : (
                <>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 10 }}>
                    {(legalResult.buy_groups || []).map(g => {
                      const p = piecesById[g.piece_id];
                      const key = `${g.offset}-${g.piece_id}`;
                      const selected = opSelectedGroupKey === key;

                      return (
                        <button
                          key={key}
                          onClick={() => {
                            setOpSelectedGroupKey(key);
                            setPreview(null);
                            setPreviewPlayer(1);
                          }}
                          style={{
                            textAlign: "left",
                            padding: 12,
                            borderRadius: 12,
                            border: selected ? "2px solid #f59e0b" : "1px solid #334155",
                            background: selected ? "#111827" : "#0b1220",
                            color: "white"
                          }}
                        >
                          <div style={{ display: "flex", justifyContent: "space-between", gap: 10 }}>
                            <div>
                              <div style={{ fontWeight: 900 }}>Offset {g.offset} • Piece {g.piece_id}</div>
                              <div style={{ color: "#cbd5e1", fontSize: 13, marginTop: 6 }}>
                                cost {g.cost_buttons}b • {g.cost_time}t • placements {g.placements?.length || 0}
                              </div>
                            </div>
                            {p?.shape ? <MiniShape shape={p.shape} cell={12} /> : null}
                          </div>
                        </button>
                      );
                    })}
                  </div>

                  {selectedGroup && opPlacementIndex && (
                    <div style={{ marginTop: 12, padding: 12, borderRadius: 12, border: "1px solid #334155", background: "#111827" }}>
                      <div style={{ fontWeight: 900 }}>
                        Step 2: choose orientation (then place by hovering/clicking the board)
                      </div>

                      <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 10 }}>
                        {opPlacementIndex.orientList.map(oi => {
                          const ex = opPlacementIndex.exampleByOrient.get(oi);
                          const mini = miniFromCells(ex?.cells);
                          const selected = opOrient === oi;
                          return (
                            <button
                              key={oi}
                              onClick={() => {
                                setOpOrient(oi);
                                if (ex) {
                                  setPreview(ex);
                                  setPreviewPlayer(1);
                                }
                              }}
                              style={{
                                padding: 10,
                                borderRadius: 12,
                                border: selected ? "2px solid #f59e0b" : "1px solid #334155",
                                background: selected ? "#0b1220" : "#111827",
                                color: "white",
                                textAlign: "left",
                                minWidth: 140
                              }}
                            >
                              <div style={{ fontWeight: 900 }}>Orient {oi}</div>
                              <div style={{ marginTop: 6 }}>
                                {mini ? <MiniShape shape={mini} cell={10} /> : null}
                              </div>
                            </button>
                          );
                        })}
                      </div>

                      <div style={{ marginTop: 10, color: "#cbd5e1" }}>
                        ✅ Now hover the <b>Player 1 board</b> to preview. Click to place.  
                        (If nothing highlights, that top-left square isn’t legal for that orientation.)
                      </div>
                    </div>
                  )}
                </>
              )}
            </>
          )}
        </div>
      )}

      {/* AI Suggestions (only on Player 0 turn) */}
      {toMove === 0 && solveResult && !solveResult.error && !solveResult.terminal && (
        <div style={{ display: "grid", gap: 12 }}>
          <MoveCard entry={solveResult.best} isBest={true} />
          <div style={{ fontWeight: 900 }}>Top candidates</div>
          <div style={{ display: "grid", gap: 10 }}>
            {(solveResult.top || []).map((x, i) => <MoveCard key={i} entry={x} isBest={false} />)}
          </div>
        </div>
      )}
    </div>
  );
}
