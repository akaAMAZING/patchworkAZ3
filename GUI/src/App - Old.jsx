import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

const API = "http://127.0.0.1:8000";
const colors = {
  bg: "#0a0e1a",
  panel: "#111827",
  panelBorder: "#1e293b",
  text: "#f1f5f9",
  textMuted: "#94a3b8",
  textDim: "#64748b",
  p0: "#3b82f6",
  p1: "#a855f7",
  warn: "#f59e0b",
  ok: "#22c55e",
};

const emptyBoard = () => Array.from({ length: 9 }, () => Array(9).fill(0));
const boardFromRows = (rows) => {
  if (!Array.isArray(rows) || rows.length !== 9) return emptyBoard();
  return rows.map((line) => Array.from({ length: 9 }, (_, c) => ((line || "")[c] === "2" ? 2 : (line || "")[c] === "1" ? 1 : 0)));
};
const boardToRows = (b) => b.map((row) => row.map((v) => (v === 0 ? "." : String(v))).join(""));
const cloneBoard = (b) => b.map((r) => r.slice());
const countCoverage = (b) => b.reduce((acc, row) => acc + row.filter((x) => x > 0).length, 0);

function MiniShape({ shape, cell = 11 }) {
  if (!shape) return null;
  return (
    <div style={{ display: "inline-grid", gridTemplateColumns: `repeat(${shape[0].length}, ${cell}px)`, gap: 1 }}>
      {shape.flatMap((row, r) =>
        row.map((v, c) => (
          <div key={`${r}-${c}`} style={{ width: cell, height: cell, borderRadius: 2, border: "1px solid #334155", background: v === 0 ? "transparent" : v === 2 ? "#16a34a" : "#2563eb" }} />
        )),
      )}
    </div>
  );
}

function Board({ board, title, active, color, preview, previewPlayer, player, patchMode, patchLegalSet, onHover, onClick, onLeave }) {
  const overlay = useMemo(() => {
    const m = new Map();
    if (!preview?.cells || previewPlayer !== player) return m;
    for (const c of preview.cells) m.set(`${c.r},${c.c}`, c.val);
    return m;
  }, [preview, previewPlayer, player]);

  return (
    <div style={{ background: colors.panel, border: `2px solid ${active ? color : colors.panelBorder}`, borderRadius: 12, padding: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
        <div style={{ fontWeight: 700 }}>{title}</div>
        <div style={{ color: colors.textMuted, fontSize: 12 }}>{countCoverage(board)}/81</div>
      </div>
      <div onMouseLeave={onLeave} style={{ display: "grid", gridTemplateColumns: "repeat(9, 34px)", gap: 2, background: "#080c16", borderRadius: 8, padding: 6 }}>
        {board.map((row, r) =>
          row.map((cell, c) => {
            const idx = r * 9 + c;
            const key = `${r},${c}`;
            const over = overlay.has(key);
            const pv = overlay.get(key);
            const patchLegal = patchMode && patchLegalSet.has(idx);
            const bg = over ? "rgba(251,191,36,0.22)" : patchLegal ? "rgba(34,197,94,0.2)" : cell === 2 ? "#16a34a" : cell === 1 ? "#2563eb" : "#0f172a";
            const border = over ? "2px solid #fbbf24" : patchLegal ? "2px solid #22c55e" : "1px solid #1e293b";
            const symbol = over ? (pv === 2 ? "●" : "■") : cell === 2 ? "●" : cell === 1 ? "■" : "";
            return (
              <button
                key={key}
                onMouseEnter={() => onHover?.(r, c, idx)}
                onClick={() => onClick?.(r, c, idx)}
                style={{ width: 34, height: 34, borderRadius: 4, border, background: bg, color: "#dbeafe", fontWeight: 700, cursor: patchLegal ? "pointer" : "default" }}
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

export default function PatchworkSolver() {
  const [p0Pos, setP0Pos] = useState(0);
  const [p0Buttons, setP0Buttons] = useState(5);
  const [p0Income, setP0Income] = useState(0);
  const [p0Board, setP0Board] = useState(emptyBoard);
  const [p1Pos, setP1Pos] = useState(0);
  const [p1Buttons, setP1Buttons] = useState(5);
  const [p1Income, setP1Income] = useState(0);
  const [p1Board, setP1Board] = useState(emptyBoard);
  const [circleArr, setCircleArr] = useState([]);
  const [neutral, setNeutral] = useState(0);
  const [bonusOwner, setBonusOwner] = useState(-1);
  const [pendingPatches, setPendingPatches] = useState(0);
  const [pendingOwner, setPendingOwner] = useState(-1);
  const [tiePlayer, setTiePlayer] = useState(0);
  const [serverToMove, setServerToMove] = useState(null);
  const [pieces, setPieces] = useState([]);
  const [legalResult, setLegalResult] = useState(null);
  const [selectedGroupKey, setSelectedGroupKey] = useState(null);
  const [selectedOrient, setSelectedOrient] = useState(null);
  const [patchMode, setPatchMode] = useState(false);
  const [patchLegalSet, setPatchLegalSet] = useState(new Set());
  const [preview, setPreview] = useState(null);
  const [previewPlayer, setPreviewPlayer] = useState(0);
  const [isThinking, setIsThinking] = useState(false);
  const [status, setStatus] = useState("Ready.");
  const [error, setError] = useState("");
  const [moveLog, setMoveLog] = useState([]);
  const [gameMode, setGameMode] = useState("human_vs_ai");
  const [humanPlayer, setHumanPlayer] = useState(1);
  const [mctsIterations, setMctsIterations] = useState(50000);
  const [mctsWorkers, setMctsWorkers] = useState(4);
  const [nnSimulations, setNnSimulations] = useState(800);
  const [engine, setEngine] = useState("nn");
  const [editMode, setEditMode] = useState(false);
  const [showSetup, setShowSetup] = useState(true);
  const [buildMode, setBuildMode] = useState(false);
  const [atlasOpen, setAtlasOpen] = useState(false);
  const [jsonText, setJsonText] = useState("");
  const [nnAvailable, setNnAvailable] = useState(false);
  const [nnStatus, setNnStatus] = useState(null);
  const [nnPath, setNnPath] = useState("");
  const [nnConfig, setNnConfig] = useState("configs/config_overnight.yaml");
  const [nnDevice, setNnDevice] = useState("cuda");
  const autoMoveLock = useRef(false);

  const piecesById = useMemo(() => Object.fromEntries(pieces.map((p) => [p.id, p])), [pieces]);
  const toMove = serverToMove ?? (p0Pos <= p1Pos ? 0 : 1);
  const isHumanTurn = gameMode === "human_vs_ai" && toMove === humanPlayer;
  const next3 = useMemo(() => {
    const n = circleArr.length;
    const out = [];
    for (let off = 1; off <= Math.min(3, n); off++) out.push({ off, id: circleArr[(neutral + off) % n] });
    return out;
  }, [circleArr, neutral]);

  const stateObj = useMemo(
    () => ({
      players: [
        { position: p0Pos, buttons: p0Buttons, income: p0Income, board: boardToRows(p0Board) },
        { position: p1Pos, buttons: p1Buttons, income: p1Income, board: boardToRows(p1Board) },
      ],
      circle: circleArr,
      neutral,
      bonus_owner: bonusOwner,
      pending_patches: pendingPatches,
      pending_owner: pendingOwner,
      tie_player: tiePlayer,
    }),
    [p0Pos, p0Buttons, p0Income, p0Board, p1Pos, p1Buttons, p1Income, p1Board, circleArr, neutral, bonusOwner, pendingPatches, pendingOwner, tiePlayer],
  );

  const applyState = useCallback((data) => {
    const st = data.state || data;
    setP0Pos(st.players?.[0]?.position ?? 0);
    setP0Buttons(st.players?.[0]?.buttons ?? 0);
    setP0Income(st.players?.[0]?.income ?? 0);
    setP0Board(boardFromRows(st.players?.[0]?.board));
    setP1Pos(st.players?.[1]?.position ?? 0);
    setP1Buttons(st.players?.[1]?.buttons ?? 0);
    setP1Income(st.players?.[1]?.income ?? 0);
    setP1Board(boardFromRows(st.players?.[1]?.board));
    setCircleArr(st.circle || []);
    setNeutral(st.neutral || 0);
    setBonusOwner(st.bonus_owner ?? -1);
    setPendingPatches(st.pending_patches ?? 0);
    setPendingOwner(st.pending_owner ?? -1);
    setTiePlayer(st.tie_player ?? 0);
    if (data.to_move !== undefined) setServerToMove(data.to_move);
    setJsonText(JSON.stringify(st, null, 2));
  }, []);

  const fetchJson = useCallback(async (url, options = {}) => {
    const res = await fetch(url, options);
    const txt = await res.text();
    const parsed = txt ? JSON.parse(txt) : {};
    if (!res.ok) throw new Error(parsed.error || txt || "Request failed");
    return parsed;
  }, []);

  const refreshNn = useCallback(async () => {
    try {
      const data = await fetchJson(`${API}/nn/status`);
      setNnStatus(data);
      setNnAvailable(Boolean(data.nn_loaded));
      if (!data.nn_loaded && engine === "nn") setEngine("mcts");
      if (data.model_path && !nnPath) setNnPath(data.model_path);
    } catch {
      // server may not have endpoint yet; keep UI usable
    }
  }, [fetchJson, engine, nnPath]);

  const loadNewGame = useCallback(async () => {
    setError("");
    setStatus("Loading new game...");
    const [newData, piecesData] = await Promise.all([fetchJson(`${API}/new`), fetchJson(`${API}/pieces`)]);
    setPieces(piecesData.pieces || []);
    applyState(newData);
    setLegalResult(null);
    setSelectedGroupKey(null);
    setSelectedOrient(null);
    setPatchMode(false);
    setPatchLegalSet(new Set());
    setPreview(null);
    setMoveLog([]);
    setStatus("New game ready.");
  }, [fetchJson, applyState]);

  const fetchLegal = useCallback(async () => {
    setError("");
    const data = await fetchJson(`${API}/legal`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ state: stateObj }) });
    setLegalResult(data);
    setPatchMode(data.mode === "patch");
    setPatchLegalSet(new Set((data.actions || []).map((a) => a.idx)));
    setServerToMove(data.to_move);
    setStatus(`Legal moves loaded for Player ${data.to_move}.`);
    return data;
  }, [fetchJson, stateObj]);

  const selectedGroup = useMemo(() => {
    if (!legalResult?.buy_groups || !selectedGroupKey) return null;
    const [off, pid] = selectedGroupKey.split("-").map((x) => parseInt(x, 10));
    return legalResult.buy_groups.find((g) => g.offset === off && g.piece_id === pid) || null;
  }, [legalResult, selectedGroupKey]);

  const placementIndex = useMemo(() => {
    if (!selectedGroup?.placements) return null;
    const map = new Map();
    const orientList = [];
    const seen = new Set();
    const examples = new Map();
    for (const a of selectedGroup.placements) {
      map.set(`${a.orient}:${a.top}:${a.left}`, a);
      if (!seen.has(a.orient)) {
        seen.add(a.orient);
        orientList.push(a.orient);
        examples.set(a.orient, a);
      }
    }
    orientList.sort((a, b) => a - b);
    return { map, orientList, examples };
  }, [selectedGroup]);

  useEffect(() => {
    if (!placementIndex) return;
    if (selectedOrient === null || !placementIndex.orientList.includes(selectedOrient)) setSelectedOrient(placementIndex.orientList[0] ?? null);
  }, [placementIndex, selectedOrient]);

  const applyAction = useCallback(
    async (actionObj, mover) => {
      setError("");
      const data = await fetchJson(`${API}/apply`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ state: stateObj, action: actionObj }) });
      const p0 = cloneBoard(p0Board);
      const p1 = cloneBoard(p1Board);
      if (actionObj?.cells) {
        const target = mover === 0 ? p0 : p1;
        for (const cell of actionObj.cells) target[cell.r][cell.c] = cell.val;
      }
      applyState(data);
      setP0Board(p0);
      setP1Board(p1);
      setLegalResult(null);
      setSelectedGroupKey(null);
      setSelectedOrient(null);
      setPatchMode(false);
      setPatchLegalSet(new Set());
      setPreview(null);
      return data;
    },
    [fetchJson, stateObj, p0Board, p1Board, applyState],
  );

  const solveAndPlay = useCallback(async (playerIdx) => {
    setIsThinking(true);
    setError("");
    try {
      const endpoint = engine === "nn" && nnAvailable ? "/solve_nn" : "/solve";
      const body =
        endpoint === "/solve_nn"
          ? { state: stateObj, simulations: nnSimulations, temperature: 0.0 }
          : { state: stateObj, iterations: mctsIterations, workers: mctsWorkers, seed: 0, exploration: 1.4, only_player0: false };
      const data = await fetchJson(`${API}${endpoint}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      if (data.best?.action) {
        const result = await applyAction(data.best.action, playerIdx);
        setMoveLog((prev) => [...prev, { player: playerIdx, pretty: data.best.pretty || data.best.action.type }]);
        setStatus(`Applied AI move: ${data.best.pretty || data.best.action.type}`);
        return result;
      }
      return null;
    } finally {
      setIsThinking(false);
    }
  }, [engine, nnAvailable, stateObj, nnSimulations, mctsIterations, mctsWorkers, fetchJson, applyAction]);

  const onHoverCell = useCallback((player, r, c) => {
    if (!isHumanTurn || player !== humanPlayer) return;
    if (patchMode || !placementIndex || selectedOrient === null) return;
    const a = placementIndex.map.get(`${selectedOrient}:${r}:${c}`);
    if (a) {
      setPreview(a);
      setPreviewPlayer(player);
    }
  }, [isHumanTurn, humanPlayer, patchMode, placementIndex, selectedOrient]);

  const cycleCell = (player, r, c) => {
    const set = player === 0 ? setP0Board : setP1Board;
    const src = player === 0 ? p0Board : p1Board;
    const out = cloneBoard(src);
    out[r][c] = (out[r][c] + 1) % 3;
    set(out);
    setJsonText(JSON.stringify({ ...stateObj, players: [{ ...stateObj.players[0], board: boardToRows(player === 0 ? out : p0Board) }, { ...stateObj.players[1], board: boardToRows(player === 1 ? out : p1Board) }] }, null, 2));
  };

  const onClickCell = useCallback(
    async (player, r, c, idx) => {
      if (editMode) return cycleCell(player, r, c);
      if (!isHumanTurn || player !== humanPlayer) return;
      if (patchMode) {
        if (!patchLegalSet.has(idx)) return;
        await applyAction({ type: "patch", idx, row: r, col: c, cells: [{ r, c, val: 1 }] }, humanPlayer);
        setMoveLog((prev) => [...prev, { player: humanPlayer, pretty: `PLACE_PATCH (${r + 1},${c + 1})` }]);
        return;
      }
      if (!placementIndex || selectedOrient === null) return;
      const a = placementIndex.map.get(`${selectedOrient}:${r}:${c}`);
      if (!a) return;
      await applyAction(a, humanPlayer);
      setMoveLog((prev) => [...prev, { player: humanPlayer, pretty: `BUY piece ${a.piece_id} at (${r + 1},${c + 1})` }]);
    },
    [editMode, isHumanTurn, humanPlayer, patchMode, patchLegalSet, applyAction, placementIndex, selectedOrient, p0Board, p1Board],
  );

  const passTurn = useCallback(async () => {
    await applyAction({ type: "pass" }, humanPlayer);
    setMoveLog((prev) => [...prev, { player: humanPlayer, pretty: "PASS" }]);
  }, [applyAction, humanPlayer]);

  useEffect(() => {
    (async () => {
      try {
        await Promise.all([loadNewGame(), refreshNn()]);
      } catch (e) {
        setError(String(e.message || e));
      }
    })();
  }, [loadNewGame, refreshNn]);

  useEffect(() => {
    if (gameMode !== "human_vs_ai") return;
    if (isThinking || autoMoveLock.current || editMode) return;
    if (toMove === humanPlayer) {
      fetchLegal().catch((e) => setError(String(e.message || e)));
      return;
    }
    autoMoveLock.current = true;
    const t = setTimeout(async () => {
      try {
        await solveAndPlay(toMove);
      } catch (e) {
        setError(String(e.message || e));
      } finally {
        autoMoveLock.current = false;
      }
    }, 250);
    return () => clearTimeout(t);
  }, [gameMode, toMove, humanPlayer, isThinking, solveAndPlay, fetchLegal, editMode]);

  const loadNnCheckpoint = async () => {
    try {
      if (!nnPath.trim()) throw new Error("Model path is empty.");
      setStatus("Loading checkpoint...");
      await fetchJson(`${API}/nn/load`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_path: nnPath.trim(), config_path: nnConfig.trim(), device: nnDevice, simulations: nnSimulations }),
      });
      await refreshNn();
      setEngine("nn");
      setStatus("Checkpoint loaded.");
    } catch (e) {
      setError(String(e.message || e));
    }
  };

  const unloadNnCheckpoint = async () => {
    try {
      await fetchJson(`${API}/nn/unload`, { method: "POST" });
      await refreshNn();
      setStatus("Checkpoint unloaded.");
    } catch (e) {
      setError(String(e.message || e));
    }
  };

  return (
    <div style={{ maxWidth: 1320, margin: "0 auto", padding: 16, fontFamily: "'Inter','Segoe UI',sans-serif", color: colors.text }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 24 }}>Patchwork Pro GUI</h1>
          <div style={{ color: colors.textMuted, fontSize: 12 }}>QoL-overhauled editor + gameplay + checkpoint control</div>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button onClick={() => setShowSetup((v) => !v)} style={{ padding: "8px 12px", borderRadius: 8 }}>{showSetup ? "Hide Setup" : "Show Setup"}</button>
          <button onClick={() => loadNewGame().catch((e) => setError(String(e.message || e)))} style={{ padding: "8px 12px", borderRadius: 8, background: colors.p0, color: "white", border: "none" }}>New Game</button>
        </div>
      </div>

      <div style={{ marginBottom: 10, padding: "8px 10px", borderRadius: 8, border: `1px solid ${error ? "#ef4444" : colors.panelBorder}`, background: error ? "#451a1a" : "#0f172a", color: error ? "#fca5a5" : colors.textMuted }}>
        {error || `${status}  |  Turn: P${toMove}${isHumanTurn ? " (human)" : " (AI)"}`}
      </div>

      <div style={{ display: "flex", gap: 8, marginBottom: 10, flexWrap: "wrap" }}>
        <select value={gameMode} onChange={(e) => setGameMode(e.target.value)} style={{ padding: 8, borderRadius: 8, background: colors.panel, color: colors.text }}>
          <option value="human_vs_ai">Human vs AI</option>
          <option value="ai_vs_ai">AI vs AI (manual step)</option>
        </select>
        <select value={humanPlayer} onChange={(e) => setHumanPlayer(parseInt(e.target.value, 10))} style={{ padding: 8, borderRadius: 8, background: colors.panel, color: colors.text }}>
          <option value={1}>You = P1</option>
          <option value={0}>You = P0</option>
        </select>
        <select value={engine} onChange={(e) => setEngine(e.target.value)} style={{ padding: 8, borderRadius: 8, background: colors.panel, color: colors.text }}>
          <option value="nn" disabled={!nnAvailable}>AlphaZero NN</option>
          <option value="mcts">Pure MCTS</option>
        </select>
        <button onClick={() => fetchLegal().catch((e) => setError(String(e.message || e)))} style={{ padding: "8px 12px", borderRadius: 8 }}>Refresh Legal</button>
        <button onClick={() => solveAndPlay(toMove).catch((e) => setError(String(e.message || e)))} disabled={isThinking} style={{ padding: "8px 12px", borderRadius: 8, background: "#4f46e5", color: "white", border: "none" }}>{isThinking ? "Thinking..." : "AI Move Now"}</button>
        {isHumanTurn && legalResult?.pass_allowed && !patchMode && <button onClick={() => passTurn().catch((e) => setError(String(e.message || e)))} style={{ padding: "8px 12px", borderRadius: 8, background: colors.warn, border: "none" }}>Pass</button>}
        <label style={{ display: "flex", gap: 6, alignItems: "center", fontSize: 12 }}><input type="checkbox" checked={editMode} onChange={(e) => setEditMode(e.target.checked)} /> Edit Mode</label>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 300px 1fr", gap: 12 }}>
        <Board
          board={p0Board}
          player={0}
          title={`Player 0 ${humanPlayer === 0 ? "(You)" : "(AI)"}`}
          active={toMove === 0}
          color={colors.p0}
          preview={preview}
          previewPlayer={previewPlayer}
          patchMode={patchMode && humanPlayer === 0}
          patchLegalSet={humanPlayer === 0 ? patchLegalSet : new Set()}
          onHover={(r, c) => onHoverCell(0, r, c)}
          onClick={(r, c, idx) => onClickCell(0, r, c, idx).catch((e) => setError(String(e.message || e)))}
          onLeave={() => setPreview(null)}
        />

        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <div style={{ background: colors.panel, border: `1px solid ${colors.panelBorder}`, borderRadius: 10, padding: 10 }}>
            <div style={{ fontSize: 11, color: colors.textDim, marginBottom: 6 }}>MARKET (next 3)</div>
            {next3.map(({ off, id }) => {
              const piece = piecesById[id];
              const key = `${off}-${id}`;
              const selected = selectedGroupKey === key;
              return (
                <button key={key} onClick={() => { setSelectedGroupKey(key); setSelectedOrient(null); setPreview(null); setPreviewPlayer(humanPlayer); }} disabled={!isHumanTurn || patchMode} style={{ width: "100%", textAlign: "left", marginBottom: 6, padding: 8, borderRadius: 8, border: selected ? `2px solid ${colors.warn}` : `1px solid ${colors.panelBorder}`, background: selected ? "#1c1917" : "#0f172a", color: colors.text }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <div>
                      <div style={{ fontSize: 11, color: colors.textDim }}>Offset {off}</div>
                      <div style={{ fontWeight: 700 }}>Piece {id}</div>
                      <div style={{ fontSize: 12, color: colors.textMuted }}>{piece?.cost_buttons ?? "?"}$ / {piece?.cost_time ?? "?"}t / +{piece?.income ?? "?"}</div>
                    </div>
                    {piece?.shape && <MiniShape shape={piece.shape} />}
                  </div>
                </button>
              );
            })}
            {selectedGroup && placementIndex && (
              <div style={{ marginTop: 8 }}>
                <div style={{ fontSize: 11, color: colors.textDim, marginBottom: 4 }}>ORIENTATION</div>
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                  {placementIndex.orientList.map((oi) => (
                    <button key={oi} onClick={() => { setSelectedOrient(oi); const ex = placementIndex.examples.get(oi); if (ex) { setPreview(ex); setPreviewPlayer(humanPlayer); } }} style={{ padding: "6px 9px", borderRadius: 6, border: selectedOrient === oi ? `2px solid ${colors.warn}` : `1px solid ${colors.panelBorder}`, background: "#0f172a", color: colors.text }}>
                      #{oi}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div style={{ background: colors.panel, border: `1px solid ${colors.panelBorder}`, borderRadius: 10, padding: 10 }}>
            <div style={{ fontSize: 11, color: colors.textDim, marginBottom: 6 }}>{engine === "nn" ? "NN SETTINGS" : "MCTS SETTINGS"}</div>
            {engine === "nn" ? (
              <label style={{ fontSize: 12, color: colors.textMuted }}>Simulations
                <input type="number" value={nnSimulations} onChange={(e) => setNnSimulations(Math.max(50, parseInt(e.target.value || "800", 10)))} style={{ width: "100%", padding: 6, marginTop: 4, borderRadius: 6, background: "#0f172a", color: colors.text, border: `1px solid ${colors.panelBorder}` }} />
              </label>
            ) : (
              <>
                <label style={{ fontSize: 12, color: colors.textMuted }}>Iterations
                  <input type="number" value={mctsIterations} onChange={(e) => setMctsIterations(Math.max(500, parseInt(e.target.value || "50000", 10)))} style={{ width: "100%", padding: 6, marginTop: 4, borderRadius: 6, background: "#0f172a", color: colors.text, border: `1px solid ${colors.panelBorder}` }} />
                </label>
                <label style={{ fontSize: 12, color: colors.textMuted }}>Workers
                  <input type="number" value={mctsWorkers} onChange={(e) => setMctsWorkers(Math.max(1, Math.min(64, parseInt(e.target.value || "4", 10))))} style={{ width: "100%", padding: 6, marginTop: 4, borderRadius: 6, background: "#0f172a", color: colors.text, border: `1px solid ${colors.panelBorder}` }} />
                </label>
              </>
            )}
          </div>
        </div>

        <Board
          board={p1Board}
          player={1}
          title={`Player 1 ${humanPlayer === 1 ? "(You)" : "(AI)"}`}
          active={toMove === 1}
          color={colors.p1}
          preview={preview}
          previewPlayer={previewPlayer}
          patchMode={patchMode && humanPlayer === 1}
          patchLegalSet={humanPlayer === 1 ? patchLegalSet : new Set()}
          onHover={(r, c) => onHoverCell(1, r, c)}
          onClick={(r, c, idx) => onClickCell(1, r, c, idx).catch((e) => setError(String(e.message || e)))}
          onLeave={() => setPreview(null)}
        />
      </div>

      {showSetup && (
        <div style={{ marginTop: 12, display: "grid", gridTemplateColumns: "1.3fr 1fr", gap: 10 }}>
          <div style={{ background: colors.panel, border: `1px solid ${colors.panelBorder}`, borderRadius: 10, padding: 10 }}>
            <div style={{ fontSize: 11, color: colors.textDim, marginBottom: 6 }}>STATE SETUP / PIECE ORDER</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 8 }}>
              <label style={{ fontSize: 12, color: colors.textMuted }}>P0 pos<input type="number" value={p0Pos} onChange={(e) => setP0Pos(Math.max(0, Math.min(53, parseInt(e.target.value || "0", 10))))} style={{ width: "100%", marginTop: 4 }} /></label>
              <label style={{ fontSize: 12, color: colors.textMuted }}>P0 buttons<input type="number" value={p0Buttons} onChange={(e) => setP0Buttons(Math.max(0, parseInt(e.target.value || "0", 10)))} style={{ width: "100%", marginTop: 4 }} /></label>
              <label style={{ fontSize: 12, color: colors.textMuted }}>P1 pos<input type="number" value={p1Pos} onChange={(e) => setP1Pos(Math.max(0, Math.min(53, parseInt(e.target.value || "0", 10))))} style={{ width: "100%", marginTop: 4 }} /></label>
              <label style={{ fontSize: 12, color: colors.textMuted }}>P1 buttons<input type="number" value={p1Buttons} onChange={(e) => setP1Buttons(Math.max(0, parseInt(e.target.value || "0", 10)))} style={{ width: "100%", marginTop: 4 }} /></label>
            </div>
            <div style={{ marginTop: 8 }}>
              <label style={{ fontSize: 12, color: colors.textMuted }}>Circle order CSV
                <input value={circleArr.join(",")} onChange={(e) => setCircleArr(e.target.value.split(",").map((x) => parseInt(x.trim(), 10)).filter((x) => !Number.isNaN(x)))} style={{ width: "100%", marginTop: 4 }} />
              </label>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 6 }}>
                <label style={{ fontSize: 12, color: colors.textMuted }}>Neutral index<input type="number" value={neutral} onChange={(e) => setNeutral(Math.max(0, parseInt(e.target.value || "0", 10)))} style={{ width: "100%", marginTop: 4 }} /></label>
                <label style={{ fontSize: 12, color: colors.textMuted }}>Set neutral before piece
                  <select value={circleArr.length ? circleArr[(neutral + 1) % circleArr.length] : ""} onChange={(e) => { const pid = parseInt(e.target.value, 10); const idx = circleArr.indexOf(pid); if (idx >= 0) setNeutral((idx - 1 + circleArr.length) % circleArr.length); }} style={{ width: "100%", marginTop: 4 }}>
                    {circleArr.map((id) => <option key={id} value={id}>ID {id}</option>)}
                  </select>
                </label>
              </div>
            </div>
            <div style={{ marginTop: 8, display: "flex", gap: 6, flexWrap: "wrap" }}>
              <button onClick={() => setBuildMode((v) => !v)} style={{ padding: "6px 10px", borderRadius: 6 }}>{buildMode ? "Build mode ON" : "Build mode OFF"}</button>
              <button onClick={() => setCircleArr((prev) => prev.slice(0, -1))} style={{ padding: "6px 10px", borderRadius: 6 }}>Undo piece</button>
              <button onClick={() => setCircleArr([])} style={{ padding: "6px 10px", borderRadius: 6 }}>Clear circle</button>
              <button onClick={() => setAtlasOpen((v) => !v)} style={{ padding: "6px 10px", borderRadius: 6 }}>{atlasOpen ? "Hide atlas" : "Show atlas"}</button>
            </div>
            {atlasOpen && (
              <div style={{ marginTop: 8, display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(140px,1fr))", gap: 6, maxHeight: 220, overflow: "auto" }}>
                {pieces.map((p) => (
                  <button key={p.id} onClick={() => buildMode && setCircleArr((prev) => (prev.includes(p.id) ? prev : [...prev, p.id]))} style={{ padding: 6, borderRadius: 6, textAlign: "left", border: circleArr.includes(p.id) ? "2px solid #22c55e" : `1px solid ${colors.panelBorder}`, background: "#0f172a", color: colors.text }}>
                    <div style={{ fontSize: 12, fontWeight: 700 }}>ID {p.id}</div>
                    <MiniShape shape={p.shape} cell={9} />
                  </button>
                ))}
              </div>
            )}
          </div>

          <div style={{ background: colors.panel, border: `1px solid ${colors.panelBorder}`, borderRadius: 10, padding: 10 }}>
            <div style={{ fontSize: 11, color: colors.textDim, marginBottom: 6 }}>CHECKPOINT / JSON</div>
            <label style={{ fontSize: 12, color: colors.textMuted }}>Checkpoint path<input value={nnPath} onChange={(e) => setNnPath(e.target.value)} style={{ width: "100%", marginTop: 4 }} /></label>
            <label style={{ fontSize: 12, color: colors.textMuted }}>Config path<input value={nnConfig} onChange={(e) => setNnConfig(e.target.value)} style={{ width: "100%", marginTop: 4 }} /></label>
            <label style={{ fontSize: 12, color: colors.textMuted }}>Device
              <select value={nnDevice} onChange={(e) => setNnDevice(e.target.value)} style={{ width: "100%", marginTop: 4 }}>
                <option value="cuda">cuda</option>
                <option value="cpu">cpu</option>
              </select>
            </label>
            <div style={{ marginTop: 6, display: "flex", gap: 6, flexWrap: "wrap" }}>
              <button onClick={() => loadNnCheckpoint()} style={{ padding: "6px 10px", borderRadius: 6, background: colors.ok, border: "none" }}>Load checkpoint</button>
              <button onClick={() => unloadNnCheckpoint()} style={{ padding: "6px 10px", borderRadius: 6 }}>Unload NN</button>
              <button onClick={() => refreshNn()} style={{ padding: "6px 10px", borderRadius: 6 }}>Refresh list</button>
            </div>
            {nnStatus?.checkpoints?.length > 0 && (
              <select onChange={(e) => setNnPath(e.target.value)} value={nnPath} style={{ width: "100%", marginTop: 6 }}>
                <option value="">Select discovered checkpoint...</option>
                {nnStatus.checkpoints.map((p) => <option key={p} value={p}>{p}</option>)}
              </select>
            )}
            <textarea value={jsonText} onChange={(e) => setJsonText(e.target.value)} style={{ width: "100%", minHeight: 140, marginTop: 8, background: "#0f172a", color: colors.text, border: `1px solid ${colors.panelBorder}`, borderRadius: 6, padding: 6 }} />
            <div style={{ display: "flex", gap: 6, marginTop: 6 }}>
              <button onClick={() => navigator.clipboard.writeText(jsonText).then(() => setStatus("JSON copied."))} style={{ padding: "6px 10px", borderRadius: 6 }}>Copy JSON</button>
              <button onClick={() => { try { applyState(JSON.parse(jsonText)); setStatus("JSON loaded."); } catch (e) { setError(String(e.message || e)); } }} style={{ padding: "6px 10px", borderRadius: 6 }}>Load JSON</button>
            </div>
          </div>
        </div>
      )}

      <div style={{ marginTop: 12, background: colors.panel, border: `1px solid ${colors.panelBorder}`, borderRadius: 10, padding: 10, maxHeight: 180, overflowY: "auto" }}>
        <div style={{ fontSize: 11, color: colors.textDim, marginBottom: 4 }}>MOVE LOG ({moveLog.length})</div>
        {moveLog.length === 0 ? <div style={{ color: colors.textDim, fontSize: 12 }}>No moves yet.</div> : moveLog.map((m, i) => <div key={i} style={{ fontSize: 12, color: colors.textMuted }}>#{i + 1} <span style={{ color: m.player === 0 ? colors.p0 : colors.p1 }}>P{m.player}</span> {m.pretty}</div>)}
      </div>
    </div>
  );
}
