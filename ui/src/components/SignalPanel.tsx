/**
 * SignalPanel.tsx — 方向・確率・閾値バー・ライブ特徴量。
 *
 * 表示のみ。シグナル方向は prediction.signal（-1/0/1）から、確率は prob_* から描画。
 * 閾値バーは各確率バーに probThreshold（execution.py 由来）のマーカーを重ねる。
 * 特徴量は受け取った値をそのまま表示（UI で再計算しない）。
 */
import type { LiveFeature, Prediction } from "../data/types";
import { directionOf } from "../data/types";
import { num, pct } from "../format";

const DIR_COLOR: Record<string, string> = {
  LONG: "var(--ax-positive)",
  SHORT: "var(--ax-negative)",
  FLAT: "var(--ax-text-secondary)",
};

// バー左のラベル幅 + gap / 右の数値幅 + gap。閾値線のオフセット計算に使う。
const LABEL_W = 42;
const VALUE_W = 48;
const GAP = 10;

/** 特徴量はスケールが幅広いので桁数を大きさで切り替える。 */
function featureValue(v: number): string {
  const a = Math.abs(v);
  if (a >= 100) return num(v, 1);
  if (a >= 1) return num(v, 2);
  return num(v, 4);
}

/** 確率バー 1 本: fill 幅 = prob、縦線 = 閾値マーカー。 */
function ProbRow({
  label,
  value,
  threshold,
  color,
}: {
  label: string;
  value: number;
  threshold: number;
  color: string;
}) {
  const fill = Math.max(0, Math.min(1, value)) * 100;
  // バー領域は左 (LABEL_W+GAP) 〜 右 (VALUE_W+GAP)。その内側で threshold 位置に縦線。
  const left = LABEL_W + GAP;
  const right = VALUE_W + GAP;
  return (
    <div style={{ position: "relative", display: "flex", alignItems: "center", gap: GAP }}>
      <span style={{ width: LABEL_W, fontSize: 11, color: "var(--ax-text-secondary)" }}>{label}</span>
      <div
        style={{
          flex: 1,
          height: 10,
          background: "var(--ax-border)",
          borderRadius: 5,
          overflow: "hidden",
        }}
      >
        <div style={{ width: `${fill}%`, height: "100%", background: color, borderRadius: 5 }} />
      </div>
      <span style={{ width: VALUE_W, textAlign: "right", fontSize: 12, color: "var(--ax-text-primary)" }}>
        {pct(value, 1)}
      </span>
      {/* 閾値の縦線（track の overflow:hidden を避け、行コンテナに絶対配置） */}
      <div
        aria-hidden
        title={`threshold ${pct(threshold, 0)}`}
        style={{
          position: "absolute",
          top: "50%",
          transform: "translateY(-50%)",
          left: `calc(${left}px + (100% - ${left + right}px) * ${threshold})`,
          width: 2,
          height: 16,
          background: "var(--ax-text-primary)",
          opacity: 0.55,
        }}
      />
    </div>
  );
}

export interface SignalPanelProps {
  prediction: Prediction;
  probThreshold: number;
  features: LiveFeature[];
}

export default function SignalPanel({ prediction, probThreshold, features }: SignalPanelProps) {
  const dir = directionOf(prediction.signal);
  return (
    <div className="ax-panel" style={{ padding: 18, display: "flex", flexDirection: "column", gap: 16 }}>
      {/* 見出し: 銘柄 + 方向バッジ */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
          <span className="ax-display" style={{ fontSize: 16, color: "var(--ax-text-primary)" }}>
            {prediction.symbol}
          </span>
          <span style={{ fontSize: 11, color: "var(--ax-text-tertiary)" }}>
            signal · {prediction.model_version}
          </span>
        </div>
        <span
          style={{
            fontSize: 13,
            letterSpacing: 1.5,
            fontWeight: 600,
            color: DIR_COLOR[dir],
            border: `1px solid ${DIR_COLOR[dir]}`,
            borderRadius: 6,
            padding: "3px 12px",
          }}
        >
          {dir}
        </span>
      </div>

      {/* 確率バー（閾値マーカー付き） */}
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        <ProbRow label="down" value={prediction.prob_down} threshold={probThreshold} color="var(--ax-negative)" />
        <ProbRow label="flat" value={prediction.prob_flat} threshold={probThreshold} color="var(--ax-text-secondary)" />
        <ProbRow label="up" value={prediction.prob_up} threshold={probThreshold} color="var(--ax-positive)" />
        <span style={{ fontSize: 10, color: "var(--ax-text-tertiary)" }}>
          threshold = {pct(probThreshold, 0)}
        </span>
      </div>

      {/* ライブ特徴量 */}
      <div>
        <span
          style={{
            fontSize: 10,
            letterSpacing: 1.2,
            textTransform: "uppercase",
            color: "var(--ax-text-secondary)",
          }}
        >
          live features
        </span>
        <div
          style={{
            marginTop: 8,
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))",
            gap: "6px 18px",
          }}
        >
          {features.map((f) => (
            <div
              key={f.name}
              style={{ display: "flex", justifyContent: "space-between", gap: 8, fontSize: 12 }}
            >
              <span style={{ color: "var(--ax-text-secondary)" }}>{f.name}</span>
              <span style={{ color: "var(--ax-text-primary)" }}>{featureValue(f.value)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
