import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type {
  HorizonKey,
  ModelKey,
  ModelPair,
  ModelScore,
  ModelSetting,
  ModelsPayload,
  Station,
} from "@/lib/research-data";
import {
  HORIZON_LABEL,
  campaignLong,
  campaignShort,
  campaignYear,
  fmt,
  int,
  pct,
  signed,
  useModelReplay,
} from "@/lib/research-data";
import { CARD_ORDER, FAMILY, MODEL_INFO } from "@/content/models";
import { HorizonPicker } from "@/components/findings";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";

const axisTick = { fill: "var(--ink-2)", fontSize: 13 };
const REPO = "https://github.com/akhilgolepu/aquasense-groundwater-forecasting/blob/main";
const REAL: ModelKey[] = ["combo", "lstm", "m2", "rf", "lstm_mem", "m0", "chronos"];

type Metric = "skill" | "mae_skill";
const SETTING_LABEL: Record<ModelSetting, string> = {
  main: "Test years 2019–2025",
  split: "Later test years 2020–2025",
  blocks: "Regions never seen",
};
const SETTING_HELP: Record<ModelSetting, string> = {
  main: "Trained on targets up to 2015, scored once on 2019–2025.",
  split:
    "Retrained on targets up to 2017 and scored on 2020–2025, so a different set of years decides.",
  blocks:
    "Wells grouped into 2° × 2° blocks (about 220 km); each block is forecast by models trained without it. Only models refitted in this setting are shown.",
};

function familyOf(data: ModelsPayload, key: ModelKey) {
  return data.models.find((m) => m.key === key)?.family ?? "reference";
}

function Figure({
  title,
  note,
  children,
}: {
  title: string;
  note?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <figure className="flex min-w-0 flex-col gap-3">
      <figcaption>
        <h3 className="type-heading text-[1.2rem] text-ink">{title}</h3>
        {note && <div className="mt-1 max-w-[70ch] text-[0.9375rem] text-ink-2">{note}</div>}
      </figcaption>
      {children}
    </figure>
  );
}

function TipBox({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-md border border-rule bg-panel px-3 py-2 text-[0.875rem] text-ink-2 shadow-sm">
      {children}
    </div>
  );
}

function FamilyChip({ family }: { family: keyof typeof FAMILY }) {
  const f = FAMILY[family];
  return (
    <span className="inline-flex items-center gap-1.5 rounded-md border border-rule px-2 py-0.5 text-[0.8125rem] text-ink-2">
      <Swatch family={family} />
      {f.badge}
    </span>
  );
}

function Swatch({ family }: { family: keyof typeof FAMILY }) {
  return family === "reference" ? (
    <svg width="12" height="12" aria-hidden className="shrink-0">
      <rect
        x="0.75"
        y="0.75"
        width="10.5"
        height="10.5"
        rx="2"
        fill="url(#hatch-ref)"
        stroke="var(--ink-3)"
        strokeWidth="1.5"
      />
    </svg>
  ) : (
    <span
      className="inline-block h-3 w-3 shrink-0 rounded-sm"
      style={{ background: FAMILY[family].color }}
      aria-hidden
    />
  );
}

/** One hatch pattern for every "reference only" mark on the page. */
function HatchDefs() {
  return (
    <svg width="0" height="0" className="absolute" aria-hidden>
      <defs>
        <pattern
          id="hatch-ref"
          width="6"
          height="6"
          patternUnits="userSpaceOnUse"
          patternTransform="rotate(45)"
        >
          <rect width="6" height="6" fill="var(--ground)" />
          <line x1="0" y1="0" x2="0" y2="6" stroke="var(--ink-3)" strokeWidth="2" />
        </pattern>
      </defs>
    </svg>
  );
}

// ── Section ─────────────────────────────────────────────────────────────────

export function ModelShowcase({
  data,
  stations,
  wellId,
}: {
  data: ModelsPayload;
  stations: Station[];
  wellId: string | null;
}) {
  const [h, setH] = useState<HorizonKey>("2");
  return (
    <div className="flex flex-col gap-16">
      <HatchDefs />
      <HowWeTested />
      <div className="flex flex-col gap-10">
        <HorizonPicker value={h} onChange={setH} id="models-horizon" />
        <Leaderboard data={data} h={h} />
      </div>
      <ModelCards data={data} h={h} />
      <HeadToHead data={data} h={h} />
      <StrataGrid data={data} h={h} />
      <ModelReplayPanel data={data} stations={stations} wellId={wellId} h={h} onHorizon={setH} />
      <p className="text-[0.8125rem] text-ink-3 [overflow-wrap:anywhere]">Source: {data.source}</p>
    </div>
  );
}

// ── How we tested ───────────────────────────────────────────────────────────

function HowWeTested() {
  const x = (year: number) => ((year - 1994) / (2026 - 1994)) * 100;
  const parts = [
    { from: 1994, to: 2016, label: "Train (targets up to 2015)", color: "var(--neutral-mid)" },
    { from: 2016, to: 2019, label: "Calibrate and choose (2016–2018)", color: "var(--none)" },
    { from: 2019, to: 2026, label: "Test once (2019–2025)", color: "var(--s1)" },
  ];
  return (
    <div className="flex flex-col gap-5">
      <div>
        <div
          className="relative h-9 w-full overflow-hidden rounded-md"
          role="img"
          aria-label="Timeline: training targets 1994 to 2015, calibration 2016 to 2018, test 2019 to 2025"
        >
          {parts.map((p) => (
            <div
              key={p.label}
              className="absolute inset-y-0 border-r-2 border-ground"
              style={{
                left: `${x(p.from)}%`,
                width: `${x(p.to) - x(p.from)}%`,
                background: p.color,
              }}
            />
          ))}
        </div>
        <div className="tnum mt-1 flex justify-between text-[0.8125rem] text-ink-3">
          <span>1994</span>
          <span>2016</span>
          <span>2019</span>
          <span>2025</span>
        </div>
        <ul className="mt-2 flex flex-wrap gap-x-6 gap-y-1 text-[0.875rem] text-ink-2">
          {parts.map((p) => (
            <li key={p.label} className="flex items-center gap-2">
              <span
                className="inline-block h-3 w-3 rounded-sm"
                style={{ background: p.color }}
                aria-hidden
              />
              {p.label}
            </li>
          ))}
        </ul>
      </div>
      <div className="grid gap-6 md:grid-cols-3">
        <p className="text-[0.9375rem] text-ink-2">
          <span className="font-semibold text-ink">Same rows for every model.</span> Each score
          below is on the same test forecasts, against the same simple rule, with a 95% interval
          from resampling whole wells.
        </p>
        <p className="text-[0.9375rem] text-ink-2">
          <span className="font-semibold text-ink">Choices made before testing.</span> Settings were
          tuned on 2013–2015 and the combination was picked on 2016–2018, so the test years were
          used once, for scoring.
        </p>
        <p className="text-[0.9375rem] text-ink-2">
          <span className="font-semibold text-ink">Checked for leakage.</span> No model sees data
          from after its forecast; tests fail if that changes.{" "}
          <a className="text-ink underline underline-offset-2" href={`${REPO}/LEAKAGE_AUDIT.md`}>
            Read the audit
          </a>
          .
        </p>
      </div>
    </div>
  );
}

// ── Leaderboard ─────────────────────────────────────────────────────────────

function Leaderboard({ data, h }: { data: ModelsPayload; h: HorizonKey }) {
  const [setting, setSetting] = useState<ModelSetting>("main");
  const [metric, setMetric] = useState<Metric>("skill");
  const [showRef, setShowRef] = useState(true);
  const [asTable, setAsTable] = useState(false);

  const scores = data.scores.filter((s) => s.h === Number(h) && s.setting === setting);
  const ceiling = scores.find((s) => s.model === "m3");
  const rows = scores
    .filter((s) => s.model !== "persistence" && s.model !== "m3")
    .filter((s) => showRef || familyOf(data, s.model) !== "reference")
    .map((s) => {
      const v = s[metric];
      const lo = metric === "skill" ? s.skill_lo : s.mae_skill_lo;
      const hi = metric === "skill" ? s.skill_hi : s.mae_skill_hi;
      const family = familyOf(data, s.model);
      return {
        key: s.model,
        label: MODEL_INFO[s.model].short,
        family,
        v,
        lo,
        hi,
        err: [v - lo, hi - v] as [number, number],
        s,
      };
    })
    .sort((a, b) => b.v - a.v);
  const n = scores[0];
  const ceilV = ceiling ? ceiling[metric] : undefined;
  const lowest = Math.min(0, ...rows.map((r) => r.lo));
  const highest = Math.max(...rows.map((r) => r.hi), ceilV ?? 0);
  const domain: [number, number] = [
    Math.floor(lowest * 20) / 20,
    Math.ceil((highest + 0.02) * 20) / 20,
  ];
  const ticks: number[] = [];
  for (let v = Math.ceil(domain[0] * 10) / 10; v <= domain[1] + 1e-9; v += 0.1)
    ticks.push(Number(v.toFixed(1)));

  return (
    <Figure
      title="Leaderboard"
      note={
        <>
          {metric === "skill"
            ? "Share of persistence's squared error each model removes."
            : "Share of persistence's absolute error each model removes; less affected by a few wild readings."}{" "}
          {SETTING_HELP[setting]}
          {n && ` ${int(n.rows)} forecasts at ${int(n.wells)} wells over ${n.campaigns} campaigns.`}
        </>
      }
    >
      <div className="flex flex-wrap items-center gap-3">
        <ToggleGroup
          type="single"
          value={setting}
          onValueChange={(v) => v && setSetting(v as ModelSetting)}
          className="flex-wrap justify-start rounded-md border border-rule"
          aria-label="Test setting"
        >
          {(Object.keys(SETTING_LABEL) as ModelSetting[]).map((k) => (
            <ToggleGroupItem key={k} value={k} className="px-3 text-[0.875rem]">
              {SETTING_LABEL[k]}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
        <ToggleGroup
          type="single"
          value={metric}
          onValueChange={(v) => v && setMetric(v as Metric)}
          className="rounded-md border border-rule"
          aria-label="Error measure"
        >
          <ToggleGroupItem value="skill" className="px-3 text-[0.875rem]">
            Squared error
          </ToggleGroupItem>
          <ToggleGroupItem value="mae_skill" className="px-3 text-[0.875rem]">
            Absolute error
          </ToggleGroupItem>
        </ToggleGroup>
        <label className="flex items-center gap-2 text-[0.875rem] text-ink-2">
          <input
            type="checkbox"
            checked={showRef}
            onChange={(e) => setShowRef(e.target.checked)}
            className="h-4 w-4 accent-[var(--s1)]"
          />
          Show reference models
        </label>
      </div>

      <ul
        className="flex flex-wrap gap-x-5 gap-y-1 text-[0.875rem] text-ink-2"
        aria-label="Model families"
      >
        {(["boosting", "learner", "combination", "baseline", "reference"] as const).map((f) => (
          <li key={f} className="flex items-center gap-2">
            <Swatch family={f} /> {FAMILY[f].label}
          </li>
        ))}
        {ceiling && (
          <li className="flex items-center gap-2">
            <svg width="18" height="12" aria-hidden>
              <line
                x1="9"
                y1="0"
                x2="9"
                y2="12"
                stroke="var(--ink)"
                strokeWidth="2"
                strokeDasharray="3 2"
              />
            </svg>
            Perfect-rainfall ceiling (not a real forecast)
          </li>
        )}
      </ul>

      {asTable ? (
        <LeaderTable rows={rows} ceiling={ceiling} metric={metric} />
      ) : (
        <div style={{ height: 44 + rows.length * 34 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={rows}
              layout="vertical"
              margin={{ top: 18, right: 4, bottom: 0, left: 4 }}
              barCategoryGap={6}
            >
              <CartesianGrid horizontal={false} stroke="var(--rule)" />
              <XAxis
                type="number"
                domain={domain}
                ticks={ticks}
                tick={axisTick}
                axisLine={false}
                tickLine={false}
                tickFormatter={(v: number) => v.toFixed(1)}
              />
              <YAxis
                type="category"
                dataKey="label"
                width={150}
                tick={{ ...axisTick, fontSize: 12 }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                yAxisId="values"
                orientation="right"
                type="category"
                dataKey="label"
                width={48}
                tickFormatter={(l: string) => rows.find((r) => r.label === l)?.v.toFixed(2) ?? ""}
                tick={{ fill: "var(--ink)", fontSize: 13 }}
                axisLine={false}
                tickLine={false}
              />
              <ReferenceLine x={0} stroke="var(--ink-3)" />
              {ceilV !== undefined && (
                <ReferenceLine
                  x={ceilV}
                  stroke="var(--ink)"
                  strokeWidth={2}
                  strokeDasharray="5 3"
                  label={{
                    value: `Perfect rainfall ${ceilV.toFixed(2)}`,
                    position: "top",
                    fill: "var(--ink-2)",
                    fontSize: 12,
                  }}
                />
              )}
              <Tooltip
                cursor={{ fill: "var(--sunk)" }}
                content={({ active, payload }) => {
                  const r = payload?.[0]?.payload as (typeof rows)[number] | undefined;
                  if (!active || !r) return null;
                  return (
                    <TipBox>
                      <div className="font-semibold text-ink">{MODEL_INFO[r.key].name}</div>
                      <div>{FAMILY[r.family].badge}</div>
                      <div className="tnum">
                        {fmt(r.v, 3)} (95%: {fmt(r.lo, 3)} to {fmt(r.hi, 3)})
                      </div>
                      <div className="tnum">
                        Mean error {fmt(r.s.mae, 2)} m, RMSE {fmt(r.s.rmse, 2)} m
                      </div>
                    </TipBox>
                  );
                }}
              />
              <Bar
                dataKey="v"
                radius={[0, 4, 4, 0]}
                isAnimationActive={false}
                stroke="var(--ground)"
                strokeWidth={2}
              >
                {rows.map((r) => (
                  <Cell
                    key={r.key}
                    fill={r.family === "reference" ? "url(#hatch-ref)" : FAMILY[r.family].color}
                    stroke={r.family === "reference" ? "var(--ink-3)" : "var(--ground)"}
                  />
                ))}
                <ErrorBar
                  dataKey="err"
                  width={4}
                  strokeWidth={1.5}
                  stroke="var(--ink)"
                  direction="x"
                />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
      <button
        type="button"
        onClick={() => setAsTable((t) => !t)}
        className="self-start text-[0.875rem] text-ink underline underline-offset-2"
      >
        {asTable ? "Show as chart" : "Show as table"}
      </button>
    </Figure>
  );
}

function LeaderTable({
  rows,
  ceiling,
  metric,
}: {
  rows: {
    key: ModelKey;
    v: number;
    lo: number;
    hi: number;
    family: keyof typeof FAMILY;
    s: ModelScore;
  }[];
  ceiling: ModelScore | undefined;
  metric: Metric;
}) {
  const all = ceiling
    ? [
        ...rows,
        {
          key: "m3" as ModelKey,
          v: ceiling[metric],
          lo: metric === "skill" ? ceiling.skill_lo : ceiling.mae_skill_lo,
          hi: metric === "skill" ? ceiling.skill_hi : ceiling.mae_skill_hi,
          family: "reference" as const,
          s: ceiling,
        },
      ]
    : rows;
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[520px] text-[0.9375rem]">
        <thead>
          <tr className="border-b border-rule-strong text-left text-ink-2">
            <th className="py-1.5 font-medium">Model</th>
            <th className="py-1.5 font-medium">Group</th>
            <th className="py-1.5 text-right font-medium">Skill</th>
            <th className="py-1.5 text-right font-medium">95% interval</th>
            <th className="py-1.5 text-right font-medium">Mean error</th>
          </tr>
        </thead>
        <tbody>
          {all.map((r) => (
            <tr key={r.key} className="border-b border-rule">
              <td className="py-1.5 text-ink">{MODEL_INFO[r.key].short}</td>
              <td className="py-1.5 text-ink-2">{FAMILY[r.family].badge}</td>
              <td className="tnum py-1.5 text-right text-ink">{fmt(r.v, 3)}</td>
              <td className="tnum py-1.5 text-right text-ink-2">
                {fmt(r.lo, 3)} to {fmt(r.hi, 3)}
              </td>
              <td className="tnum py-1.5 text-right text-ink-2">{fmt(r.s.mae, 2)} m</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Model cards ─────────────────────────────────────────────────────────────

function formatSeconds(s: number) {
  return s < 90 ? `${Math.round(s)} s` : `${(s / 60).toFixed(s < 600 ? 1 : 0)} min`;
}

function ModelCards({ data, h }: { data: ModelsPayload; h: HorizonKey }) {
  const main = data.scores.filter((s) => s.h === Number(h) && s.setting === "main");
  const ranked = REAL.map((k) => main.find((s) => s.model === k)!)
    .filter(Boolean)
    .sort((a, b) => b.skill - a.skill);
  const combo = data.combination[h]?.map((k) => MODEL_INFO[k].short).join(" + ");
  const t = data.timing;
  const timeFor = (k: ModelKey): string | null => {
    if (!t) return null;
    if (k === "persistence" || k === "climatology" || k === "anomaly") return "No training";
    if (k === "chronos" && t.chronos != null)
      return `No training; ${formatSeconds(t.chronos)} to forecast ${int(t.chronos_origins)} test origins`;
    if (k === "combo") return "No training of its own; runs its member models";
    if (k === "lstm" && t.lstm != null)
      return `${formatSeconds(t.lstm)} for three training runs with early stopping`;
    if ((k === "m0" || k === "m2" || k === "rf") && t[k] != null)
      return `${formatSeconds(t[k]!)} including ${k === "rf" ? "one fit" : "the settings search and refit"}`;
    if ((k === "cfsv2" || k === "snapshot" || k === "m3") && t.m2 != null)
      return `About as long as XGBoost M2 (${formatSeconds(t.m2)})`;
    return null;
  };
  return (
    <Figure
      title="The models"
      note={`What each model reads and learns, and how it did at ${HORIZON_LABEL[h]}.${t ? ` Times are wall-clock on our workstation (${t.cpu_threads} CPU threads${t.gpu ? `, ${t.gpu}` : ""}) for the 6-month horizon, ${int(t.train_rows)} training rows.` : ""}`}
    >
      <ul className="grid gap-x-8 gap-y-8 md:grid-cols-2 xl:grid-cols-3">
        {CARD_ORDER.map((k) => {
          const info = MODEL_INFO[k];
          const family = familyOf(data, k);
          const s = main.find((r) => r.model === k);
          const rank = ranked.findIndex((r) => r.model === k);
          const inputs = data.models.find((m) => m.key === k)?.inputs;
          const time = timeFor(k);
          return (
            <li key={k} className="flex flex-col gap-2 border-t border-rule pt-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h4 className="type-heading text-[1.05rem] text-ink">{info.name}</h4>
                <FamilyChip family={family} />
              </div>
              {s && (
                <div className="tnum flex flex-wrap items-baseline gap-x-3 text-[0.9375rem] text-ink-2">
                  <span className="type-figure text-[1.6rem] leading-none text-ink">
                    {k === "persistence" ? "0" : fmt(s.skill, 2)}
                  </span>
                  <span>
                    skill
                    {k !== "persistence" && ` (95%: ${fmt(s.skill_lo, 2)}–${fmt(s.skill_hi, 2)})`}
                  </span>
                  {rank >= 0 && (
                    <span>
                      #{rank + 1} of {ranked.length} forecasting models
                    </span>
                  )}
                </div>
              )}
              <p className="text-[0.9375rem] font-medium text-ink">{info.verdict}</p>
              <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-[0.875rem]">
                <dt className="text-ink-3">Reads</dt>
                <dd className="text-ink-2">
                  {k === "combo" && combo ? `${combo} at this horizon.` : info.reads}
                  {inputs ? ` (${inputs} inputs)` : ""}
                </dd>
                <dt className="text-ink-3">Learns</dt>
                <dd className="text-ink-2">{info.learns}</dd>
                <dt className="text-ink-3">Good at</dt>
                <dd className="text-ink-2">{info.good}</dd>
                <dt className="text-ink-3">Limits</dt>
                <dd className="text-ink-2">{info.limits}</dd>
                {time && (
                  <>
                    <dt className="text-ink-3">Time</dt>
                    <dd className="text-ink-2">{time}</dd>
                  </>
                )}
              </dl>
            </li>
          );
        })}
      </ul>
    </Figure>
  );
}

// ── Head-to-head ────────────────────────────────────────────────────────────

const PICKABLE: ModelKey[] = [
  "combo",
  "lstm",
  "m2",
  "rf",
  "lstm_mem",
  "m0",
  "chronos",
  "anomaly",
  "climatology",
  "persistence",
  "cfsv2",
  "snapshot",
  "m3",
];

function ModelSelect({
  id,
  label,
  value,
  onChange,
  exclude,
}: {
  id: string;
  label: string;
  value: ModelKey;
  onChange: (k: ModelKey) => void;
  exclude: ModelKey;
}) {
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor={id} className="text-[0.875rem] text-ink-2">
        {label}
      </label>
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value as ModelKey)}
        className="h-10 rounded-md border border-rule-strong bg-panel px-3 text-[0.9375rem] text-ink"
      >
        {PICKABLE.filter((k) => k !== exclude).map((k) => (
          <option key={k} value={k}>
            {MODEL_INFO[k].short}
          </option>
        ))}
      </select>
    </div>
  );
}

function verdict(p: ModelPair, a: string, b: string) {
  if (p.lo > 0) return `${b} is better: the whole 95% interval is above zero.`;
  if (p.hi < 0) return `${a} is better: the whole 95% interval is below zero.`;
  return "No clear winner: the difference is within noise.";
}

function HeadToHead({ data, h }: { data: ModelsPayload; h: HorizonKey }) {
  const [a, setA] = useState<ModelKey>("m2");
  const [b, setB] = useState<ModelKey>("combo");
  const pair = data.pairs.find((p) => p.h === Number(h) && p.a === a && p.b === b);
  const camps = data.campaigns.filter((c) => c.h === Number(h));
  const A = MODEL_INFO[a].short;
  const B = MODEL_INFO[b].short;
  const span = pair ? Math.max(0.05, Math.abs(pair.lo), Math.abs(pair.hi)) * 1.15 : 0.1;
  const pos = (v: number) => ((v + span) / (2 * span)) * 100;

  return (
    <Figure
      title="Head to head"
      note="Pick two models. The bar shows how much less squared error the second model makes than the first, with a 95% interval; the strip shows which one had the lower average error in each test campaign."
    >
      <div className="grid gap-4 sm:grid-cols-[1fr_auto_1fr] sm:items-end">
        <ModelSelect id="h2h-a" label="First model" value={a} onChange={setA} exclude={b} />
        <button
          type="button"
          onClick={() => {
            setA(b);
            setB(a);
          }}
          className="h-10 rounded-md border border-rule px-3 text-[0.875rem] text-ink-2 hover:text-ink"
          aria-label="Swap the two models"
        >
          Swap
        </button>
        <ModelSelect id="h2h-b" label="Second model" value={b} onChange={setB} exclude={a} />
      </div>
      {pair && (
        <div className="flex flex-col gap-5">
          <div className="flex flex-col gap-1">
            <p className="text-[1.05rem] font-semibold text-ink">{verdict(pair, A, B)}</p>
            <p className="tnum text-[0.9375rem] text-ink-2">
              {B} makes{" "}
              {pair.mse_reduction >= 0
                ? `${pct(pair.mse_reduction, 1)} less`
                : `${pct(-pair.mse_reduction, 1)} more`}{" "}
              squared error than {A} (95%: {signed(pair.lo * 100)}% to {signed(pair.hi * 100)}%) and{" "}
              {pair.mae_reduction >= 0
                ? `${pct(pair.mae_reduction, 1)} less`
                : `${pct(-pair.mae_reduction, 1)} more`}{" "}
              absolute error. It wins {pair.b_wins} of {pair.campaigns} campaigns (sign test p ={" "}
              {pair.sign_p < 0.001 ? "< 0.001" : fmt(pair.sign_p, 3)}).
            </p>
          </div>
          <div
            role="img"
            aria-label={`Error reduction ${pct(pair.mse_reduction, 1)}, interval ${pct(pair.lo, 1)} to ${pct(pair.hi, 1)}`}
          >
            <div className="relative h-8">
              <div className="absolute inset-x-0 top-1/2 h-px bg-rule-strong" />
              <div className="absolute inset-y-1 w-px bg-ink-3" style={{ left: "50%" }} />
              <div
                className="absolute top-1/2 h-1.5 -translate-y-1/2 rounded-full bg-ink-2"
                style={{ left: `${pos(pair.lo)}%`, width: `${pos(pair.hi) - pos(pair.lo)}%` }}
              />
              <div
                className="absolute top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-ground bg-ink"
                style={{ left: `${pos(pair.mse_reduction)}%` }}
              />
            </div>
            <div className="flex justify-between text-[0.8125rem] text-ink-3">
              <span>{A} better</span>
              <span>0</span>
              <span>{B} better</span>
            </div>
          </div>
          <div>
            <ul className="mb-2 flex flex-wrap gap-x-5 gap-y-1 text-[0.875rem] text-ink-2">
              <li className="flex items-center gap-2">
                <span className="inline-block h-3 w-3 rounded-sm bg-s2" aria-hidden /> {A} lower
                error
              </li>
              <li className="flex items-center gap-2">
                <span className="inline-block h-3 w-3 rounded-sm bg-s1" aria-hidden /> {B} lower
                error
              </li>
            </ul>
            <ol
              className="grid gap-0.5"
              style={{ gridTemplateColumns: `repeat(${camps.length}, minmax(0, 1fr))` }}
              aria-label="Winner of each test campaign"
            >
              {camps.map((c) => {
                const bWins = c[b] < c[a];
                return (
                  <li key={c.t} className="group relative">
                    <span
                      className={`block h-8 rounded-sm ${bWins ? "bg-s1" : "bg-s2"}`}
                      title={`${campaignLong(c.t)}: ${A} ${fmt(c[a], 2)} m, ${B} ${fmt(c[b], 2)} m (${int(c.n)} forecasts)`}
                    />
                    <span className="sr-only">
                      {campaignLong(c.t)}: {bWins ? B : A} had the lower error (
                      {fmt(Math.min(c[a], c[b]), 2)} m against {fmt(Math.max(c[a], c[b]), 2)} m)
                    </span>
                  </li>
                );
              })}
            </ol>
            <div className="mt-1 flex justify-between text-[0.8125rem] text-ink-3">
              <span>{camps[0] && campaignShort(camps[0].t)}</span>
              <span>{camps.at(-1) && campaignShort(camps.at(-1)!.t)}</span>
            </div>
          </div>
        </div>
      )}
    </Figure>
  );
}

// ── Where each model wins ───────────────────────────────────────────────────

const STRATA_ORDER: Record<"depth" | "region" | "season", string[]> = {
  depth: ["< 5 m", "5–15 m", "15–40 m", "≥ 40 m"],
  region: ["Arid north-west", "Indo-Gangetic plain", "Peninsular", "Himalaya & north-east"],
  season: ["Jan", "May", "Aug", "Nov"],
};
const STRATA_LABEL = {
  depth: "Depth to water",
  region: "Region",
  season: "Month forecast",
} as const;
const GRID_MODELS: ModelKey[] = [
  "combo",
  "lstm",
  "m2",
  "rf",
  "lstm_mem",
  "m0",
  "chronos",
  "anomaly",
  "cfsv2",
  "m3",
];

function StrataGrid({ data, h }: { data: ModelsPayload; h: HorizonKey }) {
  const [stratum, setStratum] = useState<keyof typeof STRATA_LABEL>("depth");
  const cells = data.strata.filter((s) => s.h === Number(h) && s.stratum === stratum);
  const cols = STRATA_ORDER[stratum].filter((v) => cells.some((c) => c.value === v));
  const max = Math.max(
    ...cells.filter((c) => GRID_MODELS.includes(c.model)).map((c) => c.skill),
    0.01,
  );
  const get = (m: ModelKey, v: string) => cells.find((c) => c.model === m && c.value === v);
  const best = Object.fromEntries(
    cols.map((v) => {
      const real = GRID_MODELS.filter((m) => m !== "m3" && m !== "cfsv2");
      return [
        v,
        real.reduce(
          (x, m) => ((get(m, v)?.skill ?? -9) > (get(x, v)?.skill ?? -9) ? m : x),
          "combo" as ModelKey,
        ),
      ];
    }),
  );
  return (
    <Figure
      title="Where each model does best"
      note="Skill against persistence within each group. Darker cells remove more error; the best real forecast in each column is marked."
    >
      <ToggleGroup
        type="single"
        value={stratum}
        onValueChange={(v) => v && setStratum(v as keyof typeof STRATA_LABEL)}
        className="self-start rounded-md border border-rule"
        aria-label="Group wells by"
      >
        {(Object.keys(STRATA_LABEL) as (keyof typeof STRATA_LABEL)[]).map((k) => (
          <ToggleGroupItem key={k} value={k} className="px-3 text-[0.875rem]">
            {STRATA_LABEL[k]}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[560px] border-separate border-spacing-0.5 text-[0.9375rem]">
          <thead>
            <tr className="text-ink-2">
              <th className="py-1.5 pr-3 text-left font-medium">Model</th>
              {cols.map((v) => (
                <th key={v} className="px-2 py-1.5 text-right font-medium">
                  {v}
                  <div className="tnum text-[0.75rem] font-normal text-ink-3">
                    {int(get("m2", v)?.rows ?? 0)} forecasts
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {GRID_MODELS.map((m) => (
              <tr key={m}>
                <th scope="row" className="py-1 pr-3 text-left font-normal text-ink">
                  <span className="flex items-center gap-2">
                    <Swatch family={familyOf(data, m)} />
                    {MODEL_INFO[m].short}
                  </span>
                </th>
                {cols.map((v) => {
                  const c = get(m, v);
                  const share = c ? Math.max(0, c.skill) / max : 0;
                  const isBest = best[v] === m;
                  return (
                    <td
                      key={v}
                      className="tnum rounded-sm px-2 py-1 text-right text-ink"
                      style={{
                        background: `color-mix(in srgb, var(--s1) ${Math.round(share * 45)}%, transparent)`,
                      }}
                    >
                      {c ? fmt(c.skill, 2) : "—"}
                      {isBest && (
                        <span className="ml-1 text-ink" aria-label="best real forecast">
                          ★
                        </span>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Figure>
  );
}

// ── Replay on one well ──────────────────────────────────────────────────────

const REPLAY_PICK: ModelKey[] = [
  "combo",
  "lstm",
  "m2",
  "rf",
  "m0",
  "chronos",
  "persistence",
  "cfsv2",
  "m3",
];
const DASH: Partial<Record<ModelKey, string>> = {
  rf: "6 3",
  chronos: "2 3",
  m0: "6 3",
  persistence: "2 3",
  cfsv2: "6 3",
};
const MAX_LINES = 3;

function ModelReplayPanel({
  data,
  stations,
  wellId,
  h,
  onHorizon,
}: {
  data: ModelsPayload;
  stations: Station[];
  wellId: string | null;
  h: HorizonKey;
  onHorizon: (h: HorizonKey) => void;
}) {
  const [picked, setPicked] = useState<ModelKey[]>(["combo", "m2", "persistence"]);
  const station = useMemo(() => stations.find((s) => s.id === wellId), [stations, wellId]);
  const { replay, isLoading } = useModelReplay(station);
  const series = replay?.[h];

  const rows = useMemo(
    () =>
      series?.t.map((t, i) => ({
        x: campaignYear(t),
        t,
        y: series.y[i] ?? NaN,
        ...Object.fromEntries(picked.map((k) => [k, series[k]?.[i]])),
      })) ?? [],
    [series, picked],
  );
  const errors = picked.map((k) => {
    const pred = series?.[k];
    if (!pred || !series) return { k, mae: NaN };
    return {
      k,
      mae: pred.reduce((s, p, i) => s + Math.abs(p - (series.y[i] ?? p)), 0) / pred.length,
    };
  });
  const toggle = (k: ModelKey) =>
    setPicked((p) =>
      p.includes(k)
        ? p.filter((x) => x !== k)
        : p.length >= MAX_LINES
          ? [...p.slice(1), k]
          : [...p, k],
    );
  const color = (k: ModelKey) => {
    const f = familyOf(data, k);
    return f === "reference" || f === "baseline" ? "var(--ink-3)" : FAMILY[f].color;
  };

  return (
    <Figure
      title="Replay the test years at one well"
      note={`Each model's forecasts made ${HORIZON_LABEL[h]} ahead for 2019–2025 against what was measured, at the well chosen in "Look at one well" or on the map. Pick up to three models.`}
    >
      <div className="flex flex-wrap items-center gap-4">
        <HorizonPicker value={h} onChange={onHorizon} id="replay-horizon" />
      </div>
      <div className="flex flex-wrap gap-2" role="group" aria-label="Models to show">
        {REPLAY_PICK.map((k) => {
          const on = picked.includes(k);
          return (
            <button
              key={k}
              type="button"
              aria-pressed={on}
              onClick={() => toggle(k)}
              className={`flex items-center gap-2 rounded-md border px-2.5 py-1 text-[0.875rem] ${on ? "border-ink bg-sunk text-ink" : "border-rule text-ink-2 hover:text-ink"}`}
            >
              <svg width="18" height="10" aria-hidden>
                <line
                  x1="1"
                  y1="5"
                  x2="17"
                  y2="5"
                  stroke={color(k)}
                  strokeWidth="2.5"
                  strokeDasharray={DASH[k]}
                />
              </svg>
              {MODEL_INFO[k].short}
            </button>
          );
        })}
      </div>
      {!station ? (
        <p className="text-[0.9375rem] text-ink-2">
          Choose a well on the map or in "Look at one well".
        </p>
      ) : isLoading ? (
        <div className="h-[320px] animate-pulse rounded-md bg-sunk" />
      ) : !series ? (
        <p className="text-[0.9375rem] text-ink-2">
          Well {station.id} has no {HORIZON_LABEL[h]} test forecasts (it needs readings in 2019–2025
          and a reading {HORIZON_LABEL[h]} earlier). Try another horizon or another well.
        </p>
      ) : (
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_260px]">
          <div className="h-[320px] min-w-0">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
                <CartesianGrid stroke="var(--rule)" vertical={false} />
                <XAxis
                  dataKey="x"
                  type="number"
                  domain={["dataMin", "dataMax"]}
                  tick={axisTick}
                  tickFormatter={(v: number) => String(Math.round(v))}
                  allowDecimals={false}
                  axisLine={{ stroke: "var(--rule-strong)" }}
                  tickLine={false}
                />
                <YAxis
                  reversed
                  domain={["auto", "auto"]}
                  width={52}
                  tick={axisTick}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={(v: number) => `${fmt(v, 0)} m`}
                />
                <Tooltip
                  content={({ active, payload }) => {
                    const r = payload?.[0]?.payload as (typeof rows)[number] | undefined;
                    if (!active || !r) return null;
                    return (
                      <TipBox>
                        <div className="font-semibold text-ink">{campaignLong(r.t)}</div>
                        <div className="tnum">Measured {fmt(r.y, 2)} m</div>
                        {picked.map((k) => {
                          const v = (r as unknown as Record<string, number | undefined>)[k];
                          return (
                            v !== undefined && (
                              <div key={k} className="tnum">
                                {MODEL_INFO[k].short} {fmt(v, 2)} m
                              </div>
                            )
                          );
                        })}
                      </TipBox>
                    );
                  }}
                />
                <Line
                  dataKey="y"
                  name="Measured"
                  stroke="var(--ink)"
                  strokeWidth={2}
                  isAnimationActive={false}
                  dot={{ r: 4, fill: "var(--ink)", stroke: "var(--ground)", strokeWidth: 2 }}
                />
                {picked.map((k) => (
                  <Line
                    key={k}
                    dataKey={k}
                    stroke={color(k)}
                    strokeWidth={2}
                    strokeDasharray={DASH[k]}
                    isAnimationActive={false}
                    dot={false}
                    activeDot={{ r: 4, stroke: "var(--ground)", strokeWidth: 2 }}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="flex flex-col gap-3">
            <div className="text-[0.875rem] text-ink-2">
              Well {station.id}
              {station.state ? `, ${station.state}` : ""}. {series.t.length} test forecasts,{" "}
              {campaignShort(series.t[0] ?? "")} to {campaignShort(series.t.at(-1)!)}.
            </div>
            <table className="w-full text-[0.9375rem]">
              <caption className="mb-1 text-left text-[0.875rem] font-semibold text-ink">
                Average error at this well
              </caption>
              <tbody>
                <tr className="border-b border-rule">
                  <th scope="row" className="py-1.5 text-left font-normal text-ink">
                    <span className="flex items-center gap-2">
                      <svg width="18" height="10" aria-hidden>
                        <line x1="1" y1="5" x2="17" y2="5" stroke="var(--ink)" strokeWidth="2.5" />
                      </svg>
                      Measured
                    </span>
                  </th>
                  <td className="py-1.5 text-right text-ink-3">—</td>
                </tr>
                {errors.map((e) => (
                  <tr key={e.k} className="border-b border-rule">
                    <th scope="row" className="py-1.5 text-left font-normal text-ink">
                      <span className="flex items-center gap-2">
                        <svg width="18" height="10" aria-hidden>
                          <line
                            x1="1"
                            y1="5"
                            x2="17"
                            y2="5"
                            stroke={color(e.k)}
                            strokeWidth="2.5"
                            strokeDasharray={DASH[e.k]}
                          />
                        </svg>
                        {MODEL_INFO[e.k].short}
                      </span>
                    </th>
                    <td className="tnum py-1.5 text-right text-ink">
                      {Number.isFinite(e.mae) ? `${fmt(e.mae, 2)} m` : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="text-[0.8125rem] text-ink-3">
              One well is one example: a model can win here and lose on average. The leaderboard is
              the fair comparison.
            </p>
          </div>
        </div>
      )}
    </Figure>
  );
}
