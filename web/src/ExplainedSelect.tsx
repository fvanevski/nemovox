import { useEffect, useId, useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import { ChevronDown, Check } from "lucide-react";
import type { ExplainedOption } from "./config-options";

export function ExplainedSelect({label, value, onChange, items}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  items: readonly ExplainedOption[];
}) {
  const [opened, setOpened] = useState(false);
  const [hovered, setHovered] = useState<string | null>(null);
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const listId = useId();
  const activeValue = opened ? (hovered || value) : value;
  const activeIndex = Math.max(0, items.findIndex(x => x.value === activeValue));
  const active = items.find(x => x.value === value) || items[0];
  const highlighted = items[activeIndex] || active;
  useEffect(() => {
    if (!opened) return;
    const handle = (event: MouseEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpened(false);
    };
    document.addEventListener("mousedown", handle);
    return () => document.removeEventListener("mousedown", handle);
  }, [opened]);
  function key(event: KeyboardEvent<HTMLButtonElement>) {
    const index = Math.max(0, items.findIndex(item => item.value === (hovered || value)));
    if (event.key === "Escape") {setOpened(false); setHovered(null); return;}
    if (event.key === "ArrowDown" || event.key === "ArrowUp" || event.key === "Home" || event.key === "End") {
      event.preventDefault();
      setOpened(true);
      const next = event.key === "Home" ? 0 : event.key === "End" ? items.length - 1
        : (index + (event.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
      setHovered(items[next]?.value || null);
      return;
    }
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      if (opened && hovered !== null) onChange(hovered);
      setOpened(!opened);
      setHovered(null);
    }
  }
  return <div className="relative" ref={root}>
    <button type="button" ref={trigger} role="combobox"
      className="control flex min-h-11 w-full items-center justify-between gap-3 text-left"
      aria-label={label} aria-haspopup="listbox" aria-expanded={opened}
      aria-controls={listId}
      aria-activedescendant={opened && highlighted ? `${listId}-opt-${activeIndex}` : undefined}
      onClick={() => {setOpened(v => !v); setHovered(null);}} onKeyDown={key}>
      <span className="truncate">{active?.label || "Choose an option"}</span>
      <ChevronDown size={16} className="shrink-0 text-slate-400"/>
    </button>
    {opened && <div className="absolute z-40 mt-1 max-h-[330px] w-full min-w-[260px] overflow-y-auto rounded-xl border border-[#3f6070] bg-[#111f2b] p-1 shadow-xl"
      id={listId} role="listbox" aria-label={label}>
      {items.map((item, index) => <button key={item.value} id={`${listId}-opt-${index}`}
        type="button" tabIndex={-1} role="option"
        aria-selected={item.value === value} title={item.description}
        className={"group mb-1 w-full rounded-lg p-3 text-left transition hover:bg-[#274355] focus:bg-[#274355] focus:outline-none " +
          (item.value === (hovered || value) ? "bg-[#1c3544]" : "")}
        onMouseEnter={() => setHovered(item.value)}
        onFocus={() => setHovered(item.value)}
        onClick={() => {onChange(item.value); setOpened(false); setHovered(null); trigger.current?.focus();}}>
        <span className="flex items-center justify-between gap-2 text-sm font-semibold text-slate-100">
          {item.label}{item.value === value && <Check size={14} className="text-teal-300"/>}
        </span>
        <span className="mt-1 block text-xs leading-5 text-slate-300">{item.description}</span>
      </button>)}
    </div>}
    {highlighted && <p className="mt-2 text-xs leading-5 text-slate-400" aria-live="polite">
      {opened ? `${highlighted.label}: ` : ""}{highlighted.description}
    </p>}
  </div>;
}
