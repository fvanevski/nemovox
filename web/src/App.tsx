import { useEffect, useMemo, useRef, useState } from "react";
import * as Tabs from "@radix-ui/react-tabs";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { motion } from "motion/react";
import {
  Activity, AudioLines, Captions, Check, CheckCircle2, ChevronLeft, ChevronRight,
  CircleAlert, Clapperboard, Clock3, CloudDownload, Download, FileAudio2,
  FileText, Film, Folder, FolderOpen, Globe2, HardDrive, Languages, LoaderCircle,
  MonitorPlay, Pause, Play, RotateCcw, Settings2, SlidersHorizontal, Sparkles,
  Square, UploadCloud, Waves, X
} from "lucide-react";
import { studio, secondsLabel, type Cue, type Glossary, type Job, type JobOptions, type JobRequest, type MediaKind, type Step, type TaskKind } from "./api";

const INITIAL_OPTIONS: JobOptions = {
  source_language: "auto", target_language: "en", asr_mode: "speaker",
  audio_stream: 0, chunk_seconds: 90, context_seconds: 10,
  speaker_embeddings: "titanet", alignment_fallback: "off",
  refinement: "balanced", translator: "swiftu", style: "natural",
  glossary: null, ass_layout: "standard", ass_max_lanes: 3,
  condense_high_cps: true, formats: ["srt", "ass"],
};
const TASKS: {id: TaskKind; title: string; description: string; icon: typeof FileAudio2}[] = [
  {id: "transcription", title: "Transcribe", description: "Source-language transcript with speakers and timing.", icon: AudioLines},
  {id: "translation", title: "Translate", description: "Contextual English dialogue from media or a transcript.", icon: Languages},
  {id: "subtitles", title: "Subtitles", description: "Natural English SRT and styled ASS, ready to watch.", icon: Captions},
];
const LANGUAGES = [
  ["auto", "Auto-detect"], ["zh", "Mandarin Chinese"], ["fr", "French"], ["ja", "Japanese"],
  ["ko", "Korean"], ["es", "Spanish"], ["de", "German"], ["en", "English"],
];
const stages: {id: Step; name: string; icon: typeof Film; detail: string}[] = [
  {id: "media", name: "Media & task", icon: Clapperboard, detail: "Input, language, and objective"},
  {id: "configure", name: "Configure", icon: SlidersHorizontal, detail: "Processing and presentation"},
  {id: "preview", name: "Process & preview", icon: MonitorPlay, detail: "Live results and subtitle review"},
];

function Pill({children, tone = "neutral"}: {children: React.ReactNode; tone?: "neutral" | "good" | "warning"}) {
  const colors = {neutral: "text-slate-300 border-slate-600/60", good: "text-teal-200 border-teal-600/50 bg-teal-900/20", warning: "text-amber-200 border-amber-700/60 bg-amber-900/20"};
  return <span className={"pill " + colors[tone]}>{children}</span>;
}
function SectionTitle({eyebrow, title, sub}: {eyebrow: string; title: string; sub?: string}) {
  return <div className="mb-6"><div className="eyebrow mb-2">{eyebrow}</div>
    <h2 className="text-2xl font-semibold tracking-tight text-white">{title}</h2>
    {sub && <p className="mt-2 max-w-2xl text-sm leading-6 muted">{sub}</p>}
  </div>;
}
function Field({title, hint, children}: {title: string; hint?: string; children: React.ReactNode}) {
  return <label className="block min-w-0"><span className="label">{title}</span>{children}
    {hint && <span className="mt-2 block text-xs leading-5 text-slate-500">{hint}</span>}
  </label>;
}
function Select({value, onChange, children}: {value: string | number; onChange: (v: string) => void; children: React.ReactNode}) {
  return <select className="control" value={value} onChange={e => onChange(e.target.value)}>{children}</select>;
}
function Toggle({value, onChange, title, detail}: {value: boolean; onChange: (b: boolean) => void; title: string; detail?: string}) {
  return <label className="flex cursor-pointer items-center justify-between gap-4 rounded-xl border border-[#2b3a4a] bg-[#101925] p-4">
    <span><span className="block text-sm font-semibold">{title}</span>{detail && <span className="mt-1 block text-xs muted">{detail}</span>}</span>
    <input type="checkbox" checked={value} onChange={e => onChange(e.target.checked)} className="h-5 w-5 accent-teal-400" />
  </label>;
}
function TaskCard({task, active, onClick}: {task: typeof TASKS[number]; active: boolean; onClick: () => void}) {
  const Icon = task.icon;
  return <button type="button" onClick={onClick} aria-pressed={active}
    className={"flex min-h-32 flex-col items-start rounded-xl border p-4 text-left transition " + (active
      ? "border-teal-400/70 bg-[#17363d] ring-1 ring-teal-400/20"
      : "border-[#304051] bg-[#101925] hover:border-[#49657b] hover:bg-[#192635]")}>
    <div className="mb-3 flex w-full items-center justify-between">
      <Icon className={active ? "text-teal-300" : "text-slate-400"} size={22}/>
      {active && <CheckCircle2 className="text-teal-300" size={18}/>}
    </div>
    <span className="text-sm font-bold">{task.title}</span>
    <span className="mt-1 text-xs leading-5 muted">{task.description}</span>
  </button>;
}
function StatusIcon({state}: {state: Job["state"]}) {
  if (state === "complete") return <CheckCircle2 size={17} className="text-teal-300"/>;
  if (state === "running" || state === "queued") return <LoaderCircle size={17} className="animate-spin text-sky-300"/>;
  return <CircleAlert size={17} className="text-amber-300"/>;
}

export default function App() {
  const queryClient = useQueryClient();
  const [step, setStep] = useState<Step>("media");
  const [sourceMethod, setSourceMethod] = useState<"browse" | "upload" | "url">("browse");
  const [source, setSource] = useState<{kind: MediaKind; value: string}>({kind: "path", value: ""});
  const [browserDir, setBrowserDir] = useState<string | null>(null);
  const [task, setTask] = useState<TaskKind>("subtitles");
  const [options, setOptions] = useState<JobOptions>(INITIAL_OPTIONS);
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string>("");
  const [advanced, setAdvanced] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [cueSearch, setCueSearch] = useState("");
  const videoRef = useRef<HTMLVideoElement>(null);
  const selectedJob = useQuery({queryKey: ["job", jobId], queryFn: () => studio.job(jobId!), enabled: !!jobId,
    refetchInterval: (query) => ["running", "queued"].includes(query.state.data?.state || "") ? 1500 : false});
  const jobs = useQuery({queryKey: ["jobs"], queryFn: studio.jobs, refetchInterval: 4000});
  const health = useQuery({queryKey: ["health"], queryFn: studio.health, refetchInterval: 30000});
  const listing = useQuery({queryKey: ["media", browserDir], queryFn: () => studio.browse(browserDir), enabled: sourceMethod === "browse"});
  const glossaries = useQuery({queryKey: ["glossaries"], queryFn: studio.glossaries});
  const preview = useQuery({queryKey: ["preview", jobId], queryFn: () => studio.preview(jobId!), enabled: !!jobId,
    refetchInterval: selectedJob.data?.state === "complete" ? false : 4000});
  const artifacts = useQuery({queryKey: ["artifacts", jobId], queryFn: () => studio.artifacts(jobId!), enabled: !!jobId,
    refetchInterval: selectedJob.data?.state === "complete" ? false : 5000});
  const logs = useQuery({queryKey: ["logs", jobId], queryFn: () => studio.logs(jobId!), enabled: !!jobId,
    refetchInterval: selectedJob.data?.state === "running" ? 3000 : false});

  const submit = useMutation({
    mutationFn: studio.submit,
    onSuccess: (job) => {setJobId(job.id); setStep("preview"); setError(""); queryClient.invalidateQueries({queryKey: ["jobs"]});},
    onError: (e: Error) => setError(e.message),
  });
  const upload = useMutation({
    mutationFn: studio.upload,
    onSuccess: (result) => {setSource({kind: "upload", value: result.path}); setError("");},
    onError: (e: Error) => setError(e.message),
  });
  const cancel = useMutation({mutationFn: studio.cancel, onSuccess: () => {
    queryClient.invalidateQueries({queryKey: ["job", jobId]});
  }});

  function option<K extends keyof JobOptions>(key: K, value: JobOptions[K]) {
    setOptions(prev => ({...prev, [key]: value}));
  }
  function start() {
    if (!source.value.trim()) {setError("Select a media file, upload a file, or enter a URL."); setStep("media"); return;}
    const req: JobRequest = {source, task, options: {...options, formats: task === "subtitles" ? ["srt", "ass"] : ["srt"]}};
    setError("");
    submit.mutate(req);
  }
  const ready = source.value.trim().length > 0;
  const currentJob = selectedJob.data;
  const cues = preview.data?.cues || [];
  const active = cues.filter(c => c.start !== null && c.end !== null && currentTime >= c.start && currentTime < c.end).slice(0, 3);
  const filteredCues = useMemo(() => cues.filter(c => c.text.toLowerCase().includes(cueSearch.toLowerCase())), [cues, cueSearch]);
  useEffect(() => {setCurrentTime(0);}, [jobId]);

  return <div className="min-h-screen">
    <header className="border-b border-[#273648] bg-[#0b121b]/90">
      <div className="shell flex h-[80px] items-center justify-between !pb-0">
        <div className="flex items-center gap-4">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-teal-400/40 bg-[#15383c] shadow-[0_0_35px_rgba(69,184,180,.10)]">
            <Waves className="text-teal-300" size={23}/>
          </div>
          <div><div className="text-[19px] font-bold tracking-[-.03em]">nemovox<span className="font-medium text-teal-300"> studio</span></div>
            <div className="text-[10px] font-bold uppercase tracking-[.22em] text-slate-500">Local audio intelligence</div>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <Pill tone={health.isSuccess ? "good" : "warning"}>
            <span className={"h-2 w-2 rounded-full " + (health.isSuccess ? "bg-teal-300" : "bg-amber-300")}/>
            {health.isSuccess ? "Backend connected" : "Backend unavailable"}
          </Pill>
          <span className="hidden items-center gap-2 text-xs muted sm:flex"><HardDrive size={14}/> Local processing</span>
        </div>
      </div>
    </header>

    <div className="shell grid gap-7 pt-8 lg:grid-cols-[265px_minmax(0,1fr)]">
      <aside className="space-y-6">
        <div className="panel p-3">
          <div className="eyebrow px-3 pb-3 pt-2">Workflow</div>
          <Tabs.Root value={step} onValueChange={v => setStep(v as Step)} orientation="vertical">
            <Tabs.List className="grid gap-1" aria-label="Workflow stages">
              {stages.map((item, i) => {
                const Icon = item.icon;
                return <Tabs.Trigger key={item.id} value={item.id} className="step w-full text-left" data-state={step === item.id ? "active" : i < stages.findIndex(s => s.id === step) ? "done" : "inactive"}>
                  <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[#223141]"><Icon size={17}/></span>
                  <span className="min-w-0"><span className="block">{item.name}</span>
                    <span className="mt-0.5 block text-[11px] font-normal text-slate-500">{item.detail}</span></span>
                </Tabs.Trigger>;
              })}
            </Tabs.List>
          </Tabs.Root>
        </div>
        <div className="panel p-4">
          <div className="mb-4 flex items-center justify-between"><span className="eyebrow">Recent runs</span><RotateCcw size={14} className="text-slate-500"/></div>
          <div className="space-y-1">
            {!jobs.data?.length && <p className="text-xs leading-6 muted">Completed tasks will appear here for quick review.</p>}
            {jobs.data?.slice(0, 8).map(j =>
              <button key={j.id} onClick={() => {setJobId(j.id); setStep("preview");}}
                className={"flex w-full items-center gap-3 rounded-lg px-2 py-3 text-left hover:bg-[#223343] " + (jobId === j.id ? "bg-[#1e3440]" : "")}>
                <StatusIcon state={j.state}/>
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-xs font-semibold">{j.request.source.value.split("/").pop() || "Remote media"}</span>
                  <span className="mt-1 block text-[11px] capitalize text-slate-500">{j.request.task} · {j.state}</span>
                </span>
              </button>)}
          </div>
        </div>
        <div className="px-2 text-xs leading-6 text-slate-500">
          Runs use the existing Nemotron ASR and SwiftU localization pipeline. No cloud inference or subscriptions are required.
        </div>
      </aside>

      <main className="min-w-0">
        <div className="mb-7 flex flex-wrap items-end justify-between gap-4">
          <div><span className="eyebrow text-teal-300/80">Multilingual workflow</span>
            <h1 className="mt-2 text-3xl font-bold tracking-tight sm:text-4xl">Make every story <span className="text-[#78d3c9]">understandable.</span></h1>
            <p className="mt-2 text-sm muted">From original audio to natural dialogue and watchable subtitles.</p>
          </div>
          <Pill><Sparkles size={12}/> RTX-accelerated where available</Pill>
        </div>
        {error && <div role="alert" className="mb-5 flex items-start gap-3 rounded-xl border border-rose-700/50 bg-rose-950/30 p-4 text-sm text-rose-200">
          <CircleAlert size={18}/><span className="flex-1">{error}</span><button aria-label="Dismiss error" onClick={() => setError("")}><X size={16}/></button>
        </div>}

        {step === "media" && <motion.div initial={{opacity: 0, y: 8}} animate={{opacity: 1, y: 0}} className="space-y-6">
          <section className="panel p-6">
            <SectionTitle eyebrow="01 / Source" title="Choose your media" sub="Browse a file on the processing host, upload from this browser, or import a YouTube/remote URL."/>
            <div className="mb-5 grid grid-cols-3 gap-2">
              {([
                ["browse", FolderOpen, "Host files"],
                ["upload", UploadCloud, "Upload"],
                ["url", Globe2, "Remote URL"],
              ] as const).map(([key, Icon, label]) =>
                <button key={key} onClick={() => {setSourceMethod(key); setError("");}}
                  className={"flex items-center justify-center gap-2 rounded-lg border px-3 py-3 text-xs font-semibold " +
                    (sourceMethod === key ? "border-teal-500/70 bg-[#15383c] text-teal-200" : "border-[#304052] bg-[#111a26] text-slate-400 hover:bg-[#1d2c3a]")}>
                  <Icon size={16}/>{label}
                </button>)}
            </div>

            {sourceMethod === "browse" && <div>
              <div className="mb-3 flex items-center gap-3 text-xs muted">
                <Folder size={15}/>
                <span className="min-w-0 flex-1 truncate">{listing.data?.directory || "Available media folders"}</span>
                {listing.data?.parent && <button className="btn btn-ghost !min-h-8 !px-2" onClick={() => setBrowserDir(listing.data?.parent || null)}><ChevronLeft size={16}/> Up</button>}
                {browserDir && <button className="btn btn-ghost !min-h-8 !px-2" onClick={() => setBrowserDir(null)}>Roots</button>}
              </div>
              <div className="scrollbar max-h-[275px] overflow-y-auto rounded-xl border border-[#2d3e50] bg-[#0c1420] p-2">
                {listing.isLoading && <div className="p-4 text-xs muted">Loading folders…</div>}
                {listing.isError && <div className="p-4 text-xs text-rose-300">Unable to browse the selected directory.</div>}
                {!listing.isLoading && !listing.data?.items.length && <div className="p-4 text-xs muted">No supported media files in this folder.</div>}
                {listing.data?.items.map(item =>
                  <button key={item.path} className="media-row" data-selected={source.value === item.path}
                    onClick={() => {
                      if (item.kind === "directory") setBrowserDir(item.path);
                      else setSource({kind: item.name === "recording.contract.json" ? "contract" : "path", value: item.path});
                    }}>
                    {item.kind === "directory" ? <Folder size={17} className="shrink-0 text-sky-300"/> : <Film size={17} className="shrink-0 text-teal-300"/>}
                    <span className="flex-1 truncate text-sm">{item.name}</span>
                    {source.value === item.path ? <Check size={16} className="text-teal-300"/> : item.kind === "directory" ? <ChevronRight size={16} className="text-slate-500"/> : null}
                  </button>)}
              </div>
              <p className="mt-3 text-xs muted">Browser file paths are selected from server-approved directories. This does not expose the host filesystem to the browser.</p>
            </div>}
            {sourceMethod === "upload" && <label className="flex cursor-pointer flex-col items-center gap-3 rounded-xl border border-dashed border-[#4d6e81] bg-[#101b28] p-9 text-center hover:bg-[#192b3a]">
              <UploadCloud size={30} className="text-teal-300"/>
              <span className="text-sm font-semibold">Choose audio or video</span>
              <span className="text-xs muted">The file is copied to the local processing host.</span>
              <input type="file" className="sr-only" accept="audio/*,video/*,.mkv,.webm" disabled={upload.isPending}
                onChange={e => {const file = e.target.files?.[0]; if (file) upload.mutate(file);}}/>
              {upload.isPending && <span className="flex items-center gap-2 text-xs text-teal-300"><LoaderCircle size={14} className="animate-spin"/> Uploading…</span>}
            </label>}
            {sourceMethod === "url" && <Field title="Media URL" hint="YouTube and other public URLs supported by the installed yt-dlp downloader. No playlists.">
              <input type="url" className="control" placeholder="https://www.youtube.com/watch?v=..."
                value={source.kind === "url" ? source.value : ""}
                onChange={e => setSource({kind: "url", value: e.target.value})}/>
            </Field>}
            {source.value && <div className="mt-4 flex items-center gap-3 rounded-lg border border-teal-800/50 bg-teal-950/20 p-3">
              <CheckCircle2 size={18} className="shrink-0 text-teal-300"/>
              <div className="min-w-0"><div className="text-[11px] font-semibold uppercase tracking-wide text-teal-300">Selected source</div>
                <div className="truncate text-xs text-slate-200">{source.value}</div></div>
            </div>}
          </section>

          <section className="panel p-6">
            <SectionTitle eyebrow="02 / Objective" title="What would you like to create?" sub="The interface adjusts processing and output options to match the task."/>
            <div className="grid gap-3 md:grid-cols-3">{TASKS.map(t => <TaskCard key={t.id} task={t} active={task === t.id} onClick={() => setTask(t.id)}/>)}</div>
            <div className="mt-6 grid gap-4 sm:grid-cols-2">
              <Field title="Spoken language"><Select value={options.source_language} onChange={v => option("source_language", v)}>
                {LANGUAGES.map(([code, title]) => <option key={code} value={code}>{title}</option>)}
              </Select></Field>
              <Field title="Output language"><Select value={options.target_language} onChange={v => option("target_language", v)}>
                <option value="en">English</option>
              </Select></Field>
            </div>
          </section>
          <div className="flex justify-end">
            <button className="btn btn-primary" onClick={() => {if (!ready) setError("Select media first."); else {setError(""); setStep("configure");}}}>
              Continue to settings <ChevronRight size={16}/>
            </button>
          </div>
        </motion.div>}

        {step === "configure" && <motion.div initial={{opacity: 0, y: 8}} animate={{opacity: 1, y: 0}} className="space-y-6">
          <section className="panel p-6">
            <SectionTitle eyebrow="03 / Processing" title="Tailor the workflow" sub="Only settings relevant to the selected task are shown. Defaults are optimized for personal subtitle generation."/>
            <div className="mb-5 flex flex-wrap items-center gap-2"><Pill><Film size={12}/>{source.value.split("/").pop() || "Source not selected"}</Pill>
              <Pill><Activity size={12}/>{TASKS.find(t => t.id === task)?.title}</Pill>
              <Pill>{options.source_language.toUpperCase()} → {options.target_language.toUpperCase()}</Pill>
            </div>
            <div className="grid gap-5 sm:grid-cols-2">
              <Field title="Transcription mode"><Select value={options.asr_mode} onChange={v => option("asr_mode", v as JobOptions["asr_mode"])}>
                <option value="speaker">Speaker-attributed</option>
                <option value="multispeaker_sot">Overlapping speakers</option>
                <option value="verbatim">Verbatim</option>
                <option value="rich">Rich transcription</option>
              </Select></Field>
              <Field title="Audio stream index"><Select value={options.audio_stream} onChange={v => option("audio_stream", Number(v))}>
                {[0,1,2,3].map(v => <option key={v} value={v}>Audio stream {v}</option>)}
              </Select></Field>
              {task !== "transcription" && <>
                <Field title="Contextual refinement"><Select value={options.refinement} onChange={v => option("refinement", v as JobOptions["refinement"])}>
                  <option value="off">Off — raw transcript</option><option value="conservative">Conservative</option>
                  <option value="balanced">Balanced (recommended)</option><option value="expressive">Expressive</option>
                </Select></Field>
                <Field title="Translation engine"><Select value={options.translator} onChange={v => option("translator", v as JobOptions["translator"])}>
                  <option value="swiftu">SwiftU · contextual</option><option value="riva">Riva Translate · direct</option>
                </Select></Field>
                <Field title="Localization style"><Select value={options.style} onChange={v => option("style", v as JobOptions["style"])}>
                  <option value="natural">Natural dialogue</option><option value="literal">Closer to literal</option>
                  <option value="expressive">Expressive localization</option>
                </Select></Field>
                <Field title="Character / terminology glossary"><Select value={options.glossary || ""} onChange={v => option("glossary", v || null)}>
                  <option value="">No glossary</option>{(glossaries.data || []).map((g: Glossary) => <option value={g.path} key={g.path}>{g.name}</option>)}
                </Select></Field>
              </>}
            </div>
          </section>
          {task === "subtitles" && <section className="panel p-6">
            <SectionTitle eyebrow="04 / Presentation" title="Subtitles that fit the scene" sub="Select a layout for ordinary video or footage with burned-in source-language subtitles."/>
            <div className="grid gap-3 sm:grid-cols-2">
              {([
                ["standard", "Standard", "Classic lower subtitles, with upper lanes for overlaps."],
                ["raised_main", "Raised main", "Positions English above existing bottom hardsubs."],
              ] as const).map(([value, title, desc]) => <button key={value} onClick={() => option("ass_layout", value)}
                className={"rounded-xl border p-4 text-left " + (options.ass_layout === value ? "border-teal-500 bg-[#16343b]" : "border-[#314154] bg-[#101925] hover:border-[#537085]")}>
                <div className="mb-2 flex items-center justify-between"><MonitorPlay size={21} className="text-teal-300"/>{options.ass_layout === value && <CheckCircle2 size={16} className="text-teal-300"/>}</div>
                <div className="text-sm font-bold">{title}</div><div className="mt-1 text-xs leading-5 muted">{desc}</div>
              </button>)}
            </div>
            <div className="mt-5"><Toggle title="Condense hard-to-read English cues" detail="SwiftU rewrites unusually dense lines without requiring another ASR pass."
              value={options.condense_high_cps} onChange={v => option("condense_high_cps", v)}/></div>
          </section>}
          <section className="panel p-6">
            <button onClick={() => setAdvanced(v => !v)} className="flex w-full items-center justify-between text-left">
              <div className="flex items-center gap-3"><Settings2 size={19} className="text-teal-300"/>
                <span className="text-sm font-semibold">Advanced acoustic settings</span></div>
              <ChevronRight className={"transition-transform " + (advanced ? "rotate-90" : "")} size={18}/>
            </button>
            {advanced && <div className="mt-6 grid gap-5 border-t border-[#2b3b4c] pt-6 sm:grid-cols-2">
              <Field title="ASR window (seconds)"><input type="number" className="control" min={30} max={300} value={options.chunk_seconds}
                onChange={e => option("chunk_seconds", Number(e.target.value))}/></Field>
              <Field title="Context overlap (seconds)"><input type="number" className="control" min={0} max={30} value={options.context_seconds}
                onChange={e => option("context_seconds", Number(e.target.value))}/></Field>
              <Field title="Speaker embeddings"><Select value={options.speaker_embeddings} onChange={v => option("speaker_embeddings", v as JobOptions["speaker_embeddings"])}>
                <option value="titanet">TitaNet (CPU)</option><option value="none">Off</option>
              </Select></Field>
              <Field title="Forced alignment fallback" hint="Off is recommended for full episodes. Qwen can substantially increase CPU processing time.">
                <Select value={options.alignment_fallback} onChange={v => option("alignment_fallback", v as JobOptions["alignment_fallback"])}>
                  <option value="off">Off (native CTC)</option><option value="qwen">Qwen on CTC failure</option>
                </Select></Field>
              {task === "subtitles" && <Field title="Maximum ASS speaker lanes"><Select value={options.ass_max_lanes} onChange={v => option("ass_max_lanes", Number(v) as 2 | 3)}>
                <option value={2}>Two</option><option value={3}>Three</option>
              </Select></Field>}
            </div>}
          </section>
          <div className="flex flex-wrap justify-between gap-3">
            <button className="btn btn-secondary" onClick={() => setStep("media")}><ChevronLeft size={16}/> Back</button>
            <button className="btn btn-primary" onClick={start} disabled={!ready || submit.isPending}>
              {submit.isPending ? <LoaderCircle className="animate-spin" size={16}/> : <Play size={16}/>}
              {submit.isPending ? "Creating job…" : "Start processing"}
            </button>
          </div>
        </motion.div>}

        {step === "preview" && <motion.div initial={{opacity: 0, y: 8}} animate={{opacity: 1, y: 0}} className="space-y-6">
          <SectionTitle eyebrow="05 / Output" title="Review your result" sub="Follow processing, play the source media, and inspect synchronized transcript or subtitle cues."/>
          {!jobId && <div className="panel flex flex-col items-center justify-center gap-4 px-6 py-20 text-center">
            <MonitorPlay size={38} className="text-teal-400"/>
            <h3 className="text-lg font-semibold">Nothing to preview yet</h3>
            <p className="max-w-md text-sm muted">Start a new task or select a recent run from the sidebar.</p>
            <button className="btn btn-primary" onClick={() => setStep("media")}>Choose media</button>
          </div>}
          {jobId && <>
            <section className="panel p-5">
              <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-3">
                  {currentJob && <StatusIcon state={currentJob.state}/>}
                  <div><div className="text-sm font-semibold">{currentJob?.stage || "Loading job…"}</div>
                    <div className="mt-1 text-xs muted">Job {jobId.slice(0, 12)} · {currentJob?.request.task || "—"}</div></div>
                </div>
                <div className="flex items-center gap-2">
                  <Pill tone={currentJob?.state === "complete" ? "good" : currentJob?.state === "failed" ? "warning" : "neutral"}>
                    {currentJob?.state || "Loading"}
                  </Pill>
                  {(currentJob?.state === "running" || currentJob?.state === "queued") &&
                    <button onClick={() => cancel.mutate(jobId)} disabled={cancel.isPending} className="btn btn-secondary !min-h-8"><Square size={12}/> Cancel</button>}
                </div>
              </div>
              <div className="progress-track"><div className="progress-fill" style={{width: String(currentJob?.progress || 0) + "%"}}/></div>
              <div className="mt-2 flex justify-between text-xs muted"><span>{currentJob?.progress || 0}% recorded stage progress</span><span>{currentJob?.state === "complete" ? "Ready to review" : "Long-running tasks stay on the host"}</span></div>
              {currentJob?.error && <div className="mt-4 text-sm text-rose-300">{currentJob.error}</div>}
              {(currentJob?.state === "running" || currentJob?.state === "failed") && !!logs.data?.lines.length &&
                <details className="mt-4 rounded-lg border border-[#2c3c50] bg-[#0b131d] p-3">
                  <summary className="cursor-pointer text-xs font-semibold muted">Processing log</summary>
                  <pre className="scrollbar mt-3 max-h-40 overflow-auto whitespace-pre-wrap text-[11px] leading-5 text-slate-400">{logs.data.lines.join("\n")}</pre>
                </details>}
            </section>
            <div className="grid gap-5 xl:grid-cols-[minmax(0,1.3fr)_minmax(320px,.7fr)]">
              <section className="panel min-w-0 overflow-hidden">
                <div className="flex items-center justify-between border-b border-[#2b3c4c] px-5 py-4">
                  <span className="text-sm font-semibold">Media preview</span>
                  <Pill><Clock3 size={12}/>{secondsLabel(currentTime)}</Pill>
                </div>
                <div className="preview-pane relative aspect-video w-full overflow-hidden">
                  <video ref={videoRef} controls playsInline preload="metadata"
                    src={"/api/jobs/" + jobId + "/media"} className="h-full w-full object-contain"
                    onTimeUpdate={e => setCurrentTime(e.currentTarget.currentTime)}
                    aria-label="Media preview with synchronized subtitle text"/>
                  {active.length > 0 && <div className="pointer-events-none absolute inset-x-3 bottom-14 flex flex-col items-center gap-1 text-center">
                    {active.map((cue, i) => <div key={String(cue.utterance_id || cue.start) + "-" + i} className="max-w-[90%] rounded-md bg-black/80 px-3 py-1.5 text-sm font-semibold text-white shadow-xl md:text-base"
                      style={{textShadow: "0 1px 3px black"}}>{cue.text}</div>)}
                  </div>}
                </div>
                <div className="border-t border-[#2a3b4b] p-4">
                  <div className="mb-2 flex items-center justify-between"><span className="text-xs font-bold text-slate-200">Export results</span><span className="text-[11px] muted">Files stay on your host</span></div>
                  <div className="flex flex-wrap gap-2">
                    {artifacts.data?.map(item => <a key={item.path} href={item.url} className="btn btn-secondary !min-h-9 !px-3"><Download size={13}/>{item.name.split(".").slice(-2).join(".")}</a>)}
                    {!artifacts.data?.length && <span className="text-xs muted">Exports will appear when processing finishes.</span>}
                  </div>
                </div>
              </section>
              <section className="panel flex min-h-[470px] min-w-0 flex-col overflow-hidden">
                <div className="border-b border-[#2a3b4b] p-4">
                  <div className="mb-3 flex items-center justify-between"><span className="text-sm font-semibold">Dialogue timeline</span><Pill>{cues.length} cues</Pill></div>
                  <input value={cueSearch} onChange={e => setCueSearch(e.target.value)} className="control !min-h-9 !py-2 !text-xs" placeholder="Search dialogue…"/>
                </div>
                <div className="scrollbar max-h-[540px] flex-1 overflow-auto">
                  {filteredCues.map((cue: Cue, i: number) => {
                    const isActive = cue.start !== null && cue.end !== null && currentTime >= cue.start && currentTime < cue.end;
                    return <button className="cue-row" data-active={isActive} key={String(cue.utterance_id || "cue") + String(i) + String(cue.start)}
                      onClick={() => {if (cue.start != null && videoRef.current) {videoRef.current.currentTime = cue.start; void videoRef.current.play().catch(() => {});}}}>
                      <div className="mb-1 flex items-center justify-between text-[11px] text-slate-500"><span>{secondsLabel(cue.start)} → {secondsLabel(cue.end)}</span><span className="text-teal-400/70">{cue.speaker || ""}</span></div>
                      <div className="whitespace-pre-wrap text-sm leading-5 text-slate-200">{cue.text}</div>
                    </button>;
                  })}
                  {!filteredCues.length && <div className="p-6 text-center text-xs muted">No matching dialogue yet.</div>}
                </div>
              </section>
            </div>
            <div className="flex justify-end"><button className="btn btn-secondary" onClick={() => {setStep("media"); setJobId(null);}}>New task <ChevronRight size={15}/></button></div>
          </>}
        </motion.div>}
      </main>
    </div>
  </div>;
}
