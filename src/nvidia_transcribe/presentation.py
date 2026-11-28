"""CPU-only, sampled picture observations and libass-measured placement.

Coordinates in observation documents are normalized, half-open frame regions.
Samples are not proof of caption absence or scene boundaries. No OCR, embedded
subtitle stream, or language inference is used by this module.
"""
from __future__ import annotations

import ctypes as C
import ctypes.util
import json
import hashlib
import math
import subprocess
from dataclasses import replace
from pathlib import Path

from nvidia_transcribe.media import _bounded_file_fingerprint, compute_file_sha256, visual_stream_index

SCHEMA = "nemovox-presentation-observations/v1"


def _number(x):
    return type(x) in (int, float) and math.isfinite(x)


def _region(x):
    return (isinstance(x, (tuple, list)) and len(x) == 4
            and all(_number(v) and 0 <= v <= 1 for v in x)
            and x[0] < x[2] and x[1] < x[3])


def admit_observations(doc, source):
    """Fail closed on wrong source or malformed observations, not on absence."""
    if doc is None:
        return None
    if (not isinstance(doc, dict) or doc.get("schema") != SCHEMA
            or not source.get("fingerprint_sha256")
            or doc.get("media_fingerprint_sha256") != source["fingerprint_sha256"]
            or doc.get("media_content_sha256") != source.get("content_sha256")
            or not _number(source.get("duration_s"))
            or doc.get("duration_s") != source["duration_s"]):
        raise ValueError("Presentation observation source identity mismatch")
    size = doc.get("frame_size")
    if size is not None and (not isinstance(size, list) or len(size) != 2
            or any(type(v) is not int or not 1 <= v <= 16384 for v in size)):
        raise ValueError("Invalid presentation frame dimensions")
    picture = doc.get("active_picture")
    if picture is not None and not _region(picture):
        raise ValueError("Invalid active-picture region")
    samples = doc.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Invalid presentation samples")
    for sample in samples:
        if (not isinstance(sample, dict) or not _number(sample.get("time_s"))
                or not 0 <= sample["time_s"] <= doc["duration_s"]
                or (sample.get("active_picture") is not None and not _region(sample["active_picture"]))
                or not isinstance(sample.get("caption_candidates"), list)
                or not all(_region(b) for b in sample["caption_candidates"])):
            raise ValueError("Invalid presentation sample")
    # Optional independently acquired interval-bound region evidence is shared
    # with subsequent semantic inspection. A sampled empty candidate list is
    # NOT such evidence, and no literal target-language text is admitted here.
    confirmed = doc.get("caption_observations", [])
    if not isinstance(confirmed,list):
        raise ValueError("Invalid caption observations")
    for item in confirmed:
        if not isinstance(item,dict):
            raise ValueError("Invalid caption observation")
        span,regions = item.get("interval_s"),item.get("regions")
        evidence = item.get("evidence_sha256")
        if (not isinstance(span,list) or len(span)!=2 or not all(_number(v) for v in span)
                or not 0 <= span[0] < span[1] <= doc["duration_s"]
                or item.get("status") not in ("present","absent","unknown")
                or not isinstance(regions,list) or not all(_region(b) for b in regions)
                or (item["status"]=="present") != bool(regions)
                or not isinstance(evidence,str) or len(evidence)!=64
                or any(ch not in "0123456789abcdef" for ch in evidence)
                or not isinstance(item.get("method"),str) or not item["method"].strip()):
            raise ValueError("Invalid caption observation")
    cuts = doc.get("known_scene_boundaries_s", [])
    if (not isinstance(cuts, list) or any(not _number(t) or not 0 < t < doc["duration_s"] for t in cuts)
            or cuts != sorted(set(cuts))):
        raise ValueError("Invalid known scene boundaries")
    return doc


def observe_frame(raw: bytes, width: int, height: int) -> dict:
    """Near-black border consensus candidate and lower-frame bright stroke evidence.

    This intentionally does NOT label bright graphics as confirmed subtitles.
    Black frames are indeterminate. Captions can interrupt the dark border;
    a narrow centered bright region is retained separately as a candidate.
    """
    if len(raw) != width * height or width < 32 or height < 32:
        raise ValueError("Invalid grayscale frame")
    rows = [raw[y * width:(y + 1) * width] for y in range(height)]
    # Ignore central glyphs for detecting horizontal bars, but require dark
    # outer strips on both sides. Full dark frames cannot establish geometry.
    strip = width // 5
    def dark(row):
        edge = row[:strip] + row[-strip:]
        return sum(v <= 24 for v in edge) >= .98 * len(edge)
    top = 0
    while top < height and dark(rows[top]):
        top += 1
    bottom = height
    while bottom > top and dark(rows[bottom-1]):
        bottom -= 1
    picture = None
    if bottom - top >= .45 * height and sum(not dark(row) for row in rows) >= .35 * height:
        picture = [0., top/height, 1., bottom/height]
    # Bright, high-contrast adjacent horizontal strokes, away from outer
    # edges. Candidate only; false positives reserve space, not change text.
    eligible = []
    for y in range(int(height * .68), height - 1):
        row = rows[y]
        xs = [x for x in range(width//10, width-width//10)
              if row[x] >= 190 and min(row[x-1], row[x+1], rows[y-1][x], rows[y+1][x]) <= 85]
        if len(xs) >= max(4, width//55) and max(xs)-min(xs) >= width*.09:
            eligible.append((y, min(xs), max(xs)))
    bands = []
    for y, x0, x1 in eligible:
        if bands and y <= bands[-1][3] + 3:
            b = bands[-1]
            b[0], b[2], b[3] = min(x0,b[0]), max(x1,b[2]), y
        else:
            bands.append([x0,y,x1,y])
    candidates = [[max(0,(x0-3)/width), max(0,(y0-3)/height),
                   min(1,(x1+4)/width), min(1,(y1+4)/height)]
                  for x0,y0,x1,y1 in bands if 2 <= y1-y0 <= height*.12]
    return {"active_picture": picture, "caption_candidates": candidates}


def verify_media_source(path, source):
    path = Path(path)
    if (not source.get("fingerprint_sha256")
            or _bounded_file_fingerprint(path) != source["fingerprint_sha256"]
            or (source.get("content_sha256") and compute_file_sha256(path) != source["content_sha256"])):
        raise ValueError("Media does not match presentation observation source")


def observe_media(path: str | Path, source: dict, *, max_samples: int = 24) -> dict:
    """Bounded, explicit video-only acquisition from an admitted media path."""
    path = Path(path)
    fingerprint = source.get("fingerprint_sha256")
    duration = source.get("duration_s")
    if not _number(duration) or duration <= 0:
        raise ValueError("Invalid presentation observation duration")
    verify_media_source(path,source)
    doc = {"schema": SCHEMA, "media_fingerprint_sha256": fingerprint,
           "media_content_sha256": source.get("content_sha256"),
           "duration_s": duration, "frame_size": None, "active_picture": None,
           "samples": [], "caption_observations": [], "known_scene_boundaries_s": [],
           "sample_time_basis": "requested_seek_not_exact_pts",
           "method": "bounded-grayscale-edge-and-bright-strokes/v1",
           "status": "unknown", "caption_status": "unknown"}
    try:
        stream = visual_stream_index(path)
        if stream is None:
            doc["status"] = "no_video"
            verify_media_source(path,source)
            return admit_observations(doc,source)
        probe = json.loads(subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "V:0",
             "-show_entries", "stream=index,width,height,sample_aspect_ratio:stream_side_data=rotation",
             "-of", "json", str(path)], timeout=30))
        info = probe["streams"][0]
        if info["index"] != stream:
            raise ValueError("Video stream changed")
        w,h = int(info["width"]), int(info["height"])
        sar = info.get("sample_aspect_ratio", "1:1")
        if sar not in ("1:1", "N/A", "0:1"):
            a,b = map(int,sar.split(":"))
            w = round(w*a/b)
        if any(int(x.get("rotation",0)) % 180 for x in info.get("side_data_list", [])):
            w,h = h,w
        if not (32 <= w <= 16384 and 32 <= h <= 16384):
            raise ValueError("Unsupported display dimensions")
        doc["frame_size"], doc["video_stream_index"] = [w,h], stream
        sw,sh = 480, max(32, round(480*h/w))
        n = min(24,max(1,int(max_samples)),max(1,math.ceil(duration/2)))
        for i in range(n):
            t = round(duration*(i+.5)/n, 3)
            try:
                raw = subprocess.check_output(
                    ["ffmpeg","-v","error","-ss",str(t),"-i",str(path),
                     "-map",f"0:{stream}","-an","-sn","-dn","-frames:v","1",
                     "-vf",f"scale={sw}:{sh},setsar=1,format=gray",
                     "-f","rawvideo","-pix_fmt","gray","-"], timeout=20)
                observation = observe_frame(raw,sw,sh)
                doc["samples"].append({"time_s":t, "grayscale_frame_sha256":hashlib.sha256(raw).hexdigest(), **observation})
            except (OSError,subprocess.SubprocessError,ValueError):
                continue
        pictures = [s["active_picture"] for s in doc["samples"] if s["active_picture"] is not None]
        if pictures:
            # Ignore isolated dark scenes, not a competing framing regime.
            clusters = [[p for p in pictures if abs(p[1]-anchor[1]) <= .012
                         and abs(p[3]-anchor[3]) <= .012] for anchor in pictures]
            stable = max(clusters,key=len)
            if len(stable) >= max(3,math.ceil(.75*n)):
                doc["active_picture"] = [0.,min(p[1] for p in stable),1.,max(p[3] for p in stable)]
                doc["geometry_support_samples"] = len(stable)
        doc["requested_sample_count"] = n
        doc["status"] = "sampled" if doc["samples"] else "unknown"
        doc["caption_status"] = ("candidate_regions_observed" if any(s["caption_candidates"]
                                for s in doc["samples"]) else "unknown")
    except (OSError,subprocess.SubprocessError,ValueError,KeyError,IndexError,TypeError,ZeroDivisionError,RuntimeError):
        doc["status"] = "inspection_unavailable"
    verify_media_source(path,source)
    return admit_observations(doc,source)


def adaptive_config(cfg, doc):
    if doc and doc["frame_size"]:
        w,h = doc["frame_size"]
        # Preserve relative font height while matching actual display aspect.
        return replace(cfg,play_res=(max(1,round(cfg.play_res[1]*w/h)),cfg.play_res[1]))
    return cfg


def constrain_scene_padding(cues, doc):
    """Only clamp discretionary padding at positively known scene boundaries."""
    for c in cues:
        if c.unrenderable or c.raw_end is None:
            continue
        for t in (doc or {}).get("known_scene_boundaries_s", []):
            if c.raw_end <= t < c.end and t > c.start:
                # Floor to emitted precision, without cutting source timing.
                limit = math.floor(t*100)/100
                if limit >= c.raw_end and limit > c.start:
                    c.end = limit
                    c.conflicts.append("display_clamped_at_known_scene_boundary")
                break


class _Image(C.Structure):
    pass


_Image._fields_ = [("w",C.c_int),("h",C.c_int),("stride",C.c_int),
                   ("bitmap",C.POINTER(C.c_ubyte)),("color",C.c_uint32),
                   ("dst_x",C.c_int),("dst_y",C.c_int),("next",C.POINTER(_Image))]


def measure_blocks(cues, cfg):
    """Measure the exact wrapped ASS events in one in-memory native libass track.

    Libass image rectangles conservatively enclose the rasterized glyphs,
    border and shadow. Measurement and output use the same styles and position
    override. No temporary subtitle file, model, or media access is involved.
    """
    from nvidia_transcribe.subtitles import format_ass
    library = ctypes.util.find_library("ass")
    if not library:
        raise RuntimeError("libass unavailable")
    lib = C.CDLL(library)
    P,I = C.c_void_p,C.c_int
    def bind(name, result, args):
        fn = getattr(lib,name)
        fn.restype,fn.argtypes = result,args
        return fn
    init = bind("ass_library_init",P,[])
    finish = bind("ass_library_done",None,[P])
    renderer_init = bind("ass_renderer_init",P,[P])
    renderer_done = bind("ass_renderer_done",None,[P])
    frame = bind("ass_set_frame_size",None,[P,I,I])
    storage = bind("ass_set_storage_size",None,[P,I,I])
    fonts = bind("ass_set_fonts",None,[P,C.c_char_p,C.c_char_p,I,C.c_char_p,I])
    read = bind("ass_read_memory",P,[P,C.c_char_p,C.c_size_t,C.c_char_p])
    free = bind("ass_free_track",None,[P])
    render = bind("ass_render_frame",C.POINTER(_Image),[P,P,C.c_longlong,C.POINTER(I)])
    version = bind("ass_library_version",I,[])
    width,height = cfg.play_res
    x,y = width//2,height//2
    # Identical text/style geometry; synthetic timestamps are measurement only.
    probes = [replace(c,start=i*2.,end=i*2.+1.,lane=0,ass_position=(x,y))
              for i,c in enumerate(cues)]
    data = format_ass(probes,config=cfg).encode("utf-8")
    context = init()
    if not context:
        raise RuntimeError("libass initialization failed")
    renderer = track = None
    try:
        renderer = renderer_init(context)
        if not renderer:
            raise RuntimeError("libass renderer initialization failed")
        frame(renderer,width,height)
        storage(renderer,width,height)
        fonts(renderer,None,cfg.font_name.encode("utf-8"),1,None,1)
        track = read(context,data,len(data),None)
        if not track:
            raise RuntimeError("libass measurement track failed")
        boxes = {}
        for i,c in enumerate(cues):
            image = render(renderer,track,i*2000+500,None)
            rects = []
            while image:
                im = image.contents
                if im.w > 0 and im.h > 0 and (im.color & 255) != 255:
                    rects.append((im.dst_x-x,im.dst_y-y,im.dst_x+im.w-x,im.dst_y+im.h-y))
                image = im.next
            if rects:
                boxes[c.display_order] = (min(b[0] for b in rects),min(b[1] for b in rects),
                                          max(b[2] for b in rects),max(b[3] for b in rects))
        return boxes,version()
    finally:
        if track:
            free(track)
        if renderer:
            renderer_done(renderer)
        finish(context)


def _components(cues):
    group = []
    end = -1.
    for c in sorted(cues,key=lambda c:(c.start,c.display_order)):
        if group and c.start >= end:
            yield group
            group = []
        end = max(end,c.end) if group else c.end
        group.append(c)
    if group:
        yield group


def _regions(cfg, doc, start, end):
    """Return vertical regions, preferring bottom and top letterbox space.

    The union of sampled bright-stroke candidates is reserved conservatively.
    Unobserved source captions are not asserted absent. Stable letterbox bars
    permit a distinct policy from full-frame or unknown geometry.
    """
    _,h = cfg.play_res
    gap = max(8,round(h*.012))
    picture = (doc or {}).get("active_picture")
    bottom_bar = bool(picture and picture[3] <= .94)
    top_bar = bool(picture and picture[1] >= .06)
    # A source-bound absence interval covering the whole occupancy group can
    # release the lower reserve. Conflicting present/unknown evidence defeats
    # that release. Automatically sampled emptiness never does so.
    applicable = [o for o in (doc or {}).get("caption_observations",[])
                  if o["interval_s"][0] < end and o["interval_s"][1] > start]
    absent = (any(o["status"]=="absent" and o["interval_s"][0]<=start
                  and o["interval_s"][1]>=end for o in applicable)
              and all(o["status"]=="absent" for o in applicable))
    floor = h-gap if bottom_bar or absent else int(h*.84)
    candidates = [] if absent else [b for sample in (doc or {}).get("samples",[])
                                     for b in sample["caption_candidates"]]
    candidates.extend(b for o in applicable for b in o["regions"])
    bands = [(max(gap,math.floor(b[1]*h)-gap),
              min(h-gap,math.ceil(b[3]*h)+gap)) for b in candidates]
    def subtract(lo,hi):
        free = [(lo,hi)]
        for a,b in bands:
            if b <= a:
                continue
            free = [(x,y) for l,u in free
                    for x,y in ((l,min(u,a)),(max(l,b),u))
                    if x < y] if free else []
        return sorted(free,key=lambda p:p[1],reverse=True)
    preferred = []
    if bottom_bar:
        preferred.extend(subtract(math.ceil(picture[3]*h)+gap,h-gap))
    if top_bar:
        preferred.extend(subtract(gap,math.floor(picture[1]*h)-gap))
    preferred.extend(subtract(gap,floor))
    return list(dict.fromkeys(preferred)), picture


def place_ass(cues,cfg,doc):
    """Place connected occupancy groups in fixed positions and chronological order.

    Overlapping events impose vertical precedence constraints. If the group
    cannot fit, trim only discretionary padding and solve again; never shift
    acoustic onsets or discard text silently. Unfit measured blocks are
    explicitly overflowed (SRT is independent). No mid-event motion.
    """
    from nvidia_transcribe.subtitles import _acoustically_concurrent
    report = {"layout":"auto","status":"measured","observation_status":(doc or {}).get("status","unknown"),
              "caption_status":(doc or {}).get("caption_status","unknown"),
              "active_picture":(doc or {}).get("active_picture"),"play_res":list(cfg.play_res),
              "region_policy":"stable_sampled_union_or_conservative_lower_reserve",
              "confirmed_caption_observation_count":len((doc or {}).get("caption_observations",[])),
              "scene_boundary_count":len((doc or {}).get("known_scene_boundaries_s",[]))}
    if not cues:
        return [],[],report
    try:
        boxes,version = measure_blocks(cues,cfg)
    except (OSError,RuntimeError,AttributeError,ValueError):
        for c in cues:
            c.conflicts.append("placement_measurement_unavailable")
        report["status"] = "measurement_unavailable"
        return list(cues),[],report
    report["libass_version"] = version
    report["font_name"] = cfg.font_name
    width,height = cfg.play_res
    center,gap = width//2, max(8,round(height*.012))
    picture = (doc or {}).get("active_picture")
    accepted,overflow = [],[]
    usable = []
    for c in cues:
        b = boxes.get(c.display_order)
        if b is None or center+b[0] <= 1 or center+b[2] >= width-1 or b[1] <= -height//2+1 or b[3] >= height//2-1:
            c.overflow = True
            c.conflicts.append("placement_unmeasurable_or_width_exceeded")
            overflow.append(c)
        else:
            usable.append(c)
    def solve(group, floor):
        positions = {}
        # Newer overlapping events below older ones. Disjoint events can
        # reuse position. Reverse chronological order gives latest the floor.
        for c in reversed(group):
            b = boxes[c.display_order]
            next_top = min((positions[n.display_order]+boxes[n.display_order][1]-gap
                            for n in group if n.display_order in positions
                            and n.start < c.end and n.end > c.start),default=floor)
            positions[c.display_order] = min(floor,next_top)-b[3]
        return positions
    def find_positions(group):
        regions,_ = _regions(cfg,doc,min(c.start for c in group),max(c.end for c in group))
        for lo,hi in regions:
            pos = solve(group,hi)
            if all(pos[c.display_order]+boxes[c.display_order][1] >= lo for c in group):
                return pos
        return None
    for group in _components(usable):
        positions = find_positions(group)
        if positions is None:
            # Keep real concurrency, sacrifice only sequential tail padding.
            for i,c in enumerate(group):
                for n in group[i+1:]:
                    if n.start >= c.end:
                        break
                    if (c.start < n.start and not _acoustically_concurrent(c,n,cfg.overlap_threshold_s)
                            and c.raw_end is not None and n.start >= c.raw_end):
                        c.end = n.start
                        c.conflicts.append("placement_clamped_discretionary_padding")
                        break
            positions = find_positions(group)
        if positions is None:
            positions = solve(group,height-gap)
            for c in group:
                c.conflicts.append("placement_caption_region_conflict")
        for c in group:
            b = boxes[c.display_order]
            y = positions[c.display_order]
            if y+b[1] < gap:
                c.overflow = True
                c.conflicts.append("placement_region_saturation")
                overflow.append(c)
                continue
            c.ass_position = (center,y)
            c.ass_bounds = (center+b[0],y+b[1],center+b[2],y+b[3])
            if (picture and (picture[1] >= .06 or picture[3] <= .94)
                    and c.ass_bounds[1] < math.ceil(picture[3]*height)
                    and c.ass_bounds[3] > math.floor(picture[1]*height)):
                c.conflicts.append("placement_active_picture_used")
            if len(c.text)/(c.end-c.start) > cfg.target_cps+4:
                c.conflicts = [v for v in c.conflicts if not v.startswith("high_reading_speed:")]
                c.conflicts.append(f"high_reading_speed:{len(c.text)/(c.end-c.start):.1f}cps")
            accepted.append(c)
    # Reconcile pressure flags after final placement-driven trimming.
    for c in accepted:
        c.conflicts = [v for v in c.conflicts if v != "sequential_display_overlap"
                       and v != "simultaneous_display_pressure"
                       and not v.startswith("high_reading_speed:")]
        prior = [p for p in accepted if p.display_order < c.display_order
                 and p.start < c.end and p.end > c.start]
        if any(not _acoustically_concurrent(p,c,cfg.overlap_threshold_s) for p in prior):
            c.conflicts.append("sequential_display_overlap")
        pool = [c,*prior]
        if prior and (len(pool)>2 or sum(len(p.text) for p in pool)>100
                      or sum(len(p.text)/(p.end-p.start) for p in pool)>cfg.target_cps*1.8):
            c.conflicts.append("simultaneous_display_pressure")
        cps = len(c.text)/(c.end-c.start)
        if cps>cfg.target_cps+4:
            c.conflicts.append(f"high_reading_speed:{cps:.1f}cps")
    report["placed_cues"],report["overflow_cues"] = len(accepted),len(overflow)
    return accepted,overflow,report
