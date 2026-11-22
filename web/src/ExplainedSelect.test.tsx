import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { ExplainedSelect } from "./ExplainedSelect";

const entries = [
  {value: "speaker", label: "Speakers", description: "Speaker-tagged dialogue and timing."},
  {value: "verbatim", label: "Verbatim", description: "Literal words and filler."},
];

describe("ExplainedSelect accessibility contract", () => {
  it("exposes a labeled combobox trigger with the committed description", () => {
    const html = renderToStaticMarkup(
      <ExplainedSelect label="Transcription mode" value="speaker"
        onChange={() => { /* no interactions during server rendering */ }}
        items={entries}/>
    );
    expect(html).toContain('role="combobox"');
    expect(html).toContain('aria-label="Transcription mode"');
    expect(html).toContain('aria-haspopup="listbox"');
    expect(html).toContain('aria-expanded="false"');
    expect(html).toContain('aria-controls=');
    expect(html).toContain("Speaker-tagged dialogue and timing.");
  });
});
