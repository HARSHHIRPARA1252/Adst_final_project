import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "D:/BGI_Win_Probability";
const buildDir = path.join(workspaceDir, ".model_card_build");
const outputPath = path.join(workspaceDir, "reports/model_results/BGMI_Model_Report_Card.pptx");
const skillDir = process.env.SKILL_DIR;
const { resolvePresentationFont, finalizePresentation } = await import(pathToFileURL(path.join(skillDir, "container_tools/artifact_tool_utils.mjs")).href);
const font = resolvePresentationFont();
const W = 1280;
const H = 720;
const C = {
  bg: "#0B1119", panel: "#14202B", panel2: "#17252D", line: "#2A3B46",
  lime: "#C9F36A", cyan: "#64D8CB", text: "#EDF3F5", muted: "#A7B7C0",
  orange: "#FFB46B", dark: "#091017", pale: "#D7E3E7"
};

function addRect(slide, x, y, width, height, fill, radius = 14, stroke = "none") {
  return slide.shapes.add({
    geometry: "roundRect",
    position: { left: x, top: y, width, height },
    fill,
    line: { fill: stroke, width: stroke === "none" ? 0 : 1 },
    borderRadius: radius
  });
}
function addText(slide, text, x, y, width, height, size, color = C.text, options = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width, height },
    fill: "none",
    line: { fill: "none", width: 0 }
  });
  shape.text = text;
  shape.text.style = {
    typeface: font,
    fontSize: size,
    color,
    bold: Boolean(options.bold),
    alignment: options.align || "left",
    verticalAlignment: options.valign || "middle",
    autoFit: "shrinkText",
    wrap: "square",
    insets: { left: 0, right: 0, top: 0, bottom: 0 }
  };
  return shape;
}
function addLine(slide, y, color = C.line) {
  return slide.shapes.add({
    geometry: "rect",
    position: { left: 72, top: y, width: 1136, height: 1.2 },
    fill: color,
    line: { fill: color, width: 0 }
  });
}
function addFooter(slide, sourceText, page) {
  addLine(slide, 683, "#20303A");
  addText(slide, sourceText, 74, 687, 1020, 20, 14, "#7E929D");
  addText(slide, String(page).padStart(2, "0"), 1147, 685, 55, 24, 16, C.lime, { bold: true, align: "right" });
}
function styleTable(table, values, options = {}) {
  const rows = values.length;
  const cols = values[0].length;
  for (let r = 0; r < rows; r++) {
    if (table.rows?.[r]) table.rows[r].height = options.rowHeight || 46;
    for (let c = 0; c < cols; c++) {
      const cell = table.getCell(r, c);
      cell.fill = r === 0 ? "#20313B" : (r % 2 === 0 ? "#15222C" : "#111C25");
      cell.text.style = {
        typeface: font,
        fontSize: r === 0 ? 17 : (options.bodySize || 19),
        color: r === 0 ? C.lime : (c === 1 ? C.text : C.pale),
        bold: r === 0 || c === 1,
        autoFit: "shrinkText",
        wrap: "square",
        verticalAlignment: "middle"
      };
    }
  }
  table.borders.assign({ style: "solid", fill: C.line, width: 1 });
}
function addStat(slide, x, y, width, label, value, note) {
  addRect(slide, x, y, width, 82, C.panel, 12, C.line);
  addText(slide, label.toUpperCase(), x + 16, y + 9, width - 32, 18, 14, C.muted, { bold: true });
  addText(slide, value, x + 16, y + 27, width - 32, 34, 29, C.lime, { bold: true });
  addText(slide, note, x + 16, y + 60, width - 32, 16, 13, C.muted);
}

const presentation = Presentation.create({ slideSize: { width: W, height: H } });

// Slide 1: primary team win model report card.
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  addText(slide, "MODEL REPORT CARD  ·  BGMI ESPORTS ANALYTICS", 74, 29, 740, 26, 16, C.lime, { bold: true });
  addText(slide, "Team match win model", 72, 58, 880, 60, 48, C.text, { bold: true });
  addText(slide, "One-match WWCD probability · chronological test · 2 October 2026", 75, 119, 1090, 32, 21, C.muted);
  addRect(slide, 956, 68, 250, 38, "#332719", 18, "#70502F");
  addText(slide, "PRELIMINARY EVIDENCE", 970, 70, 223, 32, 15, C.orange, { bold: true, align: "center" });
  addLine(slide, 164);
  addStat(slide, 74, 178, 354, "Historical sample", "102 matches", "1,632 team-match rows");
  addStat(slide, 463, 178, 354, "Chronological holdout", "6 matches", "later matches · 96 team rows");
  addStat(slide, 852, 178, 354, "Winner ranked first", "1 of 6", "16.7% top-one hit rate");
  addText(slide, "Random Forest · sigmoid calibrated · target = match winner (WWCD)", 76, 270, 1110, 27, 18, C.cyan, { bold: true });
  const values = [
    ["Measure", "Held-out result", "Review"],
    ["Binary accuracy", "89.6%", "An always-loss rule scores 93.75%; accuracy alone misleads."],
    ["Top-one accuracy", "16.7% · 1/6", "The model ranked the actual winner first once."],
    ["ROC-AUC", "0.5444", "Weak ranking signal; close to 0.5 on six lobbies."],
    ["Conditional match log loss", "2.7640", "Selection metric; this six-match estimate is unstable."],
    ["Brier score", "0.05854", "Compare with a baseline before trusting probability quality."]
  ];
  const table = slide.tables.add({
    rows: values.length, columns: 3, left: 74, top: 300, width: 1132, height: 258,
    columnTracks: [{ mode: "fixed", value: 255 }, { mode: "fixed", value: 205 }, { mode: "fr", value: 1 }],
    values
  });
  styleTable(table, values, { rowHeight: 43, bodySize: 18 });
  addText(slide, "Most influential features: points variability 17.1% · finishes variability 14.0% · prior average points 12.5%. Importance describes model behavior; it does not prove cause.", 76, 565, 1120, 39, 17, C.muted);
  addRect(slide, 74, 611, 1132, 54, "#272218", 11, "#5C462D");
  addText(slide, "Review: experimental only. Six test matches are too few to claim dependable win probabilities.", 91, 617, 1095, 42, 20, C.orange, { bold: true });
  addFooter(slide, "Source: project model_comparison.csv and model_metadata.json; final six complete matches held out.", 1);
  slide.speakerNotes.textFrame.setText("Primary source: D:/BGI_Win_Probability/reports/model_results/model_comparison.csv and model_metadata.json. The Random Forest was selected by the lowest held-out conditional match log loss; calibration used expanding chronological match-group folds. The test contains six later matches (96 team rows), so the result is preliminary. The always-loss accuracy baseline is 90/96 = 93.75%, based on six winning team rows among 96 team rows. Feature importance is descriptive and not causal.");
}

// Slide 2: player finishes model report card and label boundary.
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  addText(slide, "SECOND MODEL  ·  PLAYER PERFORMANCE", 74, 31, 720, 24, 16, C.cyan, { bold: true });
  addText(slide, "Player finishes forecast", 72, 60, 920, 60, 48, C.text, { bold: true });
  addText(slide, "An MVP-style finish estimate, not an official MVP-award probability", 75, 122, 1100, 34, 21, C.muted);
  addLine(slide, 173);
  addStat(slide, 74, 191, 535, "Chronological holdout", "6 matches", "384 player-match rows");
  addStat(slide, 673, 191, 533, "Selected method", "Random Forest", "chosen by lower holdout MAE");
  addText(slide, "Held-out model comparison", 76, 299, 1120, 30, 21, C.lime, { bold: true });
  const values = [
    ["Method", "MAE ↓", "RMSE ↓", "R² ↑"],
    ["Random Forest regressor", "1.092", "1.436", "−0.005"],
    ["Rolling-average baseline", "1.107", "1.424", "0.011"]
  ];
  const table = slide.tables.add({
    rows: values.length, columns: 4, left: 74, top: 336, width: 1132, height: 174,
    columnTracks: [{ mode: "fixed", value: 420 }, { mode: "fixed", value: 220 }, { mode: "fixed", value: 220 }, { mode: "fr", value: 1 }],
    values
  });
  styleTable(table, values, { rowHeight: 58, bodySize: 22 });
  addRect(slide, 74, 529, 1132, 65, C.panel, 12, C.line);
  addText(slide, "Mixed evidence: MAE improves by just 0.015 finishes per player-match, while RMSE and R² are worse than the baseline.", 95, 537, 1090, 49, 20, C.pale, { bold: true });
  addRect(slide, 74, 607, 1132, 58, "#272218", 11, "#5C462D");
  addText(slide, "Use the output as an estimated finish count / likely finish leader. No validated MVP-award labels exist in the dataset.", 94, 612, 1090, 47, 19, C.orange, { bold: true });
  addFooter(slide, "Source: project player_finish_model_comparison.csv and player_finish_model_metadata.json.", 2);
  slide.speakerNotes.textFrame.setText("Primary source: D:/BGI_Win_Probability/reports/model_results/player_finish_model_comparison.csv and player_finish_model_metadata.json. The test covers six chronological matches and 384 player-match rows. Random Forest MAE is 1.091915 and rolling-average MAE is 1.106989, a difference of 0.015073 finishes per player-match. Random Forest RMSE (1.435741) and R-squared (-0.004730) are worse than the rolling-average baseline (1.424499 and 0.010944). The source does not supply match-level MVP-award labels. This model predicts finishes only.");
}

// Slide 3: practical improvement steps and Jupyter answer.
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  addText(slide, "MODEL DEVELOPMENT", 74, 31, 720, 24, 16, C.lime, { bold: true });
  addText(slide, "What will improve the evidence", 72, 60, 1080, 60, 48, C.text, { bold: true });
  addText(slide, "More trustworthy validation matters more than adding model complexity.", 75, 122, 1120, 34, 21, C.muted);
  addLine(slide, 172);
  const items = [
    ["01", "Expand verified match history", "Add complete official scorecards; check match order, team names, and exactly one WWCD winner per match."],
    ["02", "Test across more future blocks", "Use several rolling chronological holdouts and unseen-tournament checks; keep every match group intact."],
    ["03", "Add only source-backed features", "Use stage, map, roster, and player form only when the source provides reliable values. Do not invent damage or survival data."],
    ["04", "Compare against simple baselines", "Track per-lobby log loss, Brier score, calibration, and winner-ranked-first rate; report uncertainty from the small sample."]
  ];
  let y = 190;
  for (const [number, title, description] of items) {
    addRect(slide, 74, y, 1132, 78, C.panel, 12, C.line);
    addRect(slide, 89, y + 17, 45, 44, "#26351E", 10, "#536938");
    addText(slide, number, 89, y + 17, 45, 44, 18, C.lime, { bold: true, align: "center" });
    addText(slide, title, 153, y + 9, 1024, 27, 22, C.text, { bold: true });
    addText(slide, description, 153, y + 36, 1020, 34, 18, C.muted);
    y += 88;
  }
  addRect(slide, 74, 548, 1132, 119, "#172824", 12, "#3C6056");
  addText(slide, "Jupyter Notebook: yes", 94, 559, 1090, 28, 22, C.lime, { bold: true });
  addText(slide, "Use notebooks/BGMI_Winning_Probability_Analysis.ipynb for the dataset walkthrough, cleaning, EDA, leakage-safe features, chronological evaluation, calibration, and saved artifacts. Flask loads the saved model; it does not retrain when the site starts.", 94, 589, 1090, 64, 18, C.pale);
  addFooter(slide, "Notebook and training workflow: project README.md; validation recommendations use the observed holdout limits.", 3);
  slide.speakerNotes.textFrame.setText("The Jupyter notebook exists at D:/BGI_Win_Probability/notebooks/BGMI_Winning_Probability_Analysis.ipynb and contains the analysis/training workflow. The standalone script src/model_training.py can retrain and save the model outside Flask startup. Recommendations are based on the current six-match team holdout and six-match player holdout; features should only be added when actual tournament sources contain them.");
}

await fs.mkdir(buildDir, { recursive: true });
const draft = path.join(buildDir, "model_report_card_draft.pptx");
await (await PresentationFile.exportPptx(presentation)).save(draft);
for (let i = 0; i < presentation.slides.items.length; i++) {
  const slide = presentation.slides.items[i];
  const image = await presentation.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(path.join(buildDir, `slide-${i + 1}.png`), new Uint8Array(await image.arrayBuffer()));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(buildDir, `slide-${i + 1}.layout.json`), await layout.text());
}

const result = await finalizePresentation({
  explicitTotalSlideCount: 3,
  requiredNativeTableOwnerSlides: [1, 2],
  requiredNativeChartOwnerSlides: [],
  workspaceDir,
  candidatePath: draft,
  finalPath: outputPath,
  pythonExecutable: process.env.RUNTIME_PYTHON,
  integrityValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-bullet-geometry", "--validate-heading-fit", "--require-native-table-slide", "1", "--require-native-table-slide", "2"],
  fontPolicy: { basis: "design", families: [font] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(buildDir, "model_report_card.validation.json")
});
console.log(JSON.stringify({ font, draft, outputPath, result }, null, 2));
