"""LUMEN architecture diagrams as draw.io files (one page per figure + a combined file)."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from dio import Diagram, PAL, MUTED, INK, LINE, write_drawio  # noqa: E402

M3 = "M<sup>3</sup>"
NPFE = "npFe<sup>0</sup>"
MPFE = "mpFe<sup>0</sup>"


# =========================================================================== 1
def framework():
    d = Diagram("1 Framework", 1660, 790)
    # ---- data column
    d.group(20, 50, 340, 700, "DATA", kind="data")
    relab = d.card(45, 85, 290, 130, "RELAB end-members",
                   "49 lunar mineral and glass spectra", "none from an evaluation soil", "data")
    lscc = d.card(45, 350, 290, 130, "LSCC soils",
                  "19 soils · 75 RELAB spectra", "modal mineralogy, oxides, I<sub>s</sub>/FeO", "data")
    m3 = d.card(45, 590, 290, 130, f"{M3} at returned-sample sites",
                "49 sites · laboratory FeO", "Qiu et al. (2025)", "data")

    # ---- stage 1
    d.group(400, 50, 770, 200, "PHYSICS SIMULATION &amp; PRETRAINING", badge=1, kind="phys")
    sim = d.card(430, 85, 300, 130, "Physics simulator",
                 "Hapke + space weathering<br>domain randomisation",
                 "400,000 labelled spectra (Fig. 2)", "phys")
    pre = d.card(835, 85, 300, 130, "Pretraining",
                 "1-D CNN · 5-member ensemble",
                 f"composition, oxides, I<sub>s</sub>/FeO<br>+ grain size, {NPFE} (Fig. 3)", "learn")

    # ---- stage 2
    d.group(800, 300, 370, 450, "TRANSFER", badge=2, kind="learn")
    ft = d.card(825, 335, 320, 160, "Fine-tuning on returned soils",
                "per cross-validation fold",
                "real spectra (A2)<br>or + SSA-mixup and synthetic replay (A4)", "learn")
    orb = d.card(825, 590, 320, 130, "Orbital fine-tuning",
                 f"{M3} pixels of training sites",
                 "FeO only · mixed 1:1 with lab spectra", "learn")

    # ---- stage 3
    d.group(1210, 50, 430, 700, "CALIBRATED REPORTING", badge=3, kind="rep")
    gate = d.card(1245, 85, 330, 130, "Identifiability gate",
                  "skill <i>S</i> = 1 − MAE / MAE<sub>ref</sub>",
                  "map only if the 95 % CI of <i>S</i> &gt; 0", "rep")
    unc = d.card(1245, 335, 330, 160, "Uncertainty",
                 "5-member deep ensemble<br>normalised conformal (CV+)",
                 "68 / 90 / 95 % intervals", "rep")
    out = d.card(1245, 590, 330, 130, "Outputs",
                 "mode, oxides, maturity<br>with intervals",
                 "unidentifiable targets withheld", "rep")

    d.edge(relab, sim, (1, 0.5), (0, 0.5))
    d.edge(sim, pre, (1, 0.5), (0, 0.5))
    d.edge(pre, ft, (0.5, 1), (0.5, 0), label="initial weights")
    d.edge(lscc, ft, (1, 0.5), (0, 0.5))
    d.edge(m3, orb, (1, 0.5), (0, 0.5))
    d.edge(ft, orb, (0.5, 1), (0.5, 0))
    d.edge(ft, unc, (1, 0.5), (0, 0.5))
    d.edge(orb, unc, (1, 0.5), (0, 0.8), pts=[(1190, 655), (1190, 463)])
    d.edge(unc, gate, (0.5, 0), (0.5, 1))
    d.edge(unc, out, (0.5, 1), (0.5, 0))
    d.edge(gate, out, (1, 0.5), (1, 0.5), pts=[(1610, 150), (1610, 655)], dashed=True,
           color=MUTED, label="withholds")
    return d


# =========================================================================== 2
def simulator():
    d = Diagram("2 Physics simulator", 1600, 760)
    d.group(275, 50, 1305, 210, "DOMAIN RANDOMISATION (PER SYNTHETIC SOIL)", kind="neut")
    W, Y, H = 220, 330, 140
    xs = [20, 290, 550, 810, 1070, 1330]
    em = d.card(xs[0], Y, W, H, "End-member spectra",
                "49 RELAB separates<br>and glasses, <i>R<sub>i</sub></i>(λ)", "", "data")
    go = d.card(xs[1], Y, W, H, "Grain optics",
                "invert equivalent-slab model<br>⇒ <i>k<sub>i</sub></i>(λ)", "Eqs. (1)–(4)", "phys")
    sw = d.card(xs[2], Y, W, H, "Space weathering",
                "α = 4π<i>k</i>/λ + α<sub>np</sub> + α<sub>mp</sub>",
                f"{NPFE} (Eq. 5) · {MPFE} grey absorber", "phys")
    mx = d.card(xs[3], Y, W, H, "Intimate mixing",
                "slab SSA <i>w<sub>i</sub></i>(<i>D<sub>i</sub></i>)<br>cross-section weights <i>f<sub>i</sub></i>/<i>D<sub>i</sub></i>",
                "Eqs. (3), (6)", "phys")
    hp = d.card(xs[4], Y, W, H, "Hapke reflectance",
                "<i>r</i>(<i>w</i><sub>mix</sub>)<br><i>i</i> = 30°, <i>e</i> = 0°", "Eqs. (1)–(2)", "phys")
    ms = d.card(xs[5], Y, W, H, f"{M3} sampling",
                "71 bands<br>0.54–2.50 µm", "Gaussian band responses", "phys")
    chain = [em, go, sw, mx, hp, ms]
    for a, b in zip(chain, chain[1:]):
        d.edge(a, b, (1, 0.5), (0, 0.5))

    PY, PH = 90, 140
    p_em = d.card(xs[1], PY, W, PH, "End-member choice", "one per phase", "", "neut", sw=1.2)
    p_mat = d.card(xs[2], PY, W, PH, "Maturity",
                   f"agglutinate fraction<br>{NPFE} and {MPFE} loads",
                   "AGG = clip(0.65·mat + <i>N</i>(0, 0.12), 0, 0.75)", "neut", sw=1.2)
    p_cmp = d.card(xs[3], PY, W, PH, "Composition",
                   "feldspathic, basaltic or<br>broad Dirichlet prior",
                   "ILM ≤ 20 %, OLV ≤ 60 %<br>grain size 4–150 µm", "neut", sw=1.2)
    p_ins = d.card(xs[5], PY, W, PH, "Instrument",
                   "calibration tilt<br>±7 % scale", "0.2–0.8 % additive noise", "neut", sw=1.2)
    for p, t in [(p_em, go), (p_mat, sw), (p_cmp, mx), (p_ins, ms)]:
        d.edge(p, t, (0.5, 1), (0.5, 0), dashed=True, color=MUTED)

    lab = d.card(600, 560, 520, 160, "Labels by stoichiometry",
                 "7-part mode (vol%) · FeO, TiO<sub>2</sub>, Al<sub>2</sub>O<sub>3</sub>, CaO, MgO (wt%)<br>"
                 f"log<sub>10</sub> I<sub>s</sub>/FeO with I<sub>s</sub>/FeO = 2270 · {NPFE} / FeO",
                 f"auxiliary (pretraining only): grain size, {NPFE}", "learn")
    d.edge(p_mat, lab, (1, 0.5), (0.355, 0), pts=[(785, 160)], dashed=True, color=MUTED)
    d.edge(p_cmp, lab, (1, 0.5), (0.856, 0), pts=[(1045, 160)], dashed=True, color=MUTED)

    syn = d.card(1330, 580, 220, 120, "Synthetic spectrum",
                 "network input<br>2 × 71 (log <i>R</i>, CR <i>R</i>)", "", "learn")
    d.edge(ms, syn, (0.5, 1), (0.5, 0))
    d.edge(lab, syn, (1, 0.5), (0, 0.5), label="(<i>x</i>, <i>y</i>) pair", dashed=True, color=MUTED)

    d.text(20, 725, 1560, 26,
           f"400,000 spectra, fixed seed · no end-member from an evaluation soil · "
           f"agglutinates are a synthetic-glass host carrying {NPFE} and {MPFE}",
           fs=12, color=MUTED)
    return d


# =========================================================================== 3
def network():
    d = Diagram("3 Network", 1700, 620)
    CY = 290
    inp = d.card(20, CY - 100, 160, 200, "Input",
                 "log <i>R</i><br>continuum-<br>removed <i>R</i>", "", "data")
    d.text(20, 440, 160, 22, "<b>2 × 71</b>", fs=12)
    d.text(10, 462, 180, 40, "standardised with<br>synthetic statistics", fs=11, color=MUTED)

    conv_f, conv_s = "#DCE8F8", "#2F6DB5"
    pool_f, pool_s = "#ECEEF1", "#8B95A3"
    layers = [  # x, channels, bands, kind, label
        (250, 32, 71, "c", "Conv1d<br><i>k</i> = 7"),
        (345, 32, 71, "c", "Conv1d<br><i>k</i> = 5"),
        (440, 32, 35, "p", "MaxPool<br>÷2"),
        (535, 64, 35, "c", "Conv1d<br><i>k</i> = 5"),
        (640, 64, 35, "c", "Conv1d<br><i>k</i> = 3"),
        (745, 64, 17, "p", "MaxPool<br>÷2"),
        (850, 64, 17, "c", "Conv1d<br><i>k</i> = 3"),
    ]
    ids, prev = [], inp
    for x, ch, nb, k, lab in layers:
        w = 16 if ch == 32 else 30
        h = round(nb * 3.6)
        f, s = (conv_f, conv_s) if k == "c" else (pool_f, pool_s)
        v = d.shape(x - w / 2, CY - h / 2, w, h, "rounded=1;arcSize=12", fill=f, stroke=s, sw=1.3)
        d.text(x - 45, 118, 90, 40, lab, fs=11, color=INK)
        d.text(x - 40, 438, 80, 22, f"{ch}×{nb}", fs=12)
        d.edge(prev, v, (1, 0.5), (0, 0.5), sw=1.4)
        ids.append(v)
        prev = v
    # bracket under the conv body
    d.shape(250 - 8, 470, 850 + 15 - 242, 10, "shape=partialRectangle;top=0;fillColor=none",
            fill="none", stroke=MUTED, sw=1)
    d.text(242, 484, 623, 22, "each convolution: Conv1d → GroupNorm (8 groups) → GELU", fs=11, color=MUTED)

    fc_st = f"rounded=1;arcSize=18;horizontal=0;fontStyle=1"
    fl = d.shape(950, CY - 140, 46, 280, fc_st, fill="#E9F6EE", stroke="#2E8B57", sw=1.4,
                 value="Flatten · 1088", fs=12)
    f1 = d.shape(1030, CY - 110, 46, 220, fc_st, fill="#E9F6EE", stroke="#2E8B57", sw=1.4,
                 value="FC 256", fs=12)
    f2 = d.shape(1110, CY - 80, 46, 160, fc_st, fill="#E9F6EE", stroke="#2E8B57", sw=1.4,
                 value="FC 128", fs=12)
    d.edge(prev, fl, (1, 0.5), (0, 0.5), sw=1.4)
    d.edge(fl, f1, (1, 0.5), (0, 0.5), sw=1.4)
    d.edge(f1, f2, (1, 0.5), (0, 0.5), sw=1.4)
    d.text(985, 118, 140, 40, "GELU<br>dropout 0.1", fs=11)
    d.text(940, 438, 230, 22, "shared trunk (128-d embedding)", fs=11, color=MUTED)

    hd = d.card(1250, 40, 430, 160, "Dirichlet head · 7 concentrations",
                "<b>α</b> = softplus(·) + 0.05 ; composition = <b>α</b> / α<sub>0</sub>",
                "PLG, LCP, HCP, OLV, ILM, AGG, OTH<br>loss: Dirichlet NLL + <i>L</i><sub>1</sub>, Eq. (7)",
                "rep")
    hg = d.card(1250, 225, 430, 150, "Gaussian heads · 6 × (μ, log σ²)",
                "FeO, TiO<sub>2</sub>, Al<sub>2</sub>O<sub>3</sub>, CaO, MgO, log<sub>10</sub> I<sub>s</sub>/FeO",
                "loss: β-NLL (β = 0.5), Eq. (8)", "rep")
    ha = d.card(1250, 400, 430, 90, "Auxiliary heads (pretraining only)",
                f"log grain size, log {NPFE}", "", "neut", dashed=True)
    for h in (hd, hg, ha):
        d.edge(f2, h, (1, 0.5), (0, 0.5), sw=1.4)
    d.text(1250, 500, 430, 22, "<b>0.36 M parameters</b> in total", fs=12, align="right")

    # legend
    ly = 560
    d.shape(250, ly, 16, 22, "rounded=1", fill=conv_f, stroke=conv_s)
    d.text(272, ly, 200, 22, "Conv1d + GroupNorm + GELU", fs=11, align="left")
    d.shape(480, ly, 16, 22, "rounded=1", fill=pool_f, stroke=pool_s)
    d.text(502, ly, 140, 22, "max-pooling", fs=11, align="left")
    d.shape(620, ly, 16, 22, "rounded=1", fill="#E9F6EE", stroke="#2E8B57")
    d.text(642, ly, 160, 22, "fully connected", fs=11, align="left")
    d.text(800, ly, 420, 22, "slab width ∝ channels · slab height ∝ spectral length",
           fs=11, color=MUTED, align="left")
    return d


# =========================================================================== 4
def training():
    d = Diagram("4 Training and inference", 1700, 800)

    def lane(y, h, title, kind):
        fill, stroke, tcol = PAL[kind]
        d.vertex(20, y, 1660, h, "",
                 f"rounded=1;arcSize=2;html=1;fillColor=#FCFCFD;strokeColor={stroke};strokeWidth=1.2;")
        d.vertex(20, y, 38, h, f"<b>{title}</b>",
                 f"rounded=1;arcSize=30;html=1;whiteSpace=wrap;horizontal=0;fillColor={stroke};"
                 f"strokeColor={stroke};fontColor=#FFFFFF;fontFamily=Helvetica;fontSize=13;")

    lane(30, 180, "PRETRAINING", "phys")
    lane(230, 250, "FINE-TUNING", "learn")
    lane(500, 280, "INFERENCE", "rep")

    # ---- lane A
    syn = d.card(90, 60, 240, 120, "Synthetic set",
                 "400,000 spectra + labels", "20,000 held out for validation", "phys")
    # stacked ensemble look
    for k in (2, 1):
        d.vertex(400 + 7 * k, 60 - 7 * k, 320, 120, "",
                 f"rounded=1;arcSize=7;html=1;fillColor=#F4FBF6;strokeColor=#2E8B57;strokeWidth=1;")
    mem = d.card(400, 60, 320, 120, "5 members · seeds 0–4",
                 "15 epochs · AdamW · one-cycle LR", "same architecture, different seeds", "learn")
    loss = d.card(800, 60, 400, 120, "Pretraining objective",
                  "ℒ = ℒ<sub>comp</sub> + ℒ<sub>scal</sub> + auxiliary β-NLL",
                  f"Dirichlet NLL + <i>L</i><sub>1</sub> (λ<sub>1</sub> = 10) · β-NLL on oxides, maturity, grain size, {NPFE}",
                  "neut", sw=1.2)
    ck = d.card(1290, 60, 360, 120, "Pretrained weights θ<sub>m</sub><sup>0</sup>",
                "5 checkpoints", "≈ 45 min on 4 CPU cores", "learn")
    d.edge(syn, mem, (1, 0.5), (0, 0.5))
    d.edge(mem, loss, (1, 0.5), (0, 0.5))
    d.edge(loss, ck, (1, 0.5), (0, 0.5))

    # ---- lane B
    d.group(90, 260, 500, 195, "A4 MINI-BATCH · 64 SPECTRA", kind="learn")
    r1 = d.card(110, 285, 145, 80, "16 real", "training-soil spectra", "", "data", fs=11, tfs=13)
    r3 = d.card(268, 285, 145, 80, "32 SSA-mixup", "2–3 soils mixed, Eq. (6)", "", "learn", fs=11, tfs=13)
    r2 = d.card(426, 285, 145, 80, "16 replay", "synthetic spectra", "", "phys", fs=11, tfs=13)
    rnd = d.card(110, 380, 303, 55, "Calibration + noise randomisation",
                 "", "real and mixed spectra only", "neut", fs=11, tfs=12, sw=1.2)
    for r in (r1, r3):
        d.edge(r, rnd, (0.5, 1), (r == r1 and 0.239 or 0.761, 0), sw=1.2)
    ftn = d.card(660, 275, 330, 165, "Fine-tune each member",
                 "1,500 steps per CV fold<br>AdamW · lr 3×10<sup>−4</sup> · wd 10<sup>−4</sup>",
                 "100-step warm-up, cosine decay · ≈ 6 s each", "learn")
    msk = d.card(1060, 275, 300, 165, "Masked losses",
                 "Σ<sub><i>t</i></sub> <i>m<sub>t</sub></i> · β-NLL<sub><i>t</i></sub> + ℒ<sub>comp</sub>",
                 f"<i>m<sub>t</sub></i> drops unavailable targets:<br>mode for &lt;45 µm, all but FeO for {M3}",
                 "neut", sw=1.2)
    ftd = d.card(1430, 275, 220, 165, "Fine-tuned ensemble",
                 "5 members per fold", "A2 / A4 chosen by inner 5-fold CV", "learn")
    d.edge(rnd, ftn, (1, 0.5), (0, 0.75), pts=[(620, 407), (620, 399)])
    d.edge(r2, ftn, (1, 0.5), (0, 0.3))
    d.edge(ck, ftn, (0.5, 1), (0.5, 0), pts=[(1470, 220), (825, 220)], label="initialise")
    d.edge(ftn, msk, (1, 0.5), (0, 0.5))
    d.edge(msk, ftd, (1, 0.5), (0, 0.5))

    # ---- lane C
    labsp = d.card(90, 530, 210, 90, "Lab spectrum", "RELAB, 71 bands", "", "data")
    m3sp = d.card(90, 660, 210, 90, f"{M3} pixel spectrum", "Level-2 reflectance", "", "data")
    orb = d.card(350, 640, 270, 130, "Orbital transfer",
                 "zero-shot: <i>A</i><sub>lab</sub> = 1.685 <i>A</i><sub>M3</sub> + 0.0355",
                 f"or LUMEN-M3-FT: 600 steps, 16 {M3} px (FeO only) + 16 lab spectra", "phys")
    ens = d.card(680, 545, 280, 170, "Ensemble moments",
                 "μ = mean<sub><i>m</i></sub> μ<sub><i>m</i></sub><br>σ² = mean<sub><i>m</i></sub> σ²<sub><i>m</i></sub> + Var<sub><i>m</i></sub> μ<sub><i>m</i></sub>",
                 "Dirichlet means for composition", "rep")
    cf = d.card(1020, 545, 280, 170, "Conformal calibration",
                "<i>s<sub>i</sub></i> = |<i>y<sub>i</sub></i> − μ<sub><i>i</i></sub>| / σ<sub><i>i</i></sub><br>interval μ ± <i>q</i><sub>1−<i>a</i></sub> σ",
                "out-of-fold scores of other soils (CV+)", "rep")
    gate = d.shape(1345, 570, 150, 120, "rhombus", fill="#F2EEFB", stroke="#6B4FA0", sw=1.5,
                   value="<b style='color:#4A3278'>Gate</b><br><span style='font-size:11px'>95 % CI of<br><i>S</i> &gt; 0 ?</span>")
    yes = d.card(1530, 530, 130, 90, "Map", "value ± interval", "", "learn", fs=11, tfs=13)
    no = d.card(1530, 660, 130, 90, "Withhold", "e.g. olivine, minor glasses", "", "neut", fs=11, tfs=13, dashed=True)
    d.text(1310, 715, 220, 40, "<i>S</i> = 1 − MAE / MAE<sub>ref</sub><br>bootstrap over soils",
           fs=11, color=MUTED)

    d.edge(labsp, ens, (1, 0.5), (0, 0.25), pts=[(640, 575), (640, 587)])
    d.edge(m3sp, orb, (1, 0.5), (0, 0.5), pts=[(325, 705)])
    d.edge(orb, ens, (1, 0.5), (0, 0.75), pts=[(650, 705), (650, 672)])
    d.edge(ftd, ens, (0.5, 1), (0.5, 0), pts=[(1540, 490), (820, 490)])
    d.edge(ens, cf, (1, 0.5), (0, 0.5))
    d.edge(cf, gate, (1, 0.5), (0, 0.5))
    d.edge(gate, yes, (1, 0.5), (0, 0.5), pts=[(1512, 630), (1512, 575)], label="yes", lpos=0.2)
    d.edge(gate, no, (1, 0.5), (0, 0.5), pts=[(1512, 630), (1512, 705)], label="no", lpos=0.2)
    return d


# =========================================================================== 5
def evaluation():
    d = Diagram("5 Evaluation design", 1720, 760)
    TR = ("#E9F6EE", "#2E8B57")
    HO = ("#9B88CF", "#5B4A9A")
    BL = ["#D6E4F2", "#A9C4E2", "#7FA6D1", "#5585BF", "#2F67A8"]
    TE = ("#FFFFFF", "#9CA3AF")

    def row_label(y, t, s):
        d.text(20, y - 22, 280, 24, f"<b>{t}</b>", fs=15, align="left")
        d.text(20, y + 2, 300, 20, s, fs=12, color=MUTED, align="left")

    # ---- (a) laboratory
    d.text(20, 20, 600, 28, "<b>(a) Laboratory · 19 LSCC soils</b>", fs=16, align="left")
    groups = [("A11", "mare", 1), ("A12", "mare", 2), ("A14", "highland", 4),
              ("A15", "mare", 2), ("A16", "highland", 6), ("A17", "mare", 4)]
    D, SP, GG = 28, 38, 46
    x, pos = 330, []
    for g, kind, n in groups:
        xs = [x + i * SP for i in range(n)]
        pos.append((g, kind, xs))
        x = xs[-1] + SP + GG
    inner = [0, 1, 2, 3, 4, 2, 1, 9, 3, 2, 3, 4, 0, 1, 2, 4, 0, 1, 2]  # 9 marks the outer test soil
    rows = [(100, "LOSO", "19 folds, one soil out"),
            (170, "LOMO", "6 folds, one mission out"),
            (240, "E1 (nested)", "inner 5-fold split by soil")]
    for y, t, s in rows:
        row_label(y, t, s)
    k = 0
    for gi, (g, kind, xs) in enumerate(pos):
        for j, cx in enumerate(xs):
            lo = HO if (g == "A15" and j == 0) else TR
            lm = HO if g == "A16" else TR
            e1 = TE if inner[k] == 9 else (BL[inner[k]], "#2F5D94")
            for y, (f, s) in zip((100, 170, 240), (lo, lm, e1)):
                d.shape(cx - D / 2, y - D / 2, D, D, "ellipse", fill=f, stroke=s, sw=1.3)
            k += 1
        mid = (xs[0] + xs[-1]) / 2
        d.text(mid - 60, 266, 120, 22, f"<b>{g}</b>", fs=13)
        d.text(mid - 60, 286, 120, 20, kind, fs=12, color=MUTED)
    d.text(1290, 195, 410, 70,
           "In each outer training set, an inner 5-fold split by soil chooses A2 or A4; "
           "the outer test soil (white) is never seen.", fs=12, color=MUTED, align="left")

    # ---- (b) orbital
    d.text(20, 345, 700, 28, "<b>(b) Orbital FeO · 49 returned-sample sites</b>", fs=16, align="left")
    og = [("A11", 1), ("A12", 1), ("A14", 2), ("A15", 9), ("A16", 10), ("A17", 20),
          ("L16", 1), ("L20", 1), ("L24", 1), ("CE3", 1), ("CE5", 1), ("CE6", 1)]
    d2, sp2, gg2 = 17, 21, 22
    x, opos = 330, []
    for g, n in og:
        xs = [x + i * sp2 for i in range(n)]
        opos.append((g, n, xs))
        x = xs[-1] + sp2 + gg2
    p1_ho = {("A15", 3), ("A16", 7), ("A17", 2), ("A17", 13), ("L24", 0)}
    for y, t, s in [(430, "P1", "10 random folds over sites"),
                    (500, "P2", "12 folds, one landing site out")]:
        row_label(y, t, s)
    for g, n, xs in opos:
        for j, cx in enumerate(xs):
            f1 = HO if (g, j) in p1_ho else TR
            f2 = HO if g == "A17" else TR
            for y, (f, s) in zip((430, 500), (f1, f2)):
                d.shape(cx - d2 / 2, y - d2 / 2, d2, d2, "ellipse", fill=f, stroke=s, sw=1.2)
        mid = (xs[0] + xs[-1]) / 2
        d.text(mid - 30, 525, 60, 22, f"<b>{g}</b>", fs=12)
        d.text(mid - 30, 545, 60, 20, str(n), fs=12, color=MUTED)

    # ---- legend
    ly = 610
    items = [(TR, "training"), (HO, "held out (example fold)"),
             ((BL[2], "#2F5D94"), "inner folds (E1)"), (TE, "outer test soil (E1)")]
    x = 330
    for (f, s), t in items:
        d.shape(x, ly, 22, 22, "ellipse", fill=f, stroke=s, sw=1.3)
        d.text(x + 30, ly, 220, 22, t, fs=12, align="left")
        x += 270
    d.text(20, 665, 1680, 60,
           "Metric: per-soil MAE (fractions averaged within a soil, then over 19 soils) · paired Wilcoxon "
           "signed-rank tests with Holm correction · bootstrap CIs over soils. P1 follows the published "
           "protocol of Fernández-Díaz et al.; P2 is the spatial leave-one-landing-site-out test.",
           fs=12, color=MUTED, align="left")
    return d


if __name__ == "__main__":
    out = sys.argv[1]
    prev = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    os.makedirs(prev, exist_ok=True)
    figs = [("01_framework", framework()), ("02_simulator", simulator()),
            ("03_network", network()), ("04_training_inference", training()),
            ("05_evaluation", evaluation())]
    for name, dg in figs:
        write_drawio(os.path.join(out, f"LUMEN_{name}.drawio"), [dg])
        with open(os.path.join(prev, f"{name}.html"), "w") as f:
            f.write(f"<html><body style='margin:0'>{dg.preview_html()}</body></html>")
    write_drawio(os.path.join(out, "LUMEN_all_diagrams.drawio"), [dg for _, dg in figs])
    print("ok")
