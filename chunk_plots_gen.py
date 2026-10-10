#!/usr/bin/env python3
"""
chunk_plots_gen.py

Plots kinematic variables from HH -> 2B2V -> 2L2Nu ROOT files in high-resolution ROOT graphics style.
Filters events to require valid Dilepton candidates (Dilepton_ok == 1).

In addition to the reco-level plots, makes truth-level plots from the LHE* and GenPart*
branches kept by create_dilepton_ntuple.cc (LHE_* / Gen_* output files). Use --gen-all-events
to fill the truth plots for every event instead of only Dilepton_ok == 1 events.

Every plot is made inclusively and split by the reco dilepton flavour (Dilepton_channel:
0 = mumu, 1 = ee, 2 = emu). Output goes to <outdir>/{inclusive,mumu,ee,emu}/<var>.png.
Events without a dilepton candidate (only filled with --gen-all-events) appear only in
the inclusive plots.
"""

import argparse
import os
import sys
import gc

import numpy as np
import matplotlib.pyplot as plt
import uproot
import awkward as ak

try:
    import vector
    vector.register_awkward()
    HAS_VECTOR = True
except ImportError:
    HAS_VECTOR = False

# Plot styling defaults
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "figure.autolayout": True,
})

lumi = 137 # fb-1
cross_sect = 1 # pb
sum_weight = 35202.615586500004
w_const = lumi * 1000 * cross_sect / sum_weight

# Dilepton flavour channels: (output name, Dilepton_channel code, title label)
# code None = inclusive (all events)
CHANNELS = [
    ("inclusive", None, ""),
    ("mumu", 0, "$\\mu\\mu$"),
    ("ee", 1, "$ee$"),
    ("emu", 2, "$e\\mu$"),
]

# Variable-specific binning, range, and unit configurations
VAR_CONFIGS = {
    # Mass variables
    "m_LL": (0.0, 200.0, 100, "GeV"),             # 2 GeV per bin
    "m_LL_zoom": (76.0, 106.0, 60, "GeV"),        # Z peak window (0.5 GeV per bin)
    "mbb_calc": (0.0, 300.0, 150, "GeV"),         # 2 GeV per bin
    "mHH_calc": (0.0, 1000.0, 200, "GeV"),        # 5 GeV per bin

    # pT / MET variables
    "bJet1_pt": (20.0, 250.0, 125, "GeV"),
    "bJet2_pt": (20.0, 250.0, 125, "GeV"),
    "Lepton1_pt": (0.0, 250.0, 125, "GeV"),
    "Lepton2_pt": (0.0, 200.0, 100, "GeV"),
    "Dilepton_ptLL": (0.0, 300.0, 150, "GeV"),
    "PuppiMET_pt": (0.0, 250.0, 125, "GeV"),

    # Angular variables
    "Dilepton_etaLL": (-3.0, 3.0, 60, ""),
    "Dilepton_phiLL": (-3.1416, 3.1416, 64, "rad"),
    "Dilepton_dRLL": (0.0, 5.0, 50, ""),
    "Dilepton_dPhiLL": (0.0, 3.1416, 32, "rad"),
    "PuppiMET_phi": (-3.1416, 3.1416, 64, "rad"),
    "Lepton1_phi": (-3.1416, 3.1416, 64, "rad"),
    "Lepton2_phi": (-3.1416, 3.1416, 64, "rad"),
    "Lepton1_eta": (-3.0, 3.0, 60, ""),
    "Lepton2_eta": (-3.0, 3.0, 60, ""),
    "bJet1_phi": (-3.1416, 3.1416, 64, "rad"),
    "bJet2_phi": (-3.1416, 3.1416, 64, "rad"),
    "bJet1_eta": (-3.0, 3.0, 60, ""),
    "bJet2_eta": (-3.0, 3.0, 60, ""),

    # ---- LHE-level (matrix element) variables ----
    "LHE_HT": (0.0, 500.0, 100, "GeV"),
    "LHE_HTIncoming": (0.0, 500.0, 100, "GeV"),
    "LHE_Njets": (-0.5, 5.5, 6, ""),
    "LHE_mHH": (200.0, 1200.0, 200, "GeV"),
    "LHE_ptHH": (0.0, 300.0, 150, "GeV"),
    "LHE_H1_pt": (0.0, 500.0, 125, "GeV"),
    "LHE_H2_pt": (0.0, 400.0, 100, "GeV"),
    "LHE_H1_eta": (-5.0, 5.0, 100, ""),
    "LHE_H2_eta": (-5.0, 5.0, 100, ""),
    "LHE_dRHH": (0.0, 6.0, 60, ""),
    "LHE_Parton_pt": (0.0, 300.0, 150, "GeV"),

    # ---- GenPart-level (parton shower truth) variables ----
    "Gen_mHH": (200.0, 1200.0, 200, "GeV"),
    "Gen_ptHH": (0.0, 300.0, 150, "GeV"),
    "Gen_H1_pt": (0.0, 500.0, 125, "GeV"),
    "Gen_H2_pt": (0.0, 400.0, 100, "GeV"),
    "Gen_H1_eta": (-5.0, 5.0, 100, ""),
    "Gen_H2_eta": (-5.0, 5.0, 100, ""),
    "Gen_mbb": (0.0, 300.0, 150, "GeV"),
    "Gen_b1_pt": (0.0, 250.0, 125, "GeV"),
    "Gen_b2_pt": (0.0, 250.0, 125, "GeV"),
    "Gen_b1_eta": (-5.0, 5.0, 100, ""),
    "Gen_b2_eta": (-5.0, 5.0, 100, ""),
    "Gen_dRbb": (0.0, 6.0, 60, ""),
    "Gen_V_mass": (0.0, 120.0, 120, "GeV"),
    "Gen_Lepton1_pt": (0.0, 250.0, 125, "GeV"),
    "Gen_Lepton2_pt": (0.0, 200.0, 100, "GeV"),
    "Gen_Lepton1_eta": (-3.0, 3.0, 60, ""),
    "Gen_Lepton2_eta": (-3.0, 3.0, 60, ""),
    "Gen_mLL": (0.0, 200.0, 100, "GeV"),
    "Gen_dRLL": (0.0, 5.0, 50, ""),
    "Gen_MET_pt": (0.0, 250.0, 125, "GeV"),
    "Gen_MET_phi": (-3.1416, 3.1416, 64, "rad"),
}

# NanoAOD GenPart_statusFlags bits
FLAG_IS_PROMPT = 0
FLAG_FROM_HARD_PROCESS = 8
FLAG_IS_LAST_COPY = 13


def parse_args():
    parser = argparse.ArgumentParser(description="Plot ROOT kinematics with precise ROOT-style formatting.")
    parser.add_argument("-i", "--input", type=str, nargs="+", default=["output/twolepton_GluGlutoHHto2B2Vto2L2Nu.root"],
                        help="Input ROOT file(s); several files or a shell glob are combined into one set of plots")
    parser.add_argument("-t", "--tree", type=str, default="Events", help="TTree name inside ROOT file")
    parser.add_argument("-o", "--outdir", type=str, default="plots/bb2l2nu", help="Output directory")
    parser.add_argument("-s", "--step-size", type=str, default="100MB", help="Chunk size for streaming")
    parser.add_argument("--gen-all-events", action="store_true",
                        help="Fill LHE/GenPart plots for all events instead of only Dilepton_ok == 1 events")
    parser.add_argument("--min-jets", type=int, default=2,
                        help="Require nGoodJet >= this many jets (default 2; 0 disables the cut)")
    return parser.parse_args()


def construct_4vectors(pt, eta, phi, mass):
    """Constructs 4-momenta vectors using vector.zip or native numpy components."""
    if HAS_VECTOR:
        return vector.zip({"pt": pt, "eta": eta, "phi": phi, "mass": mass})
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return {"px": px, "py": py, "pz": pz, "e": e}


def compute_inv_mass_numpy(p1, p2=None, p3=None, p4=None):
    """Computes invariant mass from 4-momentum components dictionaries."""
    px = p1["px"] + (p2["px"] if p2 is not None else 0) + (p3["px"] if p3 is not None else 0) + (p4["px"] if p4 is not None else 0)
    py = p1["py"] + (p2["py"] if p2 is not None else 0) + (p3["py"] if p3 is not None else 0) + (p4["py"] if p4 is not None else 0)
    pz = p1["pz"] + (p2["pz"] if p2 is not None else 0) + (p3["pz"] if p3 is not None else 0) + (p4["pz"] if p4 is not None else 0)
    e  = p1["e"]  + (p2["e"]  if p2 is not None else 0)  + (p3["e"]  if p3 is not None else 0)  + (p4["e"]  if p4 is not None else 0)
    return np.sqrt(np.maximum(0, e**2 - (px**2 + py**2 + pz**2)))


def extract_indexed_array(ak_array, index):
    """Extracts object at 'index' from jagged array, padded with NaNs."""
    padded = ak.pad_none(ak_array, index + 1, axis=1)
    return ak.to_numpy(ak.fill_none(padded[:, index], np.nan))


def pair_kinematics(pt1, eta1, phi1, m1, pt2, eta2, phi2, m2):
    """Returns (mass, pt, dR) of the sum of two objects given as flat numpy arrays (NaN-safe)."""
    p1 = construct_4vectors_numpy(pt1, eta1, phi1, m1)
    p2 = construct_4vectors_numpy(pt2, eta2, phi2, m2)
    mass = compute_inv_mass_numpy(p1, p2)
    pt = np.hypot(p1["px"] + p2["px"], p1["py"] + p2["py"])
    dphi = np.mod(phi1 - phi2 + np.pi, 2 * np.pi) - np.pi
    dr = np.sqrt((eta1 - eta2) ** 2 + dphi ** 2)
    return mass, pt, dr


def construct_4vectors_numpy(pt, eta, phi, mass):
    """Numpy-only 4-vector components (used for gen-level quantities)."""
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return {"px": px, "py": py, "pz": pz, "e": e}


def leading_two(collection, mask):
    """Selects particles passing 'mask', sorts by pt, and returns the leading two
    as dicts of flat numpy arrays (NaN where the particle is missing)."""
    sel = collection[mask]
    sel = sel[ak.argsort(sel["pt"], axis=1, ascending=False)]
    out = []
    for i in range(2):
        out.append({f: extract_indexed_array(sel[f], i) for f in ("pt", "eta", "phi", "mass")})
    return out


def has_flag(status_flags, bit):
    return (status_flags & (1 << bit)) != 0


def fill_lhe_histograms(chunk, mask, add):
    """LHE_* scalars and LHEPart Higgs / extra-parton kinematics."""
    for b_name in ("LHE_HT", "LHE_HTIncoming", "LHE_Njets"):
        if b_name in chunk.fields:
            add(b_name, ak.to_numpy(chunk[b_name][mask]))

    if "LHEPart_pt" not in chunk.fields:
        return
    lhe = ak.zip({
        "pt": chunk["LHEPart_pt"][mask],
        "eta": chunk["LHEPart_eta"][mask],
        "phi": chunk["LHEPart_phi"][mask],
        "mass": chunk["LHEPart_mass"][mask],
        "pdgId": chunk["LHEPart_pdgId"][mask],
        "status": chunk["LHEPart_status"][mask],
    })

    # Outgoing Higgs bosons
    h1, h2 = leading_two(lhe, (lhe["pdgId"] == 25) & (lhe["status"] == 1))
    add("LHE_H1_pt", h1["pt"])
    add("LHE_H2_pt", h2["pt"])
    add("LHE_H1_eta", h1["eta"])
    add("LHE_H2_eta", h2["eta"])
    m_hh, pt_hh, dr_hh = pair_kinematics(h1["pt"], h1["eta"], h1["phi"], h1["mass"],
                                         h2["pt"], h2["eta"], h2["phi"], h2["mass"])
    add("LHE_mHH", m_hh)
    add("LHE_ptHH", pt_hh)
    add("LHE_dRHH", dr_hh)

    # Outgoing QCD parton from the NLO real emission (if present)
    is_parton = (abs(lhe["pdgId"]) <= 5) | (lhe["pdgId"] == 21)
    p1, _ = leading_two(lhe, is_parton & (lhe["status"] == 1))
    add("LHE_Parton_pt", p1["pt"])


def fill_genpart_histograms(chunk, mask, weights, chans, add):
    """Truth Higgs, H->bb quarks, H->VV bosons, prompt leptons and neutrinos from GenPart."""
    if "GenPart_pt" not in chunk.fields:
        return
    gen = ak.zip({
        "pt": chunk["GenPart_pt"][mask],
        "eta": chunk["GenPart_eta"][mask],
        "phi": chunk["GenPart_phi"][mask],
        "mass": chunk["GenPart_mass"][mask],
        "pdgId": chunk["GenPart_pdgId"][mask],
        "status": chunk["GenPart_status"][mask],
        "flags": ak.values_astype(chunk["GenPart_statusFlags"][mask], np.int32),
        "mom": chunk["GenPart_genPartIdxMother"][mask],
    })
    abs_pdg = abs(gen["pdgId"])
    mom_pdg = ak.where(gen["mom"] >= 0, gen["pdgId"][ak.where(gen["mom"] >= 0, gen["mom"], 0)], 0)
    last_copy = has_flag(gen["flags"], FLAG_IS_LAST_COPY)
    hard = has_flag(gen["flags"], FLAG_FROM_HARD_PROCESS)
    prompt = has_flag(gen["flags"], FLAG_IS_PROMPT)

    # Higgs bosons (last copy, i.e. after ISR recoil, right before decay)
    h1, h2 = leading_two(gen, (gen["pdgId"] == 25) & last_copy)
    add("Gen_H1_pt", h1["pt"])
    add("Gen_H2_pt", h2["pt"])
    add("Gen_H1_eta", h1["eta"])
    add("Gen_H2_eta", h2["eta"])
    m_hh, pt_hh, _ = pair_kinematics(h1["pt"], h1["eta"], h1["phi"], h1["mass"],
                                     h2["pt"], h2["eta"], h2["phi"], h2["mass"])
    add("Gen_mHH", m_hh)
    add("Gen_ptHH", pt_hh)

    # b quarks directly from H -> bb
    b1, b2 = leading_two(gen, (abs_pdg == 5) & (mom_pdg == 25))
    add("Gen_b1_pt", b1["pt"])
    add("Gen_b2_pt", b2["pt"])
    add("Gen_b1_eta", b1["eta"])
    add("Gen_b2_eta", b2["eta"])
    m_bb, _, dr_bb = pair_kinematics(b1["pt"], b1["eta"], b1["phi"], b1["mass"],
                                     b2["pt"], b2["eta"], b2["phi"], b2["mass"])
    add("Gen_mbb", m_bb)
    add("Gen_dRbb", dr_bb)

    # Vector bosons (Z / W) from H -> VV (one is typically off-shell)
    is_v = ((abs_pdg == 23) | (abs_pdg == 24)) & last_copy & hard
    v_mass = ak.to_numpy(ak.flatten(gen["mass"][is_v]))
    n_v = ak.to_numpy(ak.sum(is_v, axis=1))
    v_weights = np.repeat(weights, n_v)
    v_chans = np.repeat(chans, n_v)
    add("Gen_V_mass", v_mass, weights=v_weights, chans=v_chans)

    # Prompt final-state charged leptons (e, mu) from the hard process
    is_lep = ((abs_pdg == 11) | (abs_pdg == 13)) & (gen["status"] == 1) & prompt & hard
    l1, l2 = leading_two(gen, is_lep)
    add("Gen_Lepton1_pt", l1["pt"])
    add("Gen_Lepton2_pt", l2["pt"])
    add("Gen_Lepton1_eta", l1["eta"])
    add("Gen_Lepton2_eta", l2["eta"])
    m_ll, _, dr_ll = pair_kinematics(l1["pt"], l1["eta"], l1["phi"], l1["mass"],
                                     l2["pt"], l2["eta"], l2["phi"], l2["mass"])
    add("Gen_mLL", m_ll)
    add("Gen_dRLL", dr_ll)

    # Gen MET = vector sum of prompt neutrinos from the hard process
    is_nu = ((abs_pdg == 12) | (abs_pdg == 14) | (abs_pdg == 16)) & (gen["status"] == 1) & prompt & hard
    nu = gen[is_nu]
    has_nu = ak.to_numpy(ak.num(nu, axis=1)) > 0
    met_px = ak.to_numpy(ak.sum(nu["pt"] * np.cos(nu["phi"]), axis=1))
    met_py = ak.to_numpy(ak.sum(nu["pt"] * np.sin(nu["phi"]), axis=1))
    add("Gen_MET_pt", np.where(has_nu, np.hypot(met_px, met_py), np.nan))
    add("Gen_MET_phi", np.where(has_nu, np.arctan2(met_py, met_px), np.nan))


def plot_root_style(data, var_key, title, xlabel, output_path, weights=None):
    """Renders a high-precision ROOT-style histogram from pre-binned counts or raw data."""
    # Look up precise range/bins configuration or fallback defaults
    if var_key in VAR_CONFIGS:
        xmin, xmax, nbins, unit = VAR_CONFIGS[var_key]
    elif "_pt" in var_key:
        xmin, xmax, nbins, unit = 0.0, 250.0, 125, "GeV"
    elif "_eta" in var_key:
        xmin, xmax, nbins, unit = -3.0, 3.0, 60, ""
    elif "_phi" in var_key:
        xmin, xmax, nbins, unit = -3.1416, 3.1416, 64, "rad"
    else:
        xmin, xmax, nbins, unit = 0.0, 250.0, 100, ""

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Render pre-binned counts
    bin_edges = np.linspace(xmin, xmax, nbins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    counts, _, _ = ax.hist(
        bin_centers,
        weights=data,
        bins=bin_edges,
        histtype="stepfilled",
        facecolor="#b8c7f2",   # ROOT classic light blue fill
        edgecolor="#0000d0",   # Solid dark blue outline
        linewidth=1.4,
    )

    # Dynamic Y-axis label calculation
    bin_width = (xmax - xmin) / nbins
    if unit:
        width_str = f"{int(bin_width)}" if bin_width.is_integer() else f"{bin_width:.2f}"
        ylabel = f"Events / {width_str} {unit}"
    else:
        ylabel = f"Events / {bin_width:.2f}"

    # Title & Axis Labels
    ax.set_title(title, fontsize=18, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=14, loc="right")
    ax.set_ylabel(ylabel, fontsize=14, loc="top")

    # ROOT Inward Minor & Major Ticks Configuration
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", direction="in", length=8, width=1.0, top=True, right=True, labelsize=12)
    ax.tick_params(axis="both", which="minor", direction="in", length=4, width=0.7, top=True, right=True)

    # Grid Lines
    ax.grid(True, which="major", linestyle=":", linewidth=0.8, color="#808080", alpha=0.7)

    # Axes limits & outer frame box
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(bottom=0, top=np.max(counts) * 1.08 if len(counts) > 0 and np.max(counts) > 0 else 1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("black")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    args = parse_args()

    missing = [f for f in args.input if not os.path.exists(f)]
    if missing:
        sys.exit(f"Error: File(s) not found: {', '.join(missing)}")

    print(f"Opening {len(args.input)} ROOT file(s): {args.input[0]}{' ...' if len(args.input) > 1 else ''}")

    # Initialize plot_dict with empty 1D binned counts arrays (one per channel) to keep RAM minimal
    plot_dict = {}

    direct_vars = [
        ("Dilepton_mLL", "m_LL", "Dilepton Invariant Mass", "$m_{\\ell\\ell}$ [GeV]"),
        ("Dilepton_mLL", "m_LL_zoom", "Dilepton Mass (Zoom)", "$m_{\\ell\\ell}$ [GeV]"),
        ("Dilepton_ptLL", "Dilepton_ptLL", "Dilepton System $p_T$", "$p_T^{\\ell\\ell}$ [GeV]"),
        ("Dilepton_etaLL", "Dilepton_etaLL", "Dilepton System Pseudorapidity", "$\\eta_{\\ell\\ell}$"),
        ("Dilepton_phiLL", "Dilepton_phiLL", "Dilepton System Azimuthal Angle", "$\\phi_{\\ell\\ell}$ [rad]"),
        ("Dilepton_dRLL", "Dilepton_dRLL", "Lepton Angular Separation", "$\\Delta R_{\\ell\\ell}$"),
        ("Dilepton_dPhiLL", "Dilepton_dPhiLL", "Lepton Azimuthal Separation", "$\\Delta\\phi_{\\ell\\ell}$ [rad]"),
        ("PuppiMET_pt", "PuppiMET_pt", "Puppi MET $p_T$", "Puppi MET $p_T$ [GeV]"),
        ("PuppiMET_phi", "PuppiMET_phi", "Puppi MET $\\phi$", "Puppi MET $\\phi$ [rad]"),
    ]

    all_keys = direct_vars + [
        ("mbb_calc", "mbb_calc", "Di-jet Invariant Mass", "$m_{bb}$ [GeV]"),
        ("mHH_calc", "mHH_calc", "Di-Higgs Invariant Mass", "$m_{HH}$ [GeV]"),
        ("Lepton1_pt", "Lepton1_pt", "Lepton 1 Transverse Momentum", "Lepton 1 $p_T$ [GeV]"),
        ("Lepton1_eta", "Lepton1_eta", "Lepton 1 Pseudorapidity", "Lepton 1 $\\eta$"),
        ("Lepton1_phi", "Lepton1_phi", "Lepton 1 Azimuthal Angle", "Lepton 1 $\\phi$ [rad]"),
        ("Lepton2_pt", "Lepton2_pt", "Lepton 2 Transverse Momentum", "Lepton 2 $p_T$ [GeV]"),
        ("Lepton2_eta", "Lepton2_eta", "Lepton 2 Pseudorapidity", "Lepton 2 $\\eta$"),
        ("Lepton2_phi", "Lepton2_phi", "Lepton 2 Azimuthal Angle", "Lepton 2 $\\phi$ [rad]"),
        ("bJet1_pt", "bJet1_pt", "b-Jet 1 Transverse Momentum", "b-Jet 1 $p_T$ [GeV]"),
        ("bJet1_eta", "bJet1_eta", "b-Jet 1 Pseudorapidity", "b-Jet 1 $\\eta$"),
        ("bJet1_phi", "bJet1_phi", "b-Jet 1 Azimuthal Angle", "b-Jet 1 $\\phi$ [rad]"),
        ("bJet2_pt", "bJet2_pt", "b-Jet 2 Transverse Momentum", "b-Jet 2 $p_T$ [GeV]"),
        ("bJet2_eta", "bJet2_eta", "b-Jet 2 Pseudorapidity", "b-Jet 2 $\\eta$"),
        ("bJet2_phi", "bJet2_phi", "b-Jet 2 Azimuthal Angle", "b-Jet 2 $\\phi$ [rad]"),
    ]

    lhe_keys = [
        ("LHE_HT", "LHE_HT", "LHE $H_T$", "LHE $H_T$ [GeV]"),
        ("LHE_HTIncoming", "LHE_HTIncoming", "LHE Incoming $H_T$", "LHE $H_T^{in}$ [GeV]"),
        ("LHE_Njets", "LHE_Njets", "LHE Jet Multiplicity", "LHE $N_{jets}$"),
        ("LHE_mHH", "LHE_mHH", "LHE Di-Higgs Invariant Mass", "LHE $m_{HH}$ [GeV]"),
        ("LHE_ptHH", "LHE_ptHH", "LHE Di-Higgs $p_T$", "LHE $p_T^{HH}$ [GeV]"),
        ("LHE_H1_pt", "LHE_H1_pt", "LHE Leading Higgs $p_T$", "LHE $H_1$ $p_T$ [GeV]"),
        ("LHE_H2_pt", "LHE_H2_pt", "LHE Subleading Higgs $p_T$", "LHE $H_2$ $p_T$ [GeV]"),
        ("LHE_H1_eta", "LHE_H1_eta", "LHE Leading Higgs $\\eta$", "LHE $H_1$ $\\eta$"),
        ("LHE_H2_eta", "LHE_H2_eta", "LHE Subleading Higgs $\\eta$", "LHE $H_2$ $\\eta$"),
        ("LHE_dRHH", "LHE_dRHH", "LHE Higgs Angular Separation", "LHE $\\Delta R_{HH}$"),
        ("LHE_Parton_pt", "LHE_Parton_pt", "LHE Extra Parton $p_T$", "LHE parton $p_T$ [GeV]"),
    ]

    gen_keys = [
        ("Gen_mHH", "Gen_mHH", "Gen Di-Higgs Invariant Mass", "Gen $m_{HH}$ [GeV]"),
        ("Gen_ptHH", "Gen_ptHH", "Gen Di-Higgs $p_T$", "Gen $p_T^{HH}$ [GeV]"),
        ("Gen_H1_pt", "Gen_H1_pt", "Gen Leading Higgs $p_T$", "Gen $H_1$ $p_T$ [GeV]"),
        ("Gen_H2_pt", "Gen_H2_pt", "Gen Subleading Higgs $p_T$", "Gen $H_2$ $p_T$ [GeV]"),
        ("Gen_H1_eta", "Gen_H1_eta", "Gen Leading Higgs $\\eta$", "Gen $H_1$ $\\eta$"),
        ("Gen_H2_eta", "Gen_H2_eta", "Gen Subleading Higgs $\\eta$", "Gen $H_2$ $\\eta$"),
        ("Gen_mbb", "Gen_mbb", "Gen $H \\to b\\bar{b}$ Invariant Mass", "Gen $m_{bb}$ [GeV]"),
        ("Gen_b1_pt", "Gen_b1_pt", "Gen Leading b-Quark $p_T$", "Gen $b_1$ $p_T$ [GeV]"),
        ("Gen_b2_pt", "Gen_b2_pt", "Gen Subleading b-Quark $p_T$", "Gen $b_2$ $p_T$ [GeV]"),
        ("Gen_b1_eta", "Gen_b1_eta", "Gen Leading b-Quark $\\eta$", "Gen $b_1$ $\\eta$"),
        ("Gen_b2_eta", "Gen_b2_eta", "Gen Subleading b-Quark $\\eta$", "Gen $b_2$ $\\eta$"),
        ("Gen_dRbb", "Gen_dRbb", "Gen b-Quark Angular Separation", "Gen $\\Delta R_{bb}$"),
        ("Gen_V_mass", "Gen_V_mass", "Gen Vector Boson Mass (Z/W)", "Gen $m_V$ [GeV]"),
        ("Gen_Lepton1_pt", "Gen_Lepton1_pt", "Gen Leading Lepton $p_T$", "Gen lepton 1 $p_T$ [GeV]"),
        ("Gen_Lepton2_pt", "Gen_Lepton2_pt", "Gen Subleading Lepton $p_T$", "Gen lepton 2 $p_T$ [GeV]"),
        ("Gen_Lepton1_eta", "Gen_Lepton1_eta", "Gen Leading Lepton $\\eta$", "Gen lepton 1 $\\eta$"),
        ("Gen_Lepton2_eta", "Gen_Lepton2_eta", "Gen Subleading Lepton $\\eta$", "Gen lepton 2 $\\eta$"),
        ("Gen_mLL", "Gen_mLL", "Gen Dilepton Invariant Mass", "Gen $m_{\\ell\\ell}$ [GeV]"),
        ("Gen_dRLL", "Gen_dRLL", "Gen Lepton Angular Separation", "Gen $\\Delta R_{\\ell\\ell}$"),
        ("Gen_MET_pt", "Gen_MET_pt", "Gen MET (Neutrino Sum) $p_T$", "Gen MET $p_T$ [GeV]"),
        ("Gen_MET_phi", "Gen_MET_phi", "Gen MET (Neutrino Sum) $\\phi$", "Gen MET $\\phi$ [rad]"),
    ]

    all_keys = all_keys + lhe_keys + gen_keys

    for _, key, title, xlabel in all_keys:
        xmin, xmax, nbins, _ = VAR_CONFIGS.get(key, (0.0, 250.0, 100, ""))
        plot_dict[key] = {
            "counts": {ch_name: np.zeros(nbins, dtype=float) for ch_name, _, _ in CHANNELS},
            "edges": np.linspace(xmin, xmax, nbins + 1),
            "title": title,
            "xlabel": xlabel,
        }

    # Stream file in chunks using uproot.iterate
    branches_to_read = [
        "Dilepton_ok", "Dilepton_channel", "Dilepton_mLL", "Dilepton_ptLL", "Dilepton_etaLL", "Dilepton_phiLL",
        "Dilepton_dRLL", "Dilepton_dPhiLL", "PuppiMET_pt", "PuppiMET_phi",
        "GoodLepton_pt", "GoodLepton_eta", "GoodLepton_phi",
        "nGoodJet", "GoodJet_pt", "GoodJet_eta", "GoodJet_phi", "GoodJet_mass", "genWeight",
        # LHE
        "LHE_HT", "LHE_HTIncoming", "LHE_Njets",
        "LHEPart_pt", "LHEPart_eta", "LHEPart_phi", "LHEPart_mass", "LHEPart_pdgId", "LHEPart_status",
        # GenPart
        "GenPart_pt", "GenPart_eta", "GenPart_phi", "GenPart_mass", "GenPart_pdgId",
        "GenPart_status", "GenPart_statusFlags", "GenPart_genPartIdxMother",
    ]

    total_events = 0
    kept_events = 0
    kept_per_channel = {ch_name: 0 for ch_name, _, _ in CHANNELS}
    warned_no_channel = False

    for chunk in uproot.iterate({f: args.tree for f in args.input}, filter_name=branches_to_read, step_size=args.step_size):
        chunk_size = len(chunk)
        total_events += chunk_size

        if "Dilepton_ok" in chunk.fields:
            good_ll_mask = ak.to_numpy(chunk["Dilepton_ok"]) == 1
        else:
            good_ll_mask = np.ones(chunk_size, dtype=bool)

        # Jet multiplicity cut (applies to reco plots, and to truth plots unless --gen-all-events)
        if args.min_jets > 0:
            if "nGoodJet" in chunk.fields:
                n_jets = ak.to_numpy(chunk["nGoodJet"])
            else:
                n_jets = ak.to_numpy(ak.num(chunk["GoodJet_pt"], axis=1))
            good_ll_mask = good_ll_mask & (n_jets >= args.min_jets)

        if "Dilepton_channel" in chunk.fields:
            all_chans = ak.to_numpy(chunk["Dilepton_channel"])
        else:
            if not warned_no_channel:
                print("Warning: no Dilepton_channel branch; only the inclusive plots will be filled.")
                warned_no_channel = True
            all_chans = np.full(chunk_size, -1)

        if "genWeight" in chunk.fields:
            all_weights = ak.to_numpy(chunk["genWeight"]) * w_const
        else:
            all_weights = np.full(chunk_size, w_const)

        def add_weighted(key, values, weights, chans):
            """Fills the inclusive histogram and the matching per-channel histogram."""
            val_clean = np.nan_to_num(np.array(values, dtype=float), nan=-999.0)
            valid_mask = (val_clean > -900.0)
            vals, wts, chs = val_clean[valid_mask], weights[valid_mask], chans[valid_mask]
            for ch_name, ch_code, _ in CHANNELS:
                sel = slice(None) if ch_code is None else (chs == ch_code)
                c, _ = np.histogram(vals[sel], bins=plot_dict[key]["edges"], weights=wts[sel])
                plot_dict[key]["counts"][ch_name] += c

        # 0. LHE / GenPart truth-level quantities
        gen_mask = np.ones(chunk_size, dtype=bool) if args.gen_all_events else good_ll_mask
        if np.any(gen_mask):
            gen_weights = all_weights[gen_mask]
            gen_chans = all_chans[gen_mask]
            def add_gen(key, values, weights=gen_weights, chans=gen_chans):
                add_weighted(key, values, weights, chans)
            fill_lhe_histograms(chunk, gen_mask, add_gen)
            fill_genpart_histograms(chunk, gen_mask, gen_weights, gen_chans, add_gen)

        n_pass = np.sum(good_ll_mask)
        if n_pass == 0:
            continue
        kept_events += n_pass
        event_weights = all_weights[good_ll_mask]
        event_chans = all_chans[good_ll_mask]
        for ch_name, ch_code, _ in CHANNELS:
            kept_per_channel[ch_name] += n_pass if ch_code is None else int(np.sum(event_chans == ch_code))

        def add_to_histogram(key, values):
            add_weighted(key, values, event_weights, event_chans)

        # 1. Direct Boson Branches
        for b_name, key, _, _ in direct_vars:
            if b_name in chunk.fields:
                arr = ak.to_numpy(chunk[b_name][good_ll_mask])
                add_to_histogram(key, arr)

        # 2. Individual Leptons
        if "GoodLep_pt" in chunk.fields or "GoodLepton_pt" in chunk.fields:
            lep_pt = chunk["GoodLepton_pt"][good_ll_mask]
            lep_eta = chunk["GoodLepton_eta"][good_ll_mask]
            lep_phi = chunk["GoodLepton_phi"][good_ll_mask]

            for i in range(2):
                p_t = extract_indexed_array(lep_pt, i)
                eta = extract_indexed_array(lep_eta, i)
                phi = extract_indexed_array(lep_phi, i)

                if not np.all(np.isnan(p_t)):
                    add_to_histogram(f"Lepton{i+1}_pt", p_t)
                    add_to_histogram(f"Lepton{i+1}_eta", eta)
                    add_to_histogram(f"Lepton{i+1}_phi", phi)
                   # plot_dict[f"Lepton{i+1}_pt"] = (p_t, f"Lepton {i+1} Transverse Momentum", f"Lepton {i+1} $p_T$ [GeV]")
                   # plot_dict[f"Lepton{i+1}_eta"] = (eta, f"Lepton {i+1} Pseudorapidity", f"Lepton {i+1} $\\eta$")
                   # plot_dict[f"Lepton{i+1}_phi"] = (phi, f"Lepton {i+1} Azimuthal Angle", f"Lepton {i+1} $\\phi$ [rad]")

        # 3. Individual Jets & Calculated mbb, mHH
        if "GoodJet_pt" in chunk.fields:
            j_pt = chunk["GoodJet_pt"][good_ll_mask]
            j_eta = chunk["GoodJet_eta"][good_ll_mask]
            j_phi = chunk["GoodJet_phi"][good_ll_mask]
            j_m = chunk["GoodJet_mass"][good_ll_mask]

            j1_pt, j1_eta, j1_phi, j1_m = extract_indexed_array(j_pt, 0), extract_indexed_array(j_eta, 0), extract_indexed_array(j_phi, 0), extract_indexed_array(j_m, 0)
            j2_pt, j2_eta, j2_phi, j2_m = extract_indexed_array(j_pt, 1), extract_indexed_array(j_eta, 1), extract_indexed_array(j_phi, 1), extract_indexed_array(j_m, 1)

            for i in range(2):
                p_t = extract_indexed_array(j_pt, i)
                eta = extract_indexed_array(j_eta, i)
                phi = extract_indexed_array(j_phi, i)
                if not np.all(np.isnan(p_t)):
                    add_to_histogram(f"bJet{i+1}_pt", p_t)
                    add_to_histogram(f"bJet{i+1}_eta", eta)
                    add_to_histogram(f"bJet{i+1}_phi", phi)
                   # plot_dict[f"bJet{i+1}_pt"] = (p_t, f"b-Jet {i+1} Transverse Momentum", f"b-Jet {i+1} $p_T$ [GeV]")
                   # plot_dict[f"bJet{i+1}_eta"] = (eta, f"b-Jet {i+1} Pseudorapidity", f"b-Jet {i+1} $\\eta$")
                   # plot_dict[f"bJet{i+1}_phi"] = (phi, f"b-Jet {i+1} Azimuthal Angle", f"b-Jet {i+1} $\\phi$ [rad]")

            p_j1 = construct_4vectors(j1_pt, j1_eta, j1_phi, j1_m)
            p_j2 = construct_4vectors(j2_pt, j2_eta, j2_phi, j2_m)

            if HAS_VECTOR:
                mbb_calc = ak.to_numpy((p_j1 + p_j2).mass)
            else:
                mbb_calc = compute_inv_mass_numpy(p_j1, p_j2)

            add_to_histogram("mbb_calc", mbb_calc)
            #plot_dict[f"mbb_calc"] = (mbb_calc, f"Di-Jet Invariant Mass", f"$m_{bb}$ [GeV]")

            if "Dilepton_ptLL" in chunk.fields and "Dilepton_mLL" in chunk.fields and "PuppiMET_pt" in chunk.fields:
                ll_pt = ak.to_numpy(chunk["Dilepton_ptLL"][good_ll_mask])
                ll_eta = ak.to_numpy(chunk["Dilepton_etaLL"][good_ll_mask])
                ll_phi = ak.to_numpy(chunk["Dilepton_phiLL"][good_ll_mask])
                ll_m = ak.to_numpy(chunk["Dilepton_mLL"][good_ll_mask])

                met_pt = ak.to_numpy(chunk["PuppiMET_pt"][good_ll_mask])
                met_phi = ak.to_numpy(chunk["PuppiMET_phi"][good_ll_mask])

                p_ll = construct_4vectors(ll_pt, ll_eta, ll_phi, ll_m)
                p_met = construct_4vectors(met_pt, np.zeros_like(met_pt), met_phi, np.zeros_like(met_pt))

                if HAS_VECTOR:
                    mHH_calc = ak.to_numpy((p_j1 + p_j2 + p_ll + p_met).mass)
                else:
                    mHH_calc = compute_inv_mass_numpy(p_j1, p_j2, p_ll, p_met)

                add_to_histogram("mHH_calc", mHH_calc)
                #plot_dict[f"mHH_calc"]= (mHH_calc, "Di-Higgs Invariant Mass", f"$m_{HH}$ [GeV]")

        gc.collect()

    print(f"Filtering: Keeping {kept_events} / {total_events} events with valid Dilepton candidates"
          f"{f' and nGoodJet >= {args.min_jets}' if args.min_jets > 0 else ''}.")
    for ch_name, ch_code, _ in CHANNELS:
        if ch_code is not None:
            print(f"  {ch_name:>5}: {kept_per_channel[ch_name]} events")

    # 4. Generate all ROOT-styled plots (inclusive + one directory per channel)
    print(f"\nGenerating {len(plot_dict) * len(CHANNELS)} ROOT-style plots...")
    for ch_name, ch_code, ch_label in CHANNELS:
        for var_key, data_info in plot_dict.items():
            title = data_info["title"] if ch_code is None else f"{data_info['title']} ({ch_label})"
            out_file = os.path.join(args.outdir, ch_name, f"{var_key}.png")
            plot_root_style(data_info["counts"][ch_name], var_key, title, data_info["xlabel"], out_file)

    print(f"\nCompleted! All plots saved to '{args.outdir}'")


if __name__ == "__main__":
    main()
