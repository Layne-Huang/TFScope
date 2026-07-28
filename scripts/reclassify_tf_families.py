"""Curated rule-based TF DBD-family classifier (fixes the broken Pfam rebin34 scheme).
Keeps the correct C2H2 short/medium/long split from the existing data; reclassifies all
other TFs by gene symbol to their real DBD family (Lambert et al. 2018 taxonomy)."""
import re, pandas as pd, numpy as np

# explicit gene -> family (tricky / non-prefix cases), checked first
EXPLICIT = {}
def add(fam,*genes):
    for g in genes: EXPLICIT[g]=fam
add("HMG/SOX","SRY","LEF1","HBP1","BBX","HMG20A","HMG20B","HMGA1","HMGA2","TCF7","TCF7L1","TCF7L2")
add("bHLH","MAX","MNT","MXI1","MXD4","MLX","MLXIP","MLXIPL","MGA","MITF","TFE3","TFEB","TFEC","TFAP4",
    "AHR","AHRR","ARNT","ARNT2","ARNTL","EPAS1","HIF1A","HIF3A","NPAS2","NPAS3","NPAS4","CLOCK",
    "TAL1","TAL1::TCF3","LYL1","TCF3","TCF4","TCF12","TCF21","TCFL5","MSC","MSGN1","MESP1","MESP2",
    "PTF1A","SOHLH2","FIGLA","FERD3L","MYC","MYCN","MYC2","MAX::MYC","MAX_MYC","USF1","USF2","USF3",
    "SREBF1","SREBF2","MDL-1","MYOD1","MYOG","MYF5","MYF6")
add("bZIP","NRL","XBP1","NFE2","NFE2L1","NFE2L2","NFIL3","DBP","HLF","TEF","JDP2","GCN4","SKN-1","SPZ1",
    "CG18619","CREBZF","CREM")
add("T-box","T","TBXT","TBR1","EOMES")
add("MADS/SRF","SRF","MCM1")
add("Grainyhead/CP2","TFCP2","TFCP2L1","UBP1")
add("CSL","RBPJ","SU(H)")
add("EBF","EBF1","EBF2","EBF3")
add("TBP","TBP","TAF1","SPT15")
add("ARID","ARID3A","ARID3B","ARID5A","ARID5B")
add("AP-2","TFAP2A","TFAP2B","TFAP2C","TFAP2E")
add("NFI","NFIA","NFIB","NFIC","NFIX")
add("MBD","MBD1","MBD2","MBD3","MECP2")
add("CAMTA","CAMTA1","CAMTA2")
add("SAND","SP100","SP140","SP140L","AIRE","DEAF1","GMEB1","GMEB2")
add("CxxC","CXXC1","CXXC4","KDM2A","KDM2B","KMT2A","KMT2B","DNMT1","TET1","TET3")
add("CENPB/THAP-like","CENPB","JRK","POGK","FLYWCH1","TIGD3","TIGD4","TIGD5","TIGD7","ZBED1","ZBED4","ZBED5",
    "ZBED2","FAM200B","POGZ","HARBI1","BANP")
add("Nuclear_Receptor","AR","ANDR","ECR::USP")
add("Homeodomain","HNF1A","HNF1B","CUX1","CUX2","ONECUT1","ONECUT2","ONECUT3","SATB1","SATB2","ZEB1","ZEB2",
    "ZFHX2","ZFHX3","ZHX1","NANOG","MIXL1","PROX1","PROP1","HOMEZ","HMBOX1","LEUTX","TTF1","PDX1",
    "ABD-B","ANTP","AL","BCD","EVE","EXD","SCR","UBX","VND","MATALPHA2","PHA2","TPRX1")
add("Myb/SANT","MYB","MYBL1","MYBL2","MSANTD1","MSANTD3","MSANTD4","MYPOP","TERF1","TERF2","Z")
add("WRKY","WRKY1","WRKY2","WRKY33")
add("STAT","STAT1","STAT2","STAT3","STAT4","STAT5A","STAT5B","STAT6")
add("NDT80","NDT80","MYRFL","MYRF")
add("NF-Y/CBF","NFYA","NFYB","NFYC","CEBPZ")
add("Homeodomain","EN")
# misc -> Other handled by fallback

PREFIX = [  # (regex on gene, family) — order matters, specific first
    (r"^POU\d","POU"),
    (r"^(HOX|DLX|LHX|NKX|SIX|PITX|OTX|OTP|EMX|MSX|GSX|GSC|CDX|ALX|ARX|VAX|GBX|IRX|ISL|ISX|LBX|LMX|MEIS|PBX|PKNOX|TGIF|MEOX|MNX|PHOX|PRRX|RAX|RHOXF|SHOX|TLX|TSHZ|UNCX|VENTX|VSX|BARHL|BARX|BSX|CART|CRX|DBX|DMBX|DPRX|DRGX|DUX|ESX|EVX|HESX|HLX|HMX|EN\d|NOBOX|NOTO|SEBOX|MKX|VND|ANHX|ARGFX|ZHX)","Homeodomain"),
    (r"^PAX\d","PAX"),
    (r"^FOX","Forkhead"),
    (r"^HSF","HSF"),
    (r"^(SOX|HMGB)","HMG/SOX"),
    (r"^(ETS|ELF|ELK|ETV|FLI|ERG|ERF|FEV|GABP|SPI|SPDEF|EHF|EWSR1)","ETS"),
    (r"^GATA","GATA"),
    (r"^TBX","T-box"),
    (r"^(NFKB|REL|NFAT)","RHD/NFkB"),
    (r"^IRF","IRF"),
    (r"^RUNX","Runt"),
    (r"^SMAD","SMAD"),
    (r"^(TP5|TP6|TP7|P53)","p53"),
    (r"^RFX","RFX"),
    (r"^MEF2","MADS/SRF"),
    (r"^GRHL","Grainyhead/CP2"),
    (r"^(E2F|TFDP)","E2F/DP"),
    (r"^TEAD","TEA/TEAD"),
    (r"^THAP","THAP"),
    (r"^DMRT","DMRT"),
    (r"^GCM","GCM"),
    (r"^CREB","bZIP"),(r"^ATF","bZIP"),(r"^CEBP[ABDEG]$","bZIP"),(r"^(JUN|FOS|MAF|BACH|BATF)","bZIP"),
    (r"^(ASCL|ATOH|NEUROD|NEUROG|OLIG|BHLH|HES|HEY|HAND|TWIST|NHLH)","bHLH"),
    (r"^NR\d","Nuclear_Receptor"),(r"^(ESR|ESRR|PPAR|RAR|RXR|THR|ROR|HNF4)","Nuclear_Receptor"),(r"^(VDR|PGR)$","Nuclear_Receptor"),
    (r"^WRKY","WRKY"),
    (r"^TFAP2","AP-2"),
    (r"^CAMTA","CAMTA"),
]
def classify(gene, c2h2_existing):
    g=str(gene).upper()
    if g in c2h2_existing: return c2h2_existing[g]   # keep correct C2H2 short/med/long split
    if g in EXPLICIT: return EXPLICIT[g]
    for pat,fam in PREFIX:
        if re.match(pat,g): return fam
    # remaining C2H2-type zinc fingers not in the existing split -> by ZNF/ZFP/ZBTB/etc.
    if re.match(r"^(ZNF|ZFP|ZBTB|ZSCAN|ZKSCAN|ZIC|ZIK|ZIM|ZXD|ZNF|KLF|SP\d|EGR|GLI|GLIS|SNAI|IKZF|WT1|YY|CTCF|PRDM|SALL|INSM|MZF|HIC|VEZF|REST|GFI|OVOL|SCRT|BCL11|FEZF|OSR|ZFPM|ZFAT|ZFX|RREB|MECOM|EVI1|PATZ|PLAG|CASZ|TRPS|HKR|HINFP|RBAK|RLF|MTF1|E4F1|ATMIN|BNC2|CGGBP|GTF3A)",g):
        return "C2H2_long"
    return "Other"

if __name__=="__main__":
    rb=pd.read_parquet("data/processed/tf_pwm_aug_dbd_canon_trim_rebin34.parquet")
    rb["g"]=rb["gene_symbol"].fillna("").astype(str).str.upper()
    # existing C2H2 split: gene -> family
    c2={}
    for _,r in rb[rb["family_name_rebin"].isin(["C2H2_short","C2H2_medium","C2H2_long"])].iterrows():
        c2[str(r["gene_symbol"]).upper()]=r["family_name_rebin"]
    rb["new_fam"]=rb["g"].map(lambda x:classify(x,c2))
    import collections
    print("=== NEW fine family scheme: record & gene counts ===")
    for fam,n in collections.Counter(rb["new_fam"]).most_common():
        ng=rb[rb["new_fam"]==fam]["g"].nunique()
        print(f"  {fam:<18} records={n:<5} genes={ng}")
    # show 'Other' members (should be genuinely misc now) + flag any large unexpected
    oth=sorted(set(rb[rb["new_fam"]=="Other"]["g"]))
    print(f"\n'Other' ({len(oth)} genes): {', '.join(oth)}")
    rb.drop(columns=["g"]).to_parquet("/tmp/reclassified_preview.parquet")
