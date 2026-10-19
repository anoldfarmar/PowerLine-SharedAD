#!/usr/bin/env python3
"""CPLID-T0T1T2-detector-top8-v1 data audit.
Scans real directories, validates image/XML correspondence and boxes, builds
exact-duplicate and pre-registered near-duplicate (dHash64 Hamming<=10) source
groups, audits normal-vs-defective confirmable source sharing. Machine-readable
outputs only; split/freeze happens in cplid_freeze.py AFTER this audit.
Pre-registered audit rules (frozen before any split/metric):
  - XML matched to image by stem; XML <path> field ignored; <filename> may lack extension.
  - exact duplicate = identical file sha256; representative = lexicographically min path.
  - near-duplicate edge = dHash64 Hamming distance <= 10 (conservative over-approximation
    for ISOLATION SAFETY; chains allowed via union-find).
  - adjacency in numbering is NEVER used as evidence; low similarity NEVER proves independence
    (recorded as unknown).
  - a normal source group is EXCLUDED from T1 train/val/bank candidates iff it has a
    confirmed source share with a defective test image (exact content or Hamming<=10).
"""
import hashlib, json, os, sys, time, faulthandler
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

faulthandler.dump_traceback_later(120, exit=True)
PROG = bool(os.environ.get("AUDIT_VERBOSE"))
def tick(msg):
    if PROG:
        print("[tick] %s" % msg, file=sys.stderr, flush=True)

ROOT = Path(os.environ.get("CPLID_ROOT", "/root/data_cplid_local"))
OUT = Path("/root/autodl-tmp/version10-12/CPLID-T0T1T2-detector-top8-v1")
NEAR_THRESH = 10


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def dhash64(p):
    im = Image.open(p).convert("L").resize((9, 8), Image.LANCZOS)
    a = np.asarray(im, np.int16)
    d = a[:, 1:] > a[:, :-1]
    return int(np.packbits(d.flatten()).tobytes().hex(), 16) if False else int.from_bytes(
        np.packbits(d.flatten()).tobytes(), "big")


def hamming(a, b):
    return bin(a ^ b).count("1")


def parse_voc(xml_path):
    root = ET.parse(xml_path).getroot()
    fn = (root.findtext("filename") or "").strip()
    sz = root.find("size")
    w = sz.findtext("width"); h = sz.findtext("height")
    boxes = []
    names = []
    for obj in root.findall("object"):
        nm = (obj.findtext("name") or "").strip()
        bb = obj.find("bndbox")
        try:
            x1 = float(bb.findtext("xmin")); y1 = float(bb.findtext("ymin"))
            x2 = float(bb.findtext("xmax")); y2 = float(bb.findtext("ymax"))
        except (TypeError, AttributeError, ValueError):
            boxes.append(None); names.append(nm); continue
        boxes.append([x1, y1, x2, y2]); names.append(nm)
    return dict(filename_stem=Path(fn).stem if fn else None,
                 xml_width=w, xml_height=h, boxes=boxes, names=names)


def scan_side(side):
    base = ROOT / side
    imgs = sorted(base.glob("images/*"))
    imgs = [p for p in imgs if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    xmls = sorted(p for p in base.glob("labels/*.xml")) if side == "Normal_Insulators" else \
        sorted(p for p in (base / "labels").rglob("*.xml"))
    return imgs, xmls


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    audit = dict(pre_registered=dict(near_dup_dhash_hamming_le=NEAR_THRESH,
                                      xml_stem_match=True, xml_path_field_ignored=True,
                                      numbering_adjacency_not_used=True,
                                      low_similarity_never_proves_independence=True),
                  expectations=dict(normal_images=600, normal_xml=600, defective_images=248,
                                     defect_label_xml_sets=["insulator", "defect"]),
                  observed={})
    # ---------- normal side ----------
    nimgs, nxmls = scan_side("Normal_Insulators")
    audit["observed"]["normal_images"] = len(nimgs)
    audit["observed"]["normal_xml"] = len(nxmls)
    nrec = {}
    sha_of = {}
    dh_of = {}
    for p in nimgs:
        tick("normal %s" % p.name)
        im = Image.open(p); im.load()
        sha_of[str(p)] = sha256_file(p)
        dh_of[str(p)] = dhash64(p)
        nrec[str(p)] = dict(path=str(p), side="normal", sha256=sha_of[str(p)],
                             bytes=p.stat().st_size, width=im.size[0], height=im.size[1],
                             mode=im.mode, ins_boxes=[], ins_names=[], xml=None, issues=[])
    xml_by_stem = {}
    for x in nxmls:
        r = parse_voc(x)
        stem = r["filename_stem"] or x.stem
        xml_by_stem[stem] = (x, r)
    unmatched_xml = []
    for p in nimgs:
        r = nrec[str(p)]
        stem = p.stem
        if stem not in xml_by_stem:
            r["issues"].append("no_xml")
            continue
        x, v = xml_by_stem[stem]
        r["xml"] = str(x)
        if v["xml_width"] and int(float(v["xml_width"])) != r["width"]:
            r["issues"].append("xml_size_mismatch_w")
        if v["xml_height"] and int(float(v["xml_height"])) != r["height"]:
            r["issues"].append("xml_size_mismatch_h")
        for b, nm in zip(v["boxes"], v["names"]):
            if b is None:
                r["issues"].append("unparseable_box")
                continue
            x1, y1, x2, y2 = b
            ok = (0 < x2 - x1) and (0 < y2 - y1) and x1 >= 0 and y1 >= 0 and \
                x2 <= r["width"] + 2 and y2 <= r["height"] + 2
            if not ok:
                r["issues"].append("invalid_box:%s" % nm)
                continue
            r["ins_boxes"].append([round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)])
            r["ins_names"].append(nm)
    # ---------- defective side ----------
    dimgs, _ = scan_side("Defective_Insulators")
    ins_xmls = sorted((ROOT / "Defective_Insulators/labels/insulator").glob("*.xml"))
    def_xmls = sorted((ROOT / "Defective_Insulators/labels/defect").glob("*.xml"))
    audit["observed"]["defective_images"] = len(dimgs)
    audit["observed"]["defective_xml_insulator"] = len(ins_xmls)
    audit["observed"]["defective_xml_defect"] = len(def_xmls)
    drec = {}
    for p in dimgs:
        tick("defective %s" % p.name)
        im = Image.open(p); im.load()
        sha_of[str(p)] = sha256_file(p)
        dh_of[str(p)] = dhash64(p)
        drec[str(p)] = dict(path=str(p), side="defective", sha256=sha_of[str(p)],
                             bytes=p.stat().st_size, width=im.size[0], height=im.size[1],
                             mode=im.mode, ins_boxes=[], ins_names=[], def_boxes=[], def_names=[],
                             xml=None, issues=[])
    for label_set, box_key, nm_key in (("insulator", "ins_boxes", "ins_names"),
                                        ("defect", "def_boxes", "def_names")):
        xstem = {}
        for x in (ins_xmls if label_set == "insulator" else def_xmls):
            v = parse_voc(x)
            stem = v["filename_stem"] or x.stem
            xstem[stem] = (x, v)
        for p in dimgs:
            r = drec[str(p)]
            if p.stem not in xstem:
                r["issues"].append("no_xml_%s" % label_set)
                continue
            x, v = xstem[p.stem]
            if label_set == "insulator":
                r["xml"] = str(x)
            for b, nm in zip(v["boxes"], v["names"]):
                if b is None:
                    r["issues"].append("unparseable_box_%s" % label_set)
                    continue
                x1, y1, x2, y2 = b
                ok = (0 < x2 - x1) and (0 < y2 - y1) and x1 >= 0 and y1 >= 0 and \
                    x2 <= r["width"] + 2 and y2 <= r["height"] + 2
                if not ok:
                    r["issues"].append("invalid_box_%s:%s" % (label_set, nm))
                    continue
                r[box_key].append([round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)])
                r[nm_key].append(nm)
    name_counts = {}
    for r in list(drec.values()) + list(nrec.values()):
        for nm in r.get("ins_names", []) + r.get("def_names", []):
            name_counts[nm] = name_counts.get(nm, 0) + 1
    audit["observed"]["object_name_counts"] = name_counts
    # ---------- exact duplicates ----------
    parent = {}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    allp = sorted(list(nrec) + list(drec))
    for p in allp:
        parent[p] = p
    by_sha = {}
    for p in allp:
        by_sha.setdefault(sha_of[p], []).append(p)
    exact_dup_groups = 0
    for s, ps in by_sha.items():
        if len(ps) > 1:
            exact_dup_groups += 1
            for q in ps[1:]:
                union(ps[0], q)
    # ---------- near-duplicate groups among normals (pre-registered) ----------
    npaths = sorted(nrec)
    ndh = [dh_of[p] for p in npaths]
    near_edges = 0
    for i in range(len(npaths)):
        for j in range(i + 1, len(npaths)):
            if hamming(ndh[i], ndh[j]) <= NEAR_THRESH:
                union(npaths[i], npaths[j]); near_edges += 1
    # ---------- cross-side sharing audit ----------
    dpaths = sorted(drec)
    ddh = [dh_of[p] for p in dpaths]
    cross = []
    cross_exact = 0
    for k, dp in enumerate(dpaths):
        tick("cross %d/%d %s" % (k, len(dpaths), dp))
        best = 65; bestp = None
        for i, np_ in enumerate(npaths):
            if sha_of[np_] == sha_of[dp]:
                best = 0; bestp = np_; cross_exact += 1
                break
            h = hamming(ddh[k], ndh[i])
            if h < best:
                best = h; bestp = np_
        cross.append(dict(path=dp, min_hamming=int(best), matched_normal=bestp,
                           status="confirmed_shared_source" if best <= NEAR_THRESH else "unknown"))
        if best <= NEAR_THRESH:
            union(dp, bestp)
    shared_normals = sorted(set(c["matched_normal"] for c in cross if c["status"] == "confirmed_shared_source"))
    # ---------- groups ----------
    groups = {}
    for p in allp:
        groups.setdefault(find(p), []).append(p)
    glist = []
    for root_p, members in sorted(groups.items(), key=lambda kv: kv[1][0]):
        nm = [m for m in members if m in nrec]
        dm = [m for m in members if m in drec]
        has_exact = any(len(by_sha[sha_of[m]]) > 1 for m in members)
        has_near = False
        for a_i in range(len(members)):
            for b_i in range(a_i + 1, len(members)):
                if members[a_i] in dh_of and members[b_i] in dh_of and \
                        hamming(dh_of[members[a_i]], dh_of[members[b_i]]) <= NEAR_THRESH:
                    has_near = True
        edge_types = "exact+near" if (has_exact and has_near) else \
            ("exact" if has_exact else ("near" if has_near else "single"))
        glist.append(dict(group_id="g%04d" % len(glist), representative=min(members),
                           normal_members=nm, defective_members=dm,
                           edge_types=edge_types,
                           excluded_from_T1=any(x in shared_normals for x in nm)))
    for gid, g in enumerate(glist):
        for m in g["normal_members"]:
            (nrec[m] if m in nrec else drec[m])["group_id"] = g["group_id"]
        for m in g["defective_members"]:
            drec[m]["group_id"] = g["group_id"]
    # ---------- manifest ----------
    man = OUT / "DATA_MANIFEST.jsonl"
    with man.open("w") as f:
        for p in npaths:
            f.write(json.dumps(nrec[p], ensure_ascii=False, sort_keys=True) + "\n")
        for p in dpaths:
            f.write(json.dumps(drec[p], ensure_ascii=False, sort_keys=True) + "\n")
        for x in sorted(set(list(nxmls) + list(ins_xmls) + list(def_xmls))):
            f.write(json.dumps(dict(path=str(x), kind="xml", sha256=sha256_file(x)),
                                ensure_ascii=False, sort_keys=True) + "\n")
    # ---------- summary ----------
    issue_counts = {}
    for r in list(nrec.values()) + list(drec.values()):
        for i in r["issues"]:
            issue_counts[i.split(":")[0]] = issue_counts.get(i.split(":")[0], 0) + 1
    audit["observed"]["issue_counts"] = issue_counts
    audit["groups"] = dict(total=len(glist),
                            normal_only=sum(1 for g in glist if g["normal_members"] and not g["defective_members"]),
                            mixed=sum(1 for g in glist if g["normal_members"] and g["defective_members"]),
                            defective_only=sum(1 for g in glist if g["defective_members"] and not g["normal_members"]),
                            near_edges_within_normal=near_edges, exact_dup_groups=exact_dup_groups,
                            cross_side_exact=cross_exact,
                            confirmed_shared_normals=len(shared_normals),
                            excluded_group_count=sum(1 for g in glist if g["excluded_from_T1"]))
    hist = {}
    for c in cross:
        b = c["min_hamming"]
        hist["%02d" % min(b, 21) if b <= 20 else "20+"] = hist.get("%02d" % min(b, 21) if b <= 20 else "20+", 0) + 1
    audit["cross_min_hamming_hist"] = dict(sorted(hist.items()))
    audit["shared_normal_paths"] = shared_normals
    audit["webdriver_dir_entries"] = len(list((ROOT / "OriginalImageOnWebDriver").glob("*"))) if (ROOT / "OriginalImageOnWebDriver").exists() else None
    audit["elapsed_s"] = time.time() - t0
    audit["audit_ok"] = (len(nimgs) == 600 and len(nxmls) == 600 and len(dimgs) == 248
                          and len(ins_xmls) == 248 and len(def_xmls) == 248)
    write_json = OUT / "AUDIT_SUMMARY.json"
    tmp = write_json.with_suffix(".tmp"); tmp.write_text(json.dumps(audit, ensure_ascii=False, indent=1)); tmp.replace(write_json)
    print(json.dumps({k: audit[k] for k in ("observed", "groups", "cross_min_hamming_hist", "audit_ok")},
                      ensure_ascii=False, indent=1)[:3000])


if __name__ == "__main__":
    main()
