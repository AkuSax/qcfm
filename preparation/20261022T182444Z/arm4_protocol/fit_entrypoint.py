"""Leakage-safe KLK5 nested runner. It requires explicit authorization and gate receipts."""
from __future__ import annotations
import argparse, csv, json, math, random, sys, time
import traceback
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from protocol_support import (
    ROOT, SOURCE, FoldPreprocessor, load_frozen_pooled_inputs, load_fold_splits,
    load_atom_features, require_feature_coverage, require_common_coverage,
    anchored_group_pair_loss, macro_group_mae, evaluate_predictions,
    paired_group_bootstrap_reduction, FinalReadoutEnsemble, QuantumCrossAttention,
    BoltzOnlyAtomAttention, PooledFusionReadout, AtomConcatReadout, DenseReadout,
)

METHODS = ["native", "boltz_ridge", "ordinary_readout", "boltz_atom_capacity",
           "simple_scalars", "quantum_scalars", "pooled_fusion", "atom_concat",
           "atom_attention", "quantum_only"]
SEEDS = (1729, 2718, 31415)
WEIGHT_DECAYS = (0.0001, 0.01)
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
CONTROL_METHODS = ["atom_attention_free", "atom_attention_shuffle"]
MAX_FITS = 1500

def planned_fit_count():
    """Return the frozen primary, fixed sensitivity, and control fit budget."""
    ridge_per_outer = 2 * len(RIDGE_ALPHAS) + 1
    neural_per_outer = 8 * (2 * len(WEIGHT_DECAYS) + 1)
    primary_or_sensitivity = len(SEEDS) * 5 * (ridge_per_outer + neural_per_outer)
    controls = len(SEEDS) * 5 * len(CONTROL_METHODS)
    return {"full_primary": primary_or_sensitivity,
            "fixed_sensitivity": primary_or_sensitivity,
            "no_retuning_primary_controls": controls,
            "total": 2 * primary_or_sensitivity + controls}

def reserve_fit(state, outdir, dataset, seed, fold, method, hyperparameters):
    """Count and persist an attempted fit before its first model operation."""
    if state["fit_count"] >= MAX_FITS:
        raise RuntimeError("Model-fit budget exceeded before fit attempt")
    state["fit_count"] += 1
    state["active_fit"] = {"dataset":dataset,"seed":seed,"fold":fold,
                            "method":method,"hyperparameters":hyperparameters}
    (outdir/"fit_progress.json").write_text(json.dumps(state,indent=2)+"\n",encoding="utf-8")
    return state["fit_count"]


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def grouped_pairs(cids, records):
    by_group = {}
    for i, cid in enumerate(cids):
        by_group.setdefault(records[cid]["chemical_group"], []).append(i)
    pairs = []
    for group, indices in by_group.items():
        if len(indices) > 1:
            for ii, i in enumerate(indices):
                for j in indices[ii+1:]:
                    pairs.append((group, i, j))
    return pairs


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def fit_scaler(x, rows):
    return FoldPreprocessor().fit(np.asarray(x)[rows])


def prepare_features(f, train_rows, variant="bound", return_scalers=False):
    """Fit every transformation on this inner or outer training split only."""
    f = dict(f)
    if variant == "atom_attention_free":
        f["aim_atoms"] = f["aim_atoms_free"]
        f["aim_scalars"] = f["aim_scalars_free"]
    elif variant == "atom_attention_shuffle":
        src=np.asarray(f["aim_atoms"]); shuffled=src.copy(); mask=np.asarray(f["atom_mask"],bool)
        rng=np.random.default_rng(1729)
        for i in range(len(src)):
            ix=np.flatnonzero(mask[i])
            shuffled[i,ix]=src[i,rng.permutation(ix)]
        f["aim_atoms"]=shuffled
    out = {}; scaler_state={}
    for key in ("global_boltz", "simple_scalars", "aim_scalars"):
        scaler = fit_scaler(f[key], train_rows)
        out[key] = scaler.transform(f[key]).astype(np.float32)
        scaler_state[key]={"median":scaler.median_.tolist(),"mean":scaler.mean_.tolist(),"scale":scaler.scale_.tolist()}
    mask = f["atom_mask"].astype(bool)
    for key in ("boltz_atoms", "aim_atoms"):
        arr = np.asarray(f[key], dtype=float)
        fit_atoms = arr[train_rows][mask[train_rows]]
        scaler = FoldPreprocessor().fit(fit_atoms)
        flat = arr.reshape(-1, arr.shape[-1])
        out[key] = scaler.transform(flat).reshape(arr.shape).astype(np.float32)
        scaler_state[key]={"median":scaler.median_.tolist(),"mean":scaler.mean_.tolist(),"scale":scaler.scale_.tolist()}
    out["atom_mask"] = mask
    out["allowed"] = np.asarray(f["allowed"], dtype=bool)
    return (out,scaler_state) if return_scalers else out


def pairs_for_ids(cids, records):
    """Return pairs as local row indexes for differentiable training losses."""
    return grouped_pairs(cids, records)

def metric_pairs_for_ids(cids, records):
    """Return pairs as CID keys for group-balanced validation metrics."""
    out=[]
    for group,i,j in grouped_pairs(cids,records): out.append((group,cids[i],cids[j]))
    return out


def torch_inputs(method, data, idx, device):
    ix = np.asarray(idx, dtype=int)
    def t(name, dtype=torch.float32):
        return torch.as_tensor(data[name][ix], dtype=dtype, device=device)
    b, a, g = t("boltz_atoms"), t("aim_atoms"), t("global_boltz")
    m = t("atom_mask", torch.bool); adj = t("allowed", torch.bool)
    simple, aimsc = t("simple_scalars"), t("aim_scalars")
    if method == "boltz_atom_capacity":
        return (b, b, g, adj, m)
    if method in ("atom_attention", "atom_attention_free", "atom_attention_shuffle", "pooled_fusion", "atom_concat"):
        return (b, a, g, adj, m) if method in ("atom_attention","atom_attention_free","atom_attention_shuffle") else (b, a, g, m)
    if method == "simple_scalars": return (torch.cat([g, simple], -1),)
    if method == "quantum_scalars": return (torch.cat([g, simple, aimsc], -1),)
    if method == "quantum_only":
        w = m.to(a.dtype).unsqueeze(-1)
        pooled = (a*w).sum(1)/w.sum(1).clamp_min(1.0)
        return (torch.cat([pooled, simple], -1),)
    raise ValueError(method)


def new_model(method, archive, seed, device, feature):
    set_seed(seed)
    if method == "ordinary_readout":
        model = FinalReadoutEnsemble.from_npz(archive)
    elif method == "boltz_atom_capacity": model = BoltzOnlyAtomAttention()
    elif method == "pooled_fusion": model = PooledFusionReadout()
    elif method == "atom_concat": model = AtomConcatReadout()
    elif method in ("atom_attention", "atom_attention_free", "atom_attention_shuffle"): model = QuantumCrossAttention()
    elif method == "simple_scalars": model = DenseReadout(768+feature["simple_scalars"].shape[1])
    elif method == "quantum_scalars": model = DenseReadout(768+feature["simple_scalars"].shape[1]+feature["aim_scalars"].shape[1])
    elif method == "quantum_only": model = DenseReadout(256+feature["simple_scalars"].shape[1])
    else: raise ValueError(method)
    return model.to(device)


def model_scores(method, model, data, idx, x1, x2, native, device):
    if method == "ordinary_readout":
        xa=torch.as_tensor(x1[idx],dtype=torch.float32,device=device)
        xb=torch.as_tensor(x2[idx],dtype=torch.float32,device=device)
        return model(xa,xb)
    inputs=torch_inputs(method,data,idx,device)
    residual=model(*inputs)
    if method == "quantum_only": return residual
    base=torch.as_tensor(native[idx],dtype=torch.float32,device=device)
    return base+residual


def fit_inner(method, train_ids, val_ids, cid_to_row, records, labels_by_cid,
              feature, x1, x2, native, archive, seed, wd, device,
              fit_state, outdir, dataset, fold):
    train_rows=[cid_to_row[c] for c in train_ids]; val_rows=[cid_to_row[c] for c in val_ids]
    data=prepare_features(feature,train_rows,variant=method)
    pairs=pairs_for_ids(train_ids,records)
    val_pairs=metric_pairs_for_ids(val_ids,records)
    if not pairs or not val_pairs: raise ValueError("Inner train or validation split has no pairs")
    reserve_fit(fit_state,outdir,dataset,seed,fold,method,{"weight_decay":wd,"stage":"inner"})
    model=new_model(method,archive,seed,device,feature)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=wd)
    ytrain=torch.as_tensor([labels_by_cid[c] for c in train_ids],dtype=torch.float32,device=device)
    best=float("inf"); best_state=None; best_epoch=1; best_pred=None; stale=0
    train_native=np.asarray([native[cid_to_row[c]] for c in train_ids])
    val_native=np.asarray([native[cid_to_row[c]] for c in val_ids])
    for epoch in range(1,201):
        model.train(); optimizer.zero_grad(set_to_none=True)
        scores=model_scores(method,model,data,train_rows,x1,x2,native,device)
        loss=anchored_group_pair_loss(scores,ytrain,pairs,anchor_weight=0.1)
        if not torch.isfinite(loss): raise FloatingPointError("Non-finite fit loss")
        loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            ps=model_scores(method,model,data,val_rows,x1,x2,native,device).cpu().numpy()
        pred={c:float(v) for c,v in zip(val_ids,ps)}
        yy={c:labels_by_cid[c] for c in val_ids}
        score=macro_group_mae(yy,pred,val_pairs)
        if score < best-1e-12:
            best=score; best_epoch=epoch; best_pred=pred.copy(); best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else:
            stale+=1
            if stale>=20: break
    if best_state is not None: model.load_state_dict(best_state)
    return best,best_epoch,best_pred


def build_ridge_system(train_ids, y, native_by_cid, X, id_to_row, records):
    pairs=pairs_for_ids(train_ids,records)
    if not pairs: raise ValueError("Ridge training split has no pairs")
    groups={g for g,_,_ in pairs}; pg={g:[] for g in groups}
    local={c:i for i,c in enumerate(train_ids)}
    for g,i,j in pairs: pg[g].append((i,j))
    rows=[]; targets=[]; weights=[]
    for g, plist in pg.items():
        for i,j in plist:
            ca,cb=train_ids[i],train_ids[j]
            rows.append(np.r_[X[id_to_row[ca]]-X[id_to_row[cb]],0.0])
            targets.append((y[ca]-native_by_cid[ca])-(y[cb]-native_by_cid[cb]))
            weights.append(1.0/(len(pg)*len(plist)))
    for c in train_ids:
        rows.append(np.r_[X[id_to_row[c]],1.0])
        targets.append(y[c]-native_by_cid[c])
        weights.append(0.1/len(train_ids))
    A=np.asarray(rows,float); b=np.asarray(targets,float); w=np.asarray(weights,float)
    return A,b,w


def ridge_solve(train_ids, y, native_by_cid, X, alpha, id_to_row, records):
    A,b,w=build_ridge_system(train_ids,y,native_by_cid,X,id_to_row,records)
    penalty=np.eye(A.shape[1]); penalty[-1,-1]=0
    lhs=A.T@(w[:,None]*A)+alpha*penalty
    rhs=A.T@(w*b)
    return np.linalg.solve(lhs,rhs)


def predict_ridge(coef, ids, X, native_by_cid, id_to_row):
    out={}
    for c in ids:
        v=np.r_[X[id_to_row[c]],1.0]@coef
        out[c]=native_by_cid[c]+float(v)
    return out


def require_gate_receipt(path):
    d=json.loads(Path(path).read_text(encoding="utf-8-sig"))
    required=("execution_authorized","joins_verified","aim_runtime_verified","feature_manifest_frozen")
    bad=[k for k in required if d.get(k) is not True]
    if bad: raise ValueError(f"Gate receipt lacks true fields: {bad}")
    return d


def _main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--authorize-fitting",action="store_true",help="Explicitly authorize model fitting for this run")
    ap.add_argument("--gate-receipt",type=Path)
    ap.add_argument("--features",type=Path,default=ROOT/"aim_features.npz")
    ap.add_argument("--output",type=Path,required=False)
    ap.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    ap.add_argument("--gpu-start-confirmed",action="store_true")
    args=ap.parse_args()
    if not args.authorize_fitting:
        ap.error("Fitting is disabled. Pass --authorize-fitting only after the user authorizes model execution.")
    if args.gate_receipt is None: ap.error("Pass a gate receipt after every scientific gate passes.")
    receipt=require_gate_receipt(args.gate_receipt)
    # Verify scientific evidence before the first model construction.
    sys.path.insert(0,str(ROOT.parent/'orchestrator'))
    from verify_fit_gate import verify as verify_evidence_gate
    verify_evidence_gate(receipt,args.features)
    if args.device=="cuda" and not args.gpu_start_confirmed:
        ap.error("CUDA requires explicit GPU-start confirmation when a GPU VM is needed.")
    if not args.features.exists(): ap.error(f"Missing AIM feature bundle: {args.features}")
    if args.output is None: ap.error("Pass a new output directory.")
    if args.output.exists(): ap.error("Output directory already exists. Select a new path.")
    if args.device=="cuda" and not torch.cuda.is_available(): ap.error("CUDA is not available.")
    device=torch.device(args.device)
    source_ids,x1,x2,record_rows=load_frozen_pooled_inputs()
    records={r["pubchem_cid"]:r for r in record_rows}
    feature=load_atom_features(args.features,source_ids)
    coverage=require_feature_coverage(feature["feature_valid"])
    if not np.array_equal(feature["global_boltz"],np.concatenate([x1,x2],axis=1)):
        raise ValueError("global_boltz differs from the verified saved x1+x2 inputs")
    splits=load_fold_splits()
    labels={c:float(records[c]["label"]) for c in source_ids}
    native=np.asarray([float(records[c]["unchanged"]) for c in source_ids])
    native_by_cid=dict(zip(source_ids,native))
    cid_to_row={c:i for i,c in enumerate(source_ids)}
    feature_ok=dict(zip(source_ids,feature["feature_valid"].astype(bool)))
    outdir=args.output.resolve(); outdir.mkdir(parents=True)
    ledger={r["pubchem_cid"]:r for r in read_csv(ROOT/"feature_failure_ledger_template.csv")}
    ledger_rows=[]
    for c in source_ids:
        if feature_ok[c]: ledger_rows.append({"pubchem_cid":c,"status":"covered","reason":"all required feature and join checks passed"})
        else: ledger_rows.append({"pubchem_cid":c,"status":"missing","reason":ledger[c]["failure_reason"]})
    with (outdir/"feature_coverage.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=ledger_rows[0].keys());w.writeheader();w.writerows(ledger_rows)
    all_predictions=[]; fit_state={"fit_count":0,"status":"running"}; failures=[]
    datasets={"primary":set(),"hill_sensitivity":{"899202","976329"}}
    selected_attention={}
    for ds, drop in datasets.items():
        for seed in SEEDS:
            pred_by_method={m:{} for m in METHODS + (CONTROL_METHODS if ds=="primary" else [])}
            for cid in source_ids:
                if cid not in drop: pred_by_method["native"][cid]=float(records[cid]["unchanged"])
            for fold in range(5):
                fold_rows=[s for s in splits if int(s["outer_fold"])==fold]
                role={s["pubchem_cid"]:s for s in fold_rows}
                outer_train=[c for c in source_ids if role[c]["role"]=="train" and feature_ok[c] and c not in drop]
                outer_test=[c for c in source_ids if role[c]["role"]=="test" and feature_ok[c] and c not in drop]
                active_methods=METHODS + (CONTROL_METHODS if ds=="primary" else [])
                for method in METHODS:
                    if method=="native": continue
                    if method=="boltz_ridge":
                        X=np.concatenate([x1,x2],axis=1)
                        best_alpha=None; best_loss=float("inf")
                        for alpha in RIDGE_ALPHAS:
                            merged={}
                            for iv in (0,1):
                                tr=[c for c in outer_train if int(role[c]["inner_validation_fold"])!=iv]
                                va=[c for c in outer_train if int(role[c]["inner_validation_fold"])==iv]
                                sc=FoldPreprocessor().fit(X[[cid_to_row[c] for c in tr]])
                                Xs=sc.transform(X)
                                reserve_fit(fit_state,outdir,ds,seed,fold,method,{"alpha":alpha,"inner_fold":iv})
                                coef=ridge_solve(tr,labels,native_by_cid,Xs,alpha,cid_to_row,records)
                                pv=predict_ridge(coef,va,Xs,native_by_cid,cid_to_row)
                                merged.update(pv)
                            val_ids=list(merged)
                            score=macro_group_mae({c:labels[c] for c in val_ids},merged,metric_pairs_for_ids(val_ids,records))
                            if score < best_loss: best_loss=score; best_alpha=alpha
                        sc=FoldPreprocessor().fit(X[[cid_to_row[c] for c in outer_train]]); Xs=sc.transform(X)
                        (outdir/f"scalers_{ds}_s{seed}_f{fold}_{method}.json").write_text(json.dumps({"median":sc.median_.tolist(),"mean":sc.mean_.tolist(),"scale":sc.scale_.tolist()}),encoding="utf-8")
                        reserve_fit(fit_state,outdir,ds,seed,fold,method,{"alpha":best_alpha,"stage":"outer"})
                        coef=ridge_solve(outer_train,labels,native_by_cid,Xs,best_alpha,cid_to_row,records)
                        np.save(outdir/f"coef_{ds}_s{seed}_f{fold}_{method}.npy",coef)
                        digest=hashlib.sha256("\n".join(sorted(outer_train)).encode()).hexdigest()
                        with (outdir/"outer_fit_manifest.csv").open("a",newline="",encoding="utf-8") as mf:
                            wr=csv.DictWriter(mf,fieldnames=["dataset","seed","fold","method","train_cid_sha256","weight_decay","epochs","ridge_alpha"])
                            if mf.tell()==0: wr.writeheader()
                            wr.writerow({"dataset":ds,"seed":seed,"fold":fold,"method":method,"train_cid_sha256":digest,"weight_decay":"","epochs":"","ridge_alpha":best_alpha})
                        pred_by_method[method].update(predict_ridge(coef,outer_test,Xs,native_by_cid,cid_to_row))
                        continue
                    best_wd=None;best_score=float("inf");epoch_candidates=[]
                    for wd in WEIGHT_DECAYS:
                        losses=[];epochs=[]; merged_inner={}
                        for iv in (0,1):
                            tr=[c for c in outer_train if int(role[c]["inner_validation_fold"])!=iv]
                            va=[c for c in outer_train if int(role[c]["inner_validation_fold"])==iv]
                            sc,ep,pv=fit_inner(method,tr,va,cid_to_row,records,labels,feature,x1,x2,native,
                                            SOURCE/"exact_inputs.npz",seed,wd,device,
                                            fit_state,outdir,ds,fold)
                            losses.append(sc);epochs.append(ep);merged_inner.update(pv)
                        va_ids=list(merged_inner)
                        score=macro_group_mae({c:labels[c] for c in va_ids},merged_inner,
                                              metric_pairs_for_ids(va_ids,records))
                        if score<best_score:best_score=score;best_wd=wd;epoch_candidates=epochs
                    epochs=max(1,int(round(float(np.mean(epoch_candidates)))))
                    if method=="atom_attention" and ds=="primary":
                        selected_attention[(fold,seed)]=(best_wd,epochs)
                    final_train_rows=[cid_to_row[c] for c in outer_train]
                    data,scaler_state=prepare_features(feature,final_train_rows,variant=method,return_scalers=True)
                    reserve_fit(fit_state,outdir,ds,seed,fold,method,{"weight_decay":best_wd,"epochs":epochs,"stage":"outer"})
                    model=new_model(method,SOURCE/"exact_inputs.npz",seed,device,feature)
                    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=best_wd)
                    ix=np.asarray(final_train_rows,dtype=int); y=torch.as_tensor([labels[c] for c in outer_train],dtype=torch.float32,device=device)
                    pairs=pairs_for_ids(outer_train,records)
                    if not pairs: raise ValueError(f"No outer training pairs for fold {fold}")
                    for _ in range(epochs):
                        model.train();optimizer.zero_grad(set_to_none=True)
                        score=model_scores(method,model,data,ix,x1,x2,native,device)
                        loss=anchored_group_pair_loss(score,y,pairs,0.1);loss.backward();optimizer.step()
                    model.eval()
                    with torch.no_grad():
                        scores=model_scores(method,model,data,[cid_to_row[c] for c in outer_test],x1,x2,native,device).cpu().numpy()
                    pred_by_method[method].update({c:float(v) for c,v in zip(outer_test,scores)})
                    torch.save(model.state_dict(),outdir/f"model_{ds}_s{seed}_f{fold}_{method}.pt")
                    digest=hashlib.sha256("\n".join(sorted(outer_train)).encode()).hexdigest()
                    with (outdir/"outer_fit_manifest.csv").open("a",newline="",encoding="utf-8") as mf:
                        wr=csv.DictWriter(mf,fieldnames=["dataset","seed","fold","method","train_cid_sha256","weight_decay","epochs","ridge_alpha"])
                        if mf.tell()==0: wr.writeheader()
                        wr.writerow({"dataset":ds,"seed":seed,"fold":fold,"method":method,"train_cid_sha256":digest,"weight_decay":best_wd,"epochs":epochs})
                    (outdir/f"scalers_{ds}_s{seed}_f{fold}_{method}.json").write_text(json.dumps(scaler_state),encoding="utf-8")
                if ds=="primary":
                    for control in CONTROL_METHODS:
                        wd,epochs=selected_attention[(fold,seed)]
                        rows=[cid_to_row[c] for c in outer_train]
                        data,scaler_state=prepare_features(feature,rows,variant=control,return_scalers=True)
                        reserve_fit(fit_state,outdir,ds,seed,fold,control,{"weight_decay":wd,"epochs":epochs,"stage":"outer_control"})
                        model=new_model(control,SOURCE/"exact_inputs.npz",seed,device,feature)
                        opt=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=wd)
                        yy=torch.as_tensor([labels[c] for c in outer_train],dtype=torch.float32,device=device)
                        pp=pairs_for_ids(outer_train,records)
                        for _ in range(epochs):
                            model.train();opt.zero_grad(set_to_none=True)
                            pred=model_scores(control,model,data,rows,x1,x2,native,device)
                            loss=anchored_group_pair_loss(pred,yy,pp,0.1);loss.backward();opt.step()
                        model.eval()
                        with torch.no_grad():
                            scores=model_scores(control,model,data,[cid_to_row[c] for c in outer_test],x1,x2,native,device).cpu().numpy()
                        pred_by_method[control].update({c:float(v) for c,v in zip(outer_test,scores)})
                        torch.save(model.state_dict(),outdir/f"model_{ds}_s{seed}_f{fold}_{control}.pt")
                        (outdir/f"scalers_{ds}_s{seed}_f{fold}_{control}.json").write_text(json.dumps(scaler_state),encoding="utf-8")
                        digest=hashlib.sha256("\n".join(sorted(outer_train)).encode()).hexdigest()
                        with (outdir/"outer_fit_manifest.csv").open("a",newline="",encoding="utf-8") as mf:
                            wr=csv.DictWriter(mf,fieldnames=["dataset","seed","fold","method","train_cid_sha256","weight_decay","epochs","ridge_alpha"])
                            if mf.tell()==0: wr.writeheader()
                            wr.writerow({"dataset":ds,"seed":seed,"fold":fold,"method":control,"train_cid_sha256":digest,"weight_decay":wd,"epochs":epochs,"ridge_alpha":""})
                # Retain all completed fold predictions before attempting later folds.
                (outdir/f"partial_predictions_{ds}_s{seed}.json").write_text(
                    json.dumps({m:{c:v for c,v in p.items()} for m,p in pred_by_method.items()},indent=2)+"\n",encoding="utf-8")
            try:
                primary_pairs=read_csv(ROOT/("primary_pairs.csv" if ds=="primary" else "sensitivity_pairs.csv"))
                pairs=[(int(p["outer_fold"]),p["chemical_group"],p["cid_a"],p["cid_b"]) for p in primary_pairs]
                common,common_pairs=require_common_coverage(pred_by_method,source_ids,pairs)
                ycommon={c:labels[c] for c in common}
                plain_pairs=[(g,a,b) for _,g,a,b in common_pairs]
                for m in active_methods:
                    for c in source_ids:
                        all_predictions.append({"dataset":ds,"seed":seed,"method":m,"pubchem_cid":c,
                          "outer_fold":next(int(r["outer_fold"]) for r in splits if r["pubchem_cid"]==c and r["role"]=="test"),
                          "prediction":pred_by_method[m].get(c,""),"status":"predicted" if c in pred_by_method[m] else "missing"})
                for m in active_methods:
                    scores={c:pred_by_method[m][c] for c in common}
                    met=evaluate_predictions(ycommon,scores,plain_pairs)
                    with (outdir/"metrics.csv").open("a",newline="",encoding="utf-8") as f:
                        w=csv.DictWriter(f,fieldnames=["dataset","seed","method","common_records","common_pairs",*met.keys()]);
                        if f.tell()==0:w.writeheader()
                        w.writerow({"dataset":ds,"seed":seed,"method":m,"common_records":len(common),"common_pairs":len(common_pairs),**met})
                candidates=active_methods[1:]
                comparisons=[(candidate,comp) for candidate in candidates for comp in ("boltz_ridge","ordinary_readout")]
                if ds=="primary":
                    comparisons += [("atom_attention",comp) for comp in
                                    ("pooled_fusion","atom_concat","boltz_atom_capacity","simple_scalars","quantum_scalars")]
                if ds=="primary": comparisons += [(c,"atom_attention") for c in CONTROL_METHODS]
                for candidate,comparator in comparisons:
                    if candidate==comparator: continue
                    for comparator in (comparator,):
                        ci=paired_group_bootstrap_reduction(ycommon,pred_by_method[candidate],pred_by_method[comparator],plain_pairs)
                        with (outdir/"bootstrap.csv").open("a",newline="",encoding="utf-8") as f:
                            w=csv.DictWriter(f,fieldnames=["dataset","seed","candidate","comparator",*ci.keys()]);
                            if f.tell()==0:w.writeheader()
                            w.writerow({"dataset":ds,"seed":seed,"candidate":candidate,"comparator":comparator,**ci})
                with (outdir/"common_identities.csv").open("a",newline="",encoding="utf-8") as f:
                    w=csv.DictWriter(f,fieldnames=["dataset","seed","pubchem_cid","included"]);
                    if f.tell()==0:w.writeheader()
                    w.writerows({"dataset":ds,"seed":seed,"pubchem_cid":c,"included":c in common} for c in source_ids)
            except Exception as e:
                failures.append({"dataset":ds,"seed":seed,"stage":"common_metrics","error":repr(e)})
                raise
            with (outdir/"predictions.csv").open("a",newline="",encoding="utf-8") as f:
                w=csv.DictWriter(f,fieldnames=all_predictions[0].keys());
                if f.tell()==0:w.writeheader()
                w.writerows(all_predictions);all_predictions=[]
            (outdir/"fit_progress.json").write_text(json.dumps({"fit_count":fit_state["fit_count"],"dataset":ds,"seed":seed},indent=2)+"\n")
    if fit_state["fit_count"]>MAX_FITS: raise RuntimeError("Model-fit budget exceeded")
    (outdir/"fit_manifest.json").write_text(json.dumps({"status":"complete","fit_count":fit_state["fit_count"],"fit_cap":MAX_FITS,
      "seeds":SEEDS,"outer_folds":5,"inner_folds":2,"methods":METHODS,"primary_controls":CONTROL_METHODS,"fit_schedule":{"full_primary":735,"fixed_sensitivity":735,"no_retuning_primary_controls":30,"total":1500},"feature_coverage":coverage,
      "gate_receipt":str(args.gate_receipt.resolve()),"device":str(device),"model_calls":fit_state["fit_count"],
      "limitations":["KLK5 is development-only; training overlap is unresolved.","This baseline adapts final readout layers only.","It is not published full affinity-head adaptation."]},indent=2)+"\n")
    return 0

def main():
    try:
        return _main()
    except Exception as exc:
        # Preserve a machine-readable failure receipt after any future partial run.
        try:
            argv=sys.argv[1:]
            if "--output" in argv:
                out=Path(argv[argv.index("--output")+1])
                if out.is_dir():
                    (out/"failure.json").write_text(json.dumps({
                        "status":"failed", "error":repr(exc),
                        "traceback":traceback.format_exc()
                    },indent=2)+"\n",encoding="utf-8")
        except Exception:
            pass
        raise

if __name__=="__main__": raise SystemExit(main())
