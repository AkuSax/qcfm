from pathlib import Path
p=Path('fit_entrypoint.py');s=p.read_text(encoding='utf-8-sig')
s=s.replace('def pairs_for_ids(cids, records):\n    return grouped_pairs(cids, records)', 'def pairs_for_ids(cids, records):\n    """Return pairs as local row indexes for differentiable training losses."""\n    return grouped_pairs(cids, records)\n\ndef metric_pairs_for_ids(cids, records):\n    """Return pairs as CID keys for group-balanced validation metrics."""\n    out=[]\n    for group,i,j in grouped_pairs(cids,records): out.append((group,cids[i],cids[j]))\n    return out')
s=s.replace('return (b, a, g, adj, m) if method == "atom_attention" else (b, a, g, m)', 'return (b, a, g, adj, m) if method in ("atom_attention","atom_attention_free","atom_attention_shuffle") else (b, a, g, m)')
s=s.replace('    base=torch.as_tensor(native[idx],dtype=torch.float32,device=device)\n    return base+residual', '    if method == "quantum_only": return residual\n    base=torch.as_tensor(native[idx],dtype=torch.float32,device=device)\n    return base+residual')
# Remove unused train_model entirely.
a=s.index('\ndef train_model('); b=s.index('\ndef fit_inner(',a); s=s[:a]+s[b:]
s=s.replace('feature, x1, x2, native, archive, seed, wd, device, tracker):', 'feature, x1, x2, native, archive, seed, wd, device):')
s=s.replace('    val_pairs=pairs_for_ids(val_ids,records)', '    val_pairs=metric_pairs_for_ids(val_ids,records)')
s=s.replace('    best=float("inf"); best_state=None; best_epoch=1; stale=0', '    best=float("inf"); best_state=None; best_epoch=1; best_pred=None; stale=0')
s=s.replace('            best=score; best_epoch=epoch; best_state=', '            best=score; best_epoch=epoch; best_pred=pred.copy(); best_state=')
s=s.replace('    return best,best_epoch\n\n\ndef ridge_solve', '    return best,best_epoch,best_pred\n\n\ndef build_ridge_system(train_ids, y, native_by_cid, X, id_to_row, records):')
s=s.replace('            rows.append(X[id_to_row[ca]]-X[id_to_row[cb]])', '            rows.append(np.r_[X[id_to_row[ca]]-X[id_to_row[cb]],0.0])')
s=s.replace('        rows.append(X[id_to_row[c]])', '        rows.append(np.r_[X[id_to_row[c]],1.0])')
s=s.replace('    # Include an intercept. Pair rows cancel it. The score anchor estimates it.\n    A=np.column_stack([A,np.ones(len(A))]); penalty=np.eye(A.shape[1]); penalty[-1,-1]=0\n    lhs=A.T@(w[:,None]*A)+alpha*penalty\n    rhs=A.T@(w*b)\n    coef=np.linalg.solve(lhs,rhs)\n    return coef', '    return A,b,w\n\n\ndef ridge_solve(train_ids, y, native_by_cid, X, alpha, id_to_row, records):\n    A,b,w=build_ridge_system(train_ids,y,native_by_cid,X,id_to_row,records)\n    penalty=np.eye(A.shape[1]); penalty[-1,-1]=0\n    lhs=A.T@(w[:,None]*A)+alpha*penalty\n    rhs=A.T@(w*b)\n    return np.linalg.solve(lhs,rhs)')
s=s.replace('            v=np.r_[X[id_to_row[c]],1.0]@coef', '            v=np.r_[X[id_to_row[c]],1.0]@coef')
s=s.replace('fold_losses.append(macro_group_mae({c:labels[c] for c in va},pv,pairs_for_ids(va,records)))', 'fold_losses.extend((metric_pairs_for_ids(va,records),pv))')
# Fix the ridge val aggregation after inner loops.
s=s.replace('                            if float(np.mean(fold_losses))<best_loss: best_loss=float(np.mean(fold_losses));best_alpha=alpha', '                            val_union=[c for c in outer_train if int(role[c]["inner_validation_fold"]) in (0,1)]\n                            # Score each inner group once after pooling its held-out predictions.\n                            merged={}; metric_rows=[]\n                            for pair_rows, pv0 in zip(fold_losses[::2],fold_losses[1::2]): pass')
# Undo malformed placeholder using targeted broader replacement next.
p.write_text(s,encoding='utf-8')
