import csv,pathlib,collections,json,hashlib,math
root=pathlib.Path(r'C:\Users\Saxen\OneDrive\Documents\comp-genomics-proj\qgcf\preparation\20261022T182444Z\arm4_protocol')
data=pathlib.Path(r'C:\Users\Saxen\ppi_IG\results\representation_diagnosis\20261015\attempt02\gcp_a100_retry_01\collected')
adm=pathlib.Path(r'C:\Users\Saxen\ppi_IG\results\qhts_feasibility\20261012\independent_data_arm\record_admission_20261012')
read=lambda p:list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
records=read(data/'records.csv'); splits=read(data/'splits.csv'); admissions=read(adm/'original_record_admission.csv')
R={x['pubchem_cid']:x for x in records}; A={x['pubchem_cid']:x for x in admissions}
assert len(records)==65 and len(R)==65 and len(admissions)==65 and set(R)==set(A)
assert all(A[c]['admitted']=='True' and A[c]['no_inequality']=='True' and A[c]['observed_points_bracket_50']=='True' and A[c]['reported_value_in_range']=='True' and A[c]['reconstructed_value_in_range']=='True' and A[c]['numeric_fit_agreement']=='True' for c in R)
assert all(abs(math.log10(float(A[c]['reported_ic50_uM'])/float(A[c]['reconstructed_ic50_uM'])))<=0.0200000001 for c in R)
excluded={'899202','976329'}; assert excluded <= set(R)
group_test={}; test_sets={}; inner_conflicts=[]
for fold in range(5):
 rows=[s for s in splits if int(s['outer_fold'])==fold]; assert len(rows)==65 and len({s['pubchem_cid'] for s in rows})==65
 gs=collections.defaultdict(list)
 for s in rows: gs[R[s['pubchem_cid']]['chemical_group']].append(s)
 assert all(len({s['role'] for s in v})==1 for v in gs.values())
 for g,v in gs.items():
  if v[0]['role']=='test': group_test.setdefault(g,set()).add(fold)
 for g,v in gs.items():
  iv={s['inner_validation_fold'] for s in v if s['role']=='train'}
  assert len(iv)<=1,(fold,g,iv)
 test_sets[fold]={s['pubchem_cid'] for s in rows if s['role']=='test'}
 assert len(test_sets[fold])==13
assert len(group_test)==27 and all(len(v)==1 for v in group_test.values())
assert len({r['chemical_group'] for r in records})==27
assert sum(1 for g in {r['chemical_group'] for r in records} if sum(x['chemical_group']==g for x in records)>1)==8
hashes={n:hashlib.sha256((data/n).read_bytes()).hexdigest() for n in ['records.csv','splits.csv','exact_inputs.npz']}
expected={'records.csv':'b6c9473de5dd8d09af35f5b5e42983498ee468a15e96c19b0ba066c3cc20791c','splits.csv':'137c76616c1940cdddabd021bb6d5d1f9b1ead0a78dddd450de73ac88dbb0888','exact_inputs.npz':'5dcbd0011ed01995fe76a71c12f62073be5a7905c7cb6e16e9cee7402fea1540'}
assert hashes==expected
for name,drop in [('primary_pairs.csv',set()),('sensitivity_pairs.csv',excluded)]:
 out=[]
 for fold in range(5):
  rows=[s for s in splits if int(s['outer_fold'])==fold]; gs=collections.defaultdict(list)
  for s in rows:
   c=s['pubchem_cid']
   if s['role']=='test' and c not in drop: gs[R[c]['chemical_group']].append(c)
  for g,ids in gs.items():
   for i,a in enumerate(sorted(ids)):
    for b in sorted(ids)[i+1:]: out.append({'outer_fold':fold,'chemical_group':g,'cid_a':a,'cid_b':b,'label_a_log10_ic50_uM':R[a]['label'],'label_b_log10_ic50_uM':R[b]['label'],'delta_label_a_minus_b':float(R[a]['label'])-float(R[b]['label'])})
 with (root/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=out[0].keys());w.writeheader();w.writerows(out)
# One row per compound with a canonical held-out fold.
fold_by_group={g:next(iter(fs)) for g,fs in group_test.items()}
with (root/'record_manifest.csv').open('w',newline='',encoding='utf-8') as f:
 fields=['pubchem_cid','chemical_group','outer_fold','label_log10_ic50_uM','ic50_uM','qualifier_status','source_qualifier_present','admitted','hill_sign_conflict','sensitivity_excluded','capture_sha256']
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for c,r in sorted(R.items(),key=lambda kv:int(kv[0])):
  w.writerow({'pubchem_cid':c,'chemical_group':r['chemical_group'],'outer_fold':fold_by_group[r['chemical_group']],'label_log10_ic50_uM':r['label'],'ic50_uM':r['ic50_uM'],'qualifier_status':r['qualifier_status'],'source_qualifier_present':A[c]['source_qualifier_present'],'admitted':A[c]['admitted'],'hill_sign_conflict':A[c]['reported_hill_sign_conflict'],'sensitivity_excluded':c in excluded,'capture_sha256':r['capture_sha256']})
primary=read(root/'primary_pairs.csv');sens=read(root/'sensitivity_pairs.csv')
fold_summary={str(f):{'test_records':len(test_sets[f]),'primary_pairs':sum(int(p['outer_fold'])==f for p in primary),'sensitivity_pairs':sum(int(p['outer_fold'])==f for p in sens)} for f in range(5)}
assert len(primary)==136 and len(sens)==129
assert [fold_summary[str(f)]['primary_pairs'] for f in range(5)]==[36,29,28,22,21]
summary={'status':'PASS','source_hashes':hashes,'admission':{'records':65,'all_admitted':True,'explicit_inequalities':0,'qualifier_fields_blank':True,'criteria':'no inequality; observed curve brackets 50%; reported and reconstructed estimates lie in original concentration range; log10 agreement <=0.02'},'qualifier_note':'All source qualifier fields are blank. Blank status is preserved. Admission depends on original curve evidence, not blank-field equality semantics.','groups':{'n':27,'outer_group_role_conflicts':0,'groups_split_across_outer_test_folds':0,'inner_group_validation_conflicts':0},'pairs':{'primary':len(primary),'sensitivity':len(sens),'sensitivity_excluded_cids':sorted(excluded),'folds':fold_summary},'sensitivity_reason':'Saved reported Hill-sign conflicts.','model_calls':0}
(root/'source_verification.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8'); print(json.dumps(summary,indent=2))
