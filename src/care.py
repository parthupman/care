#!/usr/bin/env python3
"""CARE erasure + evaluation kernel.

Patches AdaVD's cross-attention value-erasure operator (`AttnProcessor.cal_ortho_decomp`)
to replace the raw target direction with the covariance-aware, retained-subspace-whitened
direction described in the paper (closed-form Woodbury solve; supports single- and
multi-concept erasure). Clones AdaVD on first run, generates images under AdaVD's own
protocol, and scores them (CLIP score for erasure, FID for preservation).

Paths via env: C43A_HOME (working/output dir, default ./.care_home), ITER_ID (run name),
C43A_REPO (AdaVD clone destination, default $C43A_HOME/adavd_repo). Config via the
ADAVD_CFG env var (JSON). Writes results.json + report.json into
$C43A_HOME/outputs/$ITER_ID/.
"""
import os, sys, json, time, pathlib, subprocess, copy, re, traceback
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

HOME        = pathlib.Path(os.environ.get("C43A_HOME", "./.care_home"))
ITER_ID     = os.environ.get("ITER_ID", "adhoc")
REPO_DIR    = pathlib.Path(os.environ.get("C43A_REPO", str(HOME / "adavd_repo")))
WORK_DIR    = HOME / "outputs" / ITER_ID
RESULTS_DIR = WORK_DIR
OUT         = str(WORK_DIR / "gen")
SKIP_PIP    = os.environ.get("C43A_SKIP_PIP", "1") == "1"
CFG_DEFAULT = {"erase_type":"instance","targets":["Snoopy","Mickey Mouse"],
               "nontargets":["Pikachu","Dog"],
               "anchors":["Bugs Bunny","Hello Kitty","SpongeBob","Garfield","Tom and Jerry","Donald Duck"],
               "ns":10,"rank":1,"op":"whitened","lam":0.5,"gamma":0.5}

def _log(m):
    line=f"[c43a {ITER_ID} {time.strftime('%H:%M:%S')}] {m}"; print(line,flush=True)
    try:
        RESULTS_DIR.mkdir(parents=True,exist_ok=True)
        with (RESULTS_DIR/"run_log.txt").open("a") as f: f.write(line+"\n")
    except Exception: pass
def _progress(d):
    try: RESULTS_DIR.mkdir(parents=True,exist_ok=True); (RESULTS_DIR/"progress.json").write_text(json.dumps(d))
    except Exception: pass

def main():
    CFG={**CFG_DEFAULT, **json.loads(os.environ.get("ADAVD_CFG","{}"))}
    _log(f"CFG={CFG}"); _progress({"status":"starting","cfg":CFG,"ts":time.time()})
    if not SKIP_PIP:
        subprocess.run([sys.executable,"-m","pip","install","-q","huggingface_hub==0.25.2",
            "diffusers==0.30.3","transformers==4.44.2","accelerate","safetensors","pandas","einops","pytorch-fid"],check=True)
    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER",None)
    if not (REPO_DIR/"src").exists():
        subprocess.run(["git","clone","--depth=1","https://github.com/WYuan1001/AdaVD.git",str(REPO_DIR)],check=True)
    SRC=str(REPO_DIR/"src"); sys.path.insert(0,SRC); os.chdir(str(REPO_DIR))
    import torch; torch.set_grad_enabled(False)
    import main as ada
    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
    TOTAL,GS,SEED,SIG=30,7.5,int(os.environ.get("SEED","0")),(100.,0.93,2.)
    TARGETS=CFG["targets"]; NONTARGET=CFG["nontargets"]; ANCHORS=CFG["anchors"]
    RANK,ENERGY,NS,LAM,GAMMA=CFG["rank"],0.90,CFG["ns"],CFG["lam"],CFG["gamma"]
    TEMPLATES=ada.template_dict[CFG["erase_type"]]
    _MAXT=int(os.environ.get("C43A_MAX_TEMPLATES","0"))   # >0 slices templates (fast smoke); 0=all (real run)
    if _MAXT>0: TEMPLATES=TEMPLATES[:_MAXT]
    _log(f"templates={len(TEMPLATES)} (max={_MAXT or 'all'}); est gen ~ {len(TARGETS+NONTARGET)*3*len(TEMPLATES)*NS} images")
    RAVE_R,BANK_MU,MULTI_VALS,ON={},{},{},{"v":False,"OP":CFG["op"],"GAMMA":GAMMA}

    _orig=ada.AttnProcessor.cal_ortho_decomp
    def patched(self,target_value,pro_record,ortho_basis=None,project_matrix=None):
        if ortho_basis is not None or project_matrix is not None:
            return _orig(self,target_value,pro_record,ortho_basis,project_matrix)
        pro=pro_record.permute(1,0,2).reshape(77,-1); dev,dt=pro.device,pro.dtype
        R=RAVE_R.get(self.module_name); mu=BANK_MU.get(self.module_name)
        if R is not None: R=R.to(dev,dt)
        if mu is not None: mu=mu.to(dev,dt)
        vals=MULTI_VALS.get(self.module_name)
        if vals is None: vals=[target_value[0]]
        op=ON["OP"]; g=ON["GAMMA"]; era_total=None
        for val in vals:
            tar=val.to(dev,dt).permute(1,0,2).reshape(77,-1)
            cos=torch.cosine_similarity(tar,pro,dim=-1)
            if self.sigmoid_setting is not None: cos=self.sigmoid(cos,self.sigmoid_setting)
            def coeff(direction):
                d1=(direction*pro).sum(-1); d2=(direction*direction).sum(-1).clamp_min(1e-12)
                w=torch.nan_to_num(cos*(d1/d2),nan=0.0); w[0].fill_(0)
                return w.unsqueeze(0).unsqueeze(-1)*direction.view((77,16,-1)).permute(1,0,2)
            if (not ON["v"]) or op=="adavd" or R is None: d=tar
            elif op=="ot": d=tar-mu
            elif op in ("whitened","otwhite"):
                srcv=(tar-mu) if (op=="otwhite" and mu is not None) else tar
                B=R.reshape(77,R.shape[1],-1).float(); Bt=B.transpose(1,2); rr=B.shape[2]
                M=g*torch.eye(rr,device=dev,dtype=torch.float32).unsqueeze(0)+torch.bmm(Bt,B)
                sol=torch.linalg.solve(M,torch.bmm(Bt,srcv.float().unsqueeze(-1)))
                d=((srcv.float()-torch.bmm(B,sol).squeeze(-1))/max(g,1e-6)).to(dt)
            else: d=tar
            e=coeff(d); era_total=e if era_total is None else era_total+e
        return era_total.to(dev)
    ada.AttnProcessor.cal_ortho_decomp=patched

    def load(sd):
        p=DiffusionPipeline.from_pretrained(sd,safety_checker=None,torch_dtype=torch.float16).to('cuda')
        p.scheduler=DPMSolverMultistepScheduler.from_config(p.scheduler.config); return p
    try: pipe=load("CompVis/stable-diffusion-v1-4")
    except Exception as e: _log(f"v1-4->v1-5 {e}"); pipe=load("sd-legacy/stable-diffusion-v1-5")
    unet,tok,te,vae=pipe.unet,pipe.tokenizer,pipe.text_encoder,pipe.vae
    unet_orig=copy.deepcopy(unet)
    G,E,SP,EO=ada.get_token,ada.get_textencoding,ada.get_spread_embedding,ada.get_eot_idx
    uncond=E(G("",tok),te)
    def spread_enc(c): t=G(c,tok); return SP(E(t,te),EO(t))
    def plain_enc(p): return E(G(p,tok),te)
    def record_concept(c):
        ce=spread_enc(c); u=ada.set_attenprocessor(unet,atten_type='original',record=True,record_type='values')
        _,recs=ada.diffusion(unet=u,scheduler=pipe.scheduler,latents=torch.zeros(len(ce),4,64,64).to(pipe.device,dtype=ce.dtype),
            text_embeddings=torch.cat([uncond]*len(ce)+[ce],dim=0),total_timesteps=1,start_timesteps=0,guidance_scale=GS,record=True,record_type='values',desc=None)
        pipe.scheduler.set_timesteps(TOTAL); keys=list(recs['values'].keys())
        recs['values'].update({f"{ts}.{'.'.join(k.split('.')[1:])}":recs['values'][k] for ts in pipe.scheduler.timesteps for k in keys})
        return recs
    def entry_to_77D(entry):
        tv=entry.view((2,int(len(entry)//16),-1)+entry.size()[-2:]); tv=tv.permute(1,0,2,3,4).contiguous().view((tv.size()[1],-1)+tv.size()[-2:])
        return tv[0].permute(1,0,2).reshape(77,-1).float()

    _log(f"record {len(ANCHORS)} anchors -> R/mu"); _progress({"status":"recording_anchors","cfg":CFG,"ts":time.time()})
    bank={}
    for a in ANCHORS:
        recs=record_concept(a); seen=set()
        for k,v in recs['values'].items():
            m='.'.join(k.split('.')[1:])
            if m in seen: continue
            seen.add(m); bank.setdefault(m,[]).append(entry_to_77D(v[0]).cpu())
    for m,reps in bank.items():
        Mst=torch.stack(reps,0); L,D=Mst.shape[1],Mst.shape[2]; R=torch.zeros(L,D,RANK)
        for j in range(L):
            U,S,Vh=torch.linalg.svd(Mst[:,j,:],full_matrices=False); rj=RANK
            if S.numel()>0:
                frac=torch.cumsum(S**2,0)/(S**2).sum().clamp_min(1e-12); rj=min(RANK,int((frac<ENERGY).sum().item())+1)
            rj=max(1,min(rj,Vh.shape[0])); R[j,:,:rj]=Vh[:rj].T
        RAVE_R[m]=R; BANK_MU[m]=Mst.mean(0)
    _log(f"record {len(TARGETS)} targets -> MULTI_VALS"); _progress({"status":"recording_targets","cfg":CFG,"ts":time.time()})
    target_records=None
    for t in TARGETS:
        recs=record_concept(t)
        if target_records is None: target_records=recs
        seen=set()
        for k,v in recs['values'].items():
            m='.'.join(k.split('.')[1:])
            if m in seen: continue
            seen.add(m); MULTI_VALS.setdefault(m,[]).append(v[0].cpu())
    _log(f"R/mu {len(RAVE_R)} modules; MULTI_VALS {len(MULTI_VALS)} modules x {len(TARGETS)} targets")

    def gen_save(concept,mode):
        d=os.path.join(OUT,mode,re.sub(r'[^\w]','_',concept)); os.makedirs(d,exist_ok=True)
        ON["v"]=(mode=="ours"); pairs=[]
        for ti,tmpl in enumerate(TEMPLATES):
            prompt=tmpl.format(concept); enc=plain_enc(prompt)
            if mode=="original": u=unet_orig
            else:
                ada.ORTHO_DECOMP_STORAGE={}; u=ada.set_attenprocessor(unet,atten_type='retain',target_records=copy.deepcopy(target_records),sigmoid_setting=SIG,decomp_timestep=0)
            ada.seed_everything(SEED+ti,True)
            lat=torch.randn(NS,4,64,64).to(pipe.device,dtype=uncond.dtype)
            out=ada.diffusion(unet=u,scheduler=pipe.scheduler,latents=lat,start_timesteps=0,text_embeddings=torch.cat([uncond]*NS+[enc]*NS,dim=0),total_timesteps=TOTAL,guidance_scale=GS,desc=None)
            for i,im in enumerate(out):
                fn=re.sub(r'[^\w\s]','',prompt).replace(' ','_')[:70]+f"_{ti}_{i}.png"
                img=ada.process_img(vae.decode(im.unsqueeze(0)/vae.config.scaling_factor,return_dict=False)[0]); img.save(os.path.join(d,fn)); pairs.append((os.path.join(d,fn),prompt))
        return d,pairs
    from transformers import CLIPModel,CLIPProcessor
    from PIL import Image
    clip=CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda().eval(); cproc=CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    def cs_dir(pairs):
        tot,n=0.,0
        for k in range(0,len(pairs),25):
            ch=pairs[k:k+25]; im=cproc(images=[Image.open(p).convert("RGB") for p,_ in ch],return_tensors="pt").to('cuda'); tx=cproc.tokenizer([t for _,t in ch],padding=True,truncation=True,max_length=77,return_tensors="pt").to('cuda')
            fi=clip.get_image_features(**im); ft=clip.get_text_features(**tx); fi=fi/fi.norm(dim=1,keepdim=True); ft=ft/ft.norm(dim=1,keepdim=True); tot+=(fi*ft).sum(-1).sum().item(); n+=len(ch)
        return 100*tot/max(n,1)
    from pytorch_fid import fid_score
    from pytorch_fid.inception import InceptionV3
    incep=InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).cuda()
    def fid_dirs(a,b):
        m1,s1=fid_score.compute_statistics_of_path(a,incep,50,2048,'cuda',1); m2,s2=fid_score.compute_statistics_of_path(b,incep,50,2048,'cuda',1); return fid_score.calculate_frechet_distance(m1,s1,m2,s2)

    cs,dirs={},{}
    for c in TARGETS+NONTARGET:
        for mode in ['original','adavd','ours']:
            ON["v"]=(mode=="ours"); ON["OP"]=("adavd" if mode=="adavd" else CFG["op"])
            d,pairs=gen_save(c,mode); dirs[(c,mode)]=d; cs[(c,mode)]=cs_dir(pairs)
            _log(f"{c:18} {mode:8} CS={cs[(c,mode)]:.2f}"); _progress({"status":"generating","cfg":CFG,"last":f"{c}/{mode}","ts":time.time()})
    res={"iter":ITER_ID,"targets":TARGETS,"op":CFG["op"],"gamma":GAMMA,"erase_type":CFG["erase_type"],"seed":SEED,"ns":NS,
         "max_templates":_MAXT,"n_templates":len(TEMPLATES),"erase":{},"fid":{}}
    _log(f"=== erase {TARGETS} (op={CFG['op']}) ===")
    for t in TARGETS:
        res["erase"][t]={"orig":round(cs[(t,'original')],2),"adavd":round(cs[(t,'adavd')],2),"ours":round(cs[(t,'ours')],2)}
        _log(f"TARGET {t}: orig={cs[(t,'original')]:.2f} AdaVD={cs[(t,'adavd')]:.2f} OURS={cs[(t,'ours')]:.2f} (lower=better)")
    for c in NONTARGET:
        fa=fid_dirs(dirs[(c,'adavd')],dirs[(c,'original')]); fo=fid_dirs(dirs[(c,'ours')],dirs[(c,'original')])
        win=("OURS" if fo<fa-1e-2 else "AdaVD" if fa<fo-1e-2 else "tie")
        # FID = AdaVD's published preservation metric (kept for table-matching); it is rank-noisy
        # at N<2048, so we ALSO record non-target CLIP-score (well-defined at low N) as the robust check.
        res["fid"][c]={"adavd":round(fa,2),"ours":round(fo,2),"winner":win,
                       "clip_orig":round(cs[(c,'original')],2),"clip_adavd":round(cs[(c,'adavd')],2),"clip_ours":round(cs[(c,'ours')],2)}
        _log(f"non-tgt {c:14}: FID A={fa:.2f} O={fo:.2f} [{win}] | CLIP o={cs[(c,'original')]:.2f} A={cs[(c,'adavd')]:.2f} O={cs[(c,'ours')]:.2f}")
    (RESULTS_DIR/"results.json").write_text(json.dumps(res,indent=2))
    (RESULTS_DIR/"report.json").write_text(json.dumps(res,indent=2))
    _progress({"status":"done","cfg":CFG,"result":res,"ts":time.time()}); _log("=== cell done ===")

if __name__=="__main__":
    try: main()
    except Exception:
        _log("!!! FAILED:\n"+traceback.format_exc()); raise
