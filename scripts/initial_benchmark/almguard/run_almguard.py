from pathlib import Path
import argparse, gc, json
import torch

ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")
MANIFEST = ROOT / "data/almguard/attack_manifest.jsonl"
RESULT_ROOT = ROOT / "results/almguard"

MODEL_PATHS = {
    "qwen2_audio": ROOT / "models/qwen2-audio-7b-instruct",
    "qwen25_omni": ROOT / "models/qwen2.5-omni-7b",
    "kimi_audio": ROOT / "models/kimi-audio-7b-instruct",
    "audio_flamingo3": ROOT / "models/audio-flamingo-3",
}
ATTACKS = {"advwave","advwave_p","pair_audio","gupta","ica","pap_audio"}

def read_manifest():
    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)
    rows=[]
    with MANIFEST.open("r",encoding="utf-8") as f:
        for ln,line in enumerate(f,1):
            if not line.strip():
                continue
            r=json.loads(line)
            req={"example_id","attack","target_model","prompt","audio_path","text_prefix"}
            miss=req-set(r)
            if miss:
                raise RuntimeError(f"{MANIFEST} line {ln}: missing {sorted(miss)}")
            if r["attack"] not in ATTACKS:
                raise RuntimeError(f"{MANIFEST} line {ln}: bad attack {r['attack']!r}")
            if r["target_model"] not in MODEL_PATHS:
                raise RuntimeError(f"{MANIFEST} line {ln}: bad model {r['target_model']!r}")
            p=ROOT/str(r["audio_path"])
            if not p.exists():
                raise FileNotFoundError(p)
            if r["attack"]=="ica" and not str(r["text_prefix"]).strip():
                raise RuntimeError(f"{MANIFEST} line {ln}: ICA requires text_prefix")
            if r["attack"]!="ica" and str(r["text_prefix"]).strip():
                raise RuntimeError(f"{MANIFEST} line {ln}: text_prefix must be empty for {r['attack']}")
            rows.append(r)
    if not rows:
        raise RuntimeError("empty manifest")
    keys=[(r["target_model"],r["attack"],r["example_id"]) for r in rows]
    if len(keys)!=len(set(keys)):
        raise RuntimeError("duplicate manifest key")
    return rows

def append_jsonl(path,row):
    with path.open("a",encoding="utf-8") as f:
        f.write(json.dumps(row,ensure_ascii=False)+"\n")
        f.flush()

def load_qwen2_audio():
    import librosa
    from transformers import AutoProcessor,Qwen2AudioForConditionalGeneration
    p=str(MODEL_PATHS["qwen2_audio"])
    proc=AutoProcessor.from_pretrained(p,local_files_only=True)
    model=Qwen2AudioForConditionalGeneration.from_pretrained(
        p,device_map="auto",torch_dtype="auto",local_files_only=True
    )
    model.eval()
    def infer(audio_path,text_prefix):
        content=[]
        if text_prefix:
            content.append({"type":"text","text":text_prefix})
        content.append({"type":"audio","audio_url":"file:"+str(audio_path)})
        conv=[{"role":"user","content":content}]
        text=proc.apply_chat_template(conv,add_generation_prompt=True,tokenize=False)
        wav,_=librosa.load(str(audio_path),sr=proc.feature_extractor.sampling_rate)
        inputs=proc(
            text=text,audios=[wav],return_tensors="pt",padding=True,
            sampling_rate=proc.feature_extractor.sampling_rate
        )
        for k,v in inputs.items():
            if isinstance(v,torch.Tensor):
                inputs[k]=v.to(model.device)
        n=inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out=model.generate(
                **inputs,max_length=1024,do_sample=False,
                temperature=0.0,top_p=0,top_k=0
            )
        out=out[:,n:]
        return proc.batch_decode(
            out,skip_special_tokens=True,clean_up_tokenization_spaces=False
        )[0].strip()
    return model,infer

def load_qwen25_omni():
    from transformers import Qwen2_5OmniForConditionalGeneration,Qwen2_5OmniProcessor
    from qwen_omni_utils import process_mm_info
    p=str(MODEL_PATHS["qwen25_omni"])
    proc=Qwen2_5OmniProcessor.from_pretrained(p,local_files_only=True)
    model=Qwen2_5OmniForConditionalGeneration.from_pretrained(
        p,torch_dtype="auto",device_map="auto",local_files_only=True
    )
    if hasattr(model,"disable_talker"):
        model.disable_talker()
    model.eval()
    def infer(audio_path,text_prefix):
        content=[]
        if text_prefix:
            content.append({"type":"text","text":text_prefix})
        content.append({"type":"audio","audio":"file://"+str(audio_path.resolve())})
        conv=[{"role":"user","content":content}]
        text=proc.apply_chat_template(conv,tokenize=False,add_generation_prompt=True)
        audios,images,videos=process_mm_info(conv,use_audio_in_video=False)
        inputs=proc(
            text=text,audio=audios,images=images,videos=videos,
            return_tensors="pt",padding=True,use_audio_in_video=False
        )
        inputs=inputs.to(model.device).to(model.dtype)
        n=inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out=model.generate(
                **inputs,return_audio=False,thinker_do_sample=False,
                use_audio_in_video=False,max_new_tokens=1024
            )
        if isinstance(out,tuple):
            out=out[0]
        return proc.batch_decode(
            out[:,n:],skip_special_tokens=True,clean_up_tokenization_spaces=False
        )[0].strip()
    return model,infer

def load_kimi_audio():
    from kimia_infer.api.kimia import KimiAudio
    model=KimiAudio(
        model_path=str(MODEL_PATHS["kimi_audio"]),
        load_detokenizer=False,
    )
    params={
        "audio_temperature":0.8,
        "audio_top_k":10,
        "text_temperature":0.0,
        "text_top_k":5,
        "audio_repetition_penalty":1.0,
        "audio_repetition_window_size":64,
        "text_repetition_penalty":1.0,
        "text_repetition_window_size":16,
        "max_new_tokens":1024,
    }
    def infer(audio_path,text_prefix):
        messages=[]
        if text_prefix:
            messages.append({
                "role":"user","message_type":"text","content":text_prefix
            })
        messages.append({
            "role":"user","message_type":"audio","content":str(audio_path)
        })
        _,text=model.generate(messages,output_type="text",**params)
        return str(text).strip()
    return model,infer

def load_audio_flamingo3():
    from transformers import AudioFlamingo3ForConditionalGeneration,AutoProcessor
    p=str(MODEL_PATHS["audio_flamingo3"])
    proc=AutoProcessor.from_pretrained(p,local_files_only=True)
    model=AudioFlamingo3ForConditionalGeneration.from_pretrained(
        p,torch_dtype="auto",device_map="auto",local_files_only=True
    )
    model.eval()
    def infer(audio_path,text_prefix):
        content=[]
        if text_prefix:
            content.append({"type":"text","text":text_prefix})
        content.append({"type":"audio","path":str(audio_path)})
        conv=[{"role":"user","content":content}]
        inputs=proc.apply_chat_template(
            conv,tokenize=True,add_generation_prompt=True,return_dict=True
        )
        inputs=inputs.to(model.device,dtype=model.dtype)
        n=inputs.input_ids.shape[1]
        with torch.inference_mode():
            out=model.generate(**inputs,max_new_tokens=1024,do_sample=False)
        return proc.decode(out[0,n:],skip_special_tokens=True).strip()
    return model,infer

def load_model(name):
    return {
        "qwen2_audio":load_qwen2_audio,
        "qwen25_omni":load_qwen25_omni,
        "kimi_audio":load_kimi_audio,
        "audio_flamingo3":load_audio_flamingo3,
    }[name]()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True,choices=list(MODEL_PATHS))
    ap.add_argument("--attack",choices=sorted(ATTACKS),default=None)
    ap.add_argument("--limit",type=int,default=None)
    a=ap.parse_args()

    rows=[
        r for r in read_manifest()
        if r["target_model"]==a.model and (a.attack is None or r["attack"]==a.attack)
    ]
    if a.limit is not None:
        rows=rows[:a.limit]
    if not rows:
        raise RuntimeError("no matching manifest rows")

    outdir=RESULT_ROOT/a.model
    outdir.mkdir(parents=True,exist_ok=True)
    outfile=outdir/"responses.jsonl"

    done=set()
    if outfile.exists():
        for line in outfile.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r=json.loads(line)
                done.add((r["attack"],r["example_id"]))

    model,infer=load_model(a.model)

    for i,r in enumerate(rows,1):
        key=(r["attack"],r["example_id"])
        if key in done:
            print(f"[{i:04d}/{len(rows):04d}] skip {key}",flush=True)
            continue

        print(
            f"[{i:04d}/{len(rows):04d}] {a.model} | "
            f"{r['attack']} | {r['example_id']}",
            flush=True,
        )

        response=infer(
            ROOT/str(r["audio_path"]),
            str(r["text_prefix"]).strip(),
        )

        append_jsonl(outfile,{
            "example_id":str(r["example_id"]),
            "model":a.model,
            "benchmark":"ALMGuard",
            "attack":str(r["attack"]),
            "prompt":str(r["prompt"]),
            "audio_path":str(r["audio_path"]),
            "text_prefix":str(r["text_prefix"]).strip(),
            "response":response,
        })

        print(
            " response:",
            response[:250].replace("\n"," "),
            flush=True,
        )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__=="__main__":
    main()
