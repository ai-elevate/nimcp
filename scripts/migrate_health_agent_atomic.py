import re,os,sys,subprocess
NL=chr(10)
QT=chr(34)
INCL=chr(35)+"include"
def find_files(d):
    r=subprocess.run(["grep","-rl","static.*nimcp_health_agent_t.*g_.*health_agent.*=.*NULL","--include=*.c",d],capture_output=True,text=True)
    fs=[f for f in r.stdout.strip().split(NL) if f and "venv/" not in f and "bindings/" not in f]
    return [f for f in fs if "NIMCP_DECLARE_HEALTH_AGENT" not in open(f).read() and "DEFINE_HEALTH_AGENT" not in open(f).read()]
def xmod(c):
    m=re.search(r"g_(\w+)_health_agent\s*=\s*NULL",c)
    return m.group(1) if m else None
def mig(fp):
    with open(fp) as f:
        content=f.read()
    mn=xmod(content)
    if not mn: return False
    L=content.split(NL)
    dl=None
    for i,line in enumerate(L):
        pat=r"static\s+nimcp_health_agent_t\s*\*\s*g_"+re.escape(mn)+r"_health_agent\s*=\s*NULL\s*;"
        if re.search(pat,line): dl=i; break
    if dl is None: return False
    sl=dl
    for i in range(dl-1,max(dl-15,-1),-1):
        ln=L[i].strip()
        keep=("//===" in ln or "// Health Agent" in ln or "struct nimcp_health_agent" in ln or "typedef struct nimcp_health_agent" in ln or "extern void nimcp_health_agent_heartbeat_ex" in ln or "nimcp_health_agent_heartbeat_ex" in ln or "const char* operation" in ln or "float progress)" in ln or "stddef.h" in ln or "Global health agent" in ln or ln=="")
        if keep: sl=i
        else: break
    el=dl; hf=0; sf=0; ifn=0; bd=0
    for i in range(dl+1,min(dl+35,len(L))):
        line=L[i]; st=line.strip()
        if mn+"_heartbeat" in line: hf=1; ifn=0; bd=0
        if mn+"_set_health_agent" in line and not hf: sf=1; ifn=0; bd=0
        if hf or sf:
            if "{" in line: ifn=1
            if ifn:
                bd+=line.count("{")-line.count("}")
                if bd<=0:
                    el=i
                    if hf:
                        while el+1<len(L) and L[el+1].strip()=="": el+=1
                        break
                    else: sf=0; ifn=0; bd=0
            elif st=="" or st[:2]=="//" or st[:2]=="/*" or st[:1]=="*" or st[:3]=="/**": el=i
        elif st=="" or st[:2]=="//" or st[:2]=="/*" or st[:1]=="*" or st[:3]=="/**": el=i
    if not hf:
        sf2=0; ifn2=0; bd2=0
        for i in range(dl+1,min(dl+20,len(L))):
            line=L[i]
            if mn+"_set_health_agent" in line: sf2=1; ifn2=0; bd2=0
            if sf2:
                if "{" in line: ifn2=1
                if ifn2:
                    bd2+=line.count("{")-line.count("}")
                    if bd2<=0:
                        el=i
                        while el+1<len(L) and L[el+1].strip()=="": el+=1
                        break
    hi="nimcp_health_agent_macros.h" in content
    if hi:
        rp=NL+"NIMCP_DECLARE_HEALTH_AGENT_ATOMIC("+mn+")"+NL
    else:
        rp=INCL+" "+QT+"utils/fault_tolerance/nimcp_health_agent_macros.h"+QT+NL+NL+"NIMCP_DECLARE_HEALTH_AGENT_ATOMIC("+mn+")"+NL
    nL=L[:sl]+[rp]+L[el+1:]
    nc=NL.join(nL)
    if nc==content: return False
    with open(fp,"w") as f: f.write(nc)
    return True
sd=sys.argv[1] if len(sys.argv)>1 else "src"
files=find_files(sd)
print(f"Found {len(files)} files to migrate")
s=0; fl=[]
for fp in sorted(files):
    try:
        if mig(fp): s+=1
        else: fl.append(fp)
    except Exception as e: fl.append(f"{fp}: {e}")
print(f"Results: {s} migrated, {len(fl)} skipped/failed")
for f in fl[:20]: print(f"  SKIP: {f}")
