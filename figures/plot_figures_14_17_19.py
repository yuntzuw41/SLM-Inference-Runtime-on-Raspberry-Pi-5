import re, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
T5="../task5_camera_app/captures"
OUT="."
def colors(c):
    d={}
    for l in open(f"{T5}/capture_{c}/colors.txt"):
        m=re.match(r"(\w+)\s*:\s*([\d.]+)%",l)
        if m: d[m.group(1)]=float(m.group(2))
    return d
def metrics(c):
    return {k:v for k,v in (l.strip().split("=") for l in open(f"{T5}/capture_{c}/metrics.txt"))}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
# Fig 17
a,b=colors("004"),colors("005"); names=list(a.keys())
import numpy as np
x=np.arange(len(names)); w=0.38
fig,ax=plt.subplots(figsize=(9,4.2))
ax.bar(x-w/2,[a[n] for n in names],w,label="Capture 004 (dominant: Cyan)",color="#3B8BD4")
ax.bar(x+w/2,[b[n] for n in names],w,label="Capture 005 (dominant: Gray)",color="#888780")
ax.set_xticks(x); ax.set_xticklabels(names,rotation=0,fontsize=9)
ax.set_ylabel("Share of 307,200 pixels (%)"); ax.set_ylim(0,40)
ax.set_title("HSV color distribution of two live camera captures")
for i,n in enumerate(names):
    for off,v in ((-w/2,a[n]),(w/2,b[n])):
        if v>=5: ax.text(i+off,v+0.6,f"{v:.1f}",ha="center",fontsize=7.5)
ax.grid(axis="y",alpha=.3); ax.set_axisbelow(True); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/figure17_color_distribution_004_005.png",dpi=200); plt.close(fig)
# Fig 19
caps=["004","005","006","007","008","009"]
M=[metrics(c) for c in caps]
stages=[("capture_or_copy_ms","Camera capture","#5F5E5A"),("conversion_ms","JPEG-to-PPM conversion","#B4B2A9"),("analysis_ms","HSV color analysis","#EF9F27"),("generation_ms","NEON story generation","#185FA5")]
fig,ax=plt.subplots(figsize=(9,4.4))
labels=[f"{c}\n({'original' if c in ('004','005','006','007') else 'revised'})" for c in caps]
left=np.zeros(len(caps))
for k,lab,col in stages:
    v=np.array([float(m[k]) for m in M]); ax.barh(labels,v,left=left,color=col,label=lab); left+=v
for i,m in enumerate(M):
    ax.text(float(m["total_ms"])+30,i,f"{float(m['total_ms']):.0f} ms",va="center",fontsize=8.5)
ax.invert_yaxis(); ax.set_xlim(0,4100); ax.set_xlabel("Wall-clock time (ms)")
ax.set_title("End-to-end latency of live camera runs, by processing stage")
ax.legend(ncol=4,frameon=False,loc="upper center",bbox_to_anchor=(0.5,-0.15),fontsize=8.5)
ax.grid(axis="x",alpha=.3); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f"{OUT}/figure19_end_to_end_latency.png",dpi=200); plt.close(fig)
# Fig 14
fig,ax=plt.subplots(figsize=(10,3.6)); ax.axis("off"); ax.set_xlim(0,100); ax.set_ylim(0,36)
boxes=[("OV5647 camera\nrpicam-still","640×480 JPEG","#D3D1C7"),("FFmpeg","JPEG → RGB PPM","#D3D1C7"),
("image_analysis.c","RGB→HSV, 13 colors","#FAC775"),("Environment rules","10 rules + color\nfallback","#FAC775"),
("prompt.txt","story prefix","#FAC775"),("run_neon","TinyStories 15M\nNEON matmul","#B5D4F4"),("story.txt","generated story","#B5D4F4")]
n=len(boxes); bw=12.2; gap=(100-n*bw)/(n+1)
for i,(t,s,c) in enumerate(boxes):
    x0=gap+i*(bw+gap)
    ax.add_patch(FancyBboxPatch((x0,12),bw,14,boxstyle="round,pad=0.4,rounding_size=1.2",fc=c,ec="#444441",lw=.6))
    ax.text(x0+bw/2,21.5,t,ha="center",va="center",fontsize=8.3,weight="bold")
    ax.text(x0+bw/2,15.5,s,ha="center",va="center",fontsize=7.5)
    if i<n-1: ax.annotate("",xy=(x0+bw+gap-0.3,19),xytext=(x0+bw+0.5,19),arrowprops=dict(arrowstyle="->",lw=1))
ax.text(50,32,"camera_story.c (controller): runs each stage as a child process, checks exit status, records per-stage timing",ha="center",fontsize=9)
ax.add_patch(FancyBboxPatch((1,9),98,21,boxstyle="round,pad=0.3,rounding_size=1.5",fc="none",ec="#888780",lw=.8,ls="--"))
ax.text(50,4,"Gray: image acquisition and conversion   Amber: image analysis and prompt generation   Blue: SLM inference (Task 4 runtime)",ha="center",fontsize=8,color="#444441")
fig.tight_layout(); fig.savefig(f"{OUT}/figure14_task5_architecture.png",dpi=200); plt.close(fig)
print("ok")
