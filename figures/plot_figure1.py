import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
GRAY="#D9D9D6"; AMB="#F8CB7F"; BLUE="#B9D7F2"; EDGE="#555555"; TXT="#1F2937"
fig,ax=plt.subplots(figsize=(13,6.2)); ax.set_xlim(0,130); ax.set_ylim(0,66); ax.axis("off")
def box(x,y,w,h,title,sub="",c=GRAY,fs=10,ls="-"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.4,rounding_size=1.2",fc=c,ec=EDGE,lw=1,ls=ls))
    if sub:
        ax.text(x+w/2,y+h*0.66,title,ha="center",va="center",fontsize=fs,weight="bold",color=TXT)
        ax.text(x+w/2,y+h*0.30,sub,ha="center",va="center",fontsize=fs-1.5,color=TXT,linespacing=1.3)
    else: ax.text(x+w/2,y+h/2,title,ha="center",va="center",fontsize=fs,weight="bold",color=TXT)
def poly(pts,col=EDGE):
    xs,ys=zip(*pts[:-1]); ax.plot(xs,ys,color=col,lw=1.2,solid_capstyle="round")
    arr(pts[-2][0],pts[-2][1],pts[-1][0],pts[-1][1],col=col)
def arr(x1,y1,x2,y2,cs="arc3",col=EDGE):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=13,lw=1.2,color=col,connectionstyle=cs))
# top row
Y=46; H=10
box(1,Y,15,H,"Prompt","text string")
box(20,Y,17,H,"encode()","BPE tokenizer\nvocab 32,000",AMB)
box(41,Y,15,H,"Token IDs","BOS + prompt\n(14 tokens)")
box(85,Y,16,H,"sample()","greedy (-t 0) or\ntop-p (-p 0.9)",AMB)
box(105,Y,11,H,"decode()","token to text",AMB)
box(119,Y,10,H,"Output","story text")
for a,b in [(16.4,20),(37.4,41),(101.4,105),(116.4,119)]: arr(a+0.4,Y+H/2,b-0.4,Y+H/2)
# forward block
FX,FY,FW,FH=4,3,121,37
ax.add_patch(FancyBboxPatch((FX,FY),FW,FH,boxstyle="round,pad=0.4,rounding_size=1.5",fc="#F5F9FD",ec="#2B5C8A",lw=1.4,ls="--"))
ax.text(FX+12,FY+FH-2.4,"forward(token, pos): one pass per token, 43 matrix-vector products",fontsize=10.5,weight="bold",color="#2B5C8A",va="center")
# inside
y=10; h=9
box(7,y,13,h,"Embedding","row lookup\n32,000 x 288",GRAY,9.5)
LX,LY,LW,LH=24,6.5,64,26
ax.add_patch(FancyBboxPatch((LX,LY),LW,LH,boxstyle="round,pad=0.3,rounding_size=1.2",fc="white",ec=EDGE,lw=1))
ax.text(LX+LW/2,LY+LH-2,"Transformer layer  x 6   (dim 288, 6 heads, hidden 768)",ha="center",fontsize=9.5,weight="bold",color=TXT)
# attention row
ay=19.5; ah=7.5
box(26,ay,9,ah,"RMSNorm","",GRAY,8.5)
box(37.5,ay,12,ah,"Q, K, V","3 matmul",BLUE,8.5)
box(52,ay,9.5,ah,"RoPE","",GRAY,8.5)
box(64,ay,11.5,ah,"Attention","KV cache",GRAY,8.5)
box(78,ay,8.5,ah,"Wo","1 matmul",BLUE,8.5)
for a,b in [(35.4,37.5),(49.9,52),(61.9,64),(75.9,78)]: arr(a+0.3,ay+ah/2,b-0.3,ay+ah/2)
by=9; bh=7.5
box(26,by,9,bh,"RMSNorm","",GRAY,8.5)
box(37.5,by,14,bh,"W1, W3","2 matmul",BLUE,8.5)
box(54,by,11,bh,"SwiGLU","",GRAY,8.5)
box(67.5,by,9,bh,"W2","1 matmul",BLUE,8.5)
box(79,by,7.5,bh,"+ x","residual",GRAY,8.5)
for a,b in [(35.4,37.5),(51.9,54),(65.4,67.5),(76.9,79)]: arr(a+0.3,by+bh/2,b-0.3,by+bh/2)
poly([(82.3,ay-0.3),(82.3,ay-1.4),(30.5,ay-1.4),(30.5,by+bh+0.3)])
ax.text(87.5,ay+ah/2,"+ x",fontsize=8,color=TXT,va="center")
arr(20.4,y+h/2,24,y+h/2)
box(92,y,13,h,"Final\nRMSNorm","",GRAY,9)
box(108.5,y,14.5,h,"Classifier","32,000 x 288\n1 matmul",BLUE,9.5)
arr(88.4,y+h/2,92,y+h/2); arr(105.4,y+h/2,108.5,y+h/2)
ax.text(115.75,y+h+2.2,"Logits: 32,000 scores",ha="center",fontsize=9,color=TXT,weight="bold")
# connections
poly([(48.5,Y-0.4),(48.5,Y-3),(13.5,Y-3),(13.5,y+h+0.4)])
poly([(115.75,y+h+4.6),(115.75,Y-3),(93,Y-3),(93,Y-0.4)])
# loop
poly([(93,Y+H+0.4),(93,Y+H+3.5),(48.5,Y+H+3.5),(48.5,Y+H+0.4)],col="#2B5C8A")
ax.text(70.75,Y+H+5.2,"next token fed back to forward() at pos + 1 until -n steps (prompt tokens first fill the KV cache)",ha="center",fontsize=9,color="#2B5C8A",style="italic")
ax.text(65,0.3,"Gray: data and non-matmul operators    Amber: tokenizer and sampling    Blue: matmul() calls (7 per layer x 6 + classifier = 43 per token)",ha="center",fontsize=8.5,color="#4B5563")
fig.tight_layout(pad=0.3)
fig.savefig("figure1_inference_pipeline.png",dpi=220)
