"""Rebuild the logo reveal with smooth vector geometry; preserve the intro and audio.

Usage: python polish_intro.py SOURCE.mp4 OUTPUT.mp4 [--preview]
Requires Python, numpy, Pillow and ffmpeg. 60 fps / 1080 x 1920.
Source master: atomic-intro.mp4 from git commit f3013974a2259101203b96238e48a214bfd9903e.
Use that source master when rerendering, rather than recursively editing this output.
"""
import sys, math, subprocess, pathlib
import numpy as np
from PIL import Image, ImageDraw

SOURCE, OUTPUT = sys.argv[1:3]
FPS, W, H = 60, 1080, 1920
BOX = (250, 520, 830, 1010)
S = 4
CW, CH = BOX[2]-BOX[0], BOX[3]-BOX[1]

def smooth(a,b,t):
    u=np.clip((t-a)/(b-a),0,1)
    return float(u*u*u*(u*(u*6-15)+10))

def bez(p0,p1,p2,p3,n=300):
    u=np.linspace(0,1,n)[:,None]
    return (1-u)**3*np.array(p0)+3*(1-u)**2*u*np.array(p1)+3*(1-u)*u*u*np.array(p2)+u**3*np.array(p3)

def coords(points):
    return [((x-BOX[0])*S,(y-BOX[1])*S) for x,y in points]

def curve(points, color, width):
    layer=Image.new('L',(CW*S,CH*S))
    pts=np.asarray(points,dtype=float)
    # Continuous offset ribbon avoids short-segment stroke join spikes.
    if len(pts)<2: return layer.resize((CW,CH)).convert('RGB')
    tangent=np.gradient(pts,axis=0)
    lengths=np.maximum(np.linalg.norm(tangent,axis=1),1e-8)
    normal=np.column_stack([-tangent[:,1],tangent[:,0]])/lengths[:,None]
    widths=np.full(len(pts),float(width))
    if color==(43,179,222):
      u=np.clip((pts[:,0]-708)/46,0,1);u=u*u*(3-2*u)
      widths=widths*(1-u)+3.1*u
    ribbon=np.concatenate([pts+normal*widths[:,None]/2,(pts-normal*widths[:,None]/2)[::-1]])
    draw=ImageDraw.Draw(layer)
    draw.polygon(coords(ribbon),fill=255)
    for k in [0,-1]:
      x,y=coords([pts[k]])[0]; r=widths[k]*S/2
      draw.ellipse((x-r,y-r,x+r,y+r),fill=255)
    alpha=np.array(layer.resize((CW,CH),Image.Resampling.LANCZOS),dtype=np.float32)[...,None]/255
    colors=np.zeros((CH,CW,3),dtype=np.float32)+color
    if color==(43,179,222):
      u=np.clip((np.arange(CW)+BOX[0]-708)/46,0,1);u=u*u*(3-2*u)
      colors=colors*(1-u[None,:,None])+np.array([37,117,149])*u[None,:,None]
    return Image.fromarray(np.clip(alpha*colors,0,255).astype('uint8'))

def a_mask():
    sections=[
      bez((316,978),(315,984),(320,985),(327,985)),
      bez((327,985),(337,985),(350,985),(355,985)),
      bez((355,985),(365,985),(368,980),(374,968)),
      bez((374,968),(423,871),(486,749),(523,681)),
      bez((523,681),(534,660),(543,659),(556,681)),
      bez((556,681),(601,764),(666,889),(708,973)),
      bez((708,973),(713,983),(716,985),(725,985)),
      bez((725,985),(735,985),(748,985),(757,985)),
      bez((757,985),(769,985),(770,982),(765,972)),
      bez((765,972),(704,846),(622,677),(575,578)),
      bez((575,578),(566,559),(554,550),(539,550)),
      bez((539,550),(524,550),(512,559),(503,578)),
      bez((503,578),(450,687),(365,875),(316,978)),
    ]
    mask=Image.new('L',(CW*S,CH*S))
    ImageDraw.Draw(mask).polygon(coords(np.concatenate(sections)),fill=255)
    return np.array(mask.resize((CW,CH),Image.Resampling.LANCZOS),dtype=np.float32)/255

# One closed orbit. Its rear arc is hidden by the A, and its front arc
# passes above the squares without any rectangular erasure or patched seam.
front=bez((316,973),(190,926),(628,807),(754,722),700)
back=np.concatenate([bez((754,722),(816,668),(765,638),(673,659),300),bez((673,659),(547,675),(427,820),(316,973),500)[1:]])
split=int(np.argmin(np.linalg.norm(front-[552,813],axis=1)))
trajectory=np.concatenate([front[split:],back[1:],front[1:split+1]])
distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(trajectory,axis=0),axis=1))]
distance/=distance[-1]
AM=a_mask()
A_COLOR=np.array([204,216,222],dtype=np.float32)

def point_at(progress):
    return np.array([np.interp(progress,distance,trajectory[:,k]) for k in (0,1)])

def line_layer(points,width,color):
    return np.array(curve(points,color,width),dtype=np.float32)

FRONT=line_layer(front,6.2,(43,179,222))
BACK=line_layer(back,3.1,(37,117,149))
GAP=np.max(line_layer(front,17.8,(255,255,255)),axis=2)/255

def orb_layers(progress):
    if progress<=0: return np.zeros_like(BACK),np.zeros_like(FRONT),np.zeros_like(GAP)
    if progress>=1: return BACK,FRONT,GAP
    stop=np.searchsorted(distance,progress)
    # The drawing endpoint moves continuously between samples.
    endpoint=point_at(progress)
    pts=np.vstack([trajectory[:max(1,stop)],endpoint])
    n1=len(front)-split
    rear=np.zeros_like(BACK); fore=np.zeros_like(FRONT); gap=np.zeros_like(GAP)
    if stop<n1:
      fore=line_layer(pts,6.2,(43,179,222)); gap=np.max(line_layer(pts,17.8,(255,255,255)),2)/255
    else:
      fpts=front[split:]
      fore=line_layer(fpts,6.2,(43,179,222)); gap=np.max(line_layer(fpts,17.8,(255,255,255)),2)/255
      n2=n1+len(back)-1
      if stop<n2:
        rear=line_layer(np.vstack([trajectory[n1-1:stop],endpoint]),3.1,(37,117,149))
      else:
        rear=BACK.copy()
        fpts=np.vstack([trajectory[n2-1:stop],endpoint])
        fore=np.maximum(fore,line_layer(fpts,6.2,(43,179,222)))
        gap=np.maximum(gap,np.max(line_layer(fpts,17.8,(255,255,255)),2)/255)
    return rear,fore,gap

def electron(canvas,pos,opacity=1,radius=8):
    x,y=pos-[BOX[0],BOX[1]]
    xmin,xmax=max(0,int(x-28)),min(CW,int(x+29))
    ymin,ymax=max(0,int(y-28)),min(CH,int(y+29))
    if xmax<=xmin or ymax<=ymin: return
    yy,xx=np.mgrid[ymin:ymax,xmin:xmax]
    d=np.sqrt((xx-x)**2+(yy-y)**2)
    glow=np.exp(-d*d/(2*10.5**2))*0.22*opacity
    target=canvas[ymin:ymax,xmin:xmax]
    target[:]=target*(1-glow[...,None])+np.array([112,205,240])*glow[...,None]
    cover=np.clip(radius+.5-d,0,1)*opacity
    spec=np.exp(-((xx-x+2.3)**2+(yy-y+2.7)**2)/(2*3.6**2))
    shading=np.maximum(0,1-d/radius)
    col=np.minimum(255,np.array([43,76,93])+shading[...,None]*np.array([105,138,146])+spec[...,None]*150)
    target[:]=target*(1-cover[...,None])+col*cover[...,None]

def fourth_square(canvas,t,opacity):
    if t<=16.36: return
    growth=smooth(16.36,16.94,t)
    size=14+40*growth
    radius=size/2*(1-growth)+12*growth
    mask=Image.new('L',(68*S,68*S))
    d=ImageDraw.Draw(mask)
    lo=(68-size)/2*S;hi=(68+size)/2*S
    d.rounded_rectangle((lo,lo,hi,hi),radius=radius*S,fill=255)
    alpha=np.array(mask.resize((68,68),Image.Resampling.LANCZOS),dtype=np.float32)/255
    alpha*=smooth(16.36,16.51,t)*opacity
    yy,xx=np.mgrid[0:68,0:68]
    diagonal=np.clip((xx+yy-14)/108,0,1)[...,None]
    green=np.array([123,194,98])*(1-diagonal)+np.array([40,98,49])*diagonal
    silver=np.array([139,188,196])*(1-diagonal)+np.array([41,74,82])*diagonal
    color=silver*(1-growth)+green*growth
    x,y=541-BOX[0],839-BOX[1]
    target=canvas[y:y+68,x:x+68]
    target[:]=target*(1-alpha[...,None])+color*alpha[...,None]

def frame_at(original,t):
    if t<12.65: return original
    # The original fade is sampled from an interior point of A, not its edges.
    opacity=np.clip(original[590:610,530:550].mean()/A_COLOR.mean(),0,1)
    image=np.zeros((CH,CW,3),dtype=np.float32)
    # Retain the exact square artwork and timing, with no surrounding black masks.
    for x,y in [(478,846),(478,918),(548,918)]:
      image[y-BOX[1]:y-BOX[1]+54,x-BOX[0]:x-BOX[0]+54]=original[y:y+54,x:x+54]
    fourth_square(image,t,opacity)
    progress=smooth(12.96,16.12,t)
    rear,fore,gap=orb_layers(progress)
    # Keep the completed logo at full strength; preserve the original fade to black.
    orbit_alpha=opacity if t>20 else 1
    # At the low-left turnaround the orbit slips behind A instead of
    # punching a round hole in its foot.
    behind=np.clip((np.arange(CH)+BOX[1]-928)/12,0,1)[:,None]
    gap=gap*(1-behind)
    fore=fore*(1-AM[...,None]*behind[...,None])
    image=np.maximum(image,rear*orbit_alpha)
    image=image*(1-AM[...,None]*opacity)+A_COLOR*AM[...,None]*opacity
    image*=1-gap[...,None]*orbit_alpha*AM[...,None]
    image=np.maximum(image,fore*orbit_alpha)
    if 12.96<t<16.6:
      pos=point_at(progress)
      if t>16.12:
        u=smooth(16.12,16.36,t)
        start=point_at(1);end=np.array([575.,873.])
        pos=(1-u)**3*start+3*(1-u)**2*u*np.array([583.,795.])+3*(1-u)*u*u*np.array([607.,844.])+u**3*end
      # The electron follows the same depth ordering as its orbit.
      k=np.searchsorted(distance,progress)
      on_rear=(len(front)-split)<=k<(len(front)-split+len(back)-1)
      if on_rear:
        lit=image.copy()
        electron(lit,pos,1-smooth(16.3,16.6,t))
        image+=(lit-image)*(1-AM[...,None]*opacity)
      else:
        electron(image,pos,1-smooth(16.3,16.6,t))
    out=original.copy()
    out[BOX[1]:BOX[3],BOX[0]:BOX[2]]=np.clip(image,0,255).astype('uint8')
    return out

def read_frame(t):
    b=subprocess.check_output(['ffmpeg','-v','error','-ss',str(t),'-i',SOURCE,'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
    return np.frombuffer(b,dtype=np.uint8).reshape(H,W,3)

if '--preview' in sys.argv:
    for t in [12.8,13.2,14,14.5,15,15.5,16,16.5,17,20,22]:
      a=frame_at(read_frame(t),t)
      Image.fromarray(a).save(OUTPUT.replace('.mp4',f'-{t}.png'))
    sys.exit()

decoder=subprocess.Popen(['ffmpeg','-v','error','-i',SOURCE,'-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE)
encoder=subprocess.Popen(['ffmpeg','-y','-v','warning','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0','-i',SOURCE,'-map','0:v','-map','1:a','-c:v','libx264','-preset','slow','-crf','16','-pix_fmt','yuv420p','-c:a','copy','-movflags','+faststart','-t','22.6',OUTPUT],stdin=subprocess.PIPE)
for i in range(1356):
    data=decoder.stdout.read(W*H*3)
    if len(data)!=W*H*3: raise RuntimeError(f'Incomplete source frame {i}')
    original=np.frombuffer(data,dtype=np.uint8).reshape(H,W,3)
    out=frame_at(original,i/FPS)
    encoder.stdin.write(out.tobytes())
    if i%180==0: print(f'Render {i}/1356',flush=True)
encoder.stdin.close()
if encoder.wait() or decoder.wait(): raise RuntimeError('ffmpeg failed')
print(OUTPUT,flush=True)
