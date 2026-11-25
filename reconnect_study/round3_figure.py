"""Draw round-three per-comparison paired effects without altering measured intervals."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT=Path('runs/reconnect_supplement/round3')
GROUPS=[('23','15','23 vs 15 dimensions'),('23','BoostTrack','23 dimensions vs BoostTrack-v1 score'),
        ('15','12','15 vs 12 dimensions')]


def main():
    with (ROOT/'paired-effects.csv').open(encoding='utf-8-sig') as file:
        data=list(csv.DictReader(file))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,
                         'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    positions=[]
    for left,right,title in GROUPS:
        rows=[r for r in data if (r['left'],r['right'])==(left,right)]
        if rows:
            positions.append((title,None))
            positions.extend(('',r) for r in rows)
    height = 1.7 + len(positions) * .285
    fig,axes=plt.subplots(1,3,figsize=(7.85,height),
                          gridspec_kw={'width_ratios':[1.48,1,1]},sharey=True)
    fig.subplots_adjust(left=.025,right=.98,top=1-.5/height,bottom=.95/height,wspace=.18)
    labels,a,b=axes
    labels.set_xlim(0,1)
    labels.axis('off')
    for ax,title in ((a,'a   Action failure'),(b,'b   Natural-gap recovery')):
        ax.set_title(title,loc='left',fontweight='bold',fontsize=9,pad=8)
        ax.axvline(0,color='#737a80',lw=.8,ls='--',zorder=0)
        ax.spines['left'].set_visible(False)
        ax.set_xlabel('Difference (percentage points)',fontsize=7.5)
        ax.tick_params(axis='y',left=False,labelleft=False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(axis='x',color='#e7e9eb',lw=.4,zorder=-2)
    labels.set_title('Tracker / setting                   Accepts L / R',loc='left',fontweight='bold',fontsize=8,pad=8)
    for i,(title,row) in enumerate(positions):
        if row is None:
            labels.text(0,i,title,fontsize=7.9,fontweight='bold',va='center')
            if i:
                for ax in axes: ax.axhline(i-.62,color='#d0d5d9',lw=.6)
            continue
        tracker=row['tracker'].replace(' (motion)','')
        labels.text(.018,i,f"{tracker} · {row['setting']}",va='center',fontsize=7.7)
        if row['calibration_matched']!='True':
            labels.text(.98,i,'unmatched',ha='right',va='center',fontsize=7,color='#666666')
            continue
        labels.text(.98,i,f"{row['left_accepted']} / {row['right_accepted']}",ha='right',va='center',fontsize=7.6)
        fill='#175a76' if row['test_spread_within_tolerance']=='True' else 'white'
        for ax,metric in ((a,'failure'),(b,'recovery')):
            lo,hi=float(row[metric+'_lo']),float(row[metric+'_hi'])
            ax.hlines(i,lo,hi,color='#175a76',lw=1.15)
            ax.scatter(float(row[metric+'_pp']),i,s=21,facecolor=fill,edgecolor='#175a76',lw=1,zorder=3)
    labels.set_ylim(len(positions)-.45,-.55)
    fig.text(.028,.38/height,'Points show observed differences (left minus right). Bars show 95% source-paired bootstrap intervals.',fontsize=7.2)
    fig.text(.028,.25/height,'Filled markers indicate test acceptances within the pairwise tolerance. BoT-SORT uses motion only.',fontsize=7.2)
    fig.text(.028,.12/height,'BoostTrack-v1 uses host states. The BoT-SORT adapter uses a first-order covariance approximation.',fontsize=7.2)
    for ext in ('pdf','svg','png'):
        fig.savefig(ROOT/f'main-effects.{ext}',dpi=450,facecolor='white',bbox_inches='tight',pad_inches=.08)
    plt.close(fig)


if __name__=='__main__':
    main()
