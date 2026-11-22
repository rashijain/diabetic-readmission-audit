import warnings, pickle, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.calibration import calibration_curve
sns.set_theme(style="whitegrid", context="notebook")
df=pd.read_pickle("cohort.pkl"); y=df.target.values
OOF=pickle.load(open("outputs/oof.pkl","rb"))
mn=pd.read_csv("outputs/mnar_diagnostics.csv"); r=pd.read_csv("outputs/all_results.csv")
sub=pd.read_csv("outputs/subgroup_audit.csv")

# FIG 1 - MNAR diagnostics
fig,ax=plt.subplots(1,2,figsize=(13,4.6))
m=mn.sort_values("Missing_pct")
sns.barplot(data=m,y="Feature",x="Missing_pct",hue="Feature",palette="viridis",legend=False,ax=ax[0])
ax[0].set_title("A. Missingness prevalence",fontweight="bold"); ax[0].set_xlabel("Missing (%)"); ax[0].set_ylabel("")
ax[0].set_xlim(0,112)
for i,v in enumerate(m.Missing_pct): ax[0].text(v+1.5,i,f"{v:.1f}%",va="center",fontsize=9)
pm=m.melt(id_vars=["Feature","chi2_p"],value_vars=["Readmit_measured","Readmit_missing"],var_name="Status",value_name="Rate")
pm.Status=pm.Status.map({"Readmit_measured":"Measured / documented","Readmit_missing":"Missing / not ordered"})
sns.barplot(data=pm,y="Feature",x="Rate",hue="Status",palette="Set2",ax=ax[1])
ax[1].set_title("B. Informative missingness: 30-day readmission rate",fontweight="bold")
ax[1].set_xlabel("30-day readmission rate (%)"); ax[1].set_ylabel(""); ax[1].legend(loc="upper left",fontsize=8,framealpha=.9)
for i,f in enumerate(m.Feature):
    p=m[m.Feature==f].chi2_p.iloc[0]
    ax[1].text(13.6,i,"n.s." if p>=0.05 else f"p={p:.0e}",va="center",fontsize=8,style="italic",color="#444")
ax[1].set_xlim(0,16)
plt.tight_layout(); plt.savefig("outputs/fig1_mnar.png",dpi=200,bbox_inches="tight"); plt.close()

# FIG 2 - leakage + regimes
fig,ax=plt.subplots(1,2,figsize=(13,4.6))
lk=r[r.Regime=="explicit_mnar"].copy()
lk.Split=lk.Split.map({"leaky_encounter":"Encounter-level (leaky)","grouped_patient":"Patient-grouped"})
sns.barplot(data=lk,x="Model",y="ROC_AUC",hue="Split",palette=["#d95f02","#1b9e77"],ax=ax[0])
for c in ax[0].containers: ax[0].bar_label(c,fmt="%.4f",fontsize=8,padding=2)
ax[0].set_ylim(0.60,0.70); ax[0].set_ylabel("5-fold mean ROC-AUC")
ax[0].set_title("A. Leakage ablation — no measurable inflation",fontweight="bold"); ax[0].legend(title="",fontsize=8)
gp=r[r.Split=="grouped_patient"].copy()
gp.Regime=pd.Categorical(gp.Regime,["drop_mnar","standard_impute","explicit_mnar"],ordered=True)
sns.barplot(data=gp.sort_values("Regime"),x="Model",y="PR_AUC",hue="Regime",palette="Blues",ax=ax[1])
for c in ax[1].containers: ax[1].bar_label(c,fmt="%.4f",fontsize=8,padding=2)
ax[1].set_ylim(0.19,0.25); ax[1].set_ylabel("5-fold mean PR-AUC")
ax[1].set_title("B. Missingness regimes (patient-grouped)",fontweight="bold"); ax[1].legend(title="",fontsize=8)
plt.tight_layout(); plt.savefig("outputs/fig2_leakage_regimes.png",dpi=200,bbox_inches="tight"); plt.close()

# FIG 3 - subgroup fairness
s=sub[(sub.Attribute=="race_grp")&(sub.Regime=="explicit_mnar")].sort_values("FNR_at_15")
fig,ax=plt.subplots(1,2,figsize=(13,4.6))
sns.barplot(data=s,y="Subgroup",x="FNR_at_15",hue="Subgroup",palette="Reds",legend=False,ax=ax[0])
for i,v in enumerate(s.FNR_at_15): ax[0].text(v+.01,i,f"{v:.3f}",va="center",fontsize=9)
ax[0].set_xlim(0,1.0); ax[0].set_xlabel("False negative rate at 15% threshold"); ax[0].set_ylabel("")
ax[0].set_title("A. Missed readmissions by recorded race",fontweight="bold")
h=sub[sub.Attribute=="race_grp"].pivot(index="Subgroup",columns="Regime",values="ECE")[["drop_mnar","standard_impute","explicit_mnar"]]
sns.heatmap(h,annot=True,fmt=".4f",cmap="YlOrRd",ax=ax[1],cbar_kws={"label":"ECE"})
ax[1].set_title("B. Expected calibration error by subgroup",fontweight="bold"); ax[1].set_xlabel(""); ax[1].set_ylabel("")
plt.tight_layout(); plt.savefig("outputs/fig3_subgroup.png",dpi=200,bbox_inches="tight"); plt.close()

# FIG 4 - decision curve analysis
def nb(yt,yp,ts):
    n=len(yt); out=[]
    for t in ts:
        pr=(yp>=t).astype(int)
        tp=((pr==1)&(yt==1)).sum(); fp=((pr==1)&(yt==0)).sum()
        out.append(tp/n-(fp/n)*(t/(1-t)))
    return np.array(out)
prev=y.mean(); ts=np.linspace(0.02,0.35,67)
plt.figure(figsize=(8.6,5))
plt.plot(ts*100,prev-(1-prev)*(ts/(1-ts)),color="gray",ls="--",label="Intervene on all")
plt.plot(ts*100,np.zeros_like(ts),color="black",ls=":",label="Intervene on none")
cols={"drop_mnar":"#7570b3","standard_impute":"#d95f02","explicit_mnar":"#1b9e77"}
labs={"drop_mnar":"A: drop MNAR features","standard_impute":"B: mode imputation","explicit_mnar":"C: explicit MNAR indicators"}
for k in ["drop_mnar","standard_impute","explicit_mnar"]:
    plt.plot(ts*100,nb(y,OOF[(k,"grouped_patient")]["HistGBDT"],ts),lw=2.2,color=cols[k],label=labs[k])
plt.title("Decision curve analysis — net clinical benefit (HistGBDT, patient-grouped OOF)",fontweight="bold")
plt.xlabel("Intervention threshold (% 30-day readmission risk)"); plt.ylabel("Net clinical benefit")
plt.ylim(-0.02,prev+0.01); plt.xlim(2,35); plt.legend(); plt.tight_layout()
plt.savefig("outputs/fig4_dca.png",dpi=200,bbox_inches="tight"); plt.close()
print("figures written")
for k in ["drop_mnar","standard_impute","explicit_mnar"]:
    v=nb(y,OOF[(k,"grouped_patient")]["HistGBDT"],np.array([0.15]))[0]
    print(f"  net benefit @15% threshold, {k:16s}: {v:.5f}")
print(f"  net benefit @15%, intervene-on-all      : {prev-(1-prev)*(0.15/0.85):.5f}")
