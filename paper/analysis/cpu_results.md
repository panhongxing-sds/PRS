# CPU results (N=8121, join=8121)

Weight: base=0.594 wt=0.745
Spearman bd~hes: 0.429
SE-consistent wrong n=0 P(bd>0)=nan% (summary scan)
## Ablation
- DH-Score (full): AUROC=0.904 AUPRC=0.875
- w/o Dissent: AUROC=0.704 AUPRC=0.622
- w/o Hesitation: AUROC=0.897 AUPRC=0.850
- w/o Text Pert.: AUROC=0.862 AUPRC=0.789
- w/o Weight Pert.: AUROC=0.871 AUPRC=0.829
- w/o Fusion: AUROC=0.877 AUPRC=0.877
- w/o Prox.: AUROC=0.896 AUPRC=0.869

## Collapse LODO
- all n=8121: PANDA=0.904 bd=0.897 hes=0.704 SE=0.890
- bd0 n=2960: PANDA=0.623 bd=0.500 hes=0.623 SE=0.708
- bd0_wrong n=96: PANDA=nan bd=nan hes=nan SE=nan
- se_cons_wrong n=0
