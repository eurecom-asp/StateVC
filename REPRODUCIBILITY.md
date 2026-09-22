# ICASSP release lock

The public defaults are intentionally tied to the final StateVC paper setup:

- WavLM-Large layer 6 / D=1024
- source-derived variance ordering
- routing dimension d_c=24
- pair-specific shared diagonal GMM
- K candidates 1..4
- n_min=20 hard pooled-frame support for BIC candidate validity
- GMM: random_state=0, n_init=3, max_iter=100, reg_covar=1e-4, init_params=kmeans
- five-frame posterior smoothing (J=2), edge replication + state renormalization
- m_min=20 posterior-mass threshold in each utterance
- global reference-minus-source mean fallback
- final operator: Mean
- final routing: soft
- lambda=1
- Block analysis: source-variance-ranked full D, contiguous block size 2, Bures/MKL Gaussian covariance map
- synthesis: frozen prematched kNN-VC HiFi-GAN

Do not replace pooled GMM fitting with equal-duration sampling or frame matching.
Do not apply the final Mean transport only to the top-24 coordinates.
Do not describe Block as the final StateVC operator; it is an ablation/analysis operator.

- kNN-VC code pin: c616845c4e309e24d5927f15adbdf277a3d65358
