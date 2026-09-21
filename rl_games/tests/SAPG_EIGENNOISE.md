# Independent SAPG eigennoise scales

With `fixed_sigma: coef_cond`, additive eigennoise now has one K-vector of
learnable log standard deviations per SAPG group, matching the conditional
IID scales. The fixed eigenbasis remains shared. Sampling, Gaussian entropy,
likelihood, rollout snapshots and adaptive KL all use the scales selected by
the sample's current group identifier, including relabeled experience.

Follower entropy gradients no longer directly change the leader's eigen
covariance. Single-policy (`fixed`) models retain their K-vector. Existing
shared-scale SAPG checkpoints have an incompatible parameter shape and are
rejected; there is no implicit conversion. Use fresh scratch runs or an
explicitly designed migration in a separate experiment.

The existing aggregate noise diagnostics average across groups. Additional
`info/noise_eigadd_{eigen_rms,joint_rms,eigen_joint_ratio}/block_N` metrics
report each group's covariance budget separately. RMS includes every action
channel, including any zero eigen columns for arms.

CPU contracts: `PYTHONPATH=rl_games python rl_games/tests/test_eigen_kl.py`.
They check dense Gaussian likelihood/entropy/KL agreement, selected-group
sampling, gradient isolation, checkpoint rejection, native builder row
initialization and recurrent/feedforward PPO reference updates.
