# Goal-Oriented Influence-Maximizing Data Acquisition for Learning and Optimization

Active data acquisition is central to many learning and optimization tasks with deep
neural networks, yet remains challenging because most approaches rely on predictive uncer-
tainty estimates that are difficult to obtain reliably. To this end, we propose Goal-Oriented
Influence-Maximizing Data Acquisition (GOIMDA), an active acquisition algorithm that by-
passes explicit uncertainty estimation. GOIMDA selects inputs by maximizing their expected
influence on a user-specified goal functional, such as test loss, predictive entropy, or the
value of an optimizer-recommended design. Leveraging first-order influence functions, we
derive a tractable acquisition rule that combines the goal gradient, training-loss curvature,
and candidate sensitivity to model parameters. We show theoretically that, for generalized
linear models, GOIMDA approximates predictive-entropy minimization up to a correction
term accounting for goal alignment and prediction bias, thereby implicitly capturing uncer-
tainty without maintaining a Bayesian posterior. Empirically, across learning tasks (including
image and text classification) and optimization tasks (including noisy global optimization
benchmarks and neural-network hyperparameter tuning), GOIMDA consistently reaches
target performance with substantially fewer labeled samples or function evaluations than
uncertainty-based active learning and Gaussian-process Bayesian optimization baselines.