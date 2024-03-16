def desicionBoundaries(model, samples, labels, offset=0.1, res=0.01, set_xlabel=None, set_ylabel=None, ax=None):
  import numpy as np
  import matplotlib.pyplot as plt

  offset, res = offset, res;
  h_min, h_max = samples[:, 0].min()-offset, samples[:, 0].max()+offset;
  v_min, v_max = samples[:, 1].min()-offset, samples[:, 1].max()+offset;

  h_grid, v_grid = np.meshgrid(np.arange(h_min, h_max, res), np.arange(v_min, v_max, res));

  print(f'--> h_grid: {h_grid.shape}\n--> v_grid: {v_grid.shape}\n--> h_grid_ravel: {h_grid.ravel().shape}\n--> v_grid_ravel: {v_grid.ravel().shape}\n--> h_grid + v_grid: {np.c_[h_grid.ravel(), v_grid.ravel()].shape}')

  pred_grid = model.predict(np.c_[h_grid.ravel(), v_grid.ravel()])
  print(pred_grid.shape)

  pred_grid = pred_grid.reshape(h_grid.shape)
  print(pred_grid.shape)

  ax.pcolormesh(h_grid, v_grid, pred_grid, cmap="Paired")
  ax.scatter(samples[:, 0], samples[:, 1], c=labels, edgecolor="k", cmap="Paired")
  ax.set_xlabel(set_xlabel)
  ax.set_ylabel(set_ylabel)