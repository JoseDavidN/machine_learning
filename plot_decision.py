def plotDecision(offset, res, model, muestras, labels, feature_names):
  h_min, h_max = muestras[:, 0].min()-offset, muestras[:, 0].max()+offset;
  v_min, v_max = muestras[:, 1].min()-offset, muestras[:, 1].max()+offset;

  h_grid, v_grid = np.meshgrid(np.arange(h_min, h_max, res), np.arange(v_min, v_max, res));

  print(h_grid.shape, v_grid.shape, h_grid.ravel().shape, v_grid.ravel().shape)
  print(np.c_[h_grid.ravel(), v_grid.ravel()].shape)

  pred_grid = model.predict(np.c_[h_grid.ravel(), v_grid.ravel()])
  print(pred_grid.shape)

  pred_grid = pred_grid.reshape(h_grid.shape)
  print(pred_grid.shape)

  _, ax = plt.subplots(figsize=(8,5))
  ax.pcolormesh(h_grid, v_grid, pred_grid, cmap="Paired")
  ax.scatter(muestras[:, 0], muestras[:, 1], c=labels, edgecolor="k", cmap="Paired")
  ax.set_xlabel(feature_names[0])
  ax.set_ylabel(feature_names[1])