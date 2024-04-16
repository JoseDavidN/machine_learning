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
    
def viewMetrics(predictions, dataTest, name, viewAccuracy=False, viewPrecision=False, viewRecall=False, viewF1=False, viewCm=False, ax=None):
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    import seaborn as sns
    
    accuracy = accuracy_score( dataTest, predictions[name] );
    precision = precision_score( dataTest, predictions[name], average="micro" );
    recall = recall_score( dataTest, predictions[name], average="micro" );
    f1 = f1_score( dataTest, predictions[name], average="micro" );
    cm = confusion_matrix( dataTest, predictions[name] );

    print(f"#--------- Resultados de {name} ---------#")
    resultsTitle = ''
    results = ''

    if viewAccuracy:
        resultsTitle += '-- Acc --\t'
        results += "  {0:.3f} \t".format( accuracy )
    if viewPrecision:
        resultsTitle += '-- Prec --\t'
        results += "  {0:.3f} \t".format( precision )
    if viewRecall:
        resultsTitle += '-- Rec --\t'
        results += "  {0:.3f} \t".format( recall )
    if viewF1:
        resultsTitle += '-- F1 --\t'
        results += "  {0:.3f} \t".format( f1 )
    if viewCm:
        sns.heatmap( cm, cmap='hot', annot=True, ax=ax );
        ax.set_title( name );

    print( resultsTitle )
    print( results )

def segmentation_image(path_image, model='dbscan'):
    import numpy as np
    from matplotlib import pyplot as plt
    
    if model=='dbscan':
        import cv2
        from sklearn.cluster import DBSCAN
        print('###----- Segmentation with DBSCAN -----###')
        
        #Load image
        original_image = cv2.imread(path_image)
        print(f'Image Shape original: {original_image.shape}')
        
        #Convert image to RGB and resize
        original_image = cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
        original_image = cv2.resize(original_image,(150,100))
        print(f'Image Shape resize: {original_image.shape}')
        
        #vectorize image
        img_hsv =cv2.cvtColor(original_image,cv2.COLOR_BGR2HSV)
        vectorized_hsv =img_hsv[:,:,0].reshape(-1,1)
        print(f'Image Shape Vectorized: {vectorized_hsv.shape}')
        
        #parameters for DBSCAN
        eps=abs(img_hsv[:,:,0].min()-img_hsv[:,:,0].max())*0.01
        min_samples=int(len(vectorized_hsv)*0.01)
        print(f'eps: {eps} min_samples: {min_samples}')
        
        #DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        predictions = dbscan.fit_predict(vectorized_hsv)
        print(f'Predictions shape: {predictions.shape}')
        
        #Reshape predictions
        predictions=predictions.reshape(original_image.shape[:2])
        
        #Plot
        _, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(original_image),axes[0].set_axis_off(),axes[0].set_title('DBSCAN original_image')
        axes[1].imshow(predictions, cmap='Paired'),axes[1].set_axis_off(),axes[1].set_title('DBSAN Segmented Image')
    elif model=='meanshift':
        from PIL import Image
        from sklearn.cluster import MeanShift
        print('###----- Segmentation with MeanShift -----###')
        
        #Load image
        original_image = Image.open(path_image)
        print(f'Image Size original: {original_image.size}')
        
        #Convert image to RGB, HSV and resize
        original_image = original_image.resize((150,100))
        print(f'Image Size resize: {original_image.size}')
        img_rgb = np.array(original_image.convert('RGB'))
        img_hsv = np.array(original_image.convert('HSV'))
        print(f'Image RGB Shape: {img_rgb.shape}, Image HSV Shape: {img_hsv.shape}')
        
        #view image
        _, axes = plt.subplots (2, 2, figsize = (12,5))
        axes[0,0].imshow( img_rgb ), axes[0,0].set_axis_off(), axes[0,0].set_title("Meanshift RGB Image") #RGB
        axes[0,1].imshow( img_hsv ), axes[0,1].set_axis_off(), axes[0,1].set_title("MeanShift HSV Image") #HSV
        plt.tight_layout()
        
        #vectorize image
        vectorized_hsv = img_hsv[:,:,0].reshape(-1,1)
        vectorized_hsv = np.float32(vectorized_hsv)
        print(f'Image Shape Vectorized: {vectorized_hsv.shape}')
        
        #parameters for MeanShift with HSV
        abs(vectorized_hsv.min()-vectorized_hsv.max())
        ms = MeanShift(bandwidth=8,max_iter = 20).fit(vectorized_hsv)
        
        #Predictions
        clustered_hsv = ms.predict(vectorized_hsv)
        clustered_hsv = clustered_hsv.reshape(img_hsv.shape[:2])
        print(f'Clustered Shape: {clustered_hsv.shape}')
        
        #Plot
        axes[1,0].imshow( img_rgb ), axes[1,0].set_axis_off(), axes[1,0].set_title("MeanShift Original")
        axes[1,1].imshow( clustered_hsv,cmap='spring' ), axes[1,1].set_axis_off(), axes[1,1].set_title("MeanShift Segmentation mask")
        plt.tight_layout()
    else:
        print('###----- Model not found -----###')